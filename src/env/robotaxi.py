"""Assignment 1: bicycle dynamics on the unmodified teacher's BaseEnv maps.

World units are metres, time is seconds. Actions are [acceleration, steering]
normalised to [-1, 1]. Signed speed permits forward and reverse motion. An
opposite longitudinal command brakes to zero before accelerating the other way.
"""
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvParams, BaseEnvState


class RobotaxiParams(BaseEnvParams):
    wheelbase: float = 1.0
    max_speed: float = 3.0
    max_acceleration: float = 2.0
    max_braking: float = 4.0
    max_steering: float = 0.65
    steering_rate: float = 1.5
    rolling_deceleration: float = 0.12
    linear_drag: float = 0.08
    quadratic_drag: float = 0.04
    substeps: int = 4
    allow_reverse: bool = True
    max_reverse_speed: float = 1.5


class RobotaxiState(BaseEnvState):
    speed: jax.Array
    heading: jax.Array
    steering: jax.Array
    previous_action: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class RobotaxiObservation(eqx.Module):
    # These three fields also satisfy the teacher's renderer interface.
    distance_to_path: jax.Array
    direction_of_path: jax.Array
    collision_rays: jax.Array
    vector: jax.Array

    def as_vector(self):
        return self.vector


def body_coordinates(vector, forward):
    left = jnp.array([-forward[1], forward[0]])
    return jnp.stack([vector @ forward, vector @ left], axis=-1)


class RobotaxiEnv(BaseEnv):
    @staticmethod
    def init_params(key, map_id=1, max_steps=1800, path_length=100,
                    discretization_scale=1, perception_radius=5.0,
                    num_ray_sensors=32, fov=jnp.pi, fps=60, **dynamics):
        if fps <= 0 or max_steps <= 0 or path_length < 2 or num_ray_sensors < 1:
            raise ValueError("fps, max_steps and ray count must be positive; path_length >= 2")
        base_params, base_state = BaseEnv.init_params(
            key, map_id, max_steps, path_length, discretization_scale,
            perception_radius, num_ray_sensors, fov, fps,
        )
        params = RobotaxiParams(**vars(base_params), **dynamics)
        positive = (params.wheelbase, params.max_speed, params.max_acceleration,
                    params.max_braking, params.max_steering, params.steering_rate,
                    params.perception_radius, params.max_reverse_speed)
        if (min(positive) <= 0 or not 0 < params.max_steering < jnp.pi / 2
                or not isinstance(params.substeps, int) or params.substeps < 1
                or not isinstance(params.allow_reverse, bool)
                or min(params.rolling_deceleration, params.linear_drag,
                       params.quadratic_drag) < 0):
            raise ValueError("Invalid vehicle parameters")
        state = RobotaxiState(
            **vars(base_state), speed=jnp.float32(0),
            heading=jnp.arctan2(base_state.agent_forward_dir[1],
                                base_state.agent_forward_dir[0]),
            steering=jnp.float32(0), previous_action=jnp.zeros(2),
            terminated=jnp.bool_(False), truncated=jnp.bool_(False),
        )
        return params, state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        # No hidden RNG/global mutable state; the caller keeps the initial state.
        return RobotaxiEnv.get_observation(init_state, env_params), init_state

    @staticmethod
    def path_geometry(state):
        starts, ends = state.path_array[:-1], state.path_array[1:]
        segments = ends - starts
        lengths2 = jnp.sum(segments ** 2, axis=-1)
        t = jnp.clip(jnp.sum((state.agent_pos - starts) * segments, axis=-1)
                     / jnp.maximum(lengths2, 1e-8), 0, 1)
        projections = starts + t[:, None] * segments
        distances = jnp.linalg.norm(projections - state.agent_pos, axis=-1)
        # Repeated goal padding has no tangent. Ignore it if any real segment exists.
        index = jnp.argmin(jnp.where(lengths2 > 1e-8, distances, jnp.inf))
        goal_delta = state.goal_pos - state.agent_pos
        fallback = goal_delta / jnp.maximum(jnp.linalg.norm(goal_delta), 1e-8)
        direction = jnp.where(jnp.any(lengths2 > 1e-8),
                              segments[index] / jnp.sqrt(jnp.maximum(lengths2[index], 1e-8)),
                              fallback)
        offset = jnp.where(jnp.any(lengths2 > 1e-8),
                           projections[index] - state.agent_pos, goal_delta)
        return jnp.linalg.norm(offset), direction, offset

    @staticmethod
    def get_observation(env_state, env_params):
        s, p = env_state, env_params
        distance, direction, offset = RobotaxiEnv.path_geometry(s)
        obstacles = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles])
        rays, lengths = BaseEnv._collision_ray_intersections(
            s.agent_pos, s.agent_forward_dir, obstacles, p)
        # Sensor slots have fixed angles relative to heading, so lengths are ego-centric.
        goal = (s.goal_pos - s.agent_pos) / jnp.linalg.norm(jnp.asarray(p.map_height_width))
        speed_scale = (jnp.where(s.speed < 0, p.max_reverse_speed, p.max_speed)
                       if p.allow_reverse else p.max_speed)
        vector = jnp.concatenate([
            jnp.array([s.speed / speed_scale, s.steering / p.max_steering]),
            body_coordinates(goal, s.agent_forward_dir),
            body_coordinates(offset, s.agent_forward_dir) / p.perception_radius,
            body_coordinates(direction, s.agent_forward_dir),
            s.previous_action, jnp.atleast_1d(s.time / p.max_steps_in_episode),
            lengths / p.perception_radius,
        ]).astype(jnp.float32)
        return RobotaxiObservation(distance, direction, rays * lengths[:, None], vector)

    @staticmethod
    def _events(s, p):
        obstacles = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles])
        h, w = p.map_height_width
        outside = jnp.any(s.agent_pos < 0) | jnp.any(s.agent_pos > jnp.array([w - 1, h - 1]))
        # Preserve the teacher's collision threshold and map geometry.
        collision = BaseEnv._check_collisions(s.agent_pos, obstacles) | outside
        goal = BaseEnv._check_goal(s.agent_pos, s.goal_pos) & ~collision
        return collision, goal

    @staticmethod
    def _advance(key, state, action, p):
        dt = p.step_size / p.substeps
        collision0, goal0 = RobotaxiEnv._events(state, p)

        def integrate(_, carry):
            s, collision, goal = carry
            def move(s):
                target = action[1] * p.max_steering
                steer = s.steering + jnp.clip(target - s.steering,
                                             -p.steering_rate * dt, p.steering_rate * dt)
                if p.allow_reverse:
                    # Integrate magnitude in the current direction. Braking and
                    # passive resistance cannot numerically overshoot zero. Once
                    # stopped, a held command can engage the opposite direction
                    # on the next substep, using propulsion rather than braking.
                    direction = jnp.where(s.speed != 0, jnp.sign(s.speed), jnp.sign(action[0]))
                    opposing = s.speed * action[0] < 0
                    drive = jnp.where(opposing, p.max_braking, p.max_acceleration) * action[0]
                    magnitude = jnp.abs(s.speed)
                    resistance = (p.rolling_deceleration + p.linear_drag * magnitude
                                  + p.quadratic_drag * magnitude ** 2)
                    limit = jnp.where(direction < 0, p.max_reverse_speed, p.max_speed)
                    magnitude = jnp.clip(magnitude + (direction * drive - resistance) * dt, 0, limit)
                    speed = direction * magnitude
                else:
                    # Preserve the exact forward-only dynamics of old experiments.
                    drive = jnp.where(action[0] >= 0, p.max_acceleration, p.max_braking) * action[0]
                    resistance = (p.rolling_deceleration + p.linear_drag * s.speed
                                  + p.quadratic_drag * s.speed ** 2)
                    speed = jnp.clip(s.speed + (drive - resistance) * dt, 0, p.max_speed)
                mean_speed = (s.speed + speed) / 2
                yaw = mean_speed * jnp.tan(steer) / p.wheelbase
                middle_heading = s.heading + yaw * dt / 2
                pos = s.agent_pos + mean_speed * dt * jnp.array([
                    jnp.cos(middle_heading), jnp.sin(middle_heading)])
                heading = jnp.arctan2(jnp.sin(s.heading + yaw * dt),
                                      jnp.cos(s.heading + yaw * dt))
                return eqx.tree_at(
                    lambda z: (z.agent_pos, z.speed, z.heading, z.steering,
                               z.agent_forward_dir, z.kinematic_obstacles), s,
                    (pos, speed, heading, steer,
                     jnp.array([jnp.cos(heading), jnp.sin(heading)]),
                     jnp.mod(s.kinematic_obstacles + dt * s.kinematic_obst_velocities,
                             jnp.array(p.map_height_width[::-1]))))
            s = jax.lax.cond(collision | goal, lambda s: s, move, s)
            c, g = RobotaxiEnv._events(s, p)
            return s, collision | c, goal | g

        new, collision, goal = jax.lax.fori_loop(
            0, p.substeps, integrate, (state, collision0, goal0))
        time = state.time + 1
        truncated = (time >= p.max_steps_in_episode) & ~(collision | goal)
        path = jax.lax.cond(
            (time % p.fps == 0) & ~(collision | goal | truncated),
            lambda _: BaseEnv._find_path(new.agent_pos, new.goal_pos,
                jnp.concatenate([new.static_obstacles, new.kinematic_obstacles]), p),
            lambda _: state.path_array, operand=None)
        new = eqx.tree_at(lambda z: (z.time, z.previous_action, z.path_array,
                                   z.terminated, z.truncated), new,
                          (time, action, path, collision | goal, truncated))
        obs = RobotaxiEnv.get_observation(new, p)
        progress = (jnp.linalg.norm(state.goal_pos - state.agent_pos)
                    - jnp.linalg.norm(new.goal_pos - new.agent_pos))
        # Use the OLD reference path for shaping: replanning must not create reward jumps.
        tracking_state = eqx.tree_at(lambda z: z.path_array, new, state.path_array)
        path_distance, _, _ = RobotaxiEnv.path_geometry(tracking_state)
        obstacles = jnp.concatenate([new.static_obstacles, new.kinematic_obstacles])
        clearance = jnp.min(jnp.linalg.norm(obstacles - new.agent_pos, axis=-1)) - 1.0
        reward = (2.0 * progress - 0.02 * p.step_size
                  - 0.1 * jnp.minimum(path_distance, 3.0) * p.step_size
                  - 0.2 * jnp.maximum(0.5 - clearance, 0) * p.step_size
                  - 0.01 * jnp.sum(action ** 2) * p.step_size
                  - 0.02 * jnp.sum((action - state.previous_action) ** 2)
                  + 20.0 * goal - 20.0 * collision - 2.0 * truncated)
        return obs, new, reward, new.terminated | new.truncated, RobotaxiEnv._info(new, obs, collision, goal)

    @staticmethod
    def _info(s, obs, collision, goal):
        return dict(time=s.time, distance_to_path=obs.distance_to_path,
                    direction_of_path=obs.direction_of_path, speed=s.speed,
                    steering=s.steering, collision=collision, success=goal,
                    terminated=s.terminated, truncated=s.truncated)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        # Non-finite commands are neutralised, then clipped to the action box.
        action = jnp.clip(jnp.nan_to_num(jnp.asarray(action, dtype=jnp.float32),
                                        nan=0.0, posinf=1.0, neginf=-1.0), -1, 1)
        def absorbing(_):
            obs = RobotaxiEnv.get_observation(env_state, env_params)
            c, g = RobotaxiEnv._events(env_state, env_params)
            return obs, env_state, jnp.float32(0), jnp.bool_(True), RobotaxiEnv._info(env_state, obs, c, g)
        return jax.lax.cond(
            env_state.terminated | env_state.truncated, absorbing,
            lambda _: RobotaxiEnv._advance(key, env_state, action, env_params), None)
