"""Reproducible task banks with reflected maps, start positions and obstacle phases."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.base import BaseEnv
from env.robotaxi import RobotaxiEnv


SPLIT_IDS = {'train': 0, 'validation': 1, 'test': 2}


def position_partition(positions, map_shape):
    """Assign reflection orbits to fixed 60/20/20 buckets, independently of RNG.

    Folding coordinates ensures that a mirrored training start cannot appear as
    a validation/test start. These are spatial pools, not just different seeds.
    """
    h, w = map_shape
    doubled = np.rint(2*np.asarray(positions)).astype(np.int64)
    folded = np.minimum(doubled, 2*np.array([w-1, h-1])-doubled)
    bucket = ((folded[..., 0]*73856093) ^ (folded[..., 1]*19349663)) % 10
    return np.where(bucket < 6, 0, np.where(bucket < 8, 1, 2))


def start_candidates(base, params, split='legacy'):
    if split not in ('legacy', *SPLIT_IDS):
        raise ValueError('Unknown task split')
    h, w = params.map_height_width
    xy = np.stack(np.meshgrid(np.arange(1.5, w-1, .5),
                             np.arange(1.5, h-1, .5)), axis=-1).reshape(-1, 2)
    distance = np.linalg.norm(xy[:, None]-np.asarray(base.static_obstacles)[None], axis=-1).min(1)
    valid = (distance > 2.1) & (np.linalg.norm(xy-np.asarray(base.goal_pos), axis=-1) > 2.5)
    if split != 'legacy':
        valid &= position_partition(xy, params.map_height_width) == SPLIT_IDS[split]
    return xy[valid]


def make_bank(map_id, seed, size=128, max_steps=1800, mirrors=True, navigation='teacher', split='legacy', allow_reverse=False):
    p, base = RobotaxiEnv.init_params(jax.random.PRNGKey(seed), map_id=map_id,
                                      max_steps=max_steps, path_length=60, allow_reverse=allow_reverse)
    h, w = p.map_height_width
    # Free starts have extra clearance for a finite turning radius. Some courses
    # can still be difficult; this does not claim that every sampled task is solvable.
    candidates = start_candidates(base, p, split)
    if not len(candidates):
        raise ValueError(f'No free start candidates on map {map_id}')
    candidates = jnp.asarray(candidates, dtype=jnp.float32)

    def variant(key):
        phase_key, position_key, original_key, reflection_key = jax.random.split(key, 4)
        phases = jax.random.uniform(phase_key, (32,), minval=0., maxval=12.)
        moving = jnp.mod(base.kinematic_obstacles[None] + phases[:, None, None]
                         * base.kinematic_obst_velocities[None], jnp.array([w, h]))
        index = jax.random.randint(position_key, (), 0, len(candidates))
        # Original starts are benchmarks in their own group. Strict validation/test
        # variants contain only spatially held-out starts, never the training start.
        original = jax.random.bernoulli(original_key, .35 if split in ('legacy','train') else 0.)
        position = jnp.where(original, base.agent_pos, candidates[index])
        clearances = jnp.min(jnp.linalg.norm(moving - position, axis=-1), axis=-1)
        # Select the first safe obstacle phase; reject a bank with no valid candidate below.
        safe = clearances > 1.3
        moving = moving[jnp.argmax(safe)]
        signs = jnp.where(jax.random.bernoulli(reflection_key, .5, (2,)), -1., 1.) if mirrors else jnp.ones(2)
        center = jnp.array([w - 1, h - 1]) / 2
        reflect = lambda x: center + signs * (x - center)
        heading_vec = signs * base.agent_forward_dir
        state = eqx.tree_at(
            lambda s: (s.agent_pos, s.goal_pos, s.static_obstacles,
                       s.kinematic_obstacles, s.kinematic_obst_velocities,
                       s.agent_forward_dir, s.heading), base,
            (reflect(position), reflect(base.goal_pos), reflect(base.static_obstacles),
             jnp.mod(reflect(moving), jnp.array([w, h])),
             signs * base.kinematic_obst_velocities, heading_vec,
             jnp.arctan2(heading_vec[1], heading_vec[0])))
        path = BaseEnv._find_path(state.agent_pos, state.goal_pos,
                                 jnp.concatenate([state.static_obstacles, state.kinematic_obstacles]), p)
        return eqx.tree_at(lambda s: s.path_array, state, path), original, jnp.any(safe)

    states, originals, safe = jax.jit(jax.vmap(variant))(
        jax.random.split(jax.random.PRNGKey(seed), size))
    if not bool(jnp.all(safe)):
        raise ValueError('Could not sample collision-free initial obstacle phases')
    if navigation == 'clearance':
        from rl.navigation import attach_navigation
        states = attach_navigation(p, states, batched=True)
    elif navigation != 'teacher':
        raise ValueError('Unknown navigation mode')
    return p, states, originals


def sample_task(key, bank, originals, varied_heading=True):
    index_key, angle_key = jax.random.split(key)
    index = jax.random.randint(index_key, (), 0, len(originals))
    state = jax.tree_util.tree_map(lambda a: a[index], bank)
    if varied_heading:
        angle = jax.random.uniform(angle_key, (), minval=-jnp.pi, maxval=jnp.pi)
        heading = jnp.where(originals[index], state.heading + .15 * angle, angle)
        state = eqx.tree_at(lambda s: (s.heading, s.agent_forward_dir), state,
                            (heading, jnp.array([jnp.cos(heading), jnp.sin(heading)])))
    return state


def original_task(map_id, max_steps=1800, heading_offset=0., navigation='teacher', allow_reverse=False):
    p, state = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id,
                                      max_steps=max_steps, path_length=60, allow_reverse=allow_reverse)
    heading = state.heading + heading_offset
    state = eqx.tree_at(lambda s: (s.heading, s.agent_forward_dir), state,
                       (heading, jnp.array([jnp.cos(heading), jnp.sin(heading)])))
    if navigation == 'clearance':
        from rl.navigation import attach_navigation
        state = attach_navigation(p, state)
    elif navigation != 'teacher':
        raise ValueError('Unknown navigation mode')
    return p, state
