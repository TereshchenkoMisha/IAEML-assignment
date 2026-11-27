"""Behaviour checks, using only unittest and the teacher's dependencies."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.base import BaseEnv
from env.robotaxi import RobotaxiEnv


class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = jax.random.PRNGKey(0)
        cls.p, cls.initial = RobotaxiEnv.init_params(cls.key, path_length=30, allow_reverse=False)

    def open_state(self):
        s = self.initial
        return eqx.tree_at(lambda z: (z.agent_pos, z.goal_pos, z.static_obstacles,
                                     z.kinematic_obstacles, z.kinematic_obst_velocities,
                                     z.path_array, z.heading, z.agent_forward_dir), s,
                           (jnp.array([8., 4.]), jnp.array([18., 4.]),
                            jnp.full_like(s.static_obstacles, 100.),
                            jnp.full_like(s.kinematic_obstacles, 100.),
                            jnp.zeros_like(s.kinematic_obst_velocities),
                            jnp.stack([jnp.linspace(8., 18., 30), jnp.full(30, 4.)], axis=-1),
                            jnp.float32(0), jnp.array([1., 0.])))

    def step(self, s, a, p=None):
        return RobotaxiEnv.step(self.key, s, jnp.asarray(a, dtype=jnp.float32), p or self.p)

    def test_inherits_and_preserves_maps(self):
        self.assertTrue(issubclass(RobotaxiEnv, BaseEnv))
        for map_id in range(3):
            p, initial = RobotaxiEnv.init_params(self.key, map_id=map_id, path_length=30)
            bp, bs = BaseEnv.init_params(self.key, map_id, 1800, 30, num_ray_sensors=32)
            np.testing.assert_array_equal(initial.static_obstacles, bs.static_obstacles)
            np.testing.assert_array_equal(initial.kinematic_obstacles, bs.kinematic_obstacles)
            obs, _, _, done, _ = self.step(initial, [0, 0], p)
            self.assertFalse(bool(done))
            self.assertEqual(obs.as_vector().shape, (43,))
            self.assertTrue(np.isfinite(obs.as_vector()).all())

    def test_acceleration_braking_and_drag(self):
        s = self.open_state()
        _, powered, _, _, _ = self.step(s, [1, 0])
        self.assertGreater(float(powered.speed), 0)
        self.assertGreater(float(powered.agent_pos[0]), float(s.agent_pos[0]))
        s = eqx.tree_at(lambda z: z.speed, s, jnp.float32(1))
        _, coast, _, _, _ = self.step(s, [0, 0])
        _, brake, _, _, _ = self.step(s, [-1, 0])
        self.assertLess(float(coast.speed), 1)
        self.assertLess(float(brake.speed), float(coast.speed))
        _, stop, _, _, _ = self.step(self.open_state(), [-1, 0])
        self.assertEqual(float(stop.speed), 0)

    def test_steering_and_rate_limit(self):
        s = self.open_state()
        _, stationary, _, _, _ = self.step(s, [0, 1])
        self.assertEqual(float(stationary.heading), 0)
        self.assertAlmostEqual(float(stationary.steering), self.p.steering_rate / self.p.fps, places=6)
        s = eqx.tree_at(lambda z: z.speed, s, jnp.float32(1))
        _, left, _, _, _ = self.step(s, [0, 1])
        _, right, _, _, _ = self.step(s, [0, -1])
        self.assertGreater(float(left.heading), 0)
        self.assertLess(float(right.heading), 0)
        self.assertAlmostEqual(float(left.heading), -float(right.heading), places=6)

    def test_action_clipping_and_nonfinite_inputs(self):
        s = self.open_state()
        a = self.step(s, [1, -1])[1]
        b = self.step(s, [100, -100])[1]
        np.testing.assert_allclose(a.agent_pos, b.agent_pos)
        self.assertTrue(np.isfinite(self.step(s, [float('nan'), float('inf')])[0].as_vector()).all())

    def test_collision_checked_after_motion_and_terminal_is_absorbing(self):
        s = self.open_state()
        obstacles = s.static_obstacles.at[0].set(jnp.array([9.01, 4.]))
        s = eqx.tree_at(lambda z: (z.speed, z.static_obstacles), s, (jnp.float32(3), obstacles))
        _, terminal, r, done, info = self.step(s, [0, 0])
        self.assertTrue(bool(done))
        self.assertTrue(bool(info['collision']))
        self.assertLess(float(r), -19)
        _, again, r2, done2, _ = self.step(terminal, [1, 1])
        np.testing.assert_array_equal(again.agent_pos, terminal.agent_pos)
        self.assertEqual(float(r2), 0)
        self.assertTrue(bool(done2))

    def test_goal_timeout_and_reset(self):
        s = self.open_state()
        s = eqx.tree_at(lambda z: z.agent_pos, s, s.goal_pos - jnp.array([0.5, 0.]))
        _, _, reward, done, info = self.step(s, [0, 0])
        self.assertTrue(bool(done) and bool(info['success']))
        self.assertGreater(float(reward), 19)
        p = eqx.tree_at(lambda z: z.max_steps_in_episode, self.p, 1)
        _, terminal, _, done, info = self.step(self.open_state(), [0, 0], p)
        self.assertTrue(bool(done) and bool(info['truncated']))
        self.assertFalse(bool(info['terminated']))
        obs, reset = RobotaxiEnv.reset(self.key, self.p, self.initial)
        self.assertEqual(int(reset.time), 0)
        self.assertEqual(float(reset.speed), 0)
        self.assertFalse(bool(reset.terminated | reset.truncated))

    def test_bounds_progress_reward_and_speed_limit(self):
        s = eqx.tree_at(lambda z: z.speed, self.open_state(), jnp.float32(3))
        _, moved, progress_reward, _, _ = self.step(s, [1, 0])
        self.assertLessEqual(float(moved.speed), self.p.max_speed)
        idle_reward = self.step(self.open_state(), [0, 0])[2]
        self.assertGreater(float(progress_reward), float(idle_reward))
        outside = eqx.tree_at(lambda z: z.agent_pos, s, jnp.array([-0.1, 4.]))
        self.assertTrue(bool(self.step(outside, [0, 0])[4]['collision']))

    def test_duplicate_path_and_jit_vmap(self):
        s = self.open_state()
        s = eqx.tree_at(lambda z: z.path_array, s, jnp.broadcast_to(s.goal_pos, s.path_array.shape))
        self.assertTrue(np.isfinite(RobotaxiEnv.get_observation(s, self.p).as_vector()).all())
        actions = jnp.array([[1., 0.], [0., 1.], [-1., -1.]])
        batch = jax.jit(jax.vmap(lambda a: RobotaxiEnv.step(self.key, s, a, self.p)))(actions)
        self.assertEqual(batch[0].as_vector().shape, (3, 43))
        self.assertTrue(np.isfinite(batch[0].as_vector()).all())
        np.testing.assert_allclose(batch[1].agent_pos[0], self.step(s, actions[0])[1].agent_pos)

    def test_one_second_consistent_across_frame_rates(self):
        results = []
        for fps in (30, 60, 120):
            p = eqx.tree_at(lambda z: (z.fps, z.step_size), self.p, (fps, 1 / fps))
            def body(s, _):
                return RobotaxiEnv.step(self.key, s, jnp.array([1., 0.]), p)[1], None
            final, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=fps))(self.open_state())
            results.append(np.array([*final.agent_pos, final.speed]))
        np.testing.assert_allclose(results[0], results[2], atol=0.003, rtol=0)
        np.testing.assert_allclose(results[1], results[2], atol=0.003, rtol=0)

    def test_goal_does_not_override_collision(self):
        s = self.open_state()
        s = eqx.tree_at(lambda z: (z.goal_pos, z.static_obstacles), s,
                        (s.agent_pos, s.static_obstacles.at[0].set(s.agent_pos)))
        _, _, reward, done, info = self.step(s, [0, 0])
        self.assertTrue(bool(done) and bool(info['collision']))
        self.assertFalse(bool(info['success']))
        self.assertLess(float(reward), -19)


if __name__ == '__main__':
    unittest.main()
