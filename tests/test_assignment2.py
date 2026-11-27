import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv
from rl.ppo import Config, act, advantages, distribution, init_model, log_probability, make_update, save_checkpoint, load_checkpoint
from rl.rollout import history_reset, history_step, make_runner
from rl.tasks import make_bank, sample_task


class PPOTests(unittest.TestCase):
    def test_gae_bootstraps_timeout_but_not_terminal_and_does_not_leak_after_reset(self):
        r = jnp.array([[1., 1.], [100., 100.]])
        v = jnp.array([[2., 2.], [0., 0.]])
        nv = jnp.array([[10., 10.], [0., 0.]])
        term = jnp.array([[True, False], [True, True]])
        done = jnp.ones((2, 2), dtype=bool)
        adv, returns = advantages(r, v, nv, term, done, .9, .95)
        np.testing.assert_allclose(adv[0], [-1., 8.])
        np.testing.assert_allclose(returns[0], [1., 10.])

    def test_tanh_density_and_saturated_actions_have_finite_gradients(self):
        mu, logstd, z = jnp.array([.2, -.1]), jnp.array([-.5, .2]), jnp.array([.7, -.3])
        normal = -.5 * (((np.array(z) - mu) / jnp.exp(logstd)) ** 2 + 2 * logstd + jnp.log(2 * jnp.pi))
        expected = jnp.sum(normal - jnp.log(1 - jnp.tanh(z) ** 2))
        self.assertAlmostEqual(float(log_probability(mu, logstd, z)), float(expected), places=5)
        self.assertTrue(np.isfinite(jax.grad(lambda x: log_probability(mu, logstd, x))(jnp.array([100., -100.]))).all())

    def test_update_changes_parameters_and_checkpoint_roundtrip(self):
        cfg = Config(hidden=16, epochs=2, minibatch_size=8)
        key = jax.random.PRNGKey(3)
        model = init_model(key, 12, cfg.hidden)
        observations = jax.random.normal(key, (16, 12))
        action, latent, logp, values = act(model, observations, key)
        self.assertTrue((np.abs(action) < 1).all())
        batch = (observations, latent, logp, values, jnp.linspace(-1, 1, 16), values + 1)
        optimizer, update = make_update(cfg)
        trained, _, metrics, _ = update(model, optimizer.init(model), batch, key)
        self.assertTrue(np.isfinite(metrics).all())
        self.assertFalse(np.array_equal(model['actor'][-1]['w'], trained['actor'][-1]['w']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.npz'
            save_checkpoint(path, trained, cfg)
            restored, restored_cfg = load_checkpoint(path)
            np.testing.assert_array_equal(act(trained, observations, key, True)[0], act(restored, observations, key, True)[0])
            self.assertEqual(cfg, restored_cfg)

    def test_banks_reproducible_safe_diverse_and_rollouts_finite(self):
        cfg = Config(num_envs=2, rollout_steps=4, history=4, max_steps=12, bank_size=16, hidden=16)
        p, bank, original = make_bank(0, 5, cfg.bank_size, cfg.max_steps)
        _, again, _ = make_bank(0, 5, cfg.bank_size, cfg.max_steps)
        np.testing.assert_array_equal(bank.agent_pos, again.agent_pos)
        self.assertGreater(len(np.unique(np.array(bank.agent_pos), axis=0)), 1)
        keys = jax.random.split(jax.random.PRNGKey(123), 64)
        states = jax.vmap(lambda k: sample_task(k, bank, original))(keys)
        collision, goal = jax.vmap(lambda s: RobotaxiEnv._events(s, p))(states)
        self.assertFalse(np.asarray(collision | goal).any())
        self.assertGreater(float(jnp.std(states.heading)), 1.)
        initialize, collect = make_runner(p, bank, original, cfg)
        carry = initialize(keys[0], bank, original)
        model = init_model(keys[1], 43 * cfg.history, cfg.hidden)
        carry, batch, metrics = collect(model, carry, keys[2], bank, original)
        self.assertEqual(batch[0].shape, (8, 172))
        self.assertTrue(all(np.isfinite(a).all() for a in batch))
        self.assertGreater(float(metrics[0]), 0)
        self.assertTrue(np.isfinite(carry[1]).all())

    def test_history_resets_without_previous_episode(self):
        h = history_reset(jnp.array([1., 2.]), 4)
        h = history_step(h, jnp.array([3., 4.]))
        np.testing.assert_array_equal(h[-1], [3., 4.])
        reset = history_reset(jnp.array([9., 8.]), 4)
        np.testing.assert_array_equal(reset, [[9., 8.]] * 4)

    def test_idle_metrics_exclude_absorbing_states(self):
        import equinox as eqx
        from assignment2 import make_episode
        from rl.tasks import original_task
        cfg = Config(hidden=16, max_steps=60)
        params, initial = original_task(1, cfg.max_steps)
        model = init_model(jax.random.PRNGKey(0), 172, cfg.hidden)
        # Exactly neutral policy: vehicle remains stationary away from the goal.
        model['actor'] = jax.tree_util.tree_map(jnp.zeros_like, model['actor'])
        run = make_episode(params, cfg)
        batch = jax.tree_util.tree_map(lambda x: x[None], initial)
        result = run(model, batch)
        self.assertTrue(bool(result['timeout'][0]))
        self.assertAlmostEqual(float(result['max_idle_seconds'][0]), .9, places=5)
        # Already at the goal: all subsequent scan steps are absorbing, never idle.
        at_goal = eqx.tree_at(lambda s: s.agent_pos, initial, initial.goal_pos)
        result = run(model, jax.tree_util.tree_map(lambda x: x[None], at_goal))
        self.assertTrue(bool(result['success'][0]))
        self.assertEqual(float(result['max_idle_seconds'][0]), 0.)
        self.assertEqual(float(result['idle_fraction'][0]), 0.)

    def test_checkpoint_selection_rejects_held_out_results(self):
        from rl.select_checkpoint import validation_score
        def group(map_id, scenario, success):
            return dict(map_id=map_id, scenario=scenario, success=success,
                        collision=1-success, split='train_geometry')
        result = dict(train_maps=[0,1], groups=[group(0,'original',1.), group(1,'original',0.),
            group(0,'varied',.5), group(1,'varied',.5)])
        self.assertEqual(validation_score(result), (.5,.5))
        result['groups'].append(dict(map_id=2, scenario='varied',success=1.,collision=0.,split='held_out'))
        with self.assertRaises(ValueError):
            validation_score(result)


if __name__ == '__main__':
    unittest.main()
