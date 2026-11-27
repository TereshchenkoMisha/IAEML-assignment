"""Compiled parallel rollouts with action repeat and terminal-observation bootstrapping."""
import jax
import jax.numpy as jnp
import equinox as eqx

from env.robotaxi import RobotaxiEnv
from rl.ppo import act, value, advantages
from rl.tasks import sample_task


def history_reset(observation, length):
    return jnp.repeat(observation[None], length, axis=0)


def history_step(history, observation):
    return jnp.concatenate([history[1:], observation[None]], axis=0)


def macro_step(key, state, action, params, repeat):
    def body(carry, key):
        state, total = carry
        obs, state, reward, done, info = RobotaxiEnv.step(key, state, action, params)
        if hasattr(state, 'reference_path'):
            state = eqx.tree_at(lambda s:s.path_array,state,state.reference_path)
            obs = RobotaxiEnv.get_observation(state,params)
            info = dict(info,distance_to_path=obs.distance_to_path,
                        direction_of_path=obs.direction_of_path)
        return (state, total + reward), (obs, done, info)
    (state, reward), (observations, done, infos) = jax.lax.scan(
        body, (state, jnp.float32(0)), jax.random.split(key, repeat))
    obs = jax.tree_util.tree_map(lambda x: x[-1], observations)
    info = jax.tree_util.tree_map(lambda x: x[-1], infos)
    return obs, state, reward, done[-1], info


def make_runner(params, bank, originals, cfg):
    # Banks are passed as runtime arguments below, avoiding large XLA constants.
    def reset_one(key, bank, originals):
        state = sample_task(key, bank, originals)
        obs = RobotaxiEnv.get_observation(state, params).as_vector()
        return state, history_reset(obs, cfg.history)

    @jax.jit
    def initialize(key, bank, originals):
        states, histories = jax.vmap(reset_one, in_axes=(0, None, None))(
            jax.random.split(key, cfg.num_envs), bank, originals)
        return states, histories, jnp.zeros(cfg.num_envs), jnp.zeros(cfg.num_envs, dtype=jnp.int32)

    @jax.jit
    def collect(model, carry, key, bank, originals):
        def step(carry, key):
            states, histories, episode_returns, episode_lengths = carry
            action_key, env_key, reset_key = jax.random.split(key, 3)
            observations = histories.reshape(cfg.num_envs, -1)
            actions, latent, logp, values = act(model, observations, action_key)
            # A vmap over state-dependent lax.cond executes both branches, including
            # path planning on EVERY physics step. lax.map retains per-environment
            # branching and only replans when due. Policy evaluation stays batched.
            obs, new_states, rewards, done, infos = jax.lax.map(
                lambda x: macro_step(x[0], x[1], x[2], params, cfg.action_repeat),
                (jax.random.split(env_key, cfg.num_envs), states, actions))
            new_histories = jax.vmap(history_step)(histories, obs.as_vector())
            next_values = value(model, new_histories.reshape(cfg.num_envs, -1))
            total_returns = episode_returns + rewards
            lengths = episode_lengths + 1
            reset_states, reset_histories = jax.vmap(reset_one, in_axes=(0, None, None))(
                jax.random.split(reset_key, cfg.num_envs), bank, originals)
            select = lambda reset, current: jnp.where(done.reshape((cfg.num_envs,) + (1,) * (current.ndim - 1)), reset, current)
            new_states = jax.tree_util.tree_map(select, reset_states, new_states)
            histories = select(reset_histories, new_histories)
            result = (observations, latent, logp, values, rewards, next_values,
                      infos['terminated'], done,
                      jnp.where(done, total_returns, 0), jnp.where(done, lengths, 0),
                      infos['success'], infos['collision'], infos['truncated'])
            return (new_states, histories, jnp.where(done, 0, total_returns),
                    jnp.where(done, 0, lengths)), result
        carry, rows = jax.lax.scan(step, carry, jax.random.split(key, cfg.rollout_steps))
        obs, latent, logp, values, rewards, next_values, terminated, done = rows[:8]
        adv, targets = advantages(rewards, values, next_values, terminated, done,
                                  cfg.gamma, cfg.gae_lambda)
        flatten = lambda x: x.reshape((-1,) + x.shape[2:])
        data = tuple(map(flatten, (obs, latent, logp, values, adv, targets)))
        metrics = jnp.stack([jnp.sum(done), jnp.sum(rows[8]), jnp.sum(rows[9]),
                            jnp.sum(rows[10]), jnp.sum(rows[11]), jnp.sum(rows[12])])
        return carry, data, metrics
    return initialize, collect
