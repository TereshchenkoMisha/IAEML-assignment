"""PPO-Clip with tanh Gaussian actions, GAE and independent actor/critic MLPs.

Algorithm: Schulman et al., https://arxiv.org/abs/1707.06347
No external RL framework or additional dependencies are needed.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class Config:
    seed: int = 0
    train_maps: tuple = (0, 1)
    num_envs: int = 16  # per map, not in total
    rollout_steps: int = 128
    updates: int = 200
    epochs: int = 4
    minibatch_size: int = 256
    hidden: int = 128
    history: int = 4
    action_repeat: int = 6
    max_steps: int = 1800
    bank_size: int = 128
    gamma: float = .995  # per decision (0.1 seconds with the default repeat)
    gae_lambda: float = .95
    clip: float = .2
    learning_rate: float = 3e-4
    entropy_coef: float = .005
    value_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float = .03
    navigation: str = 'teacher'
    strict_splits: bool = False
    # Old checkpoints omit this field and retain their original dynamics.
    allow_reverse: bool = False


def init_model(key, observation_size, hidden):
    def network(key, output, scale):
        keys = jax.random.split(key, 3)
        sizes = [observation_size, hidden, hidden, output]
        layers = []
        for i, (n, m) in enumerate(zip(sizes[:-1], sizes[1:])):
            weight = jax.random.normal(keys[i], (n, m)) / jnp.sqrt(float(n))
            layers.append(dict(w=weight * (scale if i == 2 else jnp.sqrt(2.)), b=jnp.zeros(m)))
        return layers
    actor_key, critic_key = jax.random.split(key)
    return dict(actor=network(actor_key, 2, .01), critic=network(critic_key, 1, 1.),
                log_std=jnp.full(2, -.5))


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer['w'] + layer['b'])
    return x @ layers[-1]['w'] + layers[-1]['b']


def distribution(model, obs):
    mean = mlp(model['actor'], obs)
    return mean, jnp.clip(model['log_std'], -3, 1)


def value(model, obs):
    return mlp(model['critic'], obs)[..., 0]


def log_probability(mean, log_std, latent):
    normal = -.5 * (((latent - mean) * jnp.exp(-log_std)) ** 2
                    + 2 * log_std + jnp.log(2 * jnp.pi))
    # Stable log(1 - tanh(z)^2), including large |z|.
    jacobian = 2 * (jnp.log(2.) - latent - jax.nn.softplus(-2 * latent))
    return jnp.sum(normal - jacobian, axis=-1)


def act(model, obs, key, deterministic=False):
    mean, log_std = distribution(model, obs)
    latent = mean if deterministic else mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    return jnp.tanh(latent), latent, log_probability(mean, log_std, latent), value(model, obs)


def advantages(rewards, values, next_values, terminated, done, gamma, lam):
    """Bootstrap time limits from the FINAL observation; stop traces at every reset."""
    deltas = rewards + gamma * (1 - terminated) * next_values - values
    def backward(carry, row):
        delta, end = row
        advantage = delta + gamma * lam * (1 - end) * carry
        return advantage, advantage
    _, adv = jax.lax.scan(backward, jnp.zeros_like(values[0]), (deltas, done), reverse=True)
    return adv, adv + values


def loss(model, batch, key, cfg):
    obs, latent, old_logp, old_value, advantage, returns = batch
    mean, log_std = distribution(model, obs)
    logp = log_probability(mean, log_std, latent)
    log_ratio = logp - old_logp
    ratio = jnp.exp(log_ratio)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantage,
        jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantage))
    prediction = value(model, obs)
    # Ordinary squared-error critic; value clipping is not required by PPO-Clip.
    value_loss = .5 * jnp.mean((prediction - returns) ** 2)
    sample = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    entropy = -jnp.mean(log_probability(mean, log_std, sample))
    approx_kl = jnp.mean((ratio - 1) - log_ratio)
    metrics = jnp.stack([policy_loss, value_loss, entropy, approx_kl,
                         jnp.mean(jnp.abs(ratio - 1) > cfg.clip)])
    return policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy, metrics


def make_update(cfg):
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                           optax.adam(cfg.learning_rate, eps=1e-5))

    @jax.jit
    def update(model, optimizer_state, batch, key):
        n = batch[0].shape[0]
        normalized = (batch[4] - jnp.mean(batch[4])) / (jnp.std(batch[4]) + 1e-8)
        batch = (*batch[:4], normalized, batch[5])

        def epoch(carry, key):
            model, optimizer_state, stopped = carry
            permutation_key, noise_key = jax.random.split(key)
            order = jax.random.permutation(permutation_key, n).reshape(-1, cfg.minibatch_size)
            keys = jax.random.split(noise_key, len(order))

            def minibatch(carry, inputs):
                model, optimizer_state, stopped = carry
                indices, key = inputs
                data = jax.tree_util.tree_map(lambda x: x[indices], batch)
                (_, metrics), grad = jax.value_and_grad(loss, has_aux=True)(model, data, key, cfg)
                stopped = stopped | (metrics[3] > cfg.target_kl)
                def apply(args):
                    model, state = args
                    updates, state = optimizer.update(grad, state, model)
                    return optax.apply_updates(model, updates), state
                model, optimizer_state = jax.lax.cond(stopped, lambda x: x, apply,
                                                       (model, optimizer_state))
                return (model, optimizer_state, stopped), metrics
            return jax.lax.scan(minibatch, (model, optimizer_state, stopped), (order, keys))

        (model, optimizer_state, stopped), metrics = jax.lax.scan(
            epoch, (model, optimizer_state, jnp.bool_(False)), jax.random.split(key, cfg.epochs))
        return model, optimizer_state, metrics.mean(axis=(0, 1)), stopped

    return optimizer, update


def save_checkpoint(path, model, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in ('actor', 'critic'):
        for i, layer in enumerate(model[name]):
            for component, array in layer.items():
                arrays[f'{name}_{i}_{component}'] = np.asarray(array)
    arrays['log_std'] = np.asarray(model['log_std'])
    arrays['config'] = np.array(json.dumps(asdict(cfg)))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        settings = json.loads(str(data['config']))
        settings['train_maps'] = tuple(settings['train_maps'])
        cfg = Config(**settings)
        model = {name: [{part: jnp.asarray(data[f'{name}_{i}_{part}']) for part in ('w', 'b')}
                         for i in range(3)] for name in ('actor', 'critic')}
        model['log_std'] = jnp.asarray(data['log_std'])
    return model, cfg
