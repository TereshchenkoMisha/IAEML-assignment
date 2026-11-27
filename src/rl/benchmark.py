"""Compare branching-preserving lax.map against vmap on the SAME simulator states."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from rl.rollout import macro_step
from rl.tasks import make_bank, sample_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    params, bank, original = make_bank(0, 0, 16, 1800)
    keys = jax.random.split(jax.random.PRNGKey(10), 8)
    states = jax.vmap(lambda key: sample_task(key, bank, original))(keys)
    actions = jnp.tile(jnp.array([.2, 0.]), (8, 1))
    outputs, result = {}, dict(environments=8, decisions_per_environment=32, action_repeat=6, repetitions=5)
    for method in ('vmap', 'lax_map'):
        def batch(states):
            f = lambda x: macro_step(x[0], x[1], x[2], params, 6)[1]
            return (jax.vmap(lambda k, s, a: f((k, s, a)))(keys, states, actions)
                    if method == 'vmap' else jax.lax.map(f, (keys, states, actions)))
        compiled = jax.jit(lambda s: jax.lax.fori_loop(0, 32, lambda i, s: batch(s), s))
        outputs[method] = jax.block_until_ready(compiled(states))  # compilation is excluded
        times = []
        for _ in range(5):
            before = time.monotonic()
            jax.block_until_ready(compiled(states))
            times.append(time.monotonic() - before)
        duration = float(np.median(times))
        result[method] = dict(median_seconds=duration, decisions_per_second=256 / duration)
    for left, right in zip(jax.tree_util.tree_leaves(outputs['vmap']), jax.tree_util.tree_leaves(outputs['lax_map'])):
        np.testing.assert_allclose(left, right, atol=1e-4, rtol=1e-4)
    result['outputs_agree'] = True
    result['speedup'] = result['vmap']['median_seconds'] / result['lax_map']['median_seconds']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
