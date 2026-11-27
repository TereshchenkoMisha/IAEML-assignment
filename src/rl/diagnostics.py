"""Save decision-level traces to inspect stopping, acceleration and obstacle clearance."""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv
from rl.ppo import act, load_checkpoint
from rl.rollout import history_reset, history_step, macro_step
from rl.tasks import make_bank, original_task, sample_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--maps', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--varied', action='store_true', help='match play --varied at the same seed')
    parser.add_argument('--seed', type=int, default=10000)
    args = parser.parse_args()
    model, cfg = load_checkpoint(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for map_id in args.maps:
        if args.varied:
            params, bank, originals = make_bank(map_id, args.seed, 64, cfg.max_steps,navigation=cfg.navigation,
                                               split='test' if cfg.strict_splits else 'legacy',allow_reverse=cfg.allow_reverse)
            initial = sample_task(jax.random.PRNGKey(args.seed), bank, originals)
        else:
            params, initial = original_task(map_id, cfg.max_steps,navigation=cfg.navigation,allow_reverse=cfg.allow_reverse)

        @jax.jit
        def run(initial):
            obs = RobotaxiEnv.get_observation(initial, params).as_vector()
            history = history_reset(obs, cfg.history)
            def body(carry, _):
                state, history = carry
                action = act(model, history.reshape(-1), jax.random.PRNGKey(0), True)[0]
                obs, new, reward, done, info = macro_step(jax.random.PRNGKey(0), state, action,
                                                         params, cfg.action_repeat)
                obstacles = jnp.concatenate([new.static_obstacles, new.kinematic_obstacles])
                row = jnp.array([new.time / params.fps, new.agent_pos[0], new.agent_pos[1],
                    new.speed, action[0], action[1], reward,
                    jnp.linalg.norm(new.goal_pos-new.agent_pos),
                    jnp.min(jnp.linalg.norm(obstacles-new.agent_pos, axis=-1))-1.,
                    obs.distance_to_path, done, info['success'], info['collision'], info['truncated']])
                return (new, history_step(history, obs.as_vector())), row
            return jax.lax.scan(body, (initial, history), None,
                                length=(cfg.max_steps+cfg.action_repeat-1)//cfg.action_repeat)[1]
        trace = np.asarray(run(initial))
        endings = np.flatnonzero(trace[:, 10])
        if len(endings):
            trace = trace[:endings[0]+1]
        names = ['seconds', 'x', 'y', 'speed', 'acceleration_command', 'steering_command',
                 'reward', 'goal_distance', 'clearance', 'path_distance', 'done',
                 'success', 'collision', 'timeout']
        with (args.out / f'map{map_id}.csv').open('w', newline='') as f:
            writer = csv.writer(f);writer.writerow(names);writer.writerows(trace.tolist())
        idle = (np.abs(trace[:, 3]) < .05) & (trace[:, 7] > 1.25) & (trace[:, 10] == 0)
        longest = current = 0
        for stopped in idle:
            current = current+1 if stopped else 0
            longest = max(longest, current)
        summaries.append(dict(map_id=map_id, varied=args.varied, seed=args.seed, success=bool(trace[-1,11]),
            collision=bool(trace[-1,12]), timeout=bool(trace[-1,13]),
            goal_distance=float(trace[-1,7]), idle_fraction=float(idle.mean()),
            max_idle_seconds=longest*cfg.action_repeat/params.fps,
            final_speed=float(trace[-1,3]), final_acceleration_command=float(trace[-1,4])))
    (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
