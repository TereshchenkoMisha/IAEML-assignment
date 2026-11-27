"""Assignment 2 commands: train PPO, evaluate fixed tests, and replay a checkpoint."""
import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv
from rl.ppo import Config, act, init_model, load_checkpoint, make_update, save_checkpoint
from rl.rollout import history_reset, history_step, macro_step, make_runner
from rl.tasks import make_bank, original_task, sample_task


def train(args):
    cfg = Config(seed=args.seed, num_envs=args.envs, rollout_steps=args.rollout,
                 updates=args.updates, minibatch_size=args.minibatch,
                 bank_size=args.bank_size, max_steps=args.max_steps,
                 train_maps=tuple(args.train_maps), navigation=args.navigation,
                 strict_splits=args.strict_splits, allow_reverse=getattr(args,'reverse',False))
    batch_size = cfg.num_envs * cfg.rollout_steps * len(cfg.train_maps)
    if min(cfg.num_envs, cfg.rollout_steps, cfg.updates, cfg.bank_size, cfg.max_steps,
           cfg.minibatch_size, args.log_every, args.checkpoint_every) < 1:
        raise ValueError('Training counts and logging intervals must be positive')
    if batch_size % cfg.minibatch_size:
        raise ValueError('Total rollout size must be divisible by minibatch size')
    warm_start = load_checkpoint(args.init_checkpoint) if args.init_checkpoint else None
    if warm_start:
        _, previous_cfg = warm_start
        if (previous_cfg.hidden, previous_cfg.history) != (cfg.hidden, cfg.history):
            raise ValueError('Warm-start model architecture must match the new configuration')
        if cfg.strict_splits and (not previous_cfg.strict_splits or
                                   not set(previous_cfg.train_maps).issubset(cfg.train_maps)):
            raise ValueError('Strict experiment cannot use weights exposed to held-out starts or maps; train from scratch')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new output directory to preserve previous experiments')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    devices=list(map(str, jax.devices())),
                    packages={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'equinox', 'optax', 'numpy')},
                    source_hashes={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), *Path(__file__).parent.joinpath('rl').glob('*.py'),
                                             Path(__file__).parent / 'env/robotaxi.py']})
    (args.out / 'runtime.json').write_text(json.dumps(metadata, indent=2) + '\n')
    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    model = init_model(model_key, 43 * cfg.history, cfg.hidden)
    if args.init_checkpoint:
        model, previous_cfg = warm_start
        metadata['warm_start'] = dict(checkpoint=str(args.init_checkpoint),
                                     previous_config=asdict(previous_cfg),
                                     optimizer_reset=True)
        (args.out / 'runtime.json').write_text(json.dumps(metadata, indent=2) + '\n')
    optimizer, update = make_update(cfg)
    optimizer_state = optimizer.init(model)
    save_checkpoint(args.out / 'initial.npz', model, cfg)
    runners = []
    for map_id in cfg.train_maps:
        p, bank, originals = make_bank(map_id, cfg.seed + 101 * map_id, cfg.bank_size, cfg.max_steps,
                                       navigation=cfg.navigation, split='train' if cfg.strict_splits else 'legacy',
                                       allow_reverse=cfg.allow_reverse)
        initialize, collect = make_runner(p, bank, originals, cfg)
        key, reset_key = jax.random.split(key)
        carry = initialize(reset_key, bank, originals)
        runners.append([bank, originals, carry, collect])
    fieldnames = ['update', 'decisions', 'wall_seconds', 'decisions_per_second',
                  'episodes', 'mean_return', 'mean_length', 'success_rate',
                  'collision_rate', 'timeout_rate', 'policy_loss', 'value_loss',
                  'entropy', 'approx_kl', 'clip_fraction', 'kl_stopped']
    started = time.monotonic()
    with (args.out / 'training.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in range(1, cfg.updates + 1):
            before = time.monotonic()
            batches, metrics = [], []
            for runner in runners:
                bank, originals, carry, collect = runner
                key, rollout_key = jax.random.split(key)
                carry, data, stats = collect(model, carry, rollout_key, bank, originals)
                runner[2] = carry
                batches.append(data)
                metrics.append(stats)
            batch = jax.tree_util.tree_map(lambda *a: jnp.concatenate(a), *batches)
            key, update_key = jax.random.split(key)
            model, optimizer_state, loss_metrics, stopped = update(model, optimizer_state, batch, update_key)
            loss_metrics = np.asarray(loss_metrics)  # synchronise timing with completed work
            if not np.isfinite(loss_metrics).all() or not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(model)):
                raise FloatingPointError('Non-finite loss or model parameters')
            episodes, returns, lengths, successes, collisions, timeouts = np.asarray(jnp.sum(jnp.stack(metrics), axis=0))
            elapsed = time.monotonic() - before
            denominator = max(float(episodes), 1.)
            row = dict(update=iteration, decisions=iteration * batch_size,
                       wall_seconds=time.monotonic() - started,
                       decisions_per_second=batch_size / elapsed, episodes=int(episodes),
                       mean_return=float(returns / denominator), mean_length=float(lengths / denominator),
                       success_rate=float(successes / denominator), collision_rate=float(collisions / denominator),
                       timeout_rate=float(timeouts / denominator),
                       **dict(zip(fieldnames[10:15], map(float, loss_metrics))), kl_stopped=bool(stopped))
            writer.writerow(row)
            stream.flush()
            if iteration == 1 or iteration % args.log_every == 0 or iteration == cfg.updates:
                print(json.dumps(row), flush=True)
            if iteration % args.checkpoint_every == 0 or iteration == cfg.updates:
                save_checkpoint(args.out / f'checkpoint_{iteration:04d}.npz', model, cfg)
                save_checkpoint(args.out / 'last.npz', model, cfg)
    training_plot(args.out / 'training.csv', args.out / 'training.svg')


def make_episode(params, cfg):
    def episode(model, initial):
        obs = RobotaxiEnv.get_observation(initial, params).as_vector()
        history = history_reset(obs, cfg.history)
        def body(carry, index):
            state, history, total, done, length, idle_total, idle_run, idle_max = carry
            action = act(model, history.reshape(-1), jax.random.PRNGKey(0), deterministic=True)[0]
            obs, state, reward, end, info = macro_step(jax.random.PRNGKey(0), state, action,
                                                       params, cfg.action_repeat)
            history = history_step(history, obs.as_vector())
            # Count only live decisions; never count time spent in absorbing states.
            far = jnp.linalg.norm(state.goal_pos - state.agent_pos) > 1.25
            idle = (~done) & (~end) & far & (jnp.abs(state.speed) < .05)
            idle_run = jnp.where(idle, idle_run + 1, 0)
            return (state, history, total + reward, done | end, length + (~done),
                    idle_total + idle, idle_run, jnp.maximum(idle_max, idle_run)), None
        (state, history, total, done, length, idle_total, idle_run, idle_max), _ = jax.lax.scan(
            body, (initial, history, jnp.float32(0), jnp.bool_(False), jnp.int32(0),
                   jnp.int32(0), jnp.int32(0), jnp.int32(0)),
            jnp.arange(math.ceil(cfg.max_steps / cfg.action_repeat)))
        collision, goal = RobotaxiEnv._events(state, params)
        return dict(success=goal, collision=collision, timeout=state.truncated, return_=total,
                    decisions=length, goal_distance=jnp.linalg.norm(state.goal_pos - state.agent_pos),
                    idle_fraction=idle_total / jnp.maximum(length, 1),
                    max_idle_seconds=idle_max * cfg.action_repeat / params.fps)
    return jax.jit(lambda model, states: jax.lax.map(lambda state: episode(model, state), states))


def evaluate(args):
    if args.episodes < 1:
        raise ValueError('Evaluation needs at least one varied episode per map')
    model, cfg = load_checkpoint(args.checkpoint)
    if args.navigation:
        cfg = replace(cfg,navigation=args.navigation)
    task_split = args.split or ('test' if cfg.strict_splits else 'legacy')
    if task_split != 'legacy' and not cfg.strict_splits:
        raise ValueError('Legacy weights may have seen these start positions; use a strict-splits checkpoint')
    rows = []
    for map_id in args.maps:
        params, bank, originals = make_bank(map_id, args.seed + map_id * 101,
                                            max(32, args.episodes * 2), cfg.max_steps,
                                            navigation=cfg.navigation, split=task_split,allow_reverse=cfg.allow_reverse)
        varied = jax.jit(jax.vmap(lambda key: sample_task(key, bank, originals)))(
            jax.random.split(jax.random.PRNGKey(args.seed + map_id), args.episodes))
        _, nominal = original_task(map_id, cfg.max_steps, navigation=cfg.navigation,allow_reverse=cfg.allow_reverse)
        states = jax.tree_util.tree_map(lambda first, rest: jnp.concatenate([first[None], rest]), nominal, varied)
        statistics = jax.device_get(make_episode(params, cfg)(model, states))
        for index in range(args.episodes + 1):
            row = dict(map_id=map_id, split='held_out' if map_id not in cfg.train_maps else 'train_geometry',
                       scenario='original' if index == 0 else 'varied', episode=index,
                       initial_heading=float(states.heading[index]))
            row.update({name: array[index].item() for name, array in statistics.items()})
            rows.append(row)
    groups = []
    for map_id in args.maps:
        for scenario in ('original', 'varied'):
            selected = [r for r in rows if r['map_id'] == map_id and r['scenario'] == scenario]
            groups.append(dict(map_id=map_id, scenario=scenario, split=selected[0]['split'], episodes=len(selected),
                               **{name: float(np.mean([r[name] for r in selected])) for name in
                                  ('success', 'collision', 'timeout', 'return_', 'goal_distance',
                                   'idle_fraction', 'max_idle_seconds')}))
    result = dict(checkpoint=str(args.checkpoint), evaluation_seed=args.seed,
                  navigation=cfg.navigation,
                  strict_splits=cfg.strict_splits, task_split=task_split,
                  allow_reverse=cfg.allow_reverse,
                  original_scenario_note='Original-start benchmark; not a spatially held-out start',
                  train_maps=cfg.train_maps, evaluation_maps=args.maps,
                  episode_count=len(rows), groups=groups, episodes=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


def play(args):
    model, cfg = load_checkpoint(args.checkpoint)
    if args.navigation:
        cfg = replace(cfg,navigation=args.navigation)
    task_split = args.split or ('test' if cfg.strict_splits else 'legacy')
    if task_split != 'legacy' and not cfg.strict_splits:
        raise ValueError('A validation/test split requires a strict-splits checkpoint')
    if args.headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import pygame
    from utils.renderer import PygameFrontend
    if args.evaluation_episode is not None:
        if not 1 <= args.evaluation_episode <= args.evaluation_episodes:
            raise ValueError('Evaluation episode must be in 1..evaluation-episodes')
        params, bank, originals = make_bank(args.map, args.seed + args.map * 101,
            max(32, args.evaluation_episodes * 2), cfg.max_steps,navigation=cfg.navigation,split=task_split,
            allow_reverse=cfg.allow_reverse)
        key = jax.random.split(jax.random.PRNGKey(args.seed + args.map), args.evaluation_episodes)[args.evaluation_episode - 1]
        initial = sample_task(key, bank, originals)
    elif args.varied:
        params, bank, originals = make_bank(args.map, args.seed, 64, cfg.max_steps,
                                            navigation=cfg.navigation, split=task_split,allow_reverse=cfg.allow_reverse)
        initial = sample_task(jax.random.PRNGKey(args.seed), bank, originals)
    else:
        params, initial = original_task(args.map, cfg.max_steps, args.heading_offset,
                                        navigation=cfg.navigation,allow_reverse=cfg.allow_reverse)

    class Frontend(PygameFrontend):
        def draw_path(self, path_array, agent_pos_map):
            # A fixed episode route starts at the original spawn. Draw only its
            # remaining part, so the teacher's renderer does not add a misleading
            # line from the current vehicle back to that spawn.
            closest = int(np.argmin(np.linalg.norm(np.asarray(path_array)-np.asarray(agent_pos_map), axis=-1)))
            super().draw_path(path_array[closest:], agent_pos_map)

        def draw_info(self, screen, info, **kwargs):
            # Keep telemetry away from the agent and destination on reflected maps.
            panel_height = 20 * len(info) + 12
            points = np.array([self.state.agent_pos, self.state.goal_pos])
            points = np.stack([(points[:, 0] + .5) * 40,
                               (params.map_height_width[0] - .5 - points[:, 1]) * 40], axis=-1)
            corners = [(5, 5), (505, 5), (5, 395-panel_height), (505, 395-panel_height)]
            scores = [np.linalg.norm(points - np.array([x+145, y+panel_height/2]), axis=-1).min()
                      for x, y in corners]
            x, y = corners[int(np.argmax(scores))]
            pygame.draw.rect(screen, (250, 250, 250), (x, y, 290, panel_height))
            super().draw_info(screen, info, position=(x+10, y+5), line_height=20,
                              color=(20, 20, 20), **kwargs)
    frontend = Frontend(RobotaxiEnv, params, initial)
    history = history_reset(frontend.obs.as_vector(), cfg.history)
    choose = jax.jit(lambda h: act(model, h.reshape(-1), jax.random.PRNGKey(0), True)[0])
    step_fn = jax.jit(lambda state, action: macro_step(jax.random.PRNGKey(0), state, action, params, cfg.action_repeat))
    writer = None
    if args.record:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '800x400', '-r', str(params.fps / cfg.action_repeat),
            '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(args.record)], stdin=subprocess.PIPE)
    total = 0.
    info = dict(success=False, collision=False, truncated=False)
    try:
        for decision in range(math.ceil(cfg.max_steps / cfg.action_repeat)):
            if any(e.type == pygame.QUIT for e in pygame.event.get()):
                break
            action = choose(history)
            frontend.obs, frontend.state, reward, done, info = step_fn(frontend.state, action)
            total += float(reward)
            history = history_step(history, frontend.obs.as_vector())
            frontend.info = {k: info[k] for k in ('time', 'speed', 'steering', 'distance_to_path', 'success', 'collision', 'truncated')}
            frontend.draw()
            if writer:
                writer.stdin.write(np.transpose(pygame.surfarray.array3d(frontend.screen), (1, 0, 2)).tobytes())
            if not args.headless:
                frontend.clock.tick(params.fps / cfg.action_repeat)
            if bool(done):
                break
    finally:
        if writer:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError('ffmpeg failed')
        pygame.quit()
    print(json.dumps(dict(map_id=args.map, seed=args.seed, varied=bool(args.varied or args.evaluation_episode is not None),
                          evaluation_episode=args.evaluation_episode,
                          initial_heading=float(initial.heading), return_=total,
                          success=bool(info['success']), collision=bool(info['collision']),
                          timeout=bool(info['truncated']))))


def training_plot(csv_path, output):
    """Standalone SVG, no plotting package is added to the prepared environment."""
    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    width, height = 900, 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:sans-serif;font-size:14px}</style>',
           '<text x="60" y="28">PPO training: episode outcomes (10-update weighted window)</text>']
    for v in (0, .25, .5, .75, 1):
        y = 370 - v * 300
        svg.extend([f'<path d="M60 {y} H850" stroke="#ddd"/>', f'<text x="12" y="{y+5}">{v:.2f}</text>'])
    for name, color in [('success_rate', '#16803c'), ('collision_rate', '#c33d32'), ('timeout_rate', '#365cb8')]:
        points = []
        for i, row in enumerate(rows):
            window = rows[max(0, i - 9):i + 1]
            count = sum(int(r['episodes']) for r in window)
            rate = sum(float(r[name]) * int(r['episodes']) for r in window) / max(count, 1)
            points.append(f'{60 + 790 * i / max(len(rows)-1,1):.1f},{370 - rate*300:.1f}')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
    svg.extend(['<text x="60" y="405">0</text>', f'<text x="720" y="405">{int(rows[-1]["decisions"]):,} decisions</text>',
                '<text x="60" y="440" fill="#16803c">Success</text><text x="210" y="440" fill="#c33d32">Collision</text><text x="370" y="440" fill="#365cb8">Time limit</text>', '</svg>'])
    Path(output).write_text('\n'.join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--envs', type=int, default=16)
    p.add_argument('--rollout', type=int, default=128)
    p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--bank-size', type=int, default=128)
    p.add_argument('--max-steps', type=int, default=1800)
    p.add_argument('--train-maps',type=int,nargs='+',choices=range(3),default=[0,1])
    p.add_argument('--navigation',choices=['teacher','clearance'],default='teacher')
    p.add_argument('--strict-splits',action='store_true',help='disjoint spatial train/validation/test starts, including reflections')
    p.add_argument('--reverse',action='store_true',help='train with signed speed and reverse motion')
    p.add_argument('--init-checkpoint',type=Path,help='warm-start weights; Adam and episodes restart')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--checkpoint-every', type=int, default=25)
    p.set_defaults(func=train)
    p = commands.add_parser('evaluate')
    p.add_argument('--split',choices=['legacy','validation','test'],help='default: test for strict checkpoints, legacy otherwise')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--navigation',choices=['teacher','clearance'],help='diagnostic override of saved navigation mode')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=32, help='varied episodes per map, plus one original')
    p.add_argument('--maps', type=int, nargs='+', choices=range(3), default=[0, 1, 2],
                   help='use 0 1 for validation without inspecting held-out map 2')
    p.add_argument('--seed', type=int, default=10000)
    p.set_defaults(func=evaluate)
    p = commands.add_parser('play')
    p.add_argument('--split',choices=['legacy','validation','test'],help='match the split used by evaluate')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--navigation',choices=['teacher','clearance'],help='diagnostic override of saved navigation mode')
    p.add_argument('--map', type=int, choices=range(3), default=1)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--varied', action='store_true')
    p.add_argument('--evaluation-episode', type=int, help='replay a numbered varied episode from evaluate')
    p.add_argument('--evaluation-episodes', type=int, default=32, help='must match the evaluation episode count')
    p.add_argument('--heading-offset', type=float, default=0.)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--record', type=Path)
    p.set_defaults(func=play)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
