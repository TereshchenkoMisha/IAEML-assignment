"""Run Assignment 1 without modifying the teacher's entry point or renderer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv


def demo_action(state, params):
    """Simple path follower for demonstrating physics; this is not a trained agent."""
    path = state.path_array
    distances = jnp.linalg.norm(path - state.agent_pos, axis=-1)
    closest = jnp.argmin(distances)
    # Look ahead along the path, stopping at the goal padding.
    target = path[jnp.minimum(closest + 1, len(path) - 1)]
    delta = target - state.agent_pos
    error = jnp.arctan2(delta[1], delta[0]) - state.heading
    error = jnp.arctan2(jnp.sin(error), jnp.cos(error))
    steering = jnp.clip(jnp.arctan2(2 * params.wheelbase * jnp.sin(error),
                                   jnp.maximum(jnp.linalg.norm(delta), 0.5))
                        / params.max_steering, -1, 1)
    target_speed = 1.2 / (1 + 2 * jnp.abs(error))
    acceleration = jnp.clip(2 * (target_speed - state.speed), -1, 1)
    return jnp.array([acceleration, steering])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=int, choices=range(3), default=1)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--demo', action='store_true', help='deterministic path follower')
    parser.add_argument('--controls-demo', action='store_true', help='scripted acceleration, coast, brake and turn')
    parser.add_argument('--reverse-demo', action='store_true', help='scripted forward, braking, reverse turn and forward again')
    parser.add_argument('--forward-only', action='store_true', help='reproduce the original forward-only model')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record', type=Path, help='MP4 output; ffmpeg is needed only for recording')
    parser.add_argument('--summary', type=Path, help='write measured episode statistics as JSON')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.reverse_demo and args.forward_only:
        parser.error('--reverse-demo requires reverse motion')
    if args.headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import pygame
    from utils.renderer import PygameFrontend

    class CarFrontend(PygameFrontend):
        def handle_keys(self):
            keys = pygame.key.get_pressed()
            return jnp.array([
                float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(keys[pygame.K_DOWN] or keys[pygame.K_s]),
                float(keys[pygame.K_LEFT] or keys[pygame.K_a]) - float(keys[pygame.K_RIGHT] or keys[pygame.K_d]),
            ])

        def draw_info(self, screen, info, **kwargs):
            # Give telemetry a readable backing over both walls and free space.
            # Only this adapter changes; the teacher's renderer remains untouched.
            pygame.draw.rect(screen, (250, 250, 250), (505, 5, 290, 20 * len(info) + 12))
            pygame.draw.rect(screen, (70, 70, 70), (505, 5, 290, 20 * len(info) + 12), 1)
            super().draw_info(screen, info, position=(515, 10), line_height=20,
                              color=(20, 20, 20), **kwargs)

    key = jax.random.PRNGKey(args.seed)
    params, initial = RobotaxiEnv.init_params(key, map_id=args.map, max_steps=args.steps,
                                             allow_reverse=not args.forward_only)
    frontend = CarFrontend(RobotaxiEnv, params, initial)
    policy = jax.jit(lambda s: demo_action(s, params))
    writer = None
    frames = 0
    if args.record:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                parser.error('Recording needs ffmpeg on PATH (or the optional imageio-ffmpeg package).')
        args.record.parent.mkdir(parents=True, exist_ok=True)
        w, h = frontend.screen.get_size()
        writer = subprocess.Popen([
            ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}', '-r', '15', '-i', '-', '-an', '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', str(args.record),
        ], stdin=subprocess.PIPE)
    total_reward = 0.0
    peak_speed = 0.0
    minimum_speed = 0.0
    reverse_steps = 0
    max_steering = 0.0
    final_info = {}
    done = False
    step = -1
    try:
        for step in range(args.steps):
            events = pygame.event.get()
            if any(e.type == pygame.QUIT for e in events):
                break
            if args.reverse_demo:
                t = step / params.fps
                command = ([1., 0.] if t < 1 else [0., 0.] if t < 1.6 else
                           [-.45, 0.] if t < 3 else [-.45, .7] if t < 4 else
                           [1., 0.] if t < 5 else [0., 0.] if t < 6 else [-.3, 0.])
                action = jnp.array(command)
            elif args.controls_demo:
                t = step / params.fps
                # All phases use the original map and obstacle motion.
                command = ([1., 0.] if t < 1 else [0., 0.] if t < 2 else
                           [-1., 0.] if t < 3 else [0.35, 1.] if t < 5 else [-1., 0.])
                action = jnp.array(command)
            else:
                action = policy(frontend.state) if args.demo else frontend.handle_keys()
            key, step_key = jax.random.split(key)
            frontend.obs, frontend.state, reward, done, final_info = RobotaxiEnv.step(
                step_key, frontend.state, action, params)
            total_reward += float(reward)
            peak_speed = max(peak_speed, float(frontend.state.speed))
            minimum_speed = min(minimum_speed, float(frontend.state.speed))
            reverse_steps += int(float(frontend.state.speed) < -1e-5)
            max_steering = max(max_steering, abs(float(frontend.state.steering)))
            frontend.info = {k: final_info[k] for k in (
                'time', 'speed', 'steering', 'distance_to_path', 'success', 'collision', 'truncated')}
            frontend.info['acceleration command'] = f'{float(action[0]):.2f}'
            frontend.info['steering command'] = f'{float(action[1]):.2f}'
            # Physics is always 60 Hz, the video is sampled at 15 Hz.
            if step % 4 == 0 or bool(done):
                frontend.draw()
                if writer:
                    frame = np.transpose(pygame.surfarray.array3d(frontend.screen), (1, 0, 2))
                    writer.stdin.write(frame.tobytes())
                    frames += 1
            if not args.headless:
                frontend.clock.tick(params.fps)
            if bool(done):
                break
    finally:
        if writer:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError('ffmpeg failed to encode the demonstration')
        pygame.quit()
    summary = dict(map_id=args.map, seed=args.seed, steps=step + 1,
                   simulated_seconds=(step + 1) / params.fps,
                   return_=total_reward, peak_speed=peak_speed,
                   minimum_speed=minimum_speed, reverse_steps=reverse_steps,
                   reverse_enabled=params.allow_reverse,
                   max_abs_steering=max_steering, video_frames=frames,
                   goal_distance=float(jnp.linalg.norm(frontend.state.goal_pos - frontend.state.agent_pos)),
                   success=bool(final_info.get('success', False)),
                   collision=bool(final_info.get('collision', False)),
                   truncated=bool(final_info.get('truncated', False)))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
