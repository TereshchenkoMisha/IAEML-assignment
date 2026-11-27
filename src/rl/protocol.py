"""Train on maps 0/1; select on validation starts; test once on fresh starts and map 2.

Runs the existing commands with one fixed, auditable protocol. No dependencies
are installed and the teacher's files are never changed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rl.tasks import original_task, start_candidates


def spatial_audit(map_id):
    params, state = original_task(map_id)
    h, w = params.map_height_width
    center = np.array([w-1, h-1])/2
    sets, counts = {}, {}
    for split in ('train', 'validation', 'test'):
        candidates = start_candidates(state, params, split)
        expanded = set()
        for sx in (-1, 1):
            for sy in (-1, 1):
                points = center + np.array([sx, sy])*(candidates-center)
                expanded.update(map(tuple, np.rint(2*points).astype(int)))
        sets[split] = expanded
        counts[split] = dict(base_candidates=len(candidates), including_reflections=len(expanded))
    overlaps = {f'{a}/{b}': len(sets[a] & sets[b]) for a,b in
                [('train','validation'),('train','test'),('validation','test')]}
    if any(overlaps.values()):
        raise AssertionError('Spatial start pools overlap')
    return dict(map_id=map_id, pools=counts, overlapping_positions=overlaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--checkpoint-every', type=int, default=200)
    parser.add_argument('--validation-seed', type=int, default=20000)
    parser.add_argument('--test-seed', type=int, default=30000)
    parser.add_argument('--validation-episodes', type=int, default=32)
    parser.add_argument('--test-episodes', type=int, default=64)
    parser.add_argument('--reverse',action='store_true',help='enable reverse motion throughout training, validation and test')
    args = parser.parse_args()
    if min(args.updates, args.checkpoint_every, args.validation_episodes, args.test_episodes) < 1:
        parser.error('Counts must be positive')
    if len({args.seed, args.validation_seed, args.test_seed}) != 3:
        parser.error('Training, validation and test seeds must differ')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use a new output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        train_maps=[0,1], validation_maps=[0,1], geometry_test_maps=[2],
        train_seed=args.seed, validation_seed=args.validation_seed, test_seed=args.test_seed,
        strict_splits=True, allow_reverse=args.reverse,
        start_partition='reflection-orbit hash v1: 60/20/20 buckets',
        heading_and_phase='Independent random draws with different seeds; headings sampled on reset',
        original_scenario='Reported separately; the original start is not an unseen-start test',
        selection_rule='0.5 * original success + 0.5 * varied success; tie: lower collision',
        validation_episodes_per_map=args.validation_episodes, test_episodes_per_map=args.test_episodes,
        spatial_audit=[spatial_audit(mid) for mid in range(3)],
        note='Map 2 has been diagnosed before, but is excluded from training and checkpoint selection; new test episodes are reserved.')
    (args.out/'protocol.json').write_text(json.dumps(manifest, indent=2)+'\n')
    entry = Path(__file__).resolve().parents[1]/'assignment2.py'

    def run(*arguments):
        subprocess.run([sys.executable, *map(str,arguments)], check=True)

    def status(phase):
        (args.out/'status.json').write_text(json.dumps(dict(phase=phase),indent=2)+'\n')

    status('training')
    reverse_flag = ['--reverse'] if args.reverse else []
    run(entry, 'train', '--out', args.out/'train', '--seed', args.seed,
        '--updates', args.updates, '--checkpoint-every', args.checkpoint_every,
        '--envs',16,'--rollout',128,'--minibatch',256,'--bank-size',256,
        '--train-maps',0,1,'--navigation','clearance','--strict-splits','--log-every',100,*reverse_flag)
    status('validation')
    marks = list(range(args.checkpoint_every,args.updates+1,args.checkpoint_every))
    if args.updates not in marks:
        marks.append(args.updates)
    evaluations = []
    for mark in marks:
        output = args.out/f'validation_{mark:04d}.json'
        run(entry,'evaluate','--checkpoint',args.out/'train'/f'checkpoint_{mark:04d}.npz',
            '--maps',0,1,'--split','validation','--seed',args.validation_seed,
            '--episodes',args.validation_episodes,'--out',output)
        evaluations.append(output)
    status('checkpoint_selection')
    run('-m','rl.select_checkpoint',*evaluations,'--out',args.out/'selection.json',
        '--checkpoint-out',args.out/'best_validation.npz')
    status('final_test')
    run(entry,'evaluate','--checkpoint',args.out/'best_validation.npz','--maps',0,1,2,
        '--split','test','--seed',args.test_seed,'--episodes',args.test_episodes,
        '--out',args.out/'test.json')
    test = json.loads((args.out/'test.json').read_text())
    with (args.out/'test_summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(test['groups'][0]))
        writer.writeheader();writer.writerows(test['groups'])
    status('complete')
    print(json.dumps(dict(checkpoint=str(args.out/'best_validation.npz'),
        checkpoint_sha256=hashlib.sha256((args.out/'best_validation.npz').read_bytes()).hexdigest(),
        test=str(args.out/'test.json'), groups=test['groups']),indent=2),flush=True)


if __name__ == '__main__':
    main()
