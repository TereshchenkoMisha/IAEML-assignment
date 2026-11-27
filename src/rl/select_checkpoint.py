"""Select a checkpoint using train-map validation only, never the held-out geometry."""
import argparse
import json
from pathlib import Path
import shutil


def validation_score(evaluation):
    if evaluation.get('strict_splits') and evaluation.get('task_split') != 'validation':
        raise ValueError('A strict experiment must select weights on validation, never test starts')
    groups = evaluation['groups']
    maps = set(evaluation['train_maps'])
    if set(g['map_id'] for g in groups) != maps:
        raise ValueError('Selection requires exactly the training maps, without held-out geometry')
    if any(g['split'] != 'train_geometry' for g in groups):
        raise ValueError('Held-out groups must not be used to select a checkpoint')
    original = [g for g in groups if g['scenario'] == 'original']
    varied = [g for g in groups if g['scenario'] == 'varied']
    if len(original) != len(maps) or len(varied) != len(maps):
        raise ValueError('Each map needs both original and varied validation groups')
    mean = lambda rows, name: sum(g[name] for g in rows) / len(rows)
    score = .5 * mean(original, 'success') + .5 * mean(varied, 'success')
    collision = .5 * mean(original, 'collision') + .5 * mean(varied, 'collision')
    return score, collision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True, help='selection.json')
    parser.add_argument('--checkpoint-out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    identity = None
    for path in args.evaluations:
        evaluation = json.loads(path.read_text())
        protocol = (evaluation['evaluation_seed'], tuple(evaluation['train_maps']),
                    evaluation.get('task_split','legacy'), evaluation.get('navigation','teacher'),
                    tuple((g['map_id'], g['scenario'], g['episodes']) for g in evaluation['groups']))
        if identity is not None and protocol != identity:
            raise ValueError('All candidates must use the same validation protocol')
        identity = protocol
        score, collision = validation_score(evaluation)
        rows.append(dict(evaluation=str(path), checkpoint=evaluation['checkpoint'],
                         score=score, collision_score=collision))
    # Criterion is fixed before seeing held-out results. Preserve the first input on ties.
    best = max(rows, key=lambda row: (row['score'], -row['collision_score']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best['checkpoint'], args.checkpoint_out)
    args.out.write_text(json.dumps(dict(
        rule='0.5 * original success + 0.5 * varied success; tie: lower collision, then first input',
        evaluation_seed=identity[0], candidates=rows, selected=best,
        copied_checkpoint=str(args.checkpoint_out)), indent=2) + '\n')
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
