"""Optional publication/export plots; uses an already installed matplotlib only.

Training itself always writes a dependency-free SVG; this command is optional.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for directory in args.runs:
        csv_path = directory / 'training.csv'
        if not csv_path.exists():
            csv_path = directory / 'train' / 'training.csv'
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        x = np.array([int(r['decisions']) for r in rows]) / 1e3
        for ax, field in zip(axes, ('success_rate', 'mean_return')):
            smooth = []
            for i in range(len(rows)):
                window = rows[max(0, i - 19):i + 1]
                n = sum(int(r['episodes']) for r in window)
                smooth.append(sum(float(r[field]) * int(r['episodes']) for r in window) / max(n, 1))
            ax.plot(x, smooth, label=directory.name)
            ax.set_xlabel('Policy decisions (thousands)')
            ax.grid(alpha=.2)
            ax.legend()
    axes[0].set_ylabel('Success rate'); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel('Mean episode return')
    fig.suptitle('Training outcomes: 20-update episode-weighted window')
    fig.savefig(args.out / 'training.png', dpi=170)
    fig.savefig(args.out / 'training.svg')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    labels = ['Map 0: new variants', 'Map 1: new variants', 'Map 2: held out']
    width = .7 / len(args.runs)
    evaluation_counts = set()
    for i, directory in enumerate(args.runs):
        evaluation_path = directory / 'evaluation.json'
        if not evaluation_path.exists():
            evaluation_path = directory / 'test.json'
        evaluation = json.loads(evaluation_path.read_text())
        groups = [g for g in evaluation['groups'] if g['scenario'] == 'varied']
        rates = np.array([g['success'] for g in groups])
        counts = np.array([g['episodes'] for g in groups])
        evaluation_counts.update(map(int, counts))
        # Wilson 95% intervals: show finite test-set uncertainty, not training-seed uncertainty.
        z = 1.96
        center = (rates + z*z/(2*counts)) / (1 + z*z/counts)
        half = z*np.sqrt(rates*(1-rates)/counts + z*z/(4*counts*counts)) / (1+z*z/counts)
        ax.bar(np.arange(3) + (i - (len(args.runs)-1)/2)*width, rates, width,
               label=directory.name, yerr=np.maximum(0, np.array([rates - (center-half), (center+half) - rates])), capsize=4)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Deterministic policy success rate')
    count_label = str(next(iter(evaluation_counts))) if len(evaluation_counts)==1 else 'varying counts'
    ax.set_title(f'New evaluation episodes: {count_label} per map; 95% Wilson intervals')
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(args.out / 'evaluation.png', dpi=170)
    fig.savefig(args.out / 'evaluation.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
