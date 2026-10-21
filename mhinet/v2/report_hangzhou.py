"""Compact final tables and plots from existing per-epoch measurements."""
import argparse
import json
from pathlib import Path
import numpy as np


def write_report(output):
    output = Path(output)
    records = [json.loads(p.read_text()) for p in sorted(output.glob('epoch_*.json'))]
    if not records:
        return
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    chosen = [records[0], records[len(records)//2], records[-1]]
    names = ('Before', 'Middle', 'After')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    axes[0].plot([r['epoch'] for r in records], [r['mean']['loss'] for r in records], marker='.')
    axes[0].set(xlabel='Epoch', ylabel='Fixed-query validation loss', title='Joint training')
    tables = {}
    for axis, group in zip(axes[1:], ('fixed_perturbation', 'actual_candidates')):
        tables[group] = {}
        for label, record in zip(names, chosen):
            entries = [p[group] for p in record['qrru'] if p[group]['points']]
            if not entries:
                continue
            weights = np.array([p['points'] for p in entries])
            values = np.average([p['epe_px'] for p in entries], axis=0, weights=weights)
            tables[group][label.lower()] = dict(epoch=record['epoch'], epe_px=values.tolist(),
                improved_fraction=np.average([p['improved_vs_input'] for p in entries], axis=0, weights=weights).tolist(),
                damaged_fraction=np.average([p['damaged_vs_input'] for p in entries], axis=0, weights=weights).tolist())
            axis.plot(range(5), values, marker='o', label=f'{label}: epoch {record["epoch"]}')
        axis.set(xlabel='QRRU iteration (0 = input)', ylabel='Reference EPE (px)',
                 title='Fixed perturbations' if group == 'fixed_perturbation' else 'Current predicted candidates')
        axis.legend(fontsize=8)
    fig.savefig(output/'qrru_progress.png', dpi=160); plt.close(fig)
    (output/'qrru_summary.json').write_text(json.dumps(tables, indent=2)+'\n')
    lines = ['# Hangzhou fresh joint training', '',
        'Only DINOv3 is pretrained. Geographic labels are approximate; precision is reference agreement.', '',
        '| Stage | Epoch | Val loss | Pre@1 | Pre@3 | Flow EPE px |',
        '|---|---:|---:|---:|---:|---:|']
    for label, r in zip(names, chosen):
        m = r['mean']
        lines.append(f'| {label} | {r["epoch"]} | {m["loss"]:.4f} | {m["pre1"]:.4%} | {m["pre3"]:.4%} | {m["flow_epe_px"]:.3f} |')
    lines.extend(['', 'Fixed perturbation queries are identical across epochs. Actual candidate sets change with the model; their means are not a fixed-population comparison.', '',
                  '![QRRU before/middle/after](qrru_progress.png)', ''])
    (output/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    write_report(parser.parse_args().output)
