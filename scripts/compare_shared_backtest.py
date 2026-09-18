"""Compare a completed tier1 backtest against tier1's original validation."""
import argparse
import json
from pathlib import Path
import numpy as np
from mhinet.config import sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--after',default='outputs/shared_tier2_backtest_tier1')
    p.add_argument('--artifact',help='Optional version-controlled result copy')
    args=p.parse_args()
    root=Path(args.after)
    report=json.loads((root/'report.json').read_text())
    before_path=Path('outputs/shared_descriptor_frozen_seed0/validation/step_012995/summary.json')
    overlap_path=Path('outputs/shared_descriptor_frozen_seed0/h0_overlap_v1/tier1_summary.json')
    before=json.loads(before_path.read_text());overlap=json.loads(overlap_path.read_text())
    after=report['summary']
    if report['tier']!=1 or report['split']!='val' or report['manifest_sha256']!=overlap['manifest_sha256']:
        raise ValueError('Incompatible validation split')
    if not before['pairs']==after['pairs']==5772:
        raise ValueError('Full tier1 validation required')
    metrics={}
    def add(k,a,b):metrics[k]=dict(before=a,after=b,delta=b-a if a is not None and b is not None else None)
    add('H0_corner_mace_px',before['H0_mace_px'],after['H0_mace_px'])
    add('descriptor_loss',before['descriptor'],after['descriptor'])
    if overlap['H0_overlap']['support_pixels']!=after['H0_overlap']['support_pixels']:
        raise ValueError('GT overlap support changed')
    for key in ('pair_mean_px','pair_median_px','pair_p90_px','pixel_mean_px','invalid_projection_rate','failed_fit_pairs'):
        add('H0_overlap_'+key,overlap['H0_overlap'][key],after['H0_overlap'][key])
    for scale in (8,4,2):
        for key in ('mean_input_px','recall_1px','recall_3px','recall_5px'):
            add(f'D{scale}_{key}',sum(before['retrieval'][f'D{scale}_{d}'][key] for d in ('ab','ba'))/2,
                sum(after['retrieval'][f'D{scale}_{d}'][key] for d in ('ab','ba'))/2)
    a={r['pair_id']:r['mean_px'] for r in map(json.loads,overlap_path.with_name('tier1_pairs.jsonl').open())}
    rows_path=root/'validation'/f'step_{after["step"]:06d}'/'pairs.jsonl'
    b={r['pair_id']:r['H0_overlap']['mean_px'] for r in map(json.loads,rows_path.open())}
    if a.keys()!=b.keys():raise ValueError('Pair IDs differ')
    delta=np.array([b[k]-a[k] for k in a if b[k] is not None and a[k] is not None])
    result=dict(pairs=5772,metrics=metrics,H0_finite_paired_count=len(delta),
        H0_pair_improved_fraction=float((delta<0).mean()) if len(delta) else None,
        H0_pair_median_delta=float(np.median(delta)) if len(delta) else None,checkpoint_sha256=report['checkpoint_sha256'],
        baseline_summary_sha256=sha256_file(before_path),
        note='Single-seed, same tier1 val; raw deltas, not a multi-seed equivalence test.')
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    if args.artifact:
        Path(args.artifact).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
