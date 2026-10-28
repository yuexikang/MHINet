"""Single-GPU, batch-one latency of the three frozen D8/D2 inference paths."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.qrru_corners.experiment import load_models
from mhinet.v2.coarse_matcher import flow_prior_from_shared
from .model import infer, load_adapter
from .no_flow import global_infer


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--pairs', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, default=Path('reports/d2_modulation_no_flow/speed.json'))
    args = p.parse_args()
    torch.set_num_threads(2)
    spec = json.loads(Path('outputs/d2_modulation_expanded/selection.json').read_text())['test']
    ds = GeometryPairDataset(spec['root'], spec['rows'])
    images = [ds[i]['images'].to(args.device) for i in range(args.pairs)]
    model, matcher = load_models(args.device)
    matcher.requires_grad_(False).eval()
    adapter = load_adapter('outputs/d2_modulation_expanded/full_pair/best.pt', args.device)
    def tick():
        torch.cuda.synchronize(args.device)
        return time.perf_counter()
    def run(image, variant):
        model.geometry_mode = 'none' if variant == 'global_no_flow' else 'flow_d16'
        start = tick()
        shared = model(image, pyramid_scales=(8, 4, 2), compute_stage1=False)
        feature = tick()
        if variant == 'local_flow':
            result = infer(shared, matcher, adapter)
        else:
            prior = None if variant == 'global_no_flow' else flow_prior_from_shared(shared, 0, (64, 64))
            result = global_infer(shared, matcher, adapter, prior)
        end = tick()
        return dict(total_ms=1000*(end-start), feature_ms=1000*(feature-start),
                    matching_ms=1000*(end-feature), points=len(result['points_a']))
    variants = ['local_flow', 'global_flow', 'global_no_flow']
    for variant in variants:
        for _ in range(5): run(images[0], variant)
    rows = {v: [] for v in variants}
    # Rotate order to avoid systematically giving one variant the first slot.
    for repeat in range(args.repeats):
        for i, image in enumerate(images):
            order = variants[(i+repeat)%3:] + variants[:(i+repeat)%3]
            for v in order: rows[v].append(run(image, v))
        print(f'repeat {repeat+1}/{args.repeats}', flush=True)
    summary = {}
    for v, records in rows.items():
        summary[v] = {k: dict(mean=float(np.mean([r[k] for r in records])),
                             p50=float(np.median([r[k] for r in records])),
                             p95=float(np.percentile([r[k] for r in records], 95)))
                      for k in ('total_ms', 'feature_ms', 'matching_ms')}
        summary[v]['pairs_per_second'] = 1000/summary[v]['total_ms']['mean']
    result = dict(gpu=torch.cuda.get_device_name(args.device), pairs=args.pairs, repeats=args.repeats,
                  warmup_per_variant=5, batch_size=1, timing='synchronized wall clock; GPU-resident images; excludes I/O and loading',
                  precision='existing FP32 matcher; original backbone internal precision unchanged',
                  summary=summary, measurements=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
