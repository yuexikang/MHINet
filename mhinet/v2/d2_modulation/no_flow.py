"""Frozen no-Flow D8→D2 evaluation; no change to mainline inference."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from mhinet.downstream.loma_reference.coarse_matcher import match_d8 as reference_d8
from mhinet.v2.coarse_matcher import match_d8, flow_prior_from_shared, geometry_reward
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import FlowStage1Adapter, _flow_fields
from mhinet.v2.flow_d8_d2.run import measure
from mhinet.v2.joint_single.path import sample, decode, source_grid
from mhinet.v2.qrru_corners.experiment import load_models
from .model import center_indices, load_adapter, match_windows, infer


@torch.no_grad()
def global_infer(shared, matcher, adapter, prior=None, check_reference=False):
    d8 = shared['pyramid'][8][0].float()
    config = matcher.config
    common = dict(temperature=float(matcher.tau_c), threshold=config.confidence_threshold,
                  max_coarse=128, policy='mutual', force_chunked=False, chunk_rows=128)
    result = match_d8(d8[0], d8[1], flow_prior=prior,
                      flow_lambda=config.flow_lambda, flow_sigma_d8=config.flow_sigma_d8,
                      **common)
    if check_reference:
        support = torch.ones((64, 64), device=d8.device, dtype=torch.bool)
        ref = reference_d8(d8[0], d8[1], support, support, **common)
        torch.testing.assert_close(result.source_flat, ref.source_flat)
        torch.testing.assert_close(result.target_flat, ref.target_flat)
        torch.testing.assert_close(result.confidence, ref.confidence, rtol=1e-4, atol=1e-7)
    grid = source_grid(d8.device)
    source = grid[result.source_flat]
    if not len(source):
        return dict(points_a=source, points_b=source.clone(), confidence=source.new_empty(0),
                    coarse_a=source, coarse_b=source.clone())
    a = F.normalize(d8[0].flatten(1).T, dim=-1)
    b = F.normalize(d8[1].flatten(1).T, dim=-1)
    logits = a[result.source_flat] @ b.T / matcher.tau_c
    if prior is not None:
        logits += geometry_reward(prior, result.source_flat, torch.arange(4096, device=d8.device),
                                  width=64, strength=config.flow_lambda, sigma=config.flow_sigma_d8)
    # Same continuous 3x3-around-peak decode in both global variants.
    target, score, _, _ = decode(logits, grid[None].expand(len(source), -1, -1), 64)
    axis = (torch.arange(8, device=d8.device)-3.5)*2
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    delta = torch.stack((x, y), -1).reshape(1, 64, 2)
    ca, cb = source[:, None]+delta, target[:, None]+delta
    d2 = shared['pyramid'][2][0].float()
    windows = dict(a=sample(d2[0], ca), b=sample(d2[1], cb),
                   valid_a=(ca >= 0).all(-1) & (ca <= 511).all(-1),
                   valid_b=(cb >= 0).all(-1) & (cb <= 511).all(-1), candidates=cb)
    fine = match_windows(adapter, windows, matcher.tau_f)
    return dict(points_a=ca[:, center_indices(d8.device)].reshape(-1, 2),
                points_b=fine['points_b'].reshape(-1, 2),
                confidence=(fine['confidence']*score[:, None]).sqrt().flatten(),
                coarse_a=source, coarse_b=target)


def aggregate(rows, variant):
    totals = {key: sum(row[variant][key] for row in rows)
              for key in ('n', 'visible', 'error_sum', 'correct1', 'correct3', 'correct5', 'correct10')}
    return dict(**totals, pairs=len(rows), points_per_pair=totals['n']/len(rows),
                epe=totals['error_sum']/max(1, totals['visible']),
                **{f'p{t}': totals[f'correct{t}']/max(1, totals['n']) for t in (1, 3, 5, 10)})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['test', 'extra_test', 'hangzhou_test'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--selection', type=Path, default=Path('outputs/d2_modulation_expanded/selection.json'))
    parser.add_argument('--adapter', type=Path, default=Path('outputs/d2_modulation_expanded/full_pair/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/d2_modulation_no_flow'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text())[args.split]
    ds = GeometryPairDataset(selection['root'], selection['rows'])
    model, matcher = load_models(args.device)
    matcher.requires_grad_(False).eval()
    adapter = load_adapter(args.adapter, args.device)
    model.geometry_mode = 'none'
    def prohibit(*unused):
        raise RuntimeError('Disabled head executed')
    matcher.qrru.register_forward_pre_hook(prohibit)
    variants = ('global_no_flow', 'global_flow', 'local_flow')
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f'{args.split}.json'
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    for i in range(min(len(ds), args.limit or len(ds))):
        item = ds[i]
        guard = model.stage1_head.match_embedding_decoder.register_forward_pre_hook(prohibit)
        shared = model(item['images'].to(args.device), pyramid_scales=(8, 4, 2),
                       compute_stage1=False, compute_flow=False)
        assert shared['stage1'] is None and shared['call_counts']['flow_head'] == 0
        predictions = {'global_no_flow': global_infer(shared, matcher, adapter, check_reference=i == 0)}
        guard.remove()
        # Controls reuse identical frozen features; the no-Flow result already exists.
        stage1 = FlowStage1Adapter.forward(model.stage1_head, shared['context'],
                                          position_dtype=shared['pair_descriptors'].dtype)
        shared.update(_flow_fields(stage1))
        prior = flow_prior_from_shared(shared, 0, (64, 64))
        predictions['global_flow'] = global_infer(shared, matcher, adapter, prior)
        predictions['local_flow'] = infer(shared, matcher, adapter)
        gt = item['geometry'].to(args.device)
        row = dict(pair_id=item['pair_id'][0], no_flow_head_calls=0)
        for variant, prediction in predictions.items():
            assert torch.isfinite(prediction['points_b']).all()
            row[variant] = measure(prediction['points_a'], prediction['points_b'], gt)[0]
            if 'coarse_a' in prediction:
                row[variant]['d8'] = measure(prediction['coarse_a'], prediction['coarse_b'], gt)[0]
        rows.append(row)
        if i == 0 or (i+1) % 16 == 0:
            print(json.dumps(dict(split=args.split, pairs=i+1,
                                  p3={v: aggregate(rows, v)['p3'] for v in variants})), flush=True)
    torch.cuda.synchronize(args.device)
    result = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  summary={v: aggregate(rows, v) for v in variants}, pairs=rows,
                  elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                  reference_check='LoMA global no-prior outputs matched on first pair',
                  no_flow_check='Flow decoder prohibited; flow_head=0 on every no-Flow forward')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
