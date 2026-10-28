"""Fixed-input equivalence and CUDA timing for skipping discarded legacy D2 work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

import numpy as np
import torch
from mhinet.v2.coarse_matcher import flow_prior_from_shared
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.d2_modulation import model as current
from mhinet.v2.flow_d8_d2.run import measure
from mhinet.v2.qrru_corners.experiment import load_models
from .model import ModulatedCGMDP


def reference(revision, path):
    # Read pinned committed implementations; no baseline code changes or duplication.
    module = types.ModuleType('reference')
    module.__package__ = 'mhinet.v2.d2_modulation'
    source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], text=True)
    exec(compile(source, f'{revision}:{path}', 'exec'), module.__dict__)
    return module


def event():
    result = torch.cuda.Event(enable_timing=True)
    result.record()
    return result


def timed_local(module, shared, matcher, adapter, legacy=False):
    marks = {}
    parent_name = 'discrete_infer' if legacy else 'local_coarse'
    parent, coarse = getattr(module, parent_name), module.coarse
    def parent_call(*args, **kwargs):
        marks['parent_start'] = event()
        result = parent(*args, **kwargs)
        marks['parent_end'] = event()
        return result
    def coarse_call(*args, **kwargs):
        marks['coarse_start'] = event()
        result = coarse(*args, **kwargs)
        marks['coarse_end'] = event()
        return result
    start = event()
    with patch.object(module, parent_name, parent_call), patch.object(module, 'coarse', coarse_call):
        result = module.infer(shared, matcher, adapter)
    end = event(); end.synchronize()
    timing = dict(total_ms=start.elapsed_time(end),
                  parent_ms=marks['parent_start'].elapsed_time(marks['parent_end']),
                  continuous_coarse_ms=marks['coarse_start'].elapsed_time(marks['coarse_end']),
                  pair_d2_ms=marks['coarse_end'].elapsed_time(end))
    return result, timing


def summarize(rows):
    totals = {k: sum(r[k] for r in rows) for k in rows[0]}
    return dict(**totals, p1=totals['correct1']/totals['n'], p3=totals['correct3']/totals['n'],
                epe=totals['error_sum']/totals['visible'])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=['candidate', 'cumulative', 'delta'], default='candidate')
    parser.add_argument('--pairs', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('reports/cgmdp_selection_optimization'))
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(2); torch.cuda.set_device(args.device)
    old = reference('ba1ea84', 'mhinet/v2/d2_modulation/model.py')
    global_path = reference('224fbd0', 'mhinet/v2/d2_modulation/no_flow.py')
    spec = json.loads(Path('outputs/d2_modulation_expanded/selection.json').read_text())['test']
    ds = GeometryPairDataset(spec['root'], spec['rows'][:args.pairs])
    model, matcher = load_models(args.device)
    matcher.requires_grad_(False).eval()
    adapter_path = 'outputs/d2_modulation_expanded/full_pair/best.pt'
    if args.mode == 'candidate':
        adapter = current.load_adapter(adapter_path, args.device)
    else:
        model = ModulatedCGMDP(model, args.mode).to(args.device).eval()
        adapter_path = f'outputs/cgmdp_global_modulation/{args.mode}/best.pt'
        state = torch.load(adapter_path, map_location=args.device, weights_only=False)
        model.modulators.load_state_dict(state['modulators']); del state
        adapter = current.LocalPairModulation('none').to(args.device).eval()
    def prohibit(*unused): raise RuntimeError('QRRU must remain disabled')
    matcher.qrru.register_forward_pre_hook(prohibit)
    variants = ['legacy', 'optimized', 'global_flow', 'global_no_prior']
    timings = {v: [] for v in variants}; metrics = {v: [] for v in variants}
    default_metrics = {v: [] for v in ('legacy', 'optimized')}
    checks = []
    for i, item in enumerate(ds):
        shared = model(item['images'].to(args.device))
        # Exact-output checks use deterministic reductions; timing uses normal kernels.
        torch.use_deterministic_algorithms(True)
        prior = flow_prior_from_shared(shared, 0, (64, 64))
        legacy_parents = old.discrete_infer(shared, matcher, 'local')['coarse_a'][:128]
        si, _, _, _ = current.local_coarse(shared['pyramid'][8][0].float(), prior, matcher)
        parents = current.source_grid(si.device)[si[:128]]
        assert torch.equal(legacy_parents, parents), f'D8 parent selection changed: legacy={legacy_parents[:8].tolist()}, new={parents[:8].tolist()}, mismatches={(legacy_parents != parents).any(-1).sum().item()}, maxdiff={(legacy_parents-parents).abs().max().item()}'
        a, b = old.extract_windows(shared, matcher), current.extract_windows(shared, matcher)
        # Every sampled feature, mask, continuous target window and coarse score is identical.
        for key in a: torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        center_a = old.coarse(shared, matcher, legacy_parents)['p8']
        center_b = current.coarse(shared, matcher, parents)['p8']
        torch.testing.assert_close(center_a, center_b, atol=0, rtol=0)
        def run(variant):
            if variant in ('legacy', 'optimized'):
                return timed_local(old if variant == 'legacy' else current, shared, matcher, adapter,
                                   legacy=variant == 'legacy')
            start = event()
            result = global_path.global_infer(shared, matcher, adapter,
                                              flow_prior_from_shared(shared, 0, (64, 64)) if variant == 'global_flow' else None)
            end = event(); end.synchronize()
            return result, dict(total_ms=start.elapsed_time(end))
        exact_outputs = {v: run(v)[0] for v in variants}
        for key in ('points_a', 'points_b', 'confidence'):
            torch.testing.assert_close(exact_outputs['legacy'][key], exact_outputs['optimized'][key], atol=0, rtol=0)
        torch.use_deterministic_algorithms(False)
        if i == 0:
            for _ in range(5):
                for v in variants: run(v)
        outputs = {}; first_old = None; legacy_order_changes = 0
        for repeat in range(args.repeats):
            offset = (i+repeat) % len(variants)
            for v in variants[offset:]+variants[:offset]:
                outputs[v], timing = run(v)
                timings[v].append(dict(pair=i, repeat=repeat, **timing))
            if first_old is None: first_old = outputs['legacy']['points_a'].clone()
            else: legacy_order_changes += int(not torch.equal(first_old, outputs['legacy']['points_a']))
        default_order_equal = torch.equal(outputs['legacy']['points_a'], outputs['optimized']['points_a'])
        sorted_outputs = {}
        for v in ('legacy', 'optimized'):
            points = outputs[v]['points_a']
            order = (points[:, 1]*512+points[:, 0]).argsort(stable=True)
            sorted_outputs[v] = {k: outputs[v][k][order] for k in ('points_a', 'points_b')}
            default_metrics[v].append(measure(outputs[v]['points_a'], outputs[v]['points_b'], item['geometry'].to(args.device))[0])
        default_set_equal = torch.equal(sorted_outputs['legacy']['points_a'], sorted_outputs['optimized']['points_a'])
        default_coordinate_diff = (float((sorted_outputs['legacy']['points_b']-sorted_outputs['optimized']['points_b']).abs().max()) if default_set_equal else None)
        torch.use_deterministic_algorithms(True)
        outputs = exact_outputs
        # Test the actual optimized production path also never enters old fine_scores.
        with patch.object(matcher, 'fine_scores', side_effect=RuntimeError('Legacy D2 executed')):
            guard_result = current.infer(shared, matcher, adapter)
        torch.testing.assert_close(guard_result['points_b'], outputs['legacy']['points_b'], atol=0, rtol=0)
        gt = item['geometry'].to(args.device)
        for v in variants:
            metrics[v].append(measure(outputs[v]['points_a'], outputs[v]['points_b'], gt)[0])
        assert metrics['legacy'][-1] == metrics['optimized'][-1]
        checks.append(dict(pair_id=item['pair_id'][0], parents=len(parents), exact=True, legacy_order_changes=legacy_order_changes,
                           default_order_equal=default_order_equal, default_source_set_equal=default_set_equal,
                           default_aligned_coordinate_max_diff=default_coordinate_diff,
                           d8_center_max_diff=float((center_a-center_b).abs().max()),
                           d2_max_diff=float((outputs['legacy']['points_b']-outputs['optimized']['points_b']).abs().max())))
        torch.use_deterministic_algorithms(False)
        if (i+1) % 8 == 0: print(f'{args.mode}: {i+1}/{len(ds)} pairs exact', flush=True)
    summary = {v: {key: dict(mean=float(np.mean([r[key] for r in rows])),
                            p50=float(np.median([r[key] for r in rows])),
                            p95=float(np.percentile([r[key] for r in rows], 95)))
                   for key in rows[0] if key not in ('pair', 'repeat')} for v, rows in timings.items()}
    report = dict(mode=args.mode, pairs=len(ds), repeats=args.repeats, gpu=torch.cuda.get_device_name(args.device),
                  device=args.device, checkpoint=adapter_path, baseline_revision='ba1ea84', global_revision='224fbd0',
                  equivalence='Deterministic CUDA reductions and cuBLAS workspace; atol=rtol=0; production determinism settings unchanged',
                  timing='Default nondeterministic kernels, CUDA events, precomputed identical features, 5 warmups/route, rotated order; concurrent training on GPU',
                  segments='legacy parent includes discarded fine matching; optimized parent is local_coarse only; pair_d2 includes window sampling and final confidence; total includes prior and setup',
                  global_note='global_no_prior skips Flow prior in matching only; feature/Flow computation is excluded for ALL routes',
                  summary=summary, metrics={v: summarize(rows) for v, rows in metrics.items()},
                  default_metrics={v: summarize(rows) for v, rows in default_metrics.items()},
                  checks=checks, measurements=timings)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/f'{args.mode}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, metrics=report['metrics']), indent=2), flush=True)


if __name__ == '__main__': main()
