"""512 inference timing and QRRU ablations on the same small Hangzhou sample.

CUDA events time modules without a barrier at each call. End-to-end wall time
includes GPU completion, excludes image loading, warmup and metric computation.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import torch

from mhinet.downstream.qrru import QRRU
from . import semidense as matching
from .pilot import load_pair, load_system, metrics, select_rows, summarize, write_json
from .feature_provider import FlowStage1Adapter


def historical_module(name, path):
    """Keep a readable reference to the paused implementation, outside model code."""
    source = subprocess.check_output(['git', 'show', f'7dec317:{path}'], text=True)
    module = types.ModuleType(name)
    module.__package__ = 'mhinet.v2'
    sys.modules[name] = module
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


@torch.no_grad()
def timed(system, images, module):
    spans = {}
    def wrap(owner, attribute, label, stack):
        original = getattr(owner, attribute)
        owned = attribute in owner.__dict__
        def call(*args, **kwargs):
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record()
            value = original(*args, **kwargs)
            end.record()
            spans.setdefault(label, []).append((begin, end))
            return value
        setattr(owner, attribute, call)
        stack.callback(setattr if owned else delattr, owner, attribute, *([original] if owned else []))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        shared = system.shared
        for owner, attr, label in (
            (shared, '_extract_batched_descriptors', 'dino'),
            (shared, '_contextualize', 'mvt'), (shared.vgg, 'forward', 'vgg'),
            (shared, '_decode_pyramid', 'cgmdp'),
            (FlowStage1Adapter, 'forward', 'flow_head'),
            (module, 'match_d8', 'd8'), (system.matcher, 'fine_scores', 'd2_scores'),
            (system.matcher.qrru.project, 'forward', 'qrru'),
            (system.matcher.qrru, 'forward', 'qrru')):
            wrap(owner, attr, label, stack)
        begin = time.perf_counter()
        result = system.matcher.infer(system(images))[0]
        torch.cuda.synchronize()
        milliseconds = (time.perf_counter()-begin)*1000
    components = {k: sum(a.elapsed_time(b) for a, b in v) for k, v in spans.items()}
    components['other'] = milliseconds-sum(components.values())
    return result, dict(total_ms=milliseconds, modules_ms=components,
        peak_mb=torch.cuda.max_memory_allocated()/2**20,
        coarse_count=len(result['coarse_a']), qrru_input=result.get('qrru_processed', len(result['fine_all_a'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', help='Optional subset of named comparisons')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(config['seed'])
    root = Path(config['dataset'])
    rows = select_rows(root, 'val', config['val_per_group'], config['seed'])
    write_json(args.output/'selection.json', [r['pair_id'] for r in rows])
    system = load_system(config, 'cuda:0')
    if system.matcher.qrru.use_discrepancy or system.matcher.qrru.use_dgfo:
        raise ValueError('This QRRU baseline comparison uses the C architecture')
    base = replace(system.matcher.config, qrru_early_stop=False)
    old = historical_module('mhinet.v2._benchmark_old_matcher', 'mhinet/v2/semidense.py')
    old.match_d8 = historical_module('mhinet.v2._benchmark_old_coarse', 'mhinet/v2/coarse_matcher.py').match_d8
    variants = dict(legacy={}, optimized_chunk=dict(coarse_mode='chunked'), full=dict(coarse_mode='full'),
        bf16=dict(coarse_mode='full', fine_precision='bf16'),
        fp16=dict(coarse_mode='full', fine_precision='fp16'))
    variants.update({f'iter{i}': dict(coarse_mode='full', inference_iterations=i) for i in (1, 2, 3)})
    variants.update({f'top{k}': dict(coarse_mode='full', max_matches=k) for k in (2048, 4096)})
    variants.update(adaptive_chunk=dict(coarse_mode='chunked', qrru_early_stop=True),
                    adaptive_full=dict(coarse_mode='full', qrru_early_stop=True))
    if args.variants:
        variants = {name: variants[name] for name in args.variants}
    if 'legacy' not in variants:
        raise ValueError('Include legacy as the reference')
    records = {k: [] for k in variants}
    def select(name):
        system.matcher.config = replace(base, **variants[name])
        system.matcher.infer = types.MethodType(old.SemidenseMatcherV2.infer if name == 'legacy'
                                                else matching.SemidenseMatcherV2.infer, system.matcher)
        system.matcher.qrru.forward = types.MethodType(QRRU.forward if name == 'legacy'
            else type(system.matcher.qrru).forward, system.matcher.qrru)
    with torch.no_grad(), (args.output/'pairs.jsonl').open('w') as stream:
        images, _, _ = load_pair(root, rows[0], 'cuda:0')
        for name in variants:
            select(name)
            for _ in range(2):
                system.matcher.infer(system(images))
        for index, row in enumerate(rows):
            images, H, masks = load_pair(root, row, 'cuda:0')
            names = list(variants)
            names = names[index % len(names):]+names[:index % len(names)]
            predictions = {}
            for name in names:
                select(name)
                result, timing = timed(system, images, old if name == 'legacy' else matching)
                record = dict(variant=name, pair_id=row['pair_id'], group=row['G'],
                              **metrics(result, H, masks), **timing)
                predictions[name] = {k: result[k].cpu() for k in ('points_a', 'points_b', 'confidence')}
                records[name].append(record)
            reference = predictions['legacy']
            for name, pred in predictions.items():
                same = pred['points_a'].shape == reference['points_a'].shape and torch.equal(pred['points_a'], reference['points_a'])
                records[name][-1]['same_source_points'] = same
                records[name][-1]['max_target_delta_px'] = float((pred['points_b']-reference['points_b']).abs().max()) if same and len(pred['points_b']) else None
                if 'full' in predictions:
                    records[name][-1]['same_full_outputs'] = all(torch.equal(pred[k], predictions['full'][k]) for k in pred)
                stream.write(json.dumps(records[name][-1])+'\n'); stream.flush()
            print(f'benchmark {index+1}/{len(rows)}', flush=True)
    summary = {}
    for name, values in records.items():
        summary[name] = dict(**summarize(values), mean_ms=float(np.mean([r['total_ms'] for r in values])),
            p95_ms=float(np.percentile([r['total_ms'] for r in values], 95)),
            peak_mb=max(r['peak_mb'] for r in values),
            qrru_input_mean=float(np.mean([r['qrru_input'] for r in values])),
            same_source_pairs=sum(r['same_source_points'] for r in values),
            same_full_output_pairs=sum(r.get('same_full_outputs', False) for r in values),
            max_target_delta_px=max((r['max_target_delta_px'] for r in values
                                    if r['max_target_delta_px'] is not None), default=None),
            module_mean_ms={k: float(np.mean([r['modules_ms'].get(k, 0) for r in values]))
                            for k in values[0]['modules_ms']})
    write_json(args.output/'report.json', dict(config=config, pairs=len(rows),
        note='Approximate geographic reference; other includes selection/output overhead. Legacy is 7dec317.',
        gpu=torch.cuda.get_device_name(), variants=summary))


if __name__ == '__main__':
    main()
