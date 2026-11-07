"""Replay frozen V1 against the release V2 full-test records, without training."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


def aggregate(rows, totals):
    result = totals(rows)
    result['pairs'] = len(rows)
    result['matches_per_pair'] = result['n'] / max(1, len(rows))
    result['correct3_per_pair'] = result['correct3'] / max(1, len(rows))
    result['empty_pairs'] = sum(r['n'] == 0 for r in rows)
    for threshold in (1, 3, 5, 10):
        result[f'macro_p{threshold}'] = sum(
            r[f'correct{threshold}'] / max(1, r['n']) for r in rows
        ) / max(1, len(rows))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--rgb-root', type=Path, required=True)
    parser.add_argument('--hangzhou-root', type=Path, required=True)
    parser.add_argument('--v2-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    release_root = Path(__file__).resolve().parents[2]
    # V1 model, QRRU, and feature provider come exclusively from the frozen tree.
    # Append only the release package for its dataset and common scoring code.
    sys.path.insert(0, str(args.v1_root.resolve()))
    import mhinet
    mhinet.__path__.append(str(release_root / 'mhinet'))
    from mhinet.config import RuntimePaths, sha256_file
    from mhinet.pretraining.model import build_shared_network
    from mhinet.downstream.training import SemidenseSystem
    from mhinet.downstream.semidense import SemidenseConfig, SemidenseMatcher
    from mhinet.v2.data import GeometryPairDataset
    from mhinet.v2.pair_matching.metrics import measure, totals

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    runtime = replace(RuntimePaths.from_json(args.runtime),
                      repo_root=args.v1_root.resolve(), output_root=args.output.resolve(),
                      device=args.device)
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != '43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4':
        raise ValueError('Expected the registered V1 release checkpoint')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    config = replace(SemidenseConfig(**state['metadata']['config']['matcher']),
                     input_size=512, max_matches=2048)
    shared, _ = build_shared_network(runtime, lora=False)
    model = SemidenseSystem(shared, SemidenseMatcher(config)).to(args.device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    report = dict(
        v1_checkpoint=str(args.checkpoint), v1_sha256=checkpoint_sha,
        v1_commit=subprocess.check_output(
            ['git', '-C', str(args.v1_root), 'rev-parse', 'HEAD'], text=True).strip(),
        v1_module=mhinet.__file__, matcher=asdict(config),
        training=state['metadata']['config'], progress=state['progress'],
        v2_results=str(args.v2_results), device=args.device,
        gpu=torch.cuda.get_device_name(args.device), torch=torch.__version__,
        protocol='512 RGB; max 2048; all returned points denominator; known-visible EPE; '
                 'same GeometryPairDataset/measure/totals as V2; no extra filtering or padding',
        tests={}, complete=False,
    )
    del state
    # Warm up without using test geometry or changing either checkpoint.
    model.matcher.infer(model(torch.zeros(1, 2, 3, 512, 512, device=args.device)))
    torch.cuda.synchronize(args.device)
    print('V1 strict loading and warmup passed: ' + mhinet.__file__, flush=True)
    splits = {
        'test': (args.rgb_root / 'test', args.rgb_root / 'test/pairs.jsonl'),
        'hangzhou_test_real': (args.hangzhou_root, args.hangzhou_root / 'manifests/test_real.jsonl'),
    }
    for split, (root, manifest) in splits.items():
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        previous = json.loads((args.v2_results / f'{split}_pairs.json').read_text())
        if [r['pair_id'] for r in records] != [r['pair_id'] for r in previous]:
            raise ValueError(f'{split}: V1 and V2 pair lists/order differ')
        if args.limit:
            records, previous = records[:args.limit], previous[:args.limit]
        data = GeometryPairDataset(root, records)
        rows, fine_rows = [], []
        failures = Counter()
        torch.cuda.reset_peak_memory_stats(args.device)
        start = time.perf_counter()
        # Each completed pair is persisted, including empty outputs.
        with (args.output / f'{split}_pairs.jsonl').open('w', buffering=1) as stream:
            for index in range(len(data)):
                item = data[index]
                shared = model(item['images'].to(args.device))
                result = model.matcher.infer(shared)[0]
                gt = item['geometry'].to(args.device)
                counts, _, _ = measure(result['points_a'], result['points_b'], gt)
                row = dict(pair_id=item['pair_id'][0], **counts)
                if counts['n'] > 2048:
                    raise AssertionError('Output cap exceeded')
                if not (torch.isfinite(result['points_a']).all() and torch.isfinite(result['points_b']).all()):
                    raise FloatingPointError('Nonfinite returned coordinates')
                rows.append(row)
                failures[result.get('failure_reason', 'none')] += 1
                # Additional diagnosis reuses the native pre-QRRU output; no second forward.
                a = result.get('fine_all_a', result['points_a'])
                b = result.get('fine_all_b', result['points_b'])
                fine_counts, _, _ = measure(a, b, gt)
                fine_rows.append(dict(pair_id=row['pair_id'], **fine_counts))
                stream.write(json.dumps(dict(**row, pre_qrru=fine_counts)) + '\n')
                if index == 0 or (index + 1) % 128 == 0:
                    print(json.dumps(dict(split=split, pairs=index + 1, total=len(data),
                                          seconds=time.perf_counter() - start,
                                          p3=totals(rows)['p3'])), flush=True)
        v1, v2 = aggregate(rows, totals), aggregate(previous, totals)
        report['tests'][split] = dict(
            v1=v1, v2=v2, v1_pre_qrru=aggregate(fine_rows, totals),
            delta_v2_minus_v1={key: v2[key] - v1[key] for key in ('p1', 'p3', 'p5', 'p10', 'epe')},
            failures=dict(failures), seconds=time.perf_counter() - start,
            peak_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
        )
        (args.output / f'{split}_pairs.json').write_text(json.dumps(rows, indent=2))
        (args.output / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report['tests'][split]), flush=True)
    report['complete'] = not bool(args.limit)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('SMOKE COMPLETE' if args.limit else 'FULL COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
