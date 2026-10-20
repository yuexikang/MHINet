#!/usr/bin/env python3
"""Read future RGB512 PNGs into filesystem cache without touching training state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch


FIELDS = ('image_A', 'image_B', 'mask_A_overlap', 'mask_B_overlap')


def write(path, payload):
    temporary = path.with_suffix('.temporary.json')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def read_pair(root, row):
    count = 0
    # The trainer seeks a random JSONL row twice per pair, too. Warming PNGs
    # alone leaves that cold manifest block on the critical path.
    with (root/'pairs.jsonl').open('rb') as stream:
        stream.seek(row['_prefetch_manifest_offset'])
        count += len(stream.readline())
    for field in FIELDS:
        path = (root/row[field]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'PNG escapes registered split: {path}')
        with path.open('rb') as stream:
            while chunk := stream.read(1024*1024):
                count += len(chunk)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--lookahead', type=int, default=128)
    parser.add_argument('--max-pairs', type=int, help='Finite diagnostic; default follows live training')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 32 <= args.lookahead <= 4096:
        raise ValueError('workers must be1..32 and lookahead32..4096')
    if args.output.exists():
        raise FileExistsError('Keep existing prefetch evidence; choose a fresh output directory')
    args.output.mkdir(parents=True)
    run = json.loads((args.run/'run.json').read_text())
    data_root = Path(run['runtime']['data_root']).resolve()
    manifest = data_root/'train/pairs.jsonl'
    actual_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual_hash != run['train_sha256']:
        raise ValueError('Registered train manifest hash mismatch')
    rows = []
    with manifest.open('rb') as stream:
        while True:
            offset = stream.tell(); line = stream.readline()
            if not line:
                break
            if line.strip():
                row = json.loads(line)
                row['_prefetch_manifest_offset'] = offset
                rows.append(row)
    if len(rows) != run['train_pairs']:
        raise ValueError('This prefetcher requires the registered full train split')
    seed = run['config']['seed']; epochs = run['config']['epochs']
    torch.set_num_threads(1)
    orders = {e: torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed+e)).tolist()
              for e in range(epochs)}
    start = time.monotonic(); completed = total_bytes = 0
    last_write = 0
    pending = {}; seen = set(); stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    record = dict(schema='mhi.v2-readonly-prefetch.v1',pid=os.getpid(),status='running',
                  started_at=datetime.now(timezone.utc).isoformat(),run=str(args.run.resolve()),
                  workers=args.workers,lookahead_pairs=args.lookahead,train_manifest_sha256=actual_hash,
                  policy='Reads only registered train PNG bytes, in future seeded sample order; no checkpoint, data, config, math or training RNG mutation')
    error = None
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while not stop['requested']:
                progress = json.loads((args.run/'status.json').read_text())
                if progress['status'] != 'running':
                    break
                for token, future in list(pending.items()):
                    if future.done():
                        total_bytes += future.result(); completed += 1
                        del pending[token]
                if args.max_pairs and completed >= args.max_pairs:
                    break
                epoch, cursor = progress['epoch'], progress['cursor']
                future_pairs = []
                while len(future_pairs) < args.lookahead and epoch < epochs:
                    end = min(len(rows), cursor+args.lookahead-len(future_pairs))
                    future_pairs.extend(((epoch, i), orders[epoch][i]) for i in range(cursor, end))
                    # Do not evict validation data by reading the next epoch early.
                    break
                for token, index in future_pairs:
                    if len(pending) >= args.workers*4:
                        break
                    if token in seen:
                        continue
                    if args.max_pairs and len(seen) >= args.max_pairs:
                        break
                    seen.add(token)
                    pending[token] = executor.submit(read_pair, data_root/'train', rows[index])
                record.update(checked_at=datetime.now(timezone.utc).isoformat(),completed_pairs=completed,
                              read_bytes=total_bytes,pending_pairs=len(pending),elapsed_seconds=time.monotonic()-start,
                              training_snapshot=progress)
                if time.monotonic()-last_write >= 5:
                    write(args.output/'status.json',record)
                    last_write = time.monotonic()
                time.sleep(.2)
            # Context exit drains at most workers*4 queued read tasks; no new tasks.
            for future in pending.values():
                total_bytes += future.result(); completed += 1
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record.update(status='failed' if error else 'stopped' if stop['requested'] else 'completed',
                      error=error,completed_pairs=completed,read_bytes=total_bytes,
                      elapsed_seconds=time.monotonic()-start,checked_at=datetime.now(timezone.utc).isoformat())
        write(args.output/'status.json',record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
