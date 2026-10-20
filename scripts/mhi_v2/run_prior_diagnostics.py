#!/usr/bin/env python3
"""Wait for the frozen-Flow baseline, then run full-val B/C prior diagnostics.

This standard-library queue never imports torch or changes training state.
Only its evaluation children use the requested GPU after baseline completion.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PAIRS = 5106
PYTHON = Path('/root/miniconda3/envs/loma-repro/bin/python')
CACHE = Path('/home/u2ssd2_TQL/mhi_v2_cache/MHINet_RGB_512')
CHECKPOINT = ROOT/'outputs/mhi_v2/Flow0/P5/flow_head_rgb512_seed42_5ep_20261020/best.pt'
DEFAULT_OUTPUT = ROOT/'outputs/mhi_v2/diagnostics/flow_best_prior_toggle_rgb512_20261020'
POLL_SECONDS = 5


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, payload):
    temporary = path.with_suffix(path.suffix+'.temporary')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie no longer holds its CUDA context or owns useful work.
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def gpu_processes(gpu):
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    return [row.strip() for row in rows if row.split(',', 1)[0].strip() == uuid]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-status', required=True, type=Path)
    parser.add_argument('--wait-pid', required=True, type=int)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--manifest', type=Path, default=CACHE/'val/pairs.jsonl')
    parser.add_argument('--cache-status', type=Path, default=CACHE/'cache_status.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args(argv)
    if args.wait_pid < 1 or args.gpu < 0:
        raise ValueError('Invalid baseline PID/GPU')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'New output required; existing artifacts preserved: {output}')
    checkpoint, manifest = args.checkpoint.resolve(), args.manifest.resolve()
    if not checkpoint.is_file() or not manifest.is_file() or not PYTHON.is_file():
        raise FileNotFoundError('Checkpoint, NVMe val manifest, or loma-repro Python missing')
    cache = json.loads(args.cache_status.read_text())
    if cache.get('status') != 'completed' or cache.get('manifests_verified') is not True:
        raise ValueError('A completed, SHA-verified NVMe cache is required')
    manifest_sha = sha256(manifest)
    expected_manifest_sha = cache['identity']['manifests']['val']['sha256']
    if manifest_sha != expected_manifest_sha:
        raise ValueError('NVMe val manifest SHA differs from verified cache identity')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_PAIRS or any(row['size_A'] != [512, 512] or row['size_B'] != [512, 512] for row in rows):
        raise ValueError('Expected the complete 5106-pair RGB512 val manifest')
    pair_ids = [str(row['pair_id']) for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError('Duplicate val pair IDs')
    checkpoint_sha = sha256(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    lock = (output/'.queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commands = {variant: [str(PYTHON), '-u', '-m', 'mhinet.v2.evaluate',
        '--checkpoint', str(checkpoint), '--manifest', str(manifest),
        '--output', str(output/variant), '--device', 'cuda:0', '--limit', '0', '--variant', variant]
        for variant in ('B', 'C')}
    state = dict(schema='mhi.v2-prior-toggle-queue.v1', status='waiting_baseline', pid=os.getpid(),
        started_at=now(), checked_at=now(), physical_gpu=args.gpu, logical_device='cuda:0',
        python=str(PYTHON), checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        manifest=str(manifest), manifest_sha256=manifest_sha, expected_pairs=EXPECTED_PAIRS,
        max_output_matches=2048, input_size=512, poll_seconds=POLL_SECONDS,
        cache_status_path=str(args.cache_status.resolve()), cache_completed=True,
        cache_files_verified=cache.get('files_completed'), cache_bytes_verified=cache.get('bytes_verified'),
        baseline_status_path=str(args.wait_status.resolve()), baseline_pid=args.wait_pid,
        script_sha256=sha256(Path(__file__)), actual_commands=commands,
        metric_protocol='raster_mask_precision_validation_diagnostic', phases=[])
    write(output/'registration.json', state)
    active = {'process': None, 'stop': False}
    def stop(signum, frame):
        active['stop'] = True
        child = active['process']
        if child is not None and child.poll() is None:
            child.terminate()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    def persist():
        state['checked_at'] = now()
        write(output/'status.json', state)
    try:
        while not active['stop']:
            baseline = json.loads(args.wait_status.read_text()) if args.wait_status.exists() else {}
            state['baseline_status'] = baseline
            if baseline and baseline.get('schema') != 'mhi.frozen-flow-fullval.v1':
                raise ValueError('Unexpected frozen-Flow baseline status schema')
            if baseline.get('status') in ('failed', 'interrupted'):
                raise RuntimeError(f"Frozen baseline did not complete: {baseline.get('error', baseline['status'])}")
            if baseline.get('status') == 'completed':
                if baseline.get('pairs') != EXPECTED_PAIRS or baseline.get('total_pairs') != EXPECTED_PAIRS:
                    raise RuntimeError('Baseline completion lacks all 5106 validation pairs')
                if alive(args.wait_pid):
                    state['status'] = 'waiting_baseline_process_exit'
                else:
                    busy = gpu_processes(args.gpu)
                    state['gpu_processes'] = busy
                    if not busy:
                        state['baseline_completed_at'] = now()
                        break
                    state['status'] = 'waiting_gpu_idle'
            elif not alive(args.wait_pid):
                raise RuntimeError('Baseline process exited without a completed status')
            else:
                state['status'] = 'waiting_baseline'
            persist()
            time.sleep(POLL_SECONDS)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for variant in ('B', 'C'):
            if active['stop']:
                break
            if sha256(checkpoint) != checkpoint_sha or sha256(manifest) != manifest_sha:
                raise RuntimeError('Pinned checkpoint/manifest changed after queue registration')
            phase = dict(variant=variant, status='running', started_at=now(),
                actual_command=commands[variant], output=str(output/variant))
            state['status'] = f'running_{variant}'
            state['phases'].append(phase)
            with (output/f'{variant}.console.log').open('x') as stream:
                child = subprocess.Popen(commands[variant], cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT)
                active['process'] = child
                phase['pid'] = child.pid
                while child.poll() is None:
                    phase['checked_at'] = now()
                    pairs_file = output/variant/'pairs.jsonl'
                    # The evaluator flushes every completed pair; byte size is
                    # cheap progress evidence without rescanning a growing file.
                    phase['pairs_jsonl_bytes'] = pairs_file.stat().st_size if pairs_file.exists() else 0
                    persist()
                    time.sleep(POLL_SECONDS)
                phase['exit_code'] = child.returncode
            active['process'] = None
            if child.returncode != 0 or active['stop']:
                raise RuntimeError(f'Variant {variant} evaluation stopped with code {child.returncode}')
            summary = json.loads((output/variant/'summary.json').read_text())
            registration = json.loads((output/variant/'registration.json').read_text())
            if (summary.get('pairs') != EXPECTED_PAIRS or summary.get('shared_output_budget') != 2048
                    or summary.get('checkpoint_sha256') != checkpoint_sha
                    or summary.get('manifest_sha256') != manifest_sha
                    or summary.get('prior_toggle') != variant
                    or registration['matcher']['input_size'] != 512
                    or registration['matcher']['flow_guidance_enabled'] != (variant == 'C')):
                raise RuntimeError(f'Variant {variant} result identity/budget/full-val check failed')
            phase.update(status='completed', completed_at=now(), pairs=summary['pairs'],
                summary=str(output/variant/'summary.json'), code_sha256=summary['code_sha256'])
            persist()
        if active['stop']:
            state['status'] = 'interrupted'
            persist()
            return 130
        b = json.loads((output/'B/summary.json').read_text())
        c = json.loads((output/'C/summary.json').read_text())
        if b['code_sha256'] != c['code_sha256']:
            raise RuntimeError('B/C implementation identities differ')
        comparison = dict(schema='mhi.v2-frozen-prior-comparison.v1', checkpoint_sha256=checkpoint_sha,
            manifest_sha256=manifest_sha, pairs=EXPECTED_PAIRS, shared_output_budget=2048,
            metric_protocol=state['metric_protocol'], scope='Same frozen Flow-best and QRRU; only prior toggled',
            stage_macro_precision_delta_C_minus_B={name: {
                str(t): c['stages'][name]['macro_raster_mask_precision'][str(t)]
                        - b['stages'][name]['macro_raster_mask_precision'][str(t)] for t in (1., 3., 5., 10.)}
                for name in ('d8', 'd2', 'final')})
        write(output/'comparison.json', comparison)
        state.update(status='completed', completed_at=now(), comparison=str(output/'comparison.json'))
        persist()
        return 0
    except BaseException as error:
        state.update(status='interrupted' if active['stop'] else 'failed', error=f'{type(error).__name__}: {error}')
        persist()
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return 130 if active['stop'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
