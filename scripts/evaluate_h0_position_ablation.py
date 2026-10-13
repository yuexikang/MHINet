#!/usr/bin/env python3
"""Fixed-weight, shared-feature 2x2 H0 inference-position ablation.

The default invocation waits on CPU for a genuinely idle GPU.  --prepare-only
performs the complete CPU provenance/data check without creating a CUDA context.
The four variants receive exactly the same feature dictionary for each pair.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / 'weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt'
DEFAULT_BASELINE = REPO / 'outputs/semidense_rgb512_h0joint_test512_20261012'
EXPECTED_CHECKPOINT_SHA = '43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4'
LEGACY_COMMIT = 'be04d1f'
DATASETS = {
    'GoogleEarth': dict(path=Path('/home/disk1/Data/datasets/Test_Data/GoogleEarth/val/pairs.jsonl'),
                       pairs=498, sha256='55088ef26b24170cad2da2feb7917bcc3e2051e94b576a48a9018ecd12872b51'),
    'JL1Flight': dict(path=Path('/home/disk1/Data/datasets/Test_Data/JL1Flight/test/pairs.jsonl'),
                     pairs=1755, sha256='d932203978e3da5d3ae96779447afdc69dc3b1d0dc26f0ef5041d0979a3aaaf5'),
}
VARIANTS = {
    'A': dict(use_h0_d8_mask=True, use_h0_d2_warp=True),
    'B': dict(use_h0_d8_mask=True, use_h0_d2_warp=False),
    'C': dict(use_h0_d8_mask=False, use_h0_d2_warp=True),
    'D': dict(use_h0_d8_mask=False, use_h0_d2_warp=False),
}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def tensor_sha256(tensor):
    """Hash dtype, shape and raw bytes, including BF16 and scalar buffers."""
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)), sort_keys=True).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b'\0')
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def feature_sha256(features):
    return {name: tensor_sha256(value) for name, value in {
        'D8': features['pyramid'][8], 'D2': features['pyramid'][2],
        'H0_norm': features['H0_norm'], 'stage1_valid': features['stage1_valid'],
    }.items()}


def check_array_equal(prediction, expected, *, label):
    import numpy as np
    import torch
    checked = []
    for key in ('points_a', 'points_b', 'confidence'):
        actual = prediction[key].detach().cpu().numpy()
        reference = np.asarray(expected[key])
        if actual.dtype != reference.dtype or actual.shape != reference.shape or not np.array_equal(actual, reference):
            difference = None
            if actual.shape == reference.shape and actual.size:
                difference = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
            raise RuntimeError(f'A baseline changed: {label}/{key}; '
                               f'shape={actual.shape}/{reference.shape}; '
                               f'dtype={actual.dtype}/{reference.dtype}; max_abs_difference={difference}')
        if not torch.equal(prediction[key].detach().cpu(), torch.from_numpy(reference)):
            raise RuntimeError(f'A tensor equality failed: {label}/{key}')
        checked.append(key)
    return checked


def check_legacy_equal(current, legacy):
    import torch
    for key, reference in legacy.items():
        if key not in current:
            raise RuntimeError('A removed legacy output ' + key)
        if torch.is_tensor(reference):
            if not torch.equal(current[key], reference):
                raise RuntimeError('A differs from legacy infer: ' + key)
        elif current[key] != reference:
            raise RuntimeError('A differs from legacy infer: ' + key)


def legacy_source():
    return subprocess.check_output(['git', '-C', str(REPO), 'show',
                                    LEGACY_COMMIT + ':mhinet/downstream/semidense.py'], text=True)


def load_legacy_infer():
    """Bind the old function to the current matcher; do not copy its weights."""
    name = 'mhinet.downstream._frozen_h0_position_legacy'
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(legacy_source(), name, 'exec'), module.__dict__)
    return module.SemidenseMatcher.infer


def manifest_rows(dataset):
    path = DATASETS[dataset]['path']
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != DATASETS[dataset]['pairs'] or len({row['pair_id'] for row in rows}) != len(rows):
        raise ValueError('Unexpected dataset inventory: ' + dataset)
    return rows


def preflight(args):
    """Read all inputs and baseline correspondence files on CPU only."""
    import cv2
    import numpy as np
    import torch
    import PIL
    from PIL import Image
    sys.path.insert(0, str(REPO))
    from mhinet.config import RuntimePaths
    from mhinet.downstream.semidense import SemidenseConfig

    checkpoint = args.checkpoint.resolve()
    if sha_file(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError('Expected the completed H0-joint checkpoint; checkpoint identity changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    meta = state['metadata']
    if meta['task'] != 'semidense_qrru_v1' or state['progress']['optimizer_step'] != meta['total_steps']:
        raise ValueError('Checkpoint training is incomplete')
    runtime = RuntimePaths.from_json(meta['config']['runtime'])
    if sha_file(runtime.dino_checkpoint) != meta['dino_sha256']:
        raise ValueError('DINO identity mismatch')
    config = replace(SemidenseConfig(**meta['matcher']), input_size=512, inference_window_chunk=1024)
    checkpoint_record = dict(path=str(checkpoint), sha256=EXPECTED_CHECKPOINT_SHA,
                             step=state['progress']['optimizer_step'], total_steps=meta['total_steps'])
    del state
    datasets = {}
    for dataset in args.datasets:
        entry = DATASETS[dataset]
        if sha_file(entry['path']) != entry['sha256']:
            raise ValueError('Original test cohort changed: ' + dataset)
        rows = manifest_rows(dataset)
        baseline_root = args.baseline_root.resolve() / ('h0joint_' + dataset)
        baseline_report = json.loads((baseline_root / 'report.json').read_text())
        registration = baseline_report['registration']
        if not baseline_report.get('complete'):
            raise ValueError('Baseline incomplete: ' + dataset)
        if registration['checkpoint']['sha256'] != EXPECTED_CHECKPOINT_SHA:
            raise ValueError('Baseline checkpoint differs: ' + dataset)
        if registration['manifest']['sha256'] != entry['sha256'] or registration['manifest']['pairs'] != len(rows):
            raise ValueError('Baseline cohort differs: ' + dataset)
        if registration['matcher'] != asdict(config):
            raise ValueError('Baseline matcher configuration differs: ' + dataset)
        if registration['implementation']['scripts/mrsi_gt_metrics.py'] != sha_file(REPO / 'scripts/mrsi_gt_metrics.py'):
            raise ValueError('Metrics differ from the original evaluation')
        old_rows = [json.loads(line) for line in (baseline_root / 'pairs.jsonl').read_text().splitlines() if line.strip()]
        if [r['pair_id'] for r in old_rows] != [r['pair_id'] for r in rows]:
            raise ValueError('Baseline correspondence inventory/order differs: ' + dataset)
        input_digest = hashlib.sha256()
        baseline_digest = hashlib.sha256()
        checked = {}
        for index, row in enumerate(rows):
            h = np.asarray(row['H_A_to_B'])
            if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-10:
                raise ValueError('Invalid GT: ' + row['pair_id'])
            for side in ('A', 'B'):
                if row['size_' + side] != [512, 512]:
                    raise ValueError('Expected stored 512 coordinates: ' + row['pair_id'])
                for key in ('image_' + side, 'mask_' + side + '_overlap'):
                    image_path = entry['path'].parent / row[key]
                    if image_path not in checked:
                        with Image.open(image_path) as image:
                            if image.size != (512, 512):
                                raise ValueError('Unexpected image size: ' + str(image_path))
                        checked[image_path] = sha_file(image_path)
                    input_digest.update((row['pair_id'] + '/' + key + ':' + checked[image_path] + '\n').encode())
            correspondence = baseline_root / 'matches' / f'{index:05d}.npz'
            with np.load(correspondence) as stored:
                arrays = [stored[key] for key in ('points_a', 'points_b', 'confidence')]
                if arrays[0].shape != arrays[1].shape or arrays[0].shape != (len(arrays[2]), 2):
                    raise ValueError('Invalid baseline correspondence: ' + str(correspondence))
            baseline_digest.update((row['pair_id'] + ':' + sha_file(correspondence) + '\n').encode())
        datasets[dataset] = dict(path=str(entry['path']), sha256=entry['sha256'], pairs=len(rows),
                                evaluated_pairs=min(args.limit, len(rows)) if args.limit else len(rows),
                                input_inventory_sha256=input_digest.hexdigest(),
                                baseline_root=str(baseline_root),
                                baseline_report_sha256=sha_file(baseline_root / 'report.json'),
                                baseline_pairs_sha256=sha_file(baseline_root / 'pairs.jsonl'),
                                baseline_npz_inventory_sha256=baseline_digest.hexdigest())
    implementation = {str(path.relative_to(REPO)): sha_file(path) for path in (
        Path(__file__), REPO / 'mhinet/downstream/semidense.py', REPO / 'mhinet/downstream/supervision.py',
        REPO / 'mhinet/downstream/losses.py', REPO / 'mhinet/downstream/qrru.py',
        REPO / 'mhinet/downstream/loma_reference/coarse_matcher.py',
        REPO / 'mhinet/downstream/loma_reference/matching_utils.py',
        REPO / 'mhinet/downstream/loma_reference/overlap_masks.py',
        REPO / 'mhinet/models/feature_provider.py', REPO / 'mhinet/pretraining/model.py',
        REPO / 'mhinet/dataio/data.py', REPO / 'mhinet/downstream/metrics.py',
        REPO / 'scripts/mrsi_gt_metrics.py',
    )}
    plan = dict(experiment='Fixed-weight H0 inference-position 2x2 ablation',
                checkpoint=checkpoint_record, datasets=datasets, variants=VARIANTS, matcher=asdict(config),
                scope='limited_validation' if args.limit else 'full_test_cohorts', limit=args.limit,
                input_size=512, inference_window_chunk=1024,
                memory_budget_gib=args.memory_budget_gib,
                d2_support_policy='H0 D2 overlap masks follow use_h0_d2_warp; disabled means image-bounds validity only.',
                weight_policy='One strict-loaded eval model; no optimizer/backward; all parameters and buffers hashed before/after.',
                feature_policy='One shared forward per pair; A/B/C/D use the same dictionary; D8/D2/H0/valid raw bytes checked before/after.',
                baseline_policy='A exactly equals every original NPZ points_a/points_b/confidence; first pair additionally checks the old infer function.',
                legacy_commit=LEGACY_COMMIT, legacy_infer_source_sha256=hashlib.sha256(legacy_source().encode()).hexdigest(),
                coordinate_unit='stored_512_target_pixels',
                precision_protocol='GT support and <=1/3/5 target pixels; all predicted matches in denominator; pair macro.',
                sr_protocol='Predictions-only RANSAC H(3px,10000 iterations,.999 confidence,seed0), GT-visible 20x20 source grid; failures included.',
                rmse_protocol='SR@3px success: RMSE over GT-correct <=5px matches; failed pairs=10; success with zero correct matches undefined and counted.',
                raster_mask_protocol='dense_metrics uses manifest overlap masks separately from headline GT-geometric-support metrics.',
                timing_note='Shared forward measured once; variant matcher measured separately. wall_total_ms is estimated standalone shared_wall_ms+matcher_wall_ms. IO/metrics/hashes/warmup excluded.',
                googleearth_label_scope='Synthetic geometry relative to nominal cross-temporal co-registration; native registration is not independently measured.',
                versions=dict(torch=torch.__version__, cv2=cv2.__version__, numpy=np.__version__, pillow=PIL.__version__),
                implementation=implementation,
                git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    # Unrelated commits may land while queued; the actual implementation hashes
    # stay strict and are the reproducibility boundary.
    fingerprint_values = {key: value for key, value in plan.items() if key != 'git_commit'}
    plan['fingerprint'] = hashlib.sha256(json.dumps(fingerprint_values, sort_keys=True).encode()).hexdigest()
    plan['created'] = now()
    return plan


def prepare(args):
    plan = preflight(args)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'plan.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('fingerprint') != plan['fingerprint']:
            raise ValueError('Experiment inputs/implementation changed; use a new output directory')
        return existing
    write_json(path, plan)
    return plan


def gpu_status():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    busy = {line.split(',')[0].strip() for line in processes.stdout.splitlines() if ',' in line}
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, memory_mib=int(memory),
                         utilization=int(utilization), compute_process=uuid in busy))
    return rows


def idle(gpu):
    return gpu['memory_mib'] <= 512 and gpu['utilization'] <= 5 and not gpu['compute_process']


@contextmanager
def acquire_gpu(args):
    streaks = defaultdict(int)
    while True:
        try:
            rows = gpu_status()
        except (subprocess.SubprocessError, ValueError) as error:
            streaks.clear()
            write_json(args.output / 'status.json', dict(state='waiting_gpu', query_error=str(error), time=now()))
            print(json.dumps(dict(event='WAIT_GPU_QUERY_ERROR', error=str(error), time=now())), flush=True)
            time.sleep(args.poll_seconds)
            continue
        for gpu in rows:
            if args.gpu is not None and gpu['index'] not in args.gpu:
                continue
            uuid = gpu['uuid']
            streaks[uuid] = streaks[uuid] + 1 if idle(gpu) else 0
            if streaks[uuid] < args.idle_checks:
                continue
            lock = open('/tmp/mhinet_gpu_queue_' + uuid + '.lock', 'a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                current = next(row for row in gpu_status() if row['uuid'] == uuid)
                if not idle(current):
                    streaks[uuid] = 0
                    lock.close()
                    continue
                print(json.dumps(dict(event='GPU_ACQUIRED', gpu=current, time=now())), flush=True)
                yield current
                return
            finally:
                lock.close()
        write_json(args.output / 'status.json', dict(state='waiting_gpu', gpus=rows, time=now(), allowed_gpus=args.gpu))
        print(json.dumps(dict(event='WAIT_GPU', gpus=rows, time=now(), allowed_gpus=args.gpu)), flush=True)
        time.sleep(args.poll_seconds)


def measure_call(callback, device):
    import torch
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start.record(torch.cuda.current_stream(device))
    result = callback()
    end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    return result, dict(stream_ms=start.elapsed_time(end), wall_ms=(time.perf_counter() - started) * 1000)


@contextmanager
def matcher_probe():
    """Keep tiny mask references/count dictionaries, without full fine traces."""
    import mhinet.downstream.semidense as module
    original_masks = module.inference_overlap_masks
    original_coarse = module.match_d8
    captured = dict(masks={}, coarse=None)

    def masks(H, resolution, enabled, *, device):
        result = original_masks(H, resolution, enabled, device=device)
        captured['masks'][resolution] = result
        return result

    def coarse(*a, **kw):
        result = original_coarse(*a, **kw)
        captured['coarse'] = dict(result.diagnostics)
        return result

    module.inference_overlap_masks = masks
    module.match_d8 = coarse
    try:
        yield captured
    finally:
        module.inference_overlap_masks = original_masks
        module.match_d8 = original_coarse


def probe_statistics(captured):
    return dict(coarse=captured['coarse'], support={str(resolution): dict(
        cells_a=int(masks.a.sum()), cells_b=int(masks.b.sum()), total_cells=resolution * resolution,
        fraction_a=float(masks.a.float().mean()), fraction_b=float(masks.b.float().mean()))
        for resolution, masks in captured['masks'].items()})


def infer_variants(model, features, sizes, device, *, measure=measure_call):
    before = feature_sha256(features)
    predictions = {}
    for variant, flags in VARIANTS.items():
        with matcher_probe() as captured:
            results, timing = measure(lambda: model.matcher.infer(features, sizes, **flags), device)
        predictions[variant] = dict(prediction=results[0], timing=timing, diagnostics=probe_statistics(captured))
    after = feature_sha256(features)
    if before != after:
        raise RuntimeError('Inference changed shared feature bytes')
    return predictions, before


def summarize(records):
    import numpy as np
    from scripts.mrsi_gt_metrics import aggregate
    from scripts.semidense_timing import distribution
    result = aggregate(records)
    rmse = [record['metrics']['rmse_correct5_or_failure10'] for record in records
            if record['metrics']['rmse_correct5_or_failure10'] is not None]
    result.update(rmse_correct5_or_failure10=float(np.mean(rmse)) if rmse else None,
                  rmse_defined_pairs=len(rmse), rmse_undefined_pairs=len(records) - len(rmse),
                  coordinate_unit='stored_512_target_pixels',
                  timing={name: distribution([record['timing'][name] for record in records])
                          for name in ('shared_wall_ms', 'matcher_wall_ms', 'wall_total_ms',
                                       'shared_stream_ms', 'matcher_stream_ms', 'stream_total_ms')})
    return result


def stage_summary(records, stage):
    return summarize([dict(record, metrics=record['stages'][stage]) for record in records])


def worker(args, plan):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO))
    from mhinet.config import RuntimePaths
    from mhinet.pretraining.model import build_shared_network
    from mhinet.downstream.training import SemidenseSystem
    from mhinet.downstream.semidense import SemidenseConfig, SemidenseMatcher
    from mhinet.downstream.supervision import legal_h
    from mhinet.dataio.data import load_rgb_bicubic
    from mhinet.downstream.metrics import dense_metrics
    from scripts.mrsi_gt_metrics import score_matches, conditional_rmse

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    budget_bytes = int(args.memory_budget_gib * 1024 ** 3)
    reserve_bytes = 4 * 1024 ** 3
    if free_bytes < budget_bytes + reserve_bytes:
        raise RuntimeError(f'Insufficient free GPU memory for this job: free={free_bytes}, '
                           f'budget={budget_bytes}, required_neighbor_reserve={reserve_bytes}')
    torch.cuda.set_per_process_memory_fraction(budget_bytes / total_bytes, device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_registration = dict(budget_gib=args.memory_budget_gib, budget_bytes=budget_bytes,
                               allocator_fraction=budget_bytes / total_bytes, total_bytes=total_bytes,
                               free_bytes_at_start=free_bytes, required_neighbor_reserve_bytes=reserve_bytes,
                               note='PyTorch allocator cap; other jobs are never stopped. Concurrent GPU timing is descriptive only.')
    for dataset in args.datasets:
        for variant in VARIANTS:
            if (args.output / f'{variant}_{dataset}').exists():
                raise FileExistsError('Preserving existing evaluation; use a new output directory')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    runtime = replace(RuntimePaths.from_json(state['metadata']['config']['runtime']), device=str(device))
    config = SemidenseConfig(**plan['matcher'])
    shared, _ = build_shared_network(runtime, lora=False)
    model = SemidenseSystem(shared, SemidenseMatcher(config)).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    checkpoint_state_sha = state_sha256(state['model'])
    del state
    initial_state_sha = state_sha256(model.state_dict())
    if checkpoint_state_sha != initial_state_sha:
        raise RuntimeError('Loaded model raw state differs from checkpoint')
    if any(module.training for module in model.modules()):
        raise RuntimeError('Evaluation model contains training-mode modules')
    old_infer = load_legacy_infer()
    registrations = {}
    completed_reports = {}
    all_validation = {}
    with torch.no_grad():
        # Warm the feature kernels without an extra forward of any test pair.
        warm_features = model(torch.zeros((1, 2, 3, 512, 512), device=device))
        torch.cuda.synchronize(device)
        del warm_features
        for dataset in args.datasets:
            rows = manifest_rows(dataset)
            if args.limit:
                rows = rows[:args.limit]
            data_root = DATASETS[dataset]['path'].parent
            baseline_root = Path(plan['datasets'][dataset]['baseline_root'])
            streams = {}
            records = {variant: [] for variant in VARIANTS}
            validated = 0
            legacy_validated = False
            try:
                for variant, flags in VARIANTS.items():
                    output = args.output / f'{variant}_{dataset}'
                    output.mkdir()
                    (output / 'matches').mkdir()
                    registration = dict(variant=variant, dataset=dataset, flags=flags, checkpoint=plan['checkpoint'],
                                        manifest=plan['datasets'][dataset], matcher=plan['matcher'], protocol=plan,
                                        model_state_sha256_before=initial_state_sha,
                                        checkpoint_model_state_sha256=checkpoint_state_sha,
                                        versions=plan['versions'], gpu=torch.cuda.get_device_name(device),
                                        visible_device=os.environ.get('CUDA_VISIBLE_DEVICES', '<all>'),
                                        logical_device=str(device), memory_budget=memory_registration,
                                        launch_git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                                        started=now())
                    write_json(output / 'registration.json', registration)
                    registrations[(variant, dataset)] = registration
                    streams[variant] = (output / 'pairs.jsonl').open('w')
                    pair_map = [dict(index=index, pair_id=row['pair_id'], npz=f'matches/{index:05d}.npz',
                                     image_A=str(data_root / row['image_A']), image_B=str(data_root / row['image_B']))
                                for index, row in enumerate(rows)]
                    write_json(output / 'match_index.json', pair_map)
                for index, row in enumerate(rows):
                    sizes = [(tuple(row['size_A']), tuple(row['size_B']))]
                    image_paths = [data_root / row['image_' + side] for side in ('A', 'B')]
                    images = torch.stack([load_rgb_bicubic(path, 512) for path in image_paths])[None].to(device)
                    features, shared_time = measure_call(lambda: model(images), device)
                    before_legacy_hash = feature_sha256(features) if index == 0 else None
                    legacy = old_infer(model.matcher, features, sizes)[0] if index == 0 else None
                    predictions, feature_hashes = infer_variants(model, features, sizes, device)
                    if before_legacy_hash is not None and before_legacy_hash != feature_hashes:
                        raise RuntimeError('Legacy inference changed shared feature bytes')
                    h0_fit_valid = bool(features['stage1_valid'][0])
                    h0_geometry_valid = legal_h(features['H0_norm'][0])
                    h0_valid = h0_fit_valid and h0_geometry_valid
                    with np.load(baseline_root / 'matches' / f'{index:05d}.npz') as baseline:
                        checked = check_array_equal(predictions['A']['prediction'], baseline, label=row['pair_id'])
                    validated += 1
                    if legacy is not None:
                        check_legacy_equal(predictions['A']['prediction'], legacy)
                        legacy_validated = True
                        del legacy
                    input_hashes = {side: sha_file(path) for side, path in zip(('A', 'B'), image_paths)}
                    for variant in VARIANTS:
                        value = predictions[variant]
                        prediction = value['prediction']
                        a, b = [prediction[key].cpu().numpy() for key in ('points_a', 'points_b')]
                        metrics = score_matches(a, b, *sizes[0], row['H_A_to_B'])
                        if metrics['sr']['3'] and metrics['ncm']['5'] == 0:
                            metrics.update(rmse_correct5_or_failure10=None, rmse_protocol_success=True, rmse_ncm=0)
                        else:
                            metrics.update(conditional_rmse(a, b, *sizes[0], row['H_A_to_B'], success_override=metrics['sr']['3']))
                        stages = {}
                        for stage, ka, kb in [('coarse', 'coarse_a', 'coarse_b'), ('fine', 'fine_all_a', 'fine_all_b')]:
                            aa = prediction[ka].cpu().numpy() if ka in prediction else np.empty((0, 2))
                            bb = prediction[kb].cpu().numpy() if kb in prediction else np.empty((0, 2))
                            sm = score_matches(aa, bb, *sizes[0], row['H_A_to_B'])
                            if sm['sr']['3'] and sm['ncm']['5'] == 0:
                                sm.update(rmse_correct5_or_failure10=None, rmse_protocol_success=True, rmse_ncm=0)
                            else:
                                sm.update(conditional_rmse(aa, bb, *sizes[0], row['H_A_to_B'], success_override=sm['sr']['3']))
                            stages[stage] = sm
                        mask_metrics = dense_metrics(row, data_root, a, b)
                        mask_metrics['coordinate_unit'] = 'stored_512_target_pixels'
                        timing = dict(shared_wall_ms=shared_time['wall_ms'], shared_stream_ms=shared_time['stream_ms'],
                                      matcher_wall_ms=value['timing']['wall_ms'], matcher_stream_ms=value['timing']['stream_ms'],
                                      wall_total_ms=shared_time['wall_ms'] + value['timing']['wall_ms'],
                                      stream_total_ms=shared_time['stream_ms'] + value['timing']['stream_ms'],
                                      method='shared_once_cuda_events; estimated_standalone_shared_plus_matcher')
                        record = dict(index=index, pair_id=row['pair_id'], group=row.get('transform_group', 'all'),
                                      variant=variant, flags=VARIANTS[variant], metrics=metrics, stages=stages,
                                      mask_metrics=mask_metrics, timing=timing, failure_reason=prediction['failure_reason'],
                                      selection=value['diagnostics'], feature_sha256=feature_hashes,
                                      shared_features_unchanged=True, image_sha256=input_hashes,
                                      h0_valid_predicted=h0_valid, h0_fit_valid=h0_fit_valid,
                                      h0_geometry_valid=h0_geometry_valid,
                                      a_baseline_exact=dict(checked=checked, old_npz=str(baseline_root / 'matches' / f'{index:05d}.npz')),
                                      qrru_outside_samples=prediction.get('qrru_outside_samples', 0),
                                      rejected_qrru=prediction.get('rejected_qrru', 0))
                        records[variant].append(record)
                        streams[variant].write(json.dumps(record) + '\n')
                        streams[variant].flush()
                        np.savez_compressed(args.output / f'{variant}_{dataset}' / 'matches' / f'{index:05d}.npz',
                                            points_a=a, points_b=b, confidence=prediction['confidence'].cpu().numpy(),
                                            pair_id=np.asarray(row['pair_id']))
                    if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(rows):
                        progress = dict(state='running', dataset=dataset, completed=index + 1, total=len(rows),
                                        a_baseline_exact_pairs=validated, time=now(),
                                        variants={variant: dict(precision=records[variant][-1]['metrics']['precision'],
                                                               sr3=records[variant][-1]['metrics']['sr']['3'],
                                                               nm=records[variant][-1]['metrics']['matches']) for variant in VARIANTS})
                        write_json(args.output / 'status.json', progress)
                        print(json.dumps(progress), flush=True)
                    del features, predictions, images
            finally:
                for stream in streams.values():
                    stream.close()
            all_validation[dataset] = dict(a_baseline_exact_pairs=validated, first_pair_legacy_exact=legacy_validated,
                                           feature_hash_checked_pairs=len(rows))
            for variant in VARIANTS:
                values = records[variant]
                report = dict(complete=False, registration=registrations[(variant, dataset)],
                              overall=summarize(values),
                              by_group={group: summarize([record for record in values if record['group'] == group])
                                        for group in sorted({record['group'] for record in values})},
                              stages={stage: stage_summary(values, stage) for stage in ('coarse', 'fine')},
                              h0_validity=dict(valid_pairs=sum(record['h0_valid_predicted'] for record in values),
                                               invalid_pairs=sum(not record['h0_valid_predicted'] for record in values),
                                               by_validity={str(valid): summarize(selected) if selected else None
                                                            for valid in (True, False)
                                                            for selected in [[record for record in values if record['h0_valid_predicted'] == valid]]}),
                              failure_reasons=dict(Counter(record['failure_reason'] for record in values)),
                              validation=all_validation[dataset], finished=now())
                completed_reports[(variant, dataset)] = report
                write_json(args.output / f'{variant}_{dataset}' / 'report.json', report)
    final_state_sha = state_sha256(model.state_dict())
    if initial_state_sha != final_state_sha:
        raise RuntimeError('Evaluation changed model parameters or buffers')
    memory_registration.update(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
    evidence = dict(complete=True, model_state_sha256_before=initial_state_sha, model_state_sha256_after=final_state_sha,
                    checkpoint_model_state_sha256=checkpoint_state_sha, model_unchanged=True,
                    feature_hash_checked_pairs=sum(value['feature_hash_checked_pairs'] for value in all_validation.values()),
                    baseline_validation=all_validation, plan_fingerprint=plan['fingerprint'],
                    memory=memory_registration, finished=now())
    write_json(args.output / 'validation.json', evidence)
    for (variant, dataset), report in completed_reports.items():
        report['complete'] = True
        report['registration']['model_state_sha256_after'] = final_state_sha
        report['validation']['model_unchanged'] = True
        write_json(args.output / f'{variant}_{dataset}' / 'registration.json', report['registration'])
        write_json(args.output / f'{variant}_{dataset}' / 'report.json', report)
    write_json(args.output / 'status.json', dict(state='complete', groups=len(completed_reports), time=now(),
                                               scope=plan['scope'], validation=evidence))
    print('ALL_VARIANTS_COMPLETE ' + str(args.output), flush=True)


def queue(args, plan):
    if (args.output / 'validation.json').exists():
        if json.loads((args.output / 'validation.json').read_text()).get('complete'):
            print('ALREADY_COMPLETE ' + str(args.output), flush=True)
            return
    with acquire_gpu(args) as gpu:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            env.pop(key, None)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--device', 'cuda:0',
                   '--checkpoint', str(args.checkpoint), '--baseline-root', str(args.baseline_root),
                   '--output', str(args.output), '--datasets', *args.datasets, '--limit', str(args.limit),
                   '--memory-budget-gib', str(args.memory_budget_gib)]
        write_json(args.output / 'status.json', dict(state='starting', gpu=gpu, time=now(), fingerprint=plan['fingerprint']))
        with (args.output / 'worker.log').open('a') as log:
            result = subprocess.run(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(args.output / 'status.json', dict(state='failed', exit_code=result.returncode, time=now(),
                                                       log=str(args.output / 'worker.log')))
            raise RuntimeError('Ablation worker failed; see ' + str(args.output / 'worker.log'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--baseline-root', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/h0_position_ablation_512_20261013')
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Separate limited-validation scope; 0 evaluates both full cohorts.')
    parser.add_argument('--gpu', type=int, nargs='+', help='Restrict idle-GPU queue to these physical GPU indices.')
    parser.add_argument('--device', help='Explicitly run on this CUDA device instead of waiting for an idle GPU.')
    parser.add_argument('--memory-budget-gib', type=float, default=6., help='Per-process PyTorch GPU allocator budget; keeps 4 GiB free for neighboring jobs.')
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--idle-checks', type=int, default=3)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.poll_seconds < 1 or args.idle_checks < 1:
        parser.error('limit must be nonnegative; polling arguments must be positive')
    if not 0 < args.memory_budget_gib < float('inf'):
        parser.error('memory-budget-gib must be finite and positive')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate dataset names')
    if args.device is not None and not args.device.startswith('cuda:'):
        parser.error('Official evaluation requires a CUDA device; CPU is reserved for preflight/tests')
    if args.gpu is not None and any(index < 0 for index in args.gpu):
        parser.error('GPU indices must be nonnegative')
    args.output = args.output.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'queue.lock').open('a')
    try:
        if not args.worker:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
        print(json.dumps(dict(event='PREFLIGHT_OK', fingerprint=plan['fingerprint'], scope=plan['scope'],
                              datasets=plan['datasets'], variants=VARIANTS)), flush=True)
        if args.prepare_only:
            return 0
        if args.device:
            worker(args, plan)
        else:
            queue(args, plan)
        return 0
    except Exception as error:
        if not args.prepare_only:
            write_json(args.output / 'status.json', dict(state='failed', error=f'{type(error).__name__}: {error}', time=now()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
