"""Flow-loss pilot diagnostics, paired comparisons, and matching controls.

No GT enters the shared forward or matching inference. The ``flow`` command
uses only D16 Head output: it never decodes CGMDP/D8 descriptors. Its D8
statistics describe the actual interpolated Flow prior, not descriptor matches.
All new outputs are exclusive directories and never overwrite an earlier run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader, Subset

from mhinet.config import RuntimePaths, sha256_file
from mhinet.engine.checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers
from mhinet.pretraining.loss import sample
from mhinet.v2 import TASK, SCHEMA
from mhinet.v2.config import SemidenseConfigV2
from mhinet.v2.coarse_matcher import flow_prior_from_shared
from mhinet.v2.data import RGB512Dataset
from mhinet.v2.evaluate import _read_configuration, _code_identity
from mhinet.v2.feature_provider import build_v2_shared
from mhinet.v2.semidense import SemidenseMatcherV2
from mhinet.v2.supervision import to_uv, to_norm
from mhinet.v2.training import SemidenseSystemV2

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA256 = '26a24e5239a1b7a3719c638ec7b040a8490c226a7341a0606137d25f0c63c33a'
VAL_SHA256 = 'ad281e31dfd612a239ddfa63f25a42812511e22445f571359477e709c8207f5b'
SOURCE = ROOT/'outputs/mhi_v2/Flow0/P5/flow_head_rgb512_seed42_5ep_20261020/best.pt'
HARD_CASES = ROOT/'versions/v2/evidence/flow_loss_pilot_20261020/hard_cases.json'
CONFIDENCE_THRESHOLDS = (0.25, 0.5, 0.75)
TAIL_THRESHOLDS = (8., 16., 32.)
QUANTILES = (0.5, 0.9, 0.95, 0.99)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.temporary')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def reserve_output(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_hard_cases(path):
    payload = json.loads(Path(path).read_text())
    ids = payload.get('pair_ids') if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or len(ids) != 52 or len(set(ids)) != 52 or not all(isinstance(i, str) for i in ids):
        raise ValueError('The preregistered hard-case file must contain 52 unique pair_ids')
    return ids


def point_statistics(error, confidence, valid, deployed_valid=None):
    """Fixed GT denominators; confidence strata always include their coverage.

    Nonfinite predictions on GT-valid points fail instead of silently vanishing.
    The all-GT statistic is never conditioned on predicted validity/confidence.
    """
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    deployed_valid = np.ones_like(valid) if deployed_valid is None else np.asarray(deployed_valid, dtype=bool).reshape(-1)
    if not (error.shape == confidence.shape == valid.shape == deployed_valid.shape):
        raise ValueError('Point diagnostic arrays have inconsistent shapes')
    if not np.isfinite(error[valid]).all() or not np.isfinite(confidence[valid]).all():
        raise FloatingPointError('Nonfinite prediction on a fixed GT-matchable query')
    total = int(valid.sum())
    def subset(mask):
        values = error[mask]
        return dict(points=len(values), coverage_of_gt=len(values)/max(1, total),
            epe_px=float(values.mean()) if len(values) else None,
            quantiles_px={str(q): float(np.quantile(values, q)) if len(values) else None for q in QUANTILES},
            tail_fraction={str(t): float((values > t).mean()) if len(values) else None for t in TAIL_THRESHOLDS})
    denominator = float(confidence[valid].sum())
    weighted = {str(t): float(confidence[valid & (error > t)].sum()/denominator)
        if denominator > 0 else None for t in TAIL_THRESHOLDS}
    return dict(all_gt_matchable=subset(valid), deployed=subset(valid & deployed_valid),
        confidence_strata={str(t): subset(valid & (confidence >= t)) for t in CONFIDENCE_THRESHOLDS},
        deployed_confidence_strata={str(t): subset(valid & deployed_valid & (confidence >= t)) for t in CONFIDENCE_THRESHOLDS},
        confidence_weighted_tail_fraction=weighted,
        confidence_weighted_epe_px=float((confidence[valid]*error[valid]).sum()/denominator) if denominator > 0 else None,
        mean_confidence_on_gt=float(confidence[valid].mean()) if total else None)


def _point_summary(pools):
    return point_statistics(np.concatenate(pools['error']), np.concatenate(pools['confidence']),
        np.ones(sum(len(x) for x in pools['error']), dtype=bool), np.concatenate(pools['deployed']))


def _numpy(value):
    return value.detach().float().cpu().numpy()


def _boundary_distances(target_norm, mask, hw):
    """Signed distances to the image box, label center box, and GT mask.

    Overlap distance uses a signed pixel-center Euclidean distance transform,
    bilinearly sampled at the continuous GT target; it is diagnostic only.
    """
    from scipy.ndimage import distance_transform_edt
    target = np.asarray(target_norm, dtype=np.float32).reshape(-1, 2)
    xy = (target+1)*256-.5
    image = np.minimum(xy, 511-xy).min(-1)
    margin = np.array([256/hw[1]-.5, 256/hw[0]-.5], dtype=np.float32)
    box = np.minimum(xy-margin, 511-margin-xy).min(-1)
    binary = mask.detach().float().cpu().numpy().reshape(512, 512) > .999
    inside = distance_transform_edt(np.pad(binary, 1, constant_values=False))[1:-1, 1:-1]
    outside = distance_transform_edt(np.pad(~binary, 1, constant_values=False))[1:-1, 1:-1]
    signed = torch.from_numpy((inside-outside).astype(np.float32))[None]
    overlap = sample(signed, torch.from_numpy(target))[:, 0].numpy()
    return dict(gt_target_image_boundary_distance_px=image,
        gt_target_label_box_boundary_distance_px=box, gt_target_overlap_boundary_distance_px=overlap)


def load_checkpoint_payload(path, expected_source_sha256=SOURCE_SHA256):
    path = Path(path).expanduser().resolve()
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != CHECKPOINT_FORMAT or payload.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Unsupported checkpoint format/version')
    metadata = payload.get('metadata', {})
    if metadata.get('task') != TASK or metadata.get('schema') != SCHEMA:
        raise ValueError('Pilot checkpoint must retain task=mhi_semidense_v2 and schema=mhi.v2-run.v1')
    identity = sha256_file(path)
    if identity != expected_source_sha256 and metadata.get('source_sha256') != expected_source_sha256:
        raise ValueError('Pilot does not descend directly from the pinned Flow-best source')
    config, config_path, runtime_path = _read_configuration(metadata)
    matcher = SemidenseConfigV2.from_dict(metadata.get('matcher'))
    if matcher.geometry_mode != 'flow_d16' or matcher.fine_prior != 'direct_d8' or matcher.max_output_matches != 2048:
        raise ValueError('Flow-loss pilot requires flow_d16/Direct-D8/512/default 2048 output budget')
    if config.get('stage') != 'flow_head':
        raise ValueError('Flow-loss diagnostics require a Flow-head-only checkpoint')
    return payload, dict(checkpoint=str(path), checkpoint_sha256=identity,
        source_sha256=expected_source_sha256, config=config,
        config_sha256=json_hash(config), runtime=str(runtime_path), runtime_sha256=sha256_file(runtime_path),
        matcher=asdict(matcher), optimizer_step=payload.get('progress', {}).get('optimizer_step'),
        experiment_kind=metadata.get('experiment_kind'), flow_loss_pilot=metadata.get('flow_loss_pilot'))


def build_system(path, device, expected_source_sha256=SOURCE_SHA256, runtime_override=None):
    payload, registration = load_checkpoint_payload(path, expected_source_sha256)
    original_runtime = RuntimePaths.from_json(registration['runtime'])
    runtime = original_runtime
    if runtime_override is not None:
        requested = RuntimePaths.from_json(runtime_override)
        allowed = {'data_root', 'split_manifest', 'output_root', 'device', 'source_path'}
        changes = [name for name in asdict(runtime)
            if getattr(runtime, name) != getattr(requested, name)]
        if set(changes)-allowed:
            raise ValueError(f'Evaluation runtime changed model resources: {sorted(set(changes)-allowed)}')
        runtime = requested
        registration['runtime_override'] = dict(path=str(Path(runtime_override).resolve()),
            sha256=sha256_file(runtime_override), changed_fields=changes,
            permitted_effect='I/O paths and device only; prediction weights and architecture unchanged')
    runtime = replace(runtime, device=str(device))
    metadata = payload['metadata']
    if metadata.get('dino_sha256') != sha256_file(runtime.dino_checkpoint):
        raise ValueError('DINO identity differs from checkpoint metadata')
    shared, provider = build_v2_shared(runtime, geometry_mode='flow_d16')
    system = SemidenseSystemV2(shared, SemidenseMatcherV2(SemidenseConfigV2.from_dict(registration['matcher'])), stage='flow_head').to(device)
    system.load_state_dict(payload['model'], strict=True)
    system.requires_grad_(False).eval()
    return system, registration, provider


def assert_frozen_states_equal(state_a, state_b):
    """The endpoint-loss arms may change only the independently owned head."""
    if set(state_a) != set(state_b):
        raise ValueError('Checkpoint tensor keys differ')
    frozen = [key for key in state_a if not key.startswith('shared.stage1_head.')]
    mismatches = [key for key in frozen if state_a[key].dtype != state_b[key].dtype
        or not torch.equal(state_a[key], state_b[key])]
    if mismatches:
        raise ValueError(f'Non-head tensors changed: {mismatches[:10]}')
    return dict(checked_frozen_tensors=len(frozen), mismatched_tensors=0)


def _configure_determinism():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '2'))))
    torch.manual_seed(123)


def evaluate_flow(args):
    # Import the shared supervision API only for the flow subcommand; compare
    # and matching-plan remain usable while training workers use objective.py.
    from .objective import flow_point_diagnostics, flow_supervision
    _configure_determinism()
    manifest = args.manifest.expanduser().resolve()
    if sha256_file(manifest) != args.expected_manifest_sha256:
        raise ValueError('Fixed RGB512 val manifest SHA mismatch')
    dataset = RGB512Dataset(manifest, image_size=512)
    if len(dataset) != 5106:
        raise ValueError('Expected the complete 5106-pair validation manifest')
    hard_ids = read_hard_cases(args.hard_cases)
    rows = {row['pair_id']: row for row in map(json.loads, manifest.read_text().splitlines())}
    if not set(hard_ids) <= rows.keys():
        raise ValueError('Hard-case list includes non-validation pair IDs')
    total = min(args.limit, len(dataset)) if args.limit else len(dataset)
    system, model_registration, provider = build_system(args.checkpoint, args.device, args.expected_source_sha256, args.runtime)
    output = reserve_output(args.output)
    (output/'hard_cases').mkdir()
    registration = dict(schema='mhi.flow-loss.point-evaluation.v1', task=TASK,
        **model_registration, manifest=str(manifest), manifest_sha256=sha256_file(manifest),
        hard_cases=str(args.hard_cases.resolve()), hard_cases_sha256=sha256_file(args.hard_cases),
        hard_case_pair_ids=hard_ids, selected_pairs=total, full_manifest_pairs=5106, limit=args.limit,
        code=_code_identity(), provider=provider, created_at=now(),
        d16_query_policy='Exact legacy flow_supervision GT center box +/- (1-1/32), bilinear masks > .999',
        d8_query_policy='flow_supervision(hw=64) center-box GT queries; not coarse_labels mutual-nearest set',
        d8_prediction_policy='Actual flow_prior_from_shared displacement interpolation and validity; 8 input pixels per D8 unit',
        confidence_thresholds=list(CONFIDENCE_THRESHOLDS), tail_thresholds_px=list(TAIL_THRESHOLDS),
        all_gt_denominator_independent_of_prediction=True, head_only_forward=True,
        gt_use='Reporting after shared(images,pyramid_scales=()); GT never enters infer/forward',
        validation_role='Diagnostics only; hard cases must never select loss hyperparameters')
    write_json(output/'registration.json', registration)
    state = dict(status='running', pid=os.getpid(), pairs=0, total_pairs=total, checked_at=now())
    write_json(output/'status.json', state)
    records = []
    pools = {name: defaultdict(list) for name in ('d16', 'd8')}
    started = time.perf_counter()
    loader = DataLoader(Subset(dataset, range(total)), batch_size=1, num_workers=args.workers,
        pin_memory=str(args.device).startswith('cuda'))
    try:
        with torch.inference_mode(), (output/'pairs.jsonl').open('x') as stream:
            for index, batch in enumerate(loader):
                images = batch['images'].to(args.device, non_blocking=True)
                # Only image tensors enter the predictor. GT is moved afterward.
                shared = system.shared(images, pyramid_scales=())
                if shared['pyramid'] or shared['H0_norm'] is not None or shared['call_counts'].get('homography_fitter', 0):
                    raise RuntimeError('Flow diagnostic unexpectedly ran CGMDP or an H fitter')
                H = batch['H_gt_norm'].to(args.device)
                ma, mb = batch['mask_A_overlap'].to(args.device), batch['mask_B_overlap'].to(args.device)
                point = flow_point_diagnostics(shared['stage1'], H, ma, mb, input_size=512)
                if not torch.isfinite(point['pred_target_norm']).all():
                    raise FloatingPointError('Nonfinite Flow target prediction')
                err16 = _numpy(point['epe_px'][0])
                conf16 = _numpy(point['matchability'][0])
                gt16 = point['gt_valid'][0].cpu().numpy().astype(bool)
                center_valid = point['predicted_valid'][0].cpu().numpy().astype(bool)
                deployed16 = shared['flow16_valid'][0, 0].flatten().cpu().numpy().astype(bool)
                d16 = point_statistics(err16, conf16, gt16, deployed16)
                d16['predicted_center_box_coverage_of_gt'] = float((center_valid & gt16).sum()/max(1, gt16.sum()))
                prior = flow_prior_from_shared(shared, 0, (64, 64),
                    min_matchability=model_registration['matcher'].get('flow_min_matchability', 0.))
                target8, gt8_tensor = flow_supervision(H, ma, mb, hw=(64, 64))
                gt8 = gt8_tensor[0].cpu().numpy().astype(bool)
                target8 = target8[0]
                predicted8 = to_norm(prior.target_uv, (64, 64))
                residual8 = (predicted8-target8)*256
                err8 = _numpy(residual8.norm(dim=-1))
                conf8 = _numpy(prior.confidence)
                deployed8 = prior.valid.cpu().numpy().astype(bool)
                d8 = point_statistics(err8, conf8, gt8, deployed8)
                pair_id = batch['pair_id'][0]
                row = rows[pair_id]
                record = dict(pair_id=pair_id, parent_group=batch['parent_group'][0], source_dataset=row['source_dataset'],
                    flow_epe_px=d16['all_gt_matchable']['epe_px'], d16=d16, d8=d8)
                for name, error, confidence, gt, deployed in (('d16', err16, conf16, gt16, deployed16),
                        ('d8', err8, conf8, gt8, deployed8)):
                    pools[name]['error'].append(error[gt])
                    pools[name]['confidence'].append(confidence[gt])
                    pools[name]['deployed'].append(deployed[gt])
                if pair_id in hard_ids:
                    payload = dict(H_gt_norm=_numpy(H[0]))
                    data16 = dict(source_norm=_numpy(point['source_norm'][0]),
                        gt_target_norm=_numpy(point['gt_target_norm'][0]), pred_target_norm=_numpy(point['pred_target_norm'][0]),
                        residual_px=_numpy(point['residual_px'][0]), epe_px=err16,
                        matchability=conf16, gt_valid=gt16, predicted_center_valid=center_valid, deployed_valid=deployed16)
                    data8 = dict(source_norm=_numpy(normalized_cell_centers(64, 64, device=H.device).reshape(-1, 2)),
                        gt_target_norm=_numpy(target8), pred_target_norm=_numpy(predicted8),
                        residual_px=_numpy(residual8), epe_px=err8, matchability=conf8, gt_valid=gt8, deployed_valid=deployed8)
                    for name, values, hw in (('d16', data16, (32, 32)), ('d8', data8, (64, 64))):
                        values.update(_boundary_distances(values['gt_target_norm'], mb[0], hw))
                        payload.update({f'{name}_{key}': value for key, value in values.items()})
                    artifact = f'hard_cases/{hard_ids.index(pair_id):03d}.npz'
                    np.savez_compressed(output/artifact, **payload)
                    record['hard_case_npz'] = artifact
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                records.append(record)
                if len(records) % 64 == 0 or len(records) == total:
                    stream.flush()
                    state.update(pairs=len(records), checked_at=now(), elapsed_seconds=time.perf_counter()-started)
                    write_json(output/'status.json', state)
        values = np.array([r['flow_epe_px'] for r in records if r['flow_epe_px'] is not None])
        summary = dict(schema=registration['schema'], checkpoint_sha256=registration['checkpoint_sha256'],
            source_sha256=args.expected_source_sha256, manifest_sha256=registration['manifest_sha256'],
            hard_cases_sha256=registration['hard_cases_sha256'], pairs=len(records), complete=len(records) == total,
            scope='full_val' if total == 5106 else 'prefix_diagnostic',
            macro_flow_epe_px=float(values.mean()) if len(values) else None,
            per_pair_epe_quantiles_px={str(q): float(np.quantile(values, q)) if len(values) else None for q in QUANTILES},
            point_level={name: _point_summary(pools[name]) for name in ('d16', 'd8')},
            by_source_dataset={source: dict(pairs=sum(r['source_dataset'] == source for r in records),
                macro_flow_epe_px=float(np.mean([r['flow_epe_px'] for r in records
                    if r['source_dataset'] == source and r['flow_epe_px'] is not None])))
                for source in sorted({r['source_dataset'] for r in records})},
            hard_cases_evaluated=sum('hard_case_npz' in r for r in records),
            seconds=time.perf_counter()-started, code_sha256=registration['code']['code_sha256'])
        write_json(output/'summary.json', summary)
        state.update(status='completed', pairs=len(records), checked_at=now(), summary=str(output/'summary.json'))
        write_json(output/'status.json', state)
        print(json.dumps(summary, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        state.update(status='failed', checked_at=now(), error=f'{type(error).__name__}: {error}')
        write_json(output/'status.json', state)
        raise


def _read_evaluation(path):
    path = Path(path).resolve()
    summary = json.loads((path/'summary.json').read_text())
    registration = json.loads((path/'registration.json').read_text())
    rows = [json.loads(line) for line in (path/'pairs.jsonl').read_text().splitlines() if line.strip()]
    if summary.get('scope') != 'full_val' or summary.get('pairs') != 5106 or len(rows) != 5106:
        raise ValueError('Paired comparison requires completed 5106-pair full-val evaluations')
    return summary, registration, rows


def paired_group_bootstrap(rows, deltas, *, iterations=2000, seed=42):
    """Resample original source-image groups within each source dataset."""
    groups = defaultdict(lambda: defaultdict(list))
    for row, delta in zip(rows, deltas):
        if delta is not None:
            groups[row['source_dataset']][row['parent_group']].append(delta)
    rng = np.random.default_rng(seed)
    if not groups:
        return dict(mean_delta=None, source_image_group_bootstrap_95ci=None, iterations=iterations,
            seed=seed, resampling_unit='original source image, stratified by source_dataset')
    estimates = []
    for _ in range(iterations):
        total = 0.; count = 0
        for source in sorted(groups):
            values = list(groups[source].values())
            selected = rng.integers(0, len(values), len(values))
            for index in selected:
                total += sum(values[index]); count += len(values[index])
        estimates.append(total/max(1, count))
    present = [x for x in deltas if x is not None]
    return dict(mean_delta=float(np.mean(present)) if present else None,
        source_image_group_bootstrap_95ci=np.quantile(estimates, [.025, .975]).tolist() if estimates else None,
        iterations=iterations, seed=seed, resampling_unit='original source image, stratified by source_dataset')


def compare(args):
    evaluations = {name: _read_evaluation(path) for name, path in
        (('arm_a', args.arm_a), ('arm_b', args.arm_b), ('source', args.source))}
    a_sum, a_reg, a = evaluations['arm_a']; b_sum, b_reg, b = evaluations['arm_b']; source_sum, source_reg, source = evaluations['source']
    ids = [row['pair_id'] for row in source]
    for name, (summary, reg, rows) in evaluations.items():
        if [row['pair_id'] for row in rows] != ids:
            raise ValueError(f'{name} query/pair order differs')
        if reg['manifest_sha256'] != VAL_SHA256 or reg['source_sha256'] != SOURCE_SHA256:
            raise ValueError('Unpinned source or manifest in paired evaluation')
        if reg['hard_cases_sha256'] != source_reg['hard_cases_sha256']:
            raise ValueError('Hard-case registry differs')
        if any(reg.get(key) != source_reg.get(key) for key in ('d16_query_policy', 'd8_query_policy')):
            raise ValueError('Fixed GT query policies differ')
    if source_reg['checkpoint_sha256'] != SOURCE_SHA256:
        raise ValueError('Source evaluation is not the pinned Flow-best checkpoint')
    deltas = [new['flow_epe_px']-old['flow_epe_px'] if new['flow_epe_px'] is not None and old['flow_epe_px'] is not None else None
        for old, new in zip(a, b)]
    output = reserve_output(args.output)
    report = dict(schema='mhi.flow-loss.paired-comparison.v1', created_at=now(),
        source_sha256=SOURCE_SHA256, manifest_sha256=VAL_SHA256, pairs=5106,
        evaluations={name: dict(path=str(Path(path).resolve()), checkpoint_sha256=evaluations[name][1]['checkpoint_sha256'])
            for name, path in (('arm_a', args.arm_a), ('arm_b', args.arm_b), ('source', args.source))},
        macro_epe={name: value[0]['macro_flow_epe_px'] for name, value in evaluations.items()},
        delta_arm_b_minus_arm_a=paired_group_bootstrap(a, deltas, iterations=args.bootstrap),
        delta_from_source={name: paired_group_bootstrap(source,
            [row['flow_epe_px']-base['flow_epe_px'] if row['flow_epe_px'] is not None and base['flow_epe_px'] is not None else None
             for row, base in zip(evaluations[name][2], source)], iterations=args.bootstrap) for name in ('arm_a', 'arm_b')},
        point_level={name: value[0]['point_level'] for name, value in evaluations.items()},
        by_source_dataset={name: value[0]['by_source_dataset'] for name, value in evaluations.items()},
        interpretation='Flow-only diagnostics; adoption requires the separately paired frozen-QRRU prior matching evaluation. Val hard cases do not tune hyperparameters.')
    with (output/'paired.jsonl').open('x') as stream:
        for base, old, new, delta in zip(source, a, b, deltas):
            stream.write(json.dumps(dict(pair_id=base['pair_id'], source_epe_px=base['flow_epe_px'],
                arm_a_epe_px=old['flow_epe_px'], arm_b_epe_px=new['flow_epe_px'], delta_B_minus_A=delta), allow_nan=False)+'\n')
    write_json(output/'comparison.json', report)
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


def negative_control(args):
    _configure_determinism()
    if sha256_file(args.manifest) != VAL_SHA256:
        raise ValueError('Negative control requires the fixed val manifest')
    payload_a, reg_a = load_checkpoint_payload(args.arm_a)
    payload_b, reg_b = load_checkpoint_payload(args.arm_b)
    audit = assert_frozen_states_equal(payload_a['model'], payload_b['model'])
    if reg_a['matcher'] != reg_b['matcher']:
        raise ValueError('Negative-control matcher settings differ between arms')
    del payload_a, payload_b
    system, _, _ = build_system(args.arm_a, args.device, runtime_override=args.runtime)
    system.matcher.config = replace(system.matcher.config, flow_guidance_enabled=False)
    dataset = RGB512Dataset(args.manifest, image_size=512)
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    output = reserve_output(args.output)
    (output/'arm_a').mkdir()
    ids = []
    with torch.inference_mode():
        for index in range(count):
            batch = dataset[index]; ids.append(batch['pair_id'])
            shared = system.shared(batch['images'][None].to(args.device))
            result = system.matcher.infer(shared)[0]
            np.savez_compressed(output/'arm_a'/f'{index:04d}.npz', **{
                key: result[key].cpu().numpy() for key in ('points_a', 'points_b', 'confidence', 'fine_points_b', 'lineage')})
        payload_b, _ = load_checkpoint_payload(args.arm_b)
        system.load_state_dict(payload_b['model'], strict=True)
        del payload_b
        for index in range(count):
            batch = dataset[index]
            shared = system.shared(batch['images'][None].to(args.device))
            result = system.matcher.infer(shared)[0]
            with np.load(output/'arm_a'/f'{index:04d}.npz', allow_pickle=False) as reference:
                for key in ('points_a', 'points_b', 'confidence', 'fine_points_b', 'lineage'):
                    value = result[key].cpu().numpy()
                    if not np.array_equal(value, reference[key]):
                        raise AssertionError(f'Prior-off negative control differs: pair={ids[index]}, field={key}')
    report = dict(schema='mhi.flow-loss.prior-off-negative-control.v1', status='passed',
        pairs=count, pair_ids=ids, scope='full_val' if count == 5106 else 'fixed_prefix',
        arm_a=reg_a, arm_b=reg_b, frozen_state_audit=audit,
        output_fields_exactly_equal=['points_a', 'points_b', 'confidence', 'fine_points_b', 'lineage'],
        source_sha256=SOURCE_SHA256, manifest_sha256=VAL_SHA256, no_gt_in_forward=True)
    write_json(output/'negative_control.json', report)
    return 0


def matching_plan(args):
    registrations = {}
    for name, checkpoint in (('arm_a', args.arm_a), ('arm_b', args.arm_b)):
        payload, registrations[name] = load_checkpoint_payload(checkpoint)
        del payload
    if sha256_file(args.manifest) != VAL_SHA256:
        raise ValueError('Matching plan requires the fixed val manifest')
    output = reserve_output(args.output)
    commands = []
    for arm, checkpoint in (('arm_a', args.arm_a), ('arm_b', args.arm_b)):
        for variant in ('B', 'C'):
            commands.append(dict(arm=arm, prior=variant, output=str(output/f'{arm}_prior_{variant}'),
                argv=[str(Path(args.python).resolve()), '-u', '-m', 'mhinet.v2.evaluate',
                    '--checkpoint', str(Path(checkpoint).resolve()), '--manifest', str(args.manifest.resolve()),
                    '--output', str(output/f'{arm}_prior_{variant}'), '--device', args.device, '--limit', '0', '--variant', variant]))
    report = dict(schema='mhi.flow-loss.frozen-matching-2x2-plan.v1', created_at=now(),
        source_sha256=SOURCE_SHA256, manifest_sha256=VAL_SHA256, pairs=5106,
        output_cap=2048, arm_checkpoints=registrations, commands=commands,
        protocol='Same frozen descriptor/QRRU, two endpoint-loss arms x prior off/on; no training and no GT in inference',
        gpu_launch='Runner executes argv serially on its explicitly reserved GPU; this command starts no evaluation')
    write_json(output/'matching_plan.json', report)
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    flow = commands.add_parser('flow', help='Complete Head-only point-level Flow/D8-prior evaluation')
    flow.add_argument('--checkpoint', type=Path, required=True)
    flow.add_argument('--manifest', type=Path, required=True)
    flow.add_argument('--hard-cases', type=Path, default=HARD_CASES)
    flow.add_argument('--output', type=Path, required=True)
    flow.add_argument('--device', default='cuda:0')
    flow.add_argument('--runtime', type=Path, help='Optional I/O-only runtime override; resource changes rejected')
    flow.add_argument('--workers', type=int, default=2)
    flow.add_argument('--limit', type=int, default=0)
    flow.add_argument('--expected-source-sha256', default=SOURCE_SHA256)
    flow.add_argument('--expected-manifest-sha256', default=VAL_SHA256)
    flow.set_defaults(function=evaluate_flow)
    compare_parser = commands.add_parser('compare', help='Compare both selected arms with the common Flow-best source')
    for name in ('arm-a', 'arm-b', 'source', 'output'):
        compare_parser.add_argument('--'+name, type=Path, required=True)
    compare_parser.add_argument('--bootstrap', type=int, default=2000)
    compare_parser.set_defaults(function=compare)
    negative = commands.add_parser('negative-control', help='Exact prior-off point-pair agreement; default first 64 val pairs')
    for name in ('arm-a', 'arm-b', 'manifest', 'output'):
        negative.add_argument('--'+name, type=Path, required=True)
    negative.add_argument('--device', default='cuda:0')
    negative.add_argument('--runtime', type=Path, help='Optional I/O-only runtime override')
    negative.add_argument('--limit', type=int, default=64)
    negative.set_defaults(function=negative_control)
    plan = commands.add_parser('matching-plan', help='Write four legacy-evaluate commands without starting a GPU job')
    for name in ('arm-a', 'arm-b', 'manifest', 'output'):
        plan.add_argument('--'+name, type=Path, required=True)
    plan.add_argument('--python', default=sys.executable)
    plan.add_argument('--device', default='cuda:0')
    plan.set_defaults(function=matching_plan)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, 'limit', 0) < 0 or getattr(args, 'workers', 0) < 0 or getattr(args, 'bootstrap', 1) < 1:
        raise ValueError('Negative diagnostic limit/workers or nonpositive bootstrap count')
    return args.function(args)


if __name__ == '__main__':
    raise SystemExit(main())
