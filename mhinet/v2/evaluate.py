"""Standalone v2 raster-mask validation diagnostics, never a GCP headline score.

Examples:
    python -m mhinet.v2.evaluate --checkpoint RUN/last.pt \
        --manifest /data/MHINet_RGB_512/val/pairs.jsonl --output NEW_DIR
    python -m mhinet.v2.evaluate ... --variant B  # same weights, prior disabled

Only image tensors and native image sizes enter prediction. Manifest homography
and overlap masks are accessed by the reporting metrics after prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.torch_version import TorchVersion

from mhinet.config import RuntimePaths, sha256_file
from mhinet.dataio.data import iter_manifest, load_rgb_bicubic
from mhinet.downstream.evaluate_semidense import coverage
from mhinet.downstream.metrics import dense_metrics
from mhinet.engine.checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from . import TASK
from .config import SemidenseConfigV2
from .feature_provider import build_v2_shared
from .semidense import SemidenseMatcherV2
from .training import SemidenseSystemV2


REPO = Path(__file__).resolve().parents[2]
THRESHOLDS = (1., 3., 5., 10.)
METRIC_NOTE = (
    'Raster overlap masks and manifest homography define validation diagnostics only. '
    'These scores are not historical headline/GCP precision or a paper RMSE. '
    'The conditional correct-at-5px RMSE uses a failure sentinel of 10px when '
    'fewer than 20 correct matches exist; it is not all-match or estimated-H RMSE.'
)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False).encode('utf-8')).hexdigest()


def _write_json(path, value):
    # The output directory is newly reserved by this run; x also rejects an
    # accidental repeated filename rather than replacing another artifact.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _resolve_file(value, label, *, base=REPO):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f'{label} must identify an existing file')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label}: {path}')
    return path


def _read_configuration(metadata):
    """Require a real runtime JSON; do not invent missing legacy config paths."""
    config = metadata.get('config')
    config_path = None
    if isinstance(config, (str, Path)):
        config_path = _resolve_file(config, 'checkpoint training configuration')
        config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('v2 checkpoint metadata.config must be an object or JSON file path')
    base = config_path.parent if config_path else REPO
    runtime_path = _resolve_file(config.get('runtime'), 'v2 runtime configuration', base=base)
    runtime_payload = json.loads(runtime_path.read_text(encoding='utf-8'))
    if not isinstance(runtime_payload, dict):
        raise ValueError('Runtime configuration must be a JSON object')
    expected = metadata.get('runtime_sha256')
    if expected is not None and sha256_file(runtime_path) != expected:
        raise ValueError('Runtime JSON identity differs from checkpoint metadata')
    return config, config_path, runtime_path


def _code_identity():
    files = sorted((REPO / 'mhinet').rglob('*.py'))
    hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in files}
    def git(*args):
        result = subprocess.run(['git', '-C', str(REPO), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    try:
        commit = git('rev-parse', 'HEAD')
        status = git('status', '--porcelain', '--untracked-files=normal')
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, None
    return dict(commit=commit, dirty_status=status, python_source_sha256=hashes,
        code_sha256=_json_hash(hashes))


def _capped_stage(points_a, points_b, confidence, cap):
    """All stage precision comparisons use the same stable confidence budget."""
    if len(points_a) != len(points_b) or len(points_a) != len(confidence):
        raise ValueError('Stage coordinates and confidence must have equal length')
    if points_a.ndim != 2 or points_a.shape[1] != 2 or points_b.shape != points_a.shape:
        raise ValueError('Stage coordinates must be Nx2')
    if not torch.isfinite(points_a).all() or not torch.isfinite(points_b).all() or not torch.isfinite(confidence).all():
        raise ValueError('Inference stage contains nonfinite coordinates/confidence')
    order = confidence.argsort(descending=True, stable=True)[:cap]
    return dict(a=points_a[order].detach().cpu().numpy(),
        b=points_b[order].detach().cpu().numpy(),
        confidence=confidence[order].detach().cpu().numpy(),
        before_budget=len(confidence))


def _metric(row, root, stage):
    result = dense_metrics(row, root, stage['a'], stage['b'], thresholds=THRESHOLDS)
    result['raster_mask_precision'] = {
        key: value['precision'] for key, value in result['thresholds'].items()}
    result['conditional_correct5_rmse_or_failure10px'] = result.pop('legacy_RMSE_correct5_or_failure10')
    result['coverage_a_16x16'] = coverage(stage['a'], row['size_A'])
    result['coverage_b_16x16'] = coverage(stage['b'], row['size_B'])
    result['candidates_before_shared_output_budget'] = stage['before_budget']
    result['protocol'] = 'raster_mask_precision_validation_diagnostic'
    result['population'] = 'matched_topk_under_shared_output_budget'
    return result


def _summarize_stage(records, stage_name):
    values = [record['stages'][stage_name] for record in records if record['stages'][stage_name] is not None]
    if not values:
        return dict(available_pairs=0, unavailable_pairs=len(records),
            unavailable_reason='Legacy h0 candidate confidence is unavailable; no stage score fabricated')
    matches = sum(value['matches'] for value in values)
    return dict(pairs=len(values), unavailable_pairs=len(records)-len(values), matches=matches,
        mean_matches=float(np.mean([value['matches'] for value in values])),
        macro_raster_mask_precision={str(t): float(np.mean([
            value['thresholds'][str(t)]['precision'] for value in values])) for t in THRESHOLDS},
        micro_raster_mask_precision={str(t): sum(value['thresholds'][str(t)]['NCM']
            for value in values)/max(1, matches) for t in THRESHOLDS},
        mean_coverage_a_16x16=float(np.mean([value['coverage_a_16x16'] for value in values])),
        mean_coverage_b_16x16=float(np.mean([value['coverage_b_16x16'] for value in values])),
        zero_match_rate=float(np.mean([value['matches'] == 0 for value in values])),
        failure_below20_correct5_rate=float(np.mean([
            not value['success_20_correct_at_5px'] for value in values])))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True,
        help='mhi_semidense_v2 checkpoint; v1 checkpoints are rejected')
    parser.add_argument('--manifest', type=Path, required=True,
        help='RGB_512 val manifest with raster overlap masks and GT homography')
    parser.add_argument('--output', type=Path, required=True,
        help='New output directory; an existing directory is never overwritten')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0,
        help='First N manifest rows; 0 evaluates the entire manifest (default)')
    parser.add_argument('--variant', choices=('B', 'C'),
        help='Keep checkpoint weights/architecture/budgets and disable/enable Flow prior')
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.limit < 0:
        raise ValueError('--limit must be nonnegative')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Evaluation supports cpu or cuda devices')
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    checkpoint = _resolve_file(args.checkpoint, 'v2 checkpoint')
    manifest = _resolve_file(args.manifest, 'evaluation manifest')
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get('format') != CHECKPOINT_FORMAT or payload.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Expected the supported mhinet.training checkpoint format/version')
    metadata = payload.get('metadata')
    if not isinstance(metadata, dict) or metadata.get('task') != TASK:
        raise ValueError('Evaluation requires task=mhi_semidense_v2; v1 checkpoints are rejected')
    config, config_path, runtime_path = _read_configuration(metadata)
    matcher_config = SemidenseConfigV2.from_dict(metadata.get('matcher'))
    checkpoint_matcher_config = asdict(matcher_config)
    if args.variant:
        if matcher_config.fine_prior != 'direct_d8' or matcher_config.geometry_mode == 'h0':
            raise ValueError('B/C prior toggles require an existing no-H Direct-D8 configuration')
        # Compute exactly the same shared Flow output in both paired runs. The
        # only mechanism changed by --variant is whether matcher uses its reward.
        matcher_config = replace(matcher_config, geometry_mode='flow_d16',
            flow_guidance_enabled=args.variant == 'C')
    runtime = replace(RuntimePaths.from_json(runtime_path), device=str(device))
    if metadata.get('dino_sha256') is not None and sha256_file(runtime.dino_checkpoint) != metadata['dino_sha256']:
        raise ValueError('DINO checkpoint identity differs from v2 checkpoint metadata')
    rows = list(iter_manifest(manifest))
    total_rows = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty evaluation manifest/selection')
    pair_ids = [str(row['pair_id']) for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError('Duplicate pair_id values in evaluation selection')
    for row in rows:
        for side in ('A', 'B'):
            if len(row[f'size_{side}']) != 2 or any(int(v) <= 0 for v in row[f'size_{side}']):
                raise ValueError(f'Invalid native image size: {row["pair_id"]}')
    stage = metadata.get('stage', config.get('stage', 'flow_head'))
    shared, provider_report = build_v2_shared(runtime, geometry_mode=matcher_config.geometry_mode)
    system = SemidenseSystemV2(shared, SemidenseMatcherV2(matcher_config), stage=stage).to(device)
    system.load_state_dict(payload['model'], strict=True)
    system.eval()
    checkpoint_step = payload.get('progress', {}).get('optimizer_step')
    del payload
    # mkdir(exist_ok=False) reserves the run atomically after input validation.
    output.mkdir(parents=True, exist_ok=False)
    arrays = output / 'pairs'
    arrays.mkdir()
    registration = dict(schema='mhi.v2.raster_mask_val_diagnostic.v1', task=TASK,
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_step=checkpoint_step,
        manifest=str(manifest), manifest_sha256=sha256_file(manifest),
        manifest_rows=total_rows, selected_pair_ids=pair_ids,
        selected_pair_ids_sha256=_json_hash(pair_ids), limit=args.limit,
        selection_scope='full_manifest' if not args.limit else 'prefix_diagnostic',
        config=config, config_sha256=_json_hash(config),
        config_path=str(config_path) if config_path else None,
        config_file_sha256=sha256_file(config_path) if config_path else None,
        runtime=str(runtime_path), runtime_sha256=sha256_file(runtime_path),
        checkpoint_matcher=checkpoint_matcher_config, matcher=asdict(matcher_config),
        prior_toggle=args.variant, architecture_flags=dict(use_discrepancy=matcher_config.use_discrepancy,
            use_dgfo=matcher_config.use_dgfo),
        shared_stage=stage, provider=provider_report, device=str(device), input_size=512,
        shared_output_budget=matcher_config.max_output_matches,
        qrru_input_budget=matcher_config.max_matches, coarse_budget=matcher_config.max_coarse,
        thresholds_native_target_pixels=list(THRESHOLDS),
        stage_comparison_population='D8/D2/final matched_topk at the same configured output budget; raw stage counts recorded separately',
        metric_protocol='raster_mask_precision_validation_diagnostic', metric_note=METRIC_NOTE,
        validation_split_claim=manifest.parent.name == 'val',
        timing_note='One first-pair warmup excluded; synchronized shared/matcher timers exclude loading, metrics and NPZ writes. Matcher may include diagnostic materialization.',
        code=_code_identity())
    _write_json(output/'registration.json', registration)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def predict(images, sizes):
        # Deliberately use shared(images), not system(images): flow_head training
        # forward skips the descriptor pyramid, while evaluation requires D8/D2.
        sync()
        start = time.perf_counter()
        features = system.shared(images)
        sync()
        shared_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        # Until default inference exports D8 confidence, request diagnostics to
        # retain the independent D8 stage. Account for its overhead explicitly.
        prediction = system.matcher.infer(features, sizes, diagnostics=True)[0]
        sync()
        return prediction, shared_ms, (time.perf_counter()-start)*1000

    records = []
    with torch.inference_mode(), (output/'pairs.jsonl').open('x', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            images = torch.stack([load_rgb_bicubic(manifest.parent/row[f'image_{side}'], size=512)
                for side in ('A', 'B')])[None].to(device)
            sizes = [(tuple(row['size_A']), tuple(row['size_B']))]
            if index == 0:
                predict(images, sizes)
                sync()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
            result, shared_ms, matcher_ms = predict(images, sizes)
            cap = matcher_config.max_output_matches
            if len(result['confidence']) > cap:
                raise RuntimeError('Final output violated configured maximum match count')
            final = _capped_stage(result['points_a'], result['points_b'], result['confidence'], cap)
            coarse_confidence = result.get('coarse_confidence')
            if coarse_confidence is None:
                coarse_confidence = result.get('diagnostics', {}).get('coarse_confidence')
            coarse_a, coarse_b = result.get('coarse_a'), result.get('coarse_b')
            # Invalid H0 can end legacy A before coarse/fine candidate creation.
            if coarse_a is None or coarse_b is None:
                if matcher_config.geometry_mode != 'h0':
                    raise RuntimeError('V2 inference omitted D8 diagnostic coordinates')
                coarse_a, coarse_b = result['points_a'].new_empty((0, 2)), result['points_b'].new_empty((0, 2))
            # A no-fine result can still contain selected D8 parents in its
            # trace. Keep the absence of their confidence explicit below.
            diagnostic = result.get('diagnostics', {})
            if len(coarse_a) == 0 and len(diagnostic.get('coarse_source', [])):
                from mhinet.downstream.loma_reference.matching_utils import flat_indices_to_centers, centers_to_native
                coarse_a = centers_to_native(flat_indices_to_centers(diagnostic['coarse_source'], 64, 64), tuple(row['size_A']))
                coarse_b = centers_to_native(flat_indices_to_centers(diagnostic['coarse_target'], 64, 64), tuple(row['size_B']))

            if coarse_confidence is None and len(coarse_a) == 0:
                coarse_confidence = torch.empty(0, device=coarse_a.device)
            # Diagnostic confidence may have been materialized on CPU by infer.
            d8 = _capped_stage(coarse_a, coarse_b, coarse_confidence.to(coarse_a.device), cap) if coarse_confidence is not None else None
            fine_confidence = result.get('fine_all_confidence')
            if fine_confidence is not None:
                d2 = _capped_stage(result['fine_all_a'], result['fine_all_b'], fine_confidence, cap)
            elif len(result.get('fine_all_a', [])) == 0:
                empty = result['points_a'].new_empty((0, 2))
                d2 = _capped_stage(empty, empty.clone(), result['confidence'].new_empty(0), cap)
            else:
                d2 = None  # Legacy A has no pre-QRRU confidence, so no ranking is invented.
            # Reporting starts only after prediction has completed.
            stages = {name: _metric(row, manifest.parent, value) if value is not None else None
                for name, value in (('d8', d8), ('d2', d2), ('final', final))}

            artifact = f'pairs/{index:06d}.npz'
            array_payload = {f'{name}_{key}': value[key]
                for name, value in (('d8', d8), ('d2', d2), ('final', final)) if value is not None
                for key in ('a', 'b', 'confidence')}
            if 'lineage' in result:
                array_payload['final_lineage'] = result['lineage'].detach().cpu().numpy()
            np.savez_compressed(output/artifact, **array_payload)
            record = dict(pair_id=str(row['pair_id']), index=index, npz=artifact,
                failure_reason=result['failure_reason'], shared_ms=shared_ms, matcher_ms=matcher_ms,
                stage_candidate_counts=dict(
                    d8_selected_before_output_budget=len(coarse_a),
                    d2_mutual_before_qrru_budget=diagnostic.get('fine_before_qrru_budget', diagnostic.get('fine_before_cap')),
                    d2_after_qrru_budget=len(result.get('fine_all_a', [])),
                    final_valid_before_output_budget=result.get('output_before_cap')),
                d8_selection_counts=diagnostic.get('coarse_counts'),
                stage_unavailable_reasons={name: 'Candidate confidence unavailable; matched_topk score not fabricated'
                    for name, value in (('d8', d8), ('d2', d2)) if value is None},
                total_model_ms=shared_ms+matcher_ms, stages=stages,
                qrru_outside_samples=result.get('qrru_outside_samples', 0),
                rejected_qrru=result.get('rejected_qrru', 0),
                final_before_cap=result.get('output_before_cap'),
                final_output_cap=cap)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            records.append(record)
    summary = dict(schema=registration['schema'], task=TASK,
        checkpoint=str(checkpoint), checkpoint_sha256=registration['checkpoint_sha256'],
        manifest=str(manifest), manifest_sha256=registration['manifest_sha256'],
        config_sha256=registration['config_sha256'], code_sha256=registration['code']['code_sha256'],
        prior_toggle=args.variant, matcher=asdict(matcher_config), pairs=len(records),
        shared_output_budget=matcher_config.max_output_matches,
        stages={stage_name: _summarize_stage(records, stage_name) for stage_name in ('d8', 'd2', 'final')},
        inference_failure_rate=float(np.mean([r['failure_reason'] != 'none' for r in records])),
        timing_ms={name: dict(mean=float(np.mean([r[name] for r in records])),
            median=float(np.median([r[name] for r in records])), p95=float(np.percentile([r[name] for r in records], 95)))
            for name in ('shared_ms', 'matcher_ms', 'total_model_ms')},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        metric_protocol=registration['metric_protocol'], metric_note=METRIC_NOTE,
        timing_note=registration['timing_note'], validation_split_claim=registration['validation_split_claim'],
        selection_scope=registration['selection_scope'])
    _write_json(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
