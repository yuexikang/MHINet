#!/usr/bin/env python3
"""CPU-only confidence Top-K evaluation of immutable, already predicted A matches.

No model, neural-network library, image inference, or confidence recalibration is
used.  Full preserves the original array order; Top-K uses a stable descending
confidence order and keeps every available correspondence when N < K.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.mrsi_gt_metrics import conditional_rmse, score_matches

DEFAULT_SOURCE = REPO / 'outputs/h0_position_ablation_512_20261013'
DEFAULT_OUTPUT = REPO / 'outputs/topk_correspondence_ablation_512_20261015'
CHECKPOINT_SHA = '43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4'
VARIANTS = {'Full': None, 'Top-4096': 4096, 'Top-2048': 2048, 'Top-1024': 1024}
SELECTION_KEYS = {'Full': 'full_indices', 'Top-4096': 'top4096_indices',
                  'Top-2048': 'top2048_indices', 'Top-1024': 'top1024_indices'}
COHORTS = {
    'GoogleEarth': dict(pairs=498, sha256='55088ef26b24170cad2da2feb7917bcc3e2051e94b576a48a9018ecd12872b51'),
    'JL1Flight': dict(pairs=1755, sha256='d932203978e3da5d3ae96779447afdc69dc3b1d0dc26f0ef5041d0979a3aaaf5'),
}
METRICS_PATH = REPO / 'scripts/mrsi_gt_metrics.py'
FULL_TOLERANCE = 1e-9


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stable_selections(confidence):
    """Selection has no coordinates or GT argument and never modifies scores."""
    confidence = np.asarray(confidence)
    if confidence.ndim != 1 or not np.issubdtype(confidence.dtype, np.floating):
        raise ValueError('Confidence must be a one-dimensional floating array')
    if not np.isfinite(confidence).all():
        raise ValueError('Confidence contains nonfinite values')
    original = np.arange(len(confidence), dtype=np.int64)
    ranked = np.argsort(-confidence, kind='stable').astype(np.int64, copy=False)
    selections = {'Full': original}
    selections.update({variant: ranked[:k].copy() for variant, k in VARIANTS.items() if k is not None})
    if not np.array_equal(selections['Top-1024'], selections['Top-2048'][:1024]):
        raise RuntimeError('Top-1024 is not a Top-2048 prefix')
    if not np.array_equal(selections['Top-2048'], selections['Top-4096'][:2048]):
        raise RuntimeError('Top-2048 is not a Top-4096 prefix')
    return ranked, selections


def load_source_matches(path, pair_id):
    with np.load(path, allow_pickle=False) as stored:
        if stored['pair_id'].shape != () or str(stored['pair_id'].item()) != pair_id:
            raise ValueError('Source NPZ pair_id differs: ' + str(path))
        a, b, confidence = [stored[key].copy() for key in ('points_a', 'points_b', 'confidence')]
    if a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape or confidence.shape != (len(a),):
        raise ValueError('Invalid source correspondence shapes: ' + str(path))
    if len(a) > 12000:
        raise ValueError('Source exceeds the registered 12000-match cap: ' + str(path))
    if any(not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()
           for value in (a, b, confidence)):
        raise ValueError('Source coordinates/confidence must be finite floating arrays: ' + str(path))
    # Cell-center coordinates can lie in [-.5,511.5].  Do not clip, reject, or
    # filter them; the unchanged evaluation metric handles its own GT support.
    return a, b, confidence


def evaluate_matches(a, b, size_a, size_b, H_gt):
    metrics = score_matches(a, b, size_a, size_b, H_gt)
    if metrics['sr']['3'] and metrics['ncm']['5'] == 0:
        metrics.update(rmse_correct5_or_failure10=None, rmse_protocol_success=True, rmse_ncm=0)
    else:
        metrics.update(conditional_rmse(a, b, size_a, size_b, H_gt, success_override=metrics['sr']['3']))
    return metrics


def require_metrics_equal(actual, expected, path='metrics', *, tolerance=FULL_TOLERANCE):
    """Require every old metric, including RANSAC H and raw grid RMSE, to recur."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError('Full baseline metric keys differ: ' + path)
        for key in expected:
            require_metrics_equal(actual[key], expected[key], path + '.' + key, tolerance=tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError('Full baseline metric shape differs: ' + path)
        for index, value in enumerate(expected):
            require_metrics_equal(actual[index], value, f'{path}[{index}]', tolerance=tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(f'Full baseline metric differs: {path}: {actual!r} != {expected!r}')
    elif isinstance(expected, int):
        if isinstance(actual, bool) or actual != expected:
            raise RuntimeError(f'Full baseline metric differs: {path}: {actual!r} != {expected!r}')
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(actual, expected, rel_tol=0., abs_tol=tolerance):
            raise RuntimeError(f'Full baseline metric differs: {path}: {actual!r} != {expected!r}')
    else:
        raise TypeError('Unsupported metric value at ' + path)


def source_inventory(args):
    """Validate the complete source experiment; hash exactly the consumed files."""
    source = args.source_root.resolve()
    plan = json.loads((source / 'plan.json').read_text())
    validation = json.loads((source / 'validation.json').read_text())
    if not validation.get('complete') or not validation.get('model_unchanged'):
        raise ValueError('Source H0 experiment is not completed and audited')
    state_hashes = [validation[key] for key in ('model_state_sha256_before', 'model_state_sha256_after',
                                               'checkpoint_model_state_sha256')]
    if len(set(state_hashes)) != 1 or not state_hashes[0]:
        raise ValueError('Source model states differ')
    if plan.get('scope') != 'full_test_cohorts' or plan.get('input_size') != 512:
        raise ValueError('Expected the full 512-pixel source cohorts')
    if plan['checkpoint']['sha256'] != CHECKPOINT_SHA or plan['checkpoint']['step'] != plan['checkpoint']['total_steps']:
        raise ValueError('Source checkpoint identity/completion differs')
    if plan['variants']['A'] != dict(use_h0_d8_mask=True, use_h0_d2_warp=True):
        raise ValueError('Source A is not the original H0 inference configuration')
    metrics_sha = sha_file(METRICS_PATH)
    if plan['implementation']['scripts/mrsi_gt_metrics.py'] != metrics_sha:
        raise ValueError('Metric code differs from the source experiment')
    for key, version in (('cv2', cv2.__version__), ('numpy', np.__version__)):
        if plan['versions'][key] != version:
            raise ValueError(f'{key} version differs from the source; use the loma-repro environment')
    files = {}

    def add(path):
        path = Path(path).resolve()
        digest = sha_file(path)
        files[str(path)] = digest
        return digest

    add(source / 'plan.json')
    add(source / 'validation.json')
    datasets = {}
    for dataset in args.datasets:
        cohort = COHORTS[dataset]
        root = source / ('A_' + dataset)
        registration_path = root / 'registration.json'
        report_path = root / 'report.json'
        pairs_path = root / 'pairs.jsonl'
        index_path = root / 'match_index.json'
        registration = json.loads(registration_path.read_text())
        report = json.loads(report_path.read_text())
        if not report.get('complete') or report['registration'] != registration:
            raise ValueError('Source report/registration incomplete or inconsistent: ' + dataset)
        if registration['variant'] != 'A' or registration['dataset'] != dataset or registration['flags'] != plan['variants']['A']:
            raise ValueError('Wrong source variant: ' + dataset)
        if registration['checkpoint'] != plan['checkpoint'] or registration['matcher'] != plan['matcher']:
            raise ValueError('Source checkpoint/configuration differs: ' + dataset)
        if registration['matcher']['input_size'] != 512 or registration['matcher']['max_matches'] != 12000:
            raise ValueError('Source input size/cap differs: ' + dataset)
        if registration['model_state_sha256_before'] != state_hashes[0] or registration['model_state_sha256_after'] != state_hashes[0]:
            raise ValueError('Source per-group weights changed: ' + dataset)
        manifest = Path(registration['manifest']['path']).resolve()
        manifest_sha = add(manifest)
        if manifest_sha != cohort['sha256'] or registration['manifest']['sha256'] != manifest_sha:
            raise ValueError('Source cohort changed: ' + dataset)
        if registration['manifest']['pairs'] != cohort['pairs'] or registration['manifest']['evaluated_pairs'] != cohort['pairs']:
            raise ValueError('Source did not evaluate the full cohort: ' + dataset)
        rows = read_jsonl(manifest)
        old_rows = read_jsonl(pairs_path)
        indices = json.loads(index_path.read_text())
        ids = [row['pair_id'] for row in rows]
        if len(rows) != cohort['pairs'] or len(set(ids)) != len(ids):
            raise ValueError('Unexpected source inventory: ' + dataset)
        if [row['pair_id'] for row in old_rows] != ids or [row['pair_id'] for row in indices] != ids:
            raise ValueError('Source row/order mapping differs: ' + dataset)
        if report['overall']['pairs'] != len(rows):
            raise ValueError('Source report count differs: ' + dataset)
        audited = validation['baseline_validation'][dataset]
        if audited['a_baseline_exact_pairs'] != len(rows) or not audited['first_pair_legacy_exact']:
            raise ValueError('Source A baseline audit incomplete: ' + dataset)
        expected_files = {f'{index:05d}.npz' for index in range(len(rows))}
        if {path.name for path in (root / 'matches').glob('*.npz')} != expected_files:
            raise ValueError('Source NPZ count/index differs: ' + dataset)
        npz_hashes = {}
        descending = 0
        counts = []
        for index, (row, old, mapping) in enumerate(zip(rows, old_rows, indices)):
            if old['index'] != index or mapping['index'] != index or mapping['npz'] != f'matches/{index:05d}.npz':
                raise ValueError('Source original index differs: ' + row['pair_id'])
            if old['variant'] != 'A' or old['flags'] != plan['variants']['A']:
                raise ValueError('Source pair flags differ: ' + row['pair_id'])
            if not old.get('shared_features_unchanged') or set(old['a_baseline_exact']['checked']) != {'points_a', 'points_b', 'confidence'}:
                raise ValueError('Source pair input/baseline audit incomplete: ' + row['pair_id'])
            if row['size_A'] != [512, 512] or row['size_B'] != [512, 512]:
                raise ValueError('Expected stored 512 coordinates: ' + row['pair_id'])
            H = np.asarray(row['H_A_to_B'], dtype=np.float64)
            if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
                raise ValueError('Invalid metric GT: ' + row['pair_id'])
            path = root / 'matches' / f'{index:05d}.npz'
            a, b, confidence = load_source_matches(path, row['pair_id'])
            if old['metrics']['matches'] != len(a):
                raise ValueError('Source NM differs from NPZ: ' + row['pair_id'])
            stable_selections(confidence)
            descending += int(bool(np.all(confidence[:-1] >= confidence[1:])))
            counts.append(len(a))
            npz_hashes[str(path.resolve())] = add(path)
        datasets[dataset] = dict(source_root=str(root),
                                manifest=dict(path=str(manifest), sha256=manifest_sha, pairs=len(rows)),
                                evaluated_pairs=min(args.limit, len(rows)) if args.limit else len(rows),
                                source_registration_sha256=add(registration_path),
                                source_report_sha256=add(report_path), source_pairs_sha256=add(pairs_path),
                                source_index_sha256=add(index_path), source_npz_count=len(npz_hashes),
                                source_npz_inventory_sha256=mapping_sha256(npz_hashes),
                                confidence_descending_pairs=descending, source_nm_min=min(counts), source_nm_max=max(counts))
    return dict(sha256=mapping_sha256(files), files=files), datasets, plan


def preflight(args):
    inventory, datasets, source_plan = source_inventory(args)
    plan = dict(experiment='Fixed stored correspondences: confidence Top-K ablation',
                source_root=str(args.source_root.resolve()), checkpoint=source_plan['checkpoint'],
                datasets=datasets, variants=VARIANTS, selection_keys=SELECTION_KEYS,
                scope='limited_validation' if args.limit else 'full_test_cohorts', limit=args.limit,
                coordinate_unit='stored_512_target_pixels',
                source_inventory_sha256=inventory['sha256'],
                metrics_sha256=sha_file(METRICS_PATH), runner_sha256=sha_file(Path(__file__)),
                versions=dict(cv2=cv2.__version__, numpy=np.__version__, python=sys.version.split()[0]),
                selection_policy='Full preserves the original order. Top-K is stable descending stored confidence with original index as tie break; N<K keeps all ranked matches. No padding, clipping, filtering or GT selection.',
                confidence_policy='Original A coarse/fine combined confidence inherited through QRRU; no score recomputation or recalibration.',
                ransac_policy='Every group, including Full, refits H from its selected predictions using the unchanged score_matches RANSAC(3px,10000,.999,seed0).',
                full_reproduction_policy=f'Every Full metric recursively matches the old A pair metrics; absolute float tolerance={FULL_TOLERANCE}, exact bool/int/string/null.',
                precision_protocol=source_plan['precision_protocol'], sr_protocol=source_plan['sr_protocol'],
                rmse_protocol=source_plan['rmse_protocol'],
                raw_grid_rmse_protocol='registration_rmse is raw GT-visible 20x20 grid RMSE after predictions-only H fit; undefined fit failures excluded from its mean and explicitly counted.',
                googleearth_label_scope=source_plan['googleearth_label_scope'],
                execution_policy='CPU only; no neural libraries/models/checkpoints loaded, no new image inference. CPU metric time is not inference latency.',
                cpu_threads=args.cpu_threads,
                git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    plan['fingerprint'] = mapping_sha256({key: value for key, value in plan.items() if key != 'git_commit'})
    plan['created'] = now()
    return plan, inventory


def prepare(args):
    plan, inventory = preflight(args)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('fingerprint') != plan['fingerprint']:
            raise ValueError('Source/implementation changed; use a new output directory')
        saved_inventory = json.loads((args.output / 'source_inventory.json').read_text())
        if saved_inventory != inventory:
            raise ValueError('Prepared source inventory changed')
        return old, saved_inventory
    write_json(path, plan)
    write_json(args.output / 'source_inventory.json', inventory)
    return plan, inventory


def transition(current, baseline):
    if current and baseline:
        return 'both_success'
    if current:
        return 'gained_success'
    if baseline:
        return 'lost_success'
    return 'both_failure'


def summarize(records):
    metrics = [record['metrics'] for record in records]
    count = len(records)

    def avg(key):
        values = [metric[key] for metric in metrics if metric[key] is not None]
        return float(np.mean(values)) if values else None

    rmse = [metric['rmse_correct5_or_failure10'] for metric in metrics if metric['rmse_correct5_or_failure10'] is not None]
    return dict(pairs=count,
                precision={threshold: float(np.mean([metric['precision'][threshold] for metric in metrics])) for threshold in ('1', '3', '5')},
                precision_micro={threshold: sum(metric['ncm'][threshold] for metric in metrics) / max(1, sum(metric['matches'] for metric in metrics)) for threshold in ('1', '3', '5')},
                mean_ncm={threshold: float(np.mean([metric['ncm'][threshold] for metric in metrics])) for threshold in ('1', '3', '5')},
                mean_matches=avg('matches'),
                sr={threshold: float(np.mean([metric['sr'][threshold] for metric in metrics])) for threshold in ('1', '3', '5')},
                matching_rmse=avg('matching_rmse'), matching_rmse_pairs=sum(metric['matching_rmse'] is not None for metric in metrics),
                matching_epe=avg('matching_epe'), registration_rmse=avg('registration_rmse'),
                registration_rmse_pairs=sum(metric['registration_rmse'] is not None for metric in metrics),
                registration_failure_rate=float(np.mean([metric['fit_failure'] is not None for metric in metrics])),
                match_success20_at5=float(np.mean([metric['match_success20_at5'] for metric in metrics])),
                rmse_correct5_or_failure10=float(np.mean(rmse)) if rmse else None,
                rmse_defined_pairs=len(rmse), rmse_undefined_pairs=count - len(rmse),
                metric_evaluation_ms_mean=float(np.mean([record['metric_evaluation_ms'] for record in records])),
                metric_evaluation_ms_p95=float(np.percentile([record['metric_evaluation_ms'] for record in records], 95)),
                coordinate_unit='stored_512_target_pixels')


def selection_summary(records, k):
    return dict(k=k, under_k_pairs=sum(record['under_k'] for record in records) if k is not None else None,
                unchanged_match_count_pairs=sum(record['source_nm'] == record['selected_nm'] for record in records),
                source_nm_min=min(record['source_nm'] for record in records), source_nm_max=max(record['source_nm'] for record in records),
                selected_nm_min=min(record['selected_nm'] for record in records), selected_nm_max=max(record['selected_nm'] for record in records),
                source_nm_mean=float(np.mean([record['source_nm'] for record in records])),
                selected_nm_mean=float(np.mean([record['selected_nm'] for record in records])))


def transition_summary(records):
    names = ('both_success', 'both_failure', 'gained_success', 'lost_success')
    return {threshold: {name: sum(record['sr_transition_vs_full'][threshold] == name for record in records) for name in names}
            for threshold in ('1', '3', '5')}


def run(args, plan, inventory):
    cv2.setNumThreads(args.cpu_threads)
    np.random.seed(0)
    for dataset in args.datasets:
        for variant in VARIANTS:
            if (args.output / f'{variant}_{dataset}').exists():
                raise FileExistsError('Preserving existing results; use a new output directory')
    selections_root = args.output / 'selections'
    if selections_root.exists():
        raise FileExistsError('Preserving existing selections; use a new output directory')
    selections_root.mkdir()
    completed = {}
    validations = {}
    selection_inventory = {}
    started = time.perf_counter()
    for dataset in args.datasets:
        source = Path(plan['datasets'][dataset]['source_root'])
        rows = read_jsonl(plan['datasets'][dataset]['manifest']['path'])
        originals = read_jsonl(source / 'pairs.jsonl')
        if args.limit:
            rows = rows[:args.limit]
            originals = originals[:args.limit]
        output_selections = selections_root / dataset
        output_selections.mkdir()
        streams = {}
        values = {variant: [] for variant in VARIANTS}
        registrations = {}
        try:
            for variant, k in VARIANTS.items():
                output = args.output / f'{variant}_{dataset}'
                output.mkdir()
                registration = dict(variant=variant, k=k, dataset=dataset, checkpoint=plan['checkpoint'],
                                    manifest=plan['datasets'][dataset]['manifest'], source=plan['datasets'][dataset],
                                    source_inventory_sha256=inventory['sha256'], metrics_sha256=plan['metrics_sha256'],
                                    protocol=plan, versions=plan['versions'], neural_inference_run=False, started=now())
                registrations[variant] = registration
                write_json(output / 'registration.json', registration)
                streams[variant] = (output / 'pairs.jsonl').open('w')
            for index, (row, original) in enumerate(zip(rows, originals)):
                source_npz = source / 'matches' / f'{index:05d}.npz'
                source_sha = sha_file(source_npz)
                if source_sha != inventory['files'][str(source_npz.resolve())]:
                    raise RuntimeError('Source NPZ changed during evaluation: ' + str(source_npz))
                a, b, confidence = load_source_matches(source_npz, row['pair_id'])
                ranked, selections = stable_selections(confidence)
                selection_npz = output_selections / f'{index:05d}.npz'
                np.savez_compressed(selection_npz, pair_id=np.asarray(row['pair_id']),
                                    source_original_indices=np.arange(len(confidence), dtype=np.int64),
                                    ranked_original_indices=ranked,
                                    **{SELECTION_KEYS[variant]: ids for variant, ids in selections.items()})
                selection_sha = sha_file(selection_npz)
                selection_inventory[str(selection_npz.resolve())] = selection_sha
                pair_results = {}
                for variant, k in VARIANTS.items():
                    ids = selections[variant]
                    metric_start = time.perf_counter()
                    metrics = evaluate_matches(a[ids], b[ids], row['size_A'], row['size_B'], row['H_A_to_B'])
                    elapsed = (time.perf_counter() - metric_start) * 1000
                    if variant == 'Full':
                        require_metrics_equal(metrics, original['metrics'], row['pair_id'])
                    elif np.array_equal(ids, selections['Full']):
                        require_metrics_equal(metrics, original['metrics'], row['pair_id'] + '/' + variant)
                    if any(metrics['ncm'][threshold] > original['metrics']['ncm'][threshold] for threshold in ('1', '3', '5')):
                        raise RuntimeError('Subset has more correct matches than Full: ' + row['pair_id'])
                    record = dict(index=index, pair_id=row['pair_id'], group=original.get('group', 'all'),
                                  variant=variant, k=k, source_nm=len(a), selected_nm=len(ids),
                                  under_k=len(a) < k if k is not None else None,
                                  metrics=metrics, baseline_metrics=original['metrics'],
                                  source_npz=str(source_npz.resolve()), source_npz_sha256=source_sha,
                                  selection_npz=str(selection_npz.resolve()), selection_sha256=selection_sha,
                                  selection_key=SELECTION_KEYS[variant], nested_prefixes_verified=True,
                                  full_baseline_reproduced=True,
                                  source_confidence_descending=bool(np.all(confidence[:-1] >= confidence[1:])),
                                  ranking_preserves_original_order=bool(np.array_equal(ranked, np.arange(len(a)))),
                                  sr_transition_vs_full={threshold: transition(metrics['sr'][threshold], original['metrics']['sr'][threshold])
                                                         for threshold in ('1', '3', '5')},
                                  metric_evaluation_ms=elapsed)
                    values[variant].append(record)
                    pair_results[variant] = record
                for threshold in ('1', '3', '5'):
                    counts = [pair_results[variant]['metrics']['ncm'][threshold]
                              for variant in ('Top-1024', 'Top-2048', 'Top-4096', 'Full')]
                    if counts != sorted(counts):
                        raise RuntimeError('Correct-match count violates nested-set monotonicity: ' + row['pair_id'])
                for variant, record in pair_results.items():
                    streams[variant].write(json.dumps(record, allow_nan=False) + '\n')
                    streams[variant].flush()
                if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(rows):
                    progress = dict(state='running', dataset=dataset, completed=index + 1, total=len(rows),
                                    full_baseline_reproduced_pairs=index + 1, nested_pairs=index + 1, time=now(),
                                    precision5={variant: pair_results[variant]['metrics']['precision']['5'] for variant in VARIANTS},
                                    sr3={variant: pair_results[variant]['metrics']['sr']['3'] for variant in VARIANTS})
                    write_json(args.output / 'status.json', progress)
                    print(json.dumps(progress), flush=True)
        finally:
            for stream in streams.values():
                stream.close()
        validations[dataset] = dict(pairs=len(rows), group_records=len(rows) * len(VARIANTS),
                                    full_baseline_reproduced_pairs=len(rows), nested_pairs=len(rows),
                                    under_k_counts={variant: sum(record['under_k'] for record in values[variant]) if k is not None else None
                                                    for variant, k in VARIANTS.items()})
        for variant, k in VARIANTS.items():
            records = values[variant]
            report = dict(complete=False, registration=registrations[variant], overall=summarize(records),
                          by_group={group: summarize([record for record in records if record['group'] == group])
                                    for group in sorted({record['group'] for record in records})},
                          selection_counts=selection_summary(records, k), sr_transitions_vs_full=transition_summary(records),
                          failure_reasons=dict(Counter(record['metrics']['fit_failure'] or 'none' for record in records)),
                          validation=validations[dataset], finished=now())
            completed[(variant, dataset)] = report
            write_json(args.output / f'{variant}_{dataset}' / 'report.json', report)
    # Re-read every consumed source and verify the complete inventory, not just
    # cached in-memory hashes, before publishing successful completion.
    after_inventory, _, _ = source_inventory(args)
    if inventory != after_inventory:
        raise RuntimeError('Source files changed while evaluating Top-K')
    if sha_file(METRICS_PATH) != plan['metrics_sha256'] or sha_file(Path(__file__)) != plan['runner_sha256']:
        raise RuntimeError('Metric or runner implementation changed while evaluating Top-K')
    after_selections = {path: sha_file(path) for path in selection_inventory}
    if after_selections != selection_inventory:
        raise RuntimeError('Selection files changed while evaluating Top-K')
    validation = dict(complete=True, source_unchanged=True,
                      source_inventory_sha256_before=inventory['sha256'], source_inventory_sha256_after=after_inventory['sha256'],
                      selections_inventory_sha256=mapping_sha256(selection_inventory),
                      total_pairs=sum(value['pairs'] for value in validations.values()),
                      total_records=sum(value['group_records'] for value in validations.values()),
                      full_baseline_reproduced_pairs=sum(value['full_baseline_reproduced_pairs'] for value in validations.values()),
                      datasets=validations, neural_inference_run=False, model_weights_loaded=False, cuda_used=False,
                      full_metric_abs_tolerance=FULL_TOLERANCE, plan_fingerprint=plan['fingerprint'],
                      cpu_postprocessing_seconds=time.perf_counter() - started, finished=now())
    write_json(args.output / 'selection_inventory.json', dict(sha256=mapping_sha256(selection_inventory), files=selection_inventory))
    write_json(args.output / 'validation.json', validation)
    for (variant, dataset), report in completed.items():
        report['complete'] = True
        report['validation']['source_unchanged'] = True
        report['validation']['source_inventory_sha256_after'] = after_inventory['sha256']
        write_json(args.output / f'{variant}_{dataset}' / 'report.json', report)
    write_json(args.output / 'status.json', dict(state='complete', groups=len(completed), scope=plan['scope'], time=now(), validation=validation))
    print('TOPK_ALL_COMPLETE ' + str(args.output), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--datasets', choices=tuple(COHORTS), nargs='+', default=list(COHORTS))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Optional separate limited-validation output; 0 evaluates the complete cohorts.')
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.cpu_threads < 1:
        parser.error('limit must be nonnegative and cpu-threads positive')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate datasets')
    args.source_root = args.source_root.resolve()
    args.output = args.output.resolve()
    if args.output == args.source_root or args.output.is_relative_to(args.source_root):
        parser.error('Output must be outside the immutable source experiment')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan, inventory = prepare(args)
            print(json.dumps(dict(event='PREFLIGHT_OK', fingerprint=plan['fingerprint'],
                                  source_inventory_sha256=inventory['sha256'], scope=plan['scope'], datasets=plan['datasets'])), flush=True)
            if args.prepare_only:
                if not (args.output / 'validation.json').exists():
                    write_json(args.output / 'status.json', dict(state='prepared', time=now(), fingerprint=plan['fingerprint'], scope=plan['scope']))
                return 0
            if (args.output / 'validation.json').exists() and json.loads((args.output / 'validation.json').read_text()).get('complete'):
                print('ALREADY_COMPLETE ' + str(args.output), flush=True)
                return 0
            run(args, plan, inventory)
            return 0
        except Exception as error:
            write_json(args.output / 'status.json', dict(state='failed', error=f'{type(error).__name__}: {error}', time=now()))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
