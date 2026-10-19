"""Independent manual-ground-GCP evaluation of saved model matches (CPU only).

Synthetic H/GT grids never enter the metric. Human points never enter fitting.
RANSAC parameters and stable Top-K policy match the repository experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from .core import Catalog, DEFAULT_ROOT, Store, atomic_write, read_jsonl, sha_file, write_json, now

VARIANTS = {'Full': None, 'Top-4096': 4096, 'Top-2048': 2048, 'Top-1024': 1024}
RANSAC = dict(method='cv2.RANSAC', reprojection_threshold=3., max_iters=10000, confidence=.999, seed=0)


def selections(confidence):
    confidence = np.asarray(confidence)
    if confidence.ndim != 1 or not np.issubdtype(confidence.dtype, np.floating) or not np.isfinite(confidence).all():
        raise ValueError('confidence must be a finite floating 1D array')
    ranked = np.argsort(-confidence, kind='stable')
    return {name: np.arange(len(confidence)) if k is None else ranked[:k] for name, k in VARIANTS.items()}


def fit_matches(a, b):
    """Predictions ONLY; exactly the finite filter and RANSAC from score_matches."""
    finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if finite.sum() < 4:
        return None, 0, 'fewer_than_four_matches'
    cv2.setRNGSeed(0)
    try:
        H, mask = cv2.findHomography(a[finite], b[finite], cv2.RANSAC, 3., maxIters=10000, confidence=.999)
    except cv2.error:
        H, mask = None, None
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return None, 0, 'invalid_homography'
    return H, int(mask.sum()), None


def ground_metrics(H, a, b):
    """Forward A->B Euclidean error in stored B TEST-image pixels."""
    projected = np.c_[np.asarray(a, dtype=np.float64), np.ones(len(a))] @ np.asarray(H, dtype=np.float64).T
    if not np.isfinite(projected).all() or (np.abs(projected[:, 2]) <= 1e-10).any():
        raise ValueError('invalid_gcp_projection')
    predicted = projected[:, :2] / projected[:, 2:]
    errors = np.linalg.norm(predicted - np.asarray(b, dtype=np.float64), axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    return dict(ground_rmse=rmse, ground_median_error=float(np.median(errors)),
                ground_p90_error=float(np.percentile(errors, 90)), errors=errors.tolist(),
                ground_sr={str(t): bool(rmse <= t) for t in (1, 3, 5)})


def load_matches(path, pair_id):
    with np.load(path, allow_pickle=False) as stored:
        if 'pair_id' not in stored or stored['pair_id'].shape != () or str(stored['pair_id'].item()) != pair_id:
            raise ValueError('NPZ pair_id mismatch or missing')
        a, b, confidence = [stored[k].copy() for k in ('points_a', 'points_b', 'confidence')]
    if a.ndim != 2 or a.shape[1:] != (2,) or a.shape != b.shape or confidence.shape != (len(a),):
        raise ValueError('Invalid match array shapes')
    if any(not np.issubdtype(v.dtype, np.floating) for v in (a, b, confidence)):
        raise ValueError('Match arrays must be floating point')
    return a, b, confidence


def match_mapping(root, index=None):
    """Pair by explicit index or embedded NPZ ID, NEVER by filename/order."""
    root = Path(root).resolve()
    index = Path(index).resolve() if index else root / 'match_index.json'
    result = {}
    if index.exists():
        entries = json.loads(index.read_text())
        for entry in entries:
            pair_id, path = entry['pair_id'], (root / entry['npz']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('NPZ index path escapes match root')
            if pair_id in result:
                raise ValueError('Duplicate pair_id in match index')
            result[pair_id] = path
        return result, dict(path=str(index), sha256=sha_file(index))
    for path in sorted(root.rglob('*.npz')):
        with np.load(path, allow_pickle=False) as stored:
            if 'pair_id' not in stored or stored['pair_id'].shape != ():
                raise ValueError('Without an index every NPZ must have a scalar pair_id')
            pair_id = str(stored['pair_id'].item())
        if pair_id in result:
            raise ValueError('Duplicate embedded NPZ pair_id')
        result[pair_id] = path
    if not result:
        raise ValueError('No saved matches found')
    return result, dict(path=None, mapping_policy='embedded scalar NPZ pair_id')


def load_annotations(path):
    path = Path(path)
    if path.is_dir():
        if (path / 'annotations').is_dir():
            path = path / 'annotations'
        items = [(p, json.loads(p.read_text())) for p in sorted(path.glob('*.json'))]
    elif path.suffix == '.jsonl':
        items = [(path, v) for v in read_jsonl(path)]
    else:
        items = [(path, json.loads(path.read_text()))]
    seen = set()
    for _, v in items:
        if v['pair_id'] in seen:
            raise ValueError('Duplicate annotation pair_id')
        seen.add(v['pair_id'])
    return items


def aggregate(records):
    good = [r for r in records if r['fit_failure'] is None]
    errors = [e for r in good for e in r['errors']]
    n = len(records)
    return dict(pairs=n, valid_pairs=len(good), failed_pairs=n-len(good),
                ground_rmse_pair_mean=float(np.mean([r['ground_rmse'] for r in good])) if good else None,
                ground_median_pair_mean=float(np.mean([r['ground_median_error'] for r in good])) if good else None,
                ground_p90_pair_mean=float(np.mean([r['ground_p90_error'] for r in good])) if good else None,
                ground_rmse_pooled=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                ground_median_pooled=float(np.median(errors)) if errors else None,
                ground_p90_pooled=float(np.percentile(errors, 90)) if errors else None,
                ground_sr={str(t): sum(r['ground_sr'][str(t)] for r in records)/n if n else None for t in (1, 3, 5)})


def evaluate(annotations, matches_root, output, data_root=DEFAULT_ROOT, variants=None, index=None,
             reviewed_only=False, allow_version_mismatch=False):
    catalog = Catalog(data_root)
    mapping, mapping_provenance = match_mapping(matches_root, index)
    versions = dict(cv2=cv2.__version__, numpy=np.__version__, python=platform.python_version())
    plan_path = Path(matches_root).resolve().parent / 'plan.json'
    source_plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    source_versions = source_plan.get('versions')
    source_size = source_plan.get('input_size', source_plan.get('matcher', {}).get('input_size'))
    if source_size is not None and any(row['size_A'] != [source_size, source_size] or
                                       row['size_B'] != [source_size, source_size] for row in catalog.rows):
        raise ValueError('Saved prediction coordinate size differs from annotation test-image size')
    image_mapping = {r['pair_id']: r for r in json.loads(Path(mapping_provenance['path']).read_text())} if mapping_provenance.get('path') else {}
    mismatch = source_versions is not None and any(versions[k] != source_versions.get(k) for k in ('cv2', 'numpy'))
    if mismatch and not allow_version_mismatch:
        raise ValueError(f'Use source experiment environment: expected {source_versions}; got {versions}. '
                         'For exploratory comparison only, pass --allow-version-mismatch.')
    variants = variants or list(VARIANTS)
    if not variants or any(v not in VARIANTS for v in variants) or len(variants) != len(set(variants)):
        raise ValueError('Invalid variants')
    records, excluded, inputs = [], [], []
    # Validation uses the same geometry and schema as the annotation service;
    # no Store directory is created just to evaluate input data.
    validator = object.__new__(Store)
    for source, annotation in load_annotations(annotations):
        pair_id = annotation['pair_id']
        if annotation.get('status') != 'completed' or (reviewed_only and annotation.get('review', {}).get('status') != 'approved'):
            excluded.append(dict(pair_id=pair_id, status=annotation.get('status'), reason='not_completed_or_not_reviewed',
                                 skip_reason=annotation.get('skip_reason', ''),
                                 review_status=annotation.get('review', {}).get('status')))
            continue
        row = catalog.row(pair_id)
        if annotation.get('ground_truth_kind') != 'manual_ground_gcp' or annotation.get('coordinate_space') != 'test_image_pixels':
            raise ValueError('Expected manual ground GCP in test-image pixels; synthetic H GT is forbidden')
        if annotation.get('dataset') != 'GoogleEarth' or annotation['manifest']['sha256'] != catalog.sha256:
            raise ValueError('Annotation dataset/manifest mismatch')
        if annotation.get('draft') or not annotation.get('points'):
            raise ValueError('Completed annotations cannot have a draft or zero points')
        points = [validator.validate_point(p, row) for p in annotation['points']]
        if len({p['id'] for p in points}) != len(points) or any(p['needs_review'] for p in points):
            raise ValueError('Duplicate or unreviewed GCP')
        for side in ('A', 'B'):
            if annotation['image_size_' + side] != row['size_' + side]:
                raise ValueError('Annotation image-size mismatch')
            test = catalog.image_path(row, side, 'test')
            if (catalog.root / annotation['image_' + side]).resolve() != test:
                raise ValueError('Annotation test-image path mismatch')
            if annotation['image_sha256'][side]['test'] != sha_file(test):
                raise ValueError('Test image changed after annotation')
        a_gt, b_gt = [np.asarray([p[side] for p in points], dtype=np.float64) for side in ('a', 'b')]
        match = mapping.get(pair_id)
        inputs.append(dict(pair_id=pair_id, annotation_path=str(source.resolve()), annotation_sha256=sha_file(source),
                           annotation_revision=annotation['revision'], npz=str(match) if match else None,
                           npz_sha256=sha_file(match) if match and match.exists() else None))
        try:
            if match is None or not match.is_file():
                raise ValueError('missing_saved_matches')
            source_entry = image_mapping.get(pair_id, {})
            for side in ('A', 'B'):
                if 'image_' + side in source_entry and Path(source_entry['image_' + side]).resolve() != catalog.image_path(row, side, 'test'):
                    raise ValueError('Saved prediction image path differs from annotation')
            a, b, confidence = load_matches(match, pair_id)
            selected = selections(confidence)
            input_failure = None
        except (ValueError, KeyError, OSError) as e:
            input_failure = 'match_input_error: ' + str(e)
        for variant in variants:
            result = dict(pair_id=pair_id, dataset='GoogleEarth', variant=variant, gcp_count=len(points),
                          gcp_ids=[p['id'] for p in points], matches=0, selected_matches=0, ransac_inliers=0,
                          ground_rmse=None, ground_median_error=None, ground_p90_error=None, errors=[],
                          ground_sr={str(t): False for t in (1, 3, 5)}, fit_failure=input_failure, H_estimated=None)
            if input_failure is None:
                indices = selected[variant]
                H, inliers, failure = fit_matches(a[indices], b[indices])
                result.update(matches=len(a), selected_matches=len(indices), ransac_inliers=inliers, fit_failure=failure)
                if H is not None:
                    try:
                        result.update(ground_metrics(H, a_gt, b_gt), H_estimated=H.tolist())
                    except ValueError as e:
                        result['fit_failure'] = str(e)
            records.append(result)
    if not records:
        raise ValueError('No eligible completed manual GCP pairs')
    report = dict(evaluation='manual_ground_gcp', created_at=now(), coordinate_unit='stored_B_test_image_pixels',
                  ransac=RANSAC, variants=variants, versions=versions, source_versions=source_versions,
                  source_version_mismatch=bool(mismatch), manifest_sha256=catalog.sha256,
                  source_plan=dict(path=str(plan_path), sha256=sha_file(plan_path)) if source_plan else None,
                  match_mapping=mapping_provenance, inputs=inputs, excluded=excluded,
                  selection_policy='Full original order; Top-K stable descending confidence; N<K uses all ranked predictions',
                  fitting_policy='Predictions only. Manual GCP used only for forward A->B error. No synthetic H metric.',
                  aggregation_policy='SR denominator includes ALL eligible pairs; failed fits/missing matches are false. '
                                     'Error means/pooled statistics use valid pairs only, counts reported; no failure penalty imputation.',
                  implementation_sha256=sha_file(__file__),
                  overall={v: aggregate([r for r in records if r['variant'] == v]) for v in variants})
    output = Path(output).resolve()
    # Refuse mixing new metrics into original experiments or annotation records.
    protected = [Path(matches_root).resolve(), catalog.manifest.parent, Path(annotations).resolve()]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError('Evaluation output must be separate from matches, test images and annotation input')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Evaluation output already contains files; choose a new directory')
    atomic_write(output / 'pairs.jsonl', ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))
    import csv
    import io
    table = io.StringIO(newline='')
    fields = ['pair_id', 'variant', 'gcp_count', 'selected_matches', 'ransac_inliers', 'ground_rmse',
              'ground_median_error', 'ground_p90_error', 'fit_failure', 'ground_sr_1', 'ground_sr_3', 'ground_sr_5']
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    for r in records:
        writer.writerow({**{f: r.get(f) for f in fields if not f.startswith('ground_sr_')},
                         **{'ground_sr_'+t: r['ground_sr'][t] for t in ('1','3','5')}})
    atomic_write(output / 'pairs.csv', table.getvalue())
    write_json(output / 'summary.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, default=DEFAULT_ROOT / 'ground_gcp_annotations/annotations')
    p.add_argument('--matches-root', type=Path, required=True, help='Experiment dataset root with match_index.json/matches')
    p.add_argument('--match-index', type=Path)
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing results are never overwritten')
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--reviewed-only', action='store_true')
    p.add_argument('--allow-version-mismatch', action='store_true')
    args = p.parse_args()
    result = evaluate(args.annotations, args.matches_root, args.output, args.data_root, args.variants,
                      args.match_index, args.reviewed_only, args.allow_version_mismatch)
    print(json.dumps(result['overall'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
