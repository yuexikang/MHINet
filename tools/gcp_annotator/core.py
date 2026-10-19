"""Manifest geometry, reproducible sampling and crash-safe annotation storage.

Canonical a/b coordinates always refer to the stored TEST images, not the UI.
Native imagery is only a viewing coordinate system; neither the synthetic
A-to-B H nor predictions are used to suggest or move human correspondences.
"""
from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path('/home/disk1/Data/datasets/Test_Data/GoogleEarth')
STATUSES = ('not_started', 'in_progress', 'completed', 'skipped')
CONVENTION = ('x right, y down; integer coordinates are pixel centers; '
              'top-left center=(0,0); range [0,width-1] x [0,height-1]; '
              'a/b are original stored test-image pixels, float64, no rounding')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def atomic_write(path, data):
    """Same-filesystem replace with file AND parent-directory fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all() or np.linalg.matrix_rank(result) < 3:
        raise ValueError('Invalid native-to-test transform')
    return result


def project(point, transform):
    v = np.asarray(transform, dtype=np.float64) @ np.r_[np.asarray(point, dtype=np.float64), 1.]
    if not np.isfinite(v).all() or abs(v[2]) <= 1e-12:
        raise ValueError('Point projects to infinity')
    return (v[:2] / v[2]).tolist()


def coordinate(value, size, tolerance=0.):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Each coordinate must contain x and y')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError('Coordinates must be finite numbers')
    if any(v < -tolerance or v > n - 1 + tolerance for v, n in zip(value, size)):
        raise ValueError(f'Coordinate {value} outside pixel-center bounds {size}')
    return [float(min(n-1, max(0., v))) for v, n in zip(value, size)]


class Conflict(ValueError):
    pass


class Catalog:
    def __init__(self, root=DEFAULT_ROOT, manifest=None):
        self.root = Path(root).resolve()
        self.manifest = Path(manifest or self.root / 'val/pairs.jsonl').resolve()
        self.sha256 = sha_file(self.manifest)
        self.rows = read_jsonl(self.manifest)
        self.by_id = {}
        for row in self.rows:
            pair_id = row['pair_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', pair_id) or pair_id in self.by_id:
                raise ValueError('Unsafe or duplicate manifest pair_id')
            # This adapter intentionally only accepts the GoogleEarth manifest.
            if row.get('source_dataset') != 'googleearth_evaluation_cross_temporal':
                raise ValueError('Only GoogleEarth cross-temporal pairs are supported')
            for side in ('A', 'B'):
                self.image_path(row, side, 'test')
                matrix(row['T_source_to_' + side])
            self.by_id[pair_id] = row
        if not self.rows:
            raise ValueError('Empty manifest')

    def row(self, pair_id):
        if pair_id not in self.by_id:
            raise KeyError(pair_id)
        return self.by_id[pair_id]

    def image_path(self, row, side, mode):
        if side not in ('A', 'B') or mode not in ('native', 'test'):
            raise ValueError('Invalid image side or view')
        if mode == 'native':
            path = Path(row['target_path' if side == 'A' else 'source_path']).resolve()
        else:
            path = (self.manifest.parent / row['image_' + side]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Test image is outside dataset root')
        if not path.is_file():
            raise ValueError('Image does not exist: ' + str(path))
        return path

    def stratum(self, row):
        # GoogleEarth has one augmentation class, unlike JL1Flight t0..t4.
        band = '[2,3)' if float(row['area_ratio_A_to_B']) < 3 else '[3,4]'
        return 'GoogleEarth/single_quadrilateral/area' + band

    def metadata(self, pair_id):
        row = self.row(pair_id)
        images = {}
        hashes = {}
        for side in ('A', 'B'):
            native = self.image_path(row, side, 'native')
            test = self.image_path(row, side, 'test')
            with Image.open(native) as im:
                native_size = list(im.size)
            with Image.open(test) as im:
                test_size = list(im.size)
            if test_size != row['size_' + side]:
                raise ValueError('Test-image dimensions differ from manifest')
            images[side] = dict(native_path=str(native), test_path=str(test),
                                native_size=native_size, test_size=test_size,
                                native_to_test=matrix(row['T_source_to_' + side]).tolist(),
                                test_to_native=np.linalg.inv(matrix(row['T_source_to_' + side])).tolist())
            hashes[side] = dict(native=sha_file(native), test=sha_file(test))
        return dict(images=images, image_sha256=hashes, stratum=self.stratum(row))


class Store:
    def __init__(self, catalog, output=None, sample_size=20, seed=42):
        self.catalog = catalog
        self.output = Path(output or catalog.root / 'ground_gcp_annotations').resolve()
        # Never write annotations into the original manifest/image directories.
        if self.output == catalog.root or self.output.is_relative_to(catalog.manifest.parent):
            raise ValueError('Output must be a separate annotations directory')
        self.output.mkdir(parents=True, exist_ok=True)
        self.annotations = self.output / 'annotations'
        self.annotations.mkdir(exist_ok=True)
        self._metadata = {}
        with self.lock():
            selection = self.output / 'selection.json'
            if not selection.exists():
                write_json(selection, self.sample(sample_size, seed))
            self.selection = json.loads(selection.read_text())
            if self.selection['manifest_sha256'] != catalog.sha256:
                raise ValueError('Manifest changed since sample selection; use a separate output directory')
            ids = self.selection['pair_ids']
            if len(ids) != len(set(ids)) or any(p not in catalog.by_id for p in ids):
                raise ValueError('Invalid frozen selection')

    @contextmanager
    def lock(self):
        with (self.output / '.store.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def sample(self, count, seed, existing=()):
        if not 1 <= count <= len(self.catalog.rows) or count < len(existing):
            raise ValueError('Invalid sample size (cannot shrink an existing selection)')
        groups = {}
        for row in self.catalog.rows:
            groups.setdefault(self.catalog.stratum(row), []).append(row['pair_id'])
        selected = list(existing)
        for name, ids in sorted(groups.items()):
            ids.sort()
            # Per-stratum RNG independent of dictionary / manifest row order.
            random.Random(f'{seed}:{name}').shuffle(ids)
        while len(selected) < count:
            counts = {k: sum(self.catalog.stratum(self.catalog.row(p)) == k for p in selected) for k in groups}
            choices = [k for k in sorted(groups) if any(p not in selected for p in groups[k])]
            key = min(choices, key=lambda k: (counts[k], k))
            selected.append(next(p for p in groups[key] if p not in selected))
        return dict(version=1, dataset='GoogleEarth', seed=seed, created_at=now(),
                    manifest=str(self.catalog.manifest), manifest_sha256=self.catalog.sha256,
                    policy='balanced seeded sampling by existing single-quad geometry and area-ratio band; no model results',
                    strata={k: sum(self.catalog.stratum(self.catalog.row(p)) == k for p in selected) for k in groups},
                    pair_ids=selected)

    def extend(self, count):
        with self.lock():
            old = json.loads((self.output / 'selection.json').read_text())
            new = self.sample(count, old['seed'], old['pair_ids'])
            new['created_at'] = old['created_at']
            new['extended_at'] = now()
            write_json(self.output / 'selection.json', new)
            self.selection = new
        return new

    def metadata(self, pair_id):
        if pair_id not in self._metadata:
            self._metadata[pair_id] = self.catalog.metadata(pair_id)
        return copy.deepcopy(self._metadata[pair_id])

    def empty(self, pair_id):
        row = self.catalog.row(pair_id)
        meta = self.metadata(pair_id)
        return dict(schema_version=1, ground_truth_kind='manual_ground_gcp', dataset='GoogleEarth', pair_id=pair_id,
                    image_A=str(self.catalog.image_path(row, 'A', 'test').relative_to(self.catalog.root)),
                    image_B=str(self.catalog.image_path(row, 'B', 'test').relative_to(self.catalog.root)),
                    dataset_root=str(self.catalog.root), image_size_A=row['size_A'], image_size_B=row['size_B'],
                    manifest=dict(path=str(self.catalog.manifest), sha256=self.catalog.sha256),
                    coordinate_convention=CONVENTION, coordinate_space='test_image_pixels',
                    preview_geometry=meta['images'], image_sha256=meta['image_sha256'],
                    status='not_started', points=[], draft=None, annotator='', skip_reason='', notes='',
                    review=dict(status='unreviewed', reviewer='', checked_ids=[]), revision=0,
                    created_at=None, updated_at=None)

    def load(self, pair_id):
        self.catalog.row(pair_id)
        path = self.annotations / (pair_id + '.json')
        if not path.exists():
            return self.empty(pair_id)
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['manifest']['sha256'] != self.catalog.sha256 or value['pair_id'] != pair_id:
            raise ValueError('Annotation provenance mismatch')
        return value

    def validate_point(self, point, row):
        if not isinstance(point, dict):
            raise ValueError('A GCP must be an object')
        point_id = point.get('id', '')
        if not isinstance(point_id, str) or not re.fullmatch(r'GCP-\d{3,}', point_id):
            raise ValueError('Invalid GCP ID')
        if point.get('type', 'ground') != 'ground':
            raise ValueError('Only manually annotated ground points are allowed')
        a = coordinate(point.get('a'), row['size_A'])
        b = coordinate(point.get('b'), row['size_B'])
        native_a = project(a, np.linalg.inv(matrix(row['T_source_to_A'])))
        native_b = project(b, np.linalg.inv(matrix(row['T_source_to_B'])))
        # Native centers must also belong to the real source imagery.
        native_a = coordinate(native_a, row['target_image_size'], tolerance=1e-7)
        native_b = coordinate(native_b, row['source_image_size'], tolerance=1e-7)
        if not isinstance(point.get('needs_review', False), bool):
            raise ValueError('needs_review must be boolean')
        return dict(id=point_id, a=a, b=b, native_a=native_a, native_b=native_b,
                    type='ground', needs_review=point.get('needs_review', False),
                    note=str(point.get('note', ''))[:2000])

    def save(self, pair_id, body):
        row = self.catalog.row(pair_id)
        with self.lock():
            old = self.load(pair_id)
            if body.get('revision') != old['revision']:
                raise Conflict('This pair changed in another tab; reload before saving. Local edits are retained.')
            if body.get('pair_id', pair_id) != pair_id:
                raise ValueError('pair_id mismatch')
            raw = body.get('points', [])
            if not isinstance(raw, list) or len(raw) > 1000:
                raise ValueError('Invalid point list')
            points = [self.validate_point(p, row) for p in raw]
            ids = [p['id'] for p in points]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate GCP IDs')
            draft = body.get('draft')
            if draft is not None:
                if not isinstance(draft, dict) or draft.get('b') is not None:
                    raise ValueError('A draft must have only the A coordinate')
                draft = dict(a=coordinate(draft.get('a'), row['size_A']))
            status = body.get('status', 'in_progress')
            if status not in STATUSES:
                raise ValueError('Invalid pair status')
            if status == 'not_started' and (points or draft):
                status = 'in_progress'
            if status == 'completed' and (not points or draft or any(p['needs_review'] for p in points)):
                raise ValueError('Completion needs at least one complete point, no draft and no points awaiting review')
            reason = str(body.get('skip_reason', '')).strip()[:4000]
            if status == 'skipped' and not reason:
                raise ValueError('A skip reason is required')
            review = body.get('review', {})
            if not isinstance(review, dict):
                raise ValueError('Invalid review record')
            reviewer = str(review.get('reviewer', '')).strip()[:200]
            checked = review.get('checked_ids', [])
            if not isinstance(checked, list) or not all(isinstance(x, str) and x in ids for x in checked):
                raise ValueError('Invalid reviewed point IDs')
            review_status = review.get('status', 'unreviewed')
            if review_status not in ('unreviewed', 'in_review', 'approved'):
                raise ValueError('Invalid review status')
            # Editing any geometry or review flags invalidates prior approval.
            changed = points != old['points'] or draft != old['draft'] or status != old['status']
            if changed:
                review_status, checked = 'unreviewed', []
            if review_status == 'approved' and (status != 'completed' or not reviewer or set(checked) != set(ids)):
                raise ValueError('Review approval requires completed pair, reviewer and all points checked')
            value = self.empty(pair_id)
            value.update(points=points, draft=draft, status=status, skip_reason=reason,
                         annotator=str(body.get('annotator', ''))[:200], notes=str(body.get('notes', ''))[:8000],
                         review=dict(status=review_status, reviewer=reviewer, checked_ids=list(dict.fromkeys(checked))),
                         revision=old['revision'] + 1, created_at=old['created_at'] or now(), updated_at=now())
            path = self.annotations / (pair_id + '.json')
            # Each revision is a complete recoverable snapshot, never an append of partial JSON.
            write_json(self.output / 'history' / pair_id / f'{value["revision"]:06d}.json', value)
            if path.exists():
                atomic_write(path.with_suffix('.json.bak'), path.read_text())
            write_json(path, value)
            return value

    def overview(self):
        counts = {s: 0 for s in STATUSES}
        selected_counts = {s: 0 for s in STATUSES}
        rows = []
        for row in self.catalog.rows:
            path = self.annotations / (row['pair_id'] + '.json')
            value = json.loads(path.read_text()) if path.exists() else {}
            status = value.get('status', 'not_started')
            counts[status] += 1
            selected = row['pair_id'] in self.selection['pair_ids']
            if selected:
                selected_counts[status] += 1
            rows.append(dict(pair_id=row['pair_id'], status=status, selected=selected,
                             points=len(value.get('points', [])), review=value.get('review', {}).get('status', 'unreviewed'),
                             stratum=self.catalog.stratum(row)))
        return dict(dataset='GoogleEarth', total=len(rows), pairs=rows, counts=counts,
                    selected_counts=selected_counts, selection=self.selection, output=str(self.output))

    def export(self):
        with self.lock():
            values = [self.load(p.stem) for p in sorted(self.annotations.glob('*.json'))]
            values = [v for v in values if v['status'] == 'completed']
            # Drafts never enter an official export, even if hand-edited on disk.
            for value in values:
                if value.get('draft'):
                    raise ValueError('Completed annotation contains a draft')
                value['points'] = [self.validate_point(p, self.catalog.row(value['pair_id'])) for p in value['points']]
            text = ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values)
            table = io.StringIO(newline='')
            fields = ['pair_id', 'dataset', 'image_A', 'image_B', 'id', 'a_x', 'a_y', 'b_x', 'b_y',
                      'type', 'needs_review', 'review_status', 'coordinate_space']
            writer = csv.DictWriter(table, fieldnames=fields)
            writer.writeheader()
            for value in values:
                for p in value['points']:
                    writer.writerow(dict(pair_id=value['pair_id'], dataset='GoogleEarth', image_A=value['image_A'],
                                         image_B=value['image_B'], id=p['id'], a_x=p['a'][0], a_y=p['a'][1],
                                         b_x=p['b'][0], b_y=p['b'][1], type='ground', needs_review=p['needs_review'],
                                         review_status=value['review']['status'], coordinate_space='test_image_pixels'))
            atomic_write(self.output / 'exports/ground_gcps.jsonl', text)
            atomic_write(self.output / 'exports/ground_gcps.csv', table.getvalue())
        return dict(pairs=len(values), points=sum(len(v['points']) for v in values),
                    jsonl='/api/exports/jsonl', csv='/api/exports/csv')
