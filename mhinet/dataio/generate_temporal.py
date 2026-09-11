"""Four-pair temporal groups, geographically split before augmentation.

Reuses LoMa's generator without modifying it. Cross-time H labels assume the
user-confirmed approximately coregistered parent frame; not exact physical GT.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

from tqdm import tqdm
from mhinet.dataio.data import parse_geo_region


def load_generator(root: Path):
    path = root.resolve() / 'generate_pairs.py'
    spec = importlib.util.spec_from_file_location('mhinet_legacy_pair_generator', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_groups(training_root: Path, legacy):
    groups = []
    seen = set()
    for original_split in ('Train', 'Val'):
        domains = {}
        for domain in ('past', 'current'):
            paths = legacy.discover_images(training_root / domain / original_split, False)
            indexed = {}
            for path in paths:
                if path.stem in indexed:
                    raise ValueError(f'Duplicate parent stem: {path}')
                indexed[path.stem] = path.relative_to(training_root).as_posix()
            domains[domain] = indexed
        if set(domains['past']) != set(domains['current']):
            raise ValueError(f'Unpaired parent images in {original_split}; refusing silent omissions')
        for stem in sorted(domains['past']):
            if stem in seen:
                raise ValueError(f'Parent appears in both original splits: {stem}')
            seen.add(stem)
            groups.append({'group_id': stem, 'past': domains['past'][stem],
                           'current': domains['current'][stem], 'original_split': original_split,
                           'geo_group': parse_geo_region({'source': domains['past'][stem]})})
    return groups


def split_groups(groups, seed=42, val_fraction=.1):
    """Choose a reproducible subset of entire geo cells nearest the target size."""
    if len(groups) < 2 or not 0 < val_fraction < 1:
        raise ValueError('Need at least two groups and 0 < val_fraction < 1')
    by_geo = defaultdict(list)
    for group in groups:
        by_geo[group['geo_group']].append(group)
    keys = sorted(by_geo)
    if len(keys) < 2:
        raise ValueError('At least two geographic cells are required')
    random.Random(seed).shuffle(keys)
    target = round(len(groups) * val_fraction)
    # Subset sum, store a predecessor once. Exact nearest attainable group count
    # avoids splitting a geographic cell merely to satisfy a rounded ratio.
    reachable = {0: None}
    limit = min(len(groups) - 1, target + max(map(len, by_geo.values())))
    for key in keys:
        size = len(by_geo[key])
        for previous in list(reachable):
            total = previous + size
            if total <= limit and total not in reachable:
                reachable[total] = (previous, key)
    total = min((n for n in reachable if 0 < n < len(groups)), key=lambda n: (abs(n-target), n))
    chosen = set()
    cursor = total
    while cursor:
        cursor, key = reachable[cursor]
        chosen.add(key)
    return {split: sorted([g for g in groups if (g['geo_group'] in chosen) == (split == 'val')],
                          key=lambda g: g['group_id']) for split in ('train', 'val')}


def recipes(group_id, legacy, seed):
    # Balance difficulty without assigning current imagery systematically harder
    # supervision than past. Cross-time pairs use independent draws/directions.
    flip = legacy.derive_seed(seed, group_id, 'same_time_difficulty') % 2
    return [('same_past', 'past', 'past', flip),
            ('same_current', 'current', 'current', 1-flip),
            ('cross_past_current', 'past', 'current', 0),
            ('cross_current_past', 'current', 'past', 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loma-root', type=Path, default=Path('/home/disk1/LoMa'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/disk1/Data/datasets/GoogleEarth'))
    parser.add_argument('--existing-test', type=Path, default=Path('/home/disk1/Data/datasets/GoogleEarth_scale_pairs/test'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate', action='store_true', help='Without this flag, print plan only (no writes).')
    parser.add_argument('--smoke', action='store_true', help='Only one parent group per split; NOT a training dataset.')
    args = parser.parse_args(argv)
    legacy = load_generator(args.loma_root)
    training_root = args.dataset_root.resolve() / 'training_data'
    groups = discover_groups(training_root, legacy)
    splits = split_groups(groups, args.seed)
    if args.smoke:
        splits = {key: value[:1] for key, value in splits.items()}
    test_root = args.existing_test.resolve()
    if not (test_root / 'pairs.jsonl').is_file():
        raise FileNotFoundError(test_root / 'pairs.jsonl')
    config = legacy.SamplingConfig(enable_photometric_aug=True)
    summary = {
        'version': 'temporal_four_pairs_v1', 'status': 'planned',
        'smoke_only': args.smoke, 'seed': args.seed,
        'dataset_root': str(args.dataset_root.resolve()), 'training_root': str(training_root),
        'output_dir': str(args.output_dir.resolve()),
        'generator_reference': str(args.loma_root.resolve() / 'generate_pairs.py'),
        'generator_reference_sha256': hashlib.sha256((args.loma_root / 'generate_pairs.py').read_bytes()).hexdigest(),
        'generator_implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sampling_config': {**asdict(config), 'pairs_per_temporal_group': 4, 'pairs_per_image': None},
        'parent_coordinate_assumption': 'user-confirmed approximately coregistered; base_H=identity; no exact cross-time GT claim',
        'csv_affine_fields_used': False,
        'split_policy': 'merge original Train/Val parent pool; whole 0.01-degree cells; nearest attainable val fraction 0.1',
        'source_parent_groups': len(groups),
        'planned': {k: {'parent_groups': len(v), 'pairs': 4*len(v), 'geo_groups': len({g['geo_group'] for g in v})} for k,v in splits.items()},
        'actual_val_fraction': len(splits['val']) / sum(map(len, splits.values())),
        'test_policy': 'unchanged reuse via symlink; generator never writes to test; no test resampling or repartition',
        'existing_test': str(test_root),
        'test_manifest_sha256': hashlib.sha256((test_root / 'pairs.jsonl').read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.generate:
        return 0
    out = args.output_dir.resolve()
    legacy.ensure_new_output_directory(out)
    (out / 'parent_split_manifest.json').write_text(json.dumps(splits, ensure_ascii=False, indent=2)+'\n')
    (out / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    for split, members in splits.items():
        split_out = out / split
        for folder in ('images', 'masks', 'metadata', 'visualization'):
            (split_out / folder).mkdir(parents=True, exist_ok=True)
        count = 0
        previews = legacy.VisualizationReservoir(20, legacy.derive_seed(args.seed, split, 'previews'))
        with (split_out / 'pairs.jsonl').open('w') as stream:
            for group in tqdm(members, desc=f'Generating {split}', unit='parent group'):
                images = {domain: legacy.read_image(training_root / group[domain], config.min_input_side)
                          for domain in ('past', 'current')}
                for kind, a, b, difficulty in recipes(group['group_id'], legacy, args.seed):
                    pair_id = f"{kind}__{legacy._safe_identifier(group['group_id'])}"
                    seed = legacy.derive_seed(args.seed, split, group['group_id'], kind)
                    metadata = {
                        'split': split, 'source': group[a], 'parent_image_A': group[a], 'parent_image_B': group[b],
                        'parent_temporal_group': group['group_id'], 'geo_group': group['geo_group'],
                        'input_mode': 'single_image' if a == b else 'paired_coregistered_images',
                        'temporal_domain': a if a == b else f'{a}_to_{b}', 'pair_kind': kind,
                        'label_provenance': 'exact_synthetic_same_parent' if a == b else 'synthetic_with_approximate_identity_parent_alignment',
                        'common_coordinate_assumption': summary['parent_coordinate_assumption'] if a != b else 'same parent image',
                        'original_parent_split': group['original_split'],
                    }
                    # Fail loudly: never silently produce incomplete four-pair groups.
                    row = legacy.generate_and_save_pair(images[a], images[b], split_out,
                        pair_id, difficulty, seed, config, metadata)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    previews.consider(split_out / row['metadata'])
                    count += 1
                stream.flush()
        assert count == len(members)*4
        summary['planned'][split]['generated_pairs'] = count
        for index, path in enumerate(previews.metadata_paths):
            legacy.visualize_pair(path, split_out, index, legacy.derive_seed(args.seed, split, 'preview', index))
        summary['planned'][split]['visualizations'] = len(previews.metadata_paths)
    (out / 'test').symlink_to(test_root, target_is_directory=True)
    summary['status'] = 'completed'
    (out / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
