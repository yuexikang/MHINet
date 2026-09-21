"""Single-parent synthetic pairs; geographic split before augmentation.

Historical module name retained for command compatibility. No cross-time labels.
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
    # Two independent synthetic draws per individual parent, never cross-time.
    return [('same_past_normal', 'past', 'past', 0),
            ('same_past_hard', 'past', 'past', 1),
            ('same_current_normal', 'current', 'current', 0),
            ('same_current_hard', 'current', 'current', 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loma-root', type=Path, default=Path('/home/disk1/LoMa'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/disk1/Data/datasets/GoogleEarth'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate', action='store_true', help='Without this flag, print plan only (no writes).')
    parser.add_argument('--smoke', action='store_true', help='Only one parent group per split; NOT a training dataset.')
    parser.add_argument('--three-tiers', action='store_true', help='Quadrants, rotation, radiation and visibility-aware occlusion; seven pairs per parent.')
    parser.add_argument('--stable-geometry', action='store_true', help='Opt-in bidirectional normalized geometry bounds; new dataset only.')
    args = parser.parse_args(argv)
    if args.stable_geometry and not args.three_tiers:
        parser.error('--stable-geometry requires --three-tiers')
    legacy = load_generator(args.loma_root)
    training_root = args.dataset_root.resolve() / 'training_data'
    groups = discover_groups(training_root, legacy)
    splits = split_groups(groups, args.seed)
    if args.smoke:
        splits = {key: value[:1] for key, value in splits.items()}
    test_root = args.dataset_root.resolve() / 'evaluation_data'
    test_rows = legacy.load_evaluation_rows(test_root)
    config = legacy.SamplingConfig(enable_photometric_aug=True)
    per_group = 14 if args.three_tiers else 4
    summary = {
        'version': 'quadrant_three_tiers_v1' if args.three_tiers else 'single_parent_v2', 'status': 'planned',
        'smoke_only': args.smoke, 'seed': args.seed,
        'dataset_root': str(args.dataset_root.resolve()), 'training_root': str(training_root),
        'output_dir': str(args.output_dir.resolve()),
        'generator_reference': str(args.loma_root.resolve() / 'generate_pairs.py'),
        'generator_reference_sha256': hashlib.sha256((args.loma_root / 'generate_pairs.py').read_bytes()).hexdigest(),
        'generator_implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'quadrant_implementation_sha256': hashlib.sha256(Path(__file__).with_name('quadrant_tiers.py').read_bytes()).hexdigest() if args.three_tiers else None,
        'sampling_config': ({'tiers': [3,2,2], 'ratios': [.8,.6,.4], 'rotation_degrees': [-30,30],
            'minimum_geometric_overlap_each': .5, 'minimum_visible_overlap_each_tier3': .3,
            'occlusion_area_fraction': [.1,.3], 'pairs_per_temporal_group':14, 'pairs_per_image':7}
            if args.three_tiers else {**asdict(config), 'pairs_per_temporal_group':4, 'pairs_per_image':2}),
        'parent_coordinate_assumption': 'A and B always generated from the identical parent image',
        'csv_affine_fields_used': False,
        'split_policy': 'merge original Train/Val parent pool; whole 0.01-degree cells; nearest attainable val fraction 0.1',
        'source_parent_groups': len(groups),
        'planned': {k: {'parent_groups': len(v), 'pairs': per_group*len(v), 'geo_groups': len({g['geo_group'] for g in v})} for k,v in splits.items()},
        'actual_val_fraction': len(splits['val']) / sum(map(len, splits.values())),
        'test_policy': 'original CSV pairs, visualization only, no ground truth or precision metrics',
        'existing_test': str(test_root),
        'test_csv_sha256': hashlib.sha256((test_root / 'test_pairs.csv').read_bytes()).hexdigest(),
    }
    if args.stable_geometry:
        from mhinet.dataio.stable_geometry import POLICY
        summary['version'] = 'quadrant_three_tiers_stable_v2'
        summary['sampling_config']['stable_geometry'] = POLICY
        summary['stability_implementation_sha256'] = hashlib.sha256(Path(__file__).with_name('stable_geometry.py').read_bytes()).hexdigest()
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
            for group_index, group in enumerate(tqdm(members, desc=f'Generating {split}', unit='parent group')):
                images = {domain: legacy.read_image(training_root / group[domain], config.min_input_side)
                          for domain in ('past', 'current')}
                selected_recipes = recipes(group['group_id'], legacy, args.seed)
                if args.three_tiers:
                    selected_recipes = [(f'{domain}_tier{tier}_{j}', domain, domain, j)
                        for domain in ('past','current') for tier,n in ((1,3),(2,2),(3,2)) for j in range(n)]
                for kind, a, b, difficulty in selected_recipes:
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
                    generator = legacy.generate_and_save_pair
                    if args.three_tiers:
                        from mhinet.dataio.quadrant_tiers import generate
                        generator = generate
                        tier = int(kind.split('tier')[1].split('_')[0])
                        ratio_index = difficulty if tier==1 else (group_index+difficulty+(a=='current'))%3
                        metadata.update(tier=tier, resolution_ratio=(.8,.6,.4)[ratio_index], stable_geometry=args.stable_geometry)
                    row = generator(images[a], images[b], split_out,
                        pair_id, difficulty, seed, config, metadata)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    previews.consider(split_out / row['metadata'])
                    count += 1
                stream.flush()
        assert count == len(members)*per_group
        summary['planned'][split]['generated_pairs'] = count
        for index, path in enumerate([] if args.three_tiers else previews.metadata_paths):
            legacy.visualize_pair(path, split_out, index, legacy.derive_seed(args.seed, split, 'preview', index))
        summary['planned'][split]['visualizations'] = 0 if args.three_tiers else len(previews.metadata_paths)
    (out / 'test').mkdir()
    with (out / 'test/pairs.jsonl').open('w') as stream:
        for index, row in enumerate(test_rows):
            stream.write(json.dumps({'pair_id': f'original_{index:04d}', 'split': 'test',
                'input_pair_row_index': index, 'image_A': str(test_root / row['Target']),
                'image_B': str(test_root / row['Source']), 'has_ground_truth': False,
                'evaluation_policy': 'visualization_only'})+'\n')
    summary['status'] = 'completed'
    (out / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
