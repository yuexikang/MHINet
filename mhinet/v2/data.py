"""RGB512 manifests use source-image identity, not legacy geographic filenames."""
from collections import Counter
import json
from pathlib import Path

from mhinet.dataio.data import _IndexEntry
from mhinet.pretraining.data import SharedPairDataset
import numpy as np
import torch
from PIL import Image
from mhinet.dataio.data import load_rgb_bicubic
from mhinet.ops.geometry import pixel_homography_to_normalized
from .geometry_gt import GeometryGT


def read_geometry(row, root, masks):
    """Explicit geometry metadata; legacy synthetic H rows retain their meaning.

    Dense NPZ keys: map_ab, known_a, visible_a, optional map_ba/known_b/visible_b.
    Sparse NPZ keys: points_a, points_b, optional visible. Coordinates must be
    declared as pixel or normalized; dense representation is absolute targets.
    """
    spec = row.get('geometry', {})
    kind = spec.get('kind', 'homography' if 'H_A_to_B' in row else 'unlabeled')
    hw_a, hw_b = tuple(reversed(row['size_A'])), tuple(reversed(row['size_B']))
    weak = row.get('gt_type') == 'geo_approx' or spec.get('exact') is False
    options = dict(hw_a=hw_a, hw_b=hw_b, mask_a=masks[0], mask_b=masks[1], exact=not weak,
                   tolerance_px=float(row.get('soft_radius_B_px', spec.get('tolerance_px', 0.))) if weak else 0.)
    if kind == 'homography':
        H = pixel_homography_to_normalized(torch.tensor(row['H_A_to_B'], dtype=torch.float64), hw_a, hw_b).float()
        return GeometryGT(kind, H=H, **options)
    if kind == 'unlabeled':
        return GeometryGT(kind, **options)
    coordinates = spec.get('coordinates')
    if coordinates not in ('pixel', 'normalized'):
        raise ValueError('Dense/sparse geometry requires coordinates=pixel or normalized')
    with np.load(Path(root)/spec['path'], allow_pickle=False) as archive:
        values = {k: torch.from_numpy(archive[k].copy()) for k in archive.files}
    def convert(value, hw, dense=False):
        value = value.float()
        if dense:
            if value.shape != (2, *hw_a) and value.shape != (2, *hw_b):
                raise ValueError('Dense fields use native 2xHxW image-center grids')
            value = value.permute(1, 2, 0)
        if coordinates == 'pixel':
            value = 2*(value+.5)/value.new_tensor([hw[1], hw[0]])-1
        return value.permute(2, 0, 1) if dense else value
    if kind == 'dense_flow':
        if spec.get('representation') != 'absolute':
            raise ValueError('Dense flow supervision requires an absolute target-coordinate map')
        payload = {k: values[k] for k in ('known_a', 'known_b', 'visible_a', 'visible_b') if k in values}
        payload['map_ab'] = convert(values['map_ab'], hw_b, True)
        if 'map_ba' in values:
            payload['map_ba'] = convert(values['map_ba'], hw_a, True)
        return GeometryGT(kind, **payload, **options)
    if kind == 'sparse_points':
        a, b = convert(values['points_a'], hw_a), convert(values['points_b'], hw_b)
        if a.ndim != 2 or a.shape[-1] != 2 or a.shape != b.shape:
            raise ValueError('Sparse annotations must be matching Nx2 points')
        return GeometryGT(kind, points_a=a, points_b=b, point_visible=values.get('visible'), **options)
    raise ValueError(f'Unknown geometry kind {kind}')


class GeometryPairDataset(torch.utils.data.Dataset):
    """B=1 variable-label loader; no manufactured H or dense sparse-point labels."""
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row['size_A'] != [512, 512] or row['size_B'] != [512, 512]:
            raise ValueError('v2 geometry training requires 512 images')
        images = torch.stack([load_rgb_bicubic(self.root/row[f'image_{s}'], 512) for s in 'AB'])
        masks = []
        for side in 'AB':
            path = row.get(f'valid_{side}', row.get(f'mask_{side}_overlap'))
            if path:
                with Image.open(self.root/path) as image:
                    mask = torch.from_numpy(np.array(image.convert('L'), copy=True)).float()[None]/255
            else:
                mask = torch.ones(1, 512, 512)
            masks.append(mask)
        geometry = read_geometry(row, self.root, masks)
        return dict(images=images[None], geometry=geometry, pair_id=[row['pair_id']],
                    group=row.get('G', 'unknown'))


class RGB512Dataset(SharedPairDataset):
    """Retain the existing image/GT loader with an explicit RGB manifest index.

    The original RGB512 split is by source image. It does not establish spatial
    independence between neighbouring FLAIR tiles; report that limitation rather
    than inventing geographic coordinates or changing the frozen split.
    """
    grouping_policy = "source_dataset + source_relative_path (fallback source_path); source-image isolation, not geographic-neighbour isolation"

    def __init__(self, manifest, *, image_size=512, max_pairs=None):
        if image_size != 512:
            raise ValueError("RGB512Dataset only accepts 512")
        self.manifest = Path(manifest).resolve()
        self.split_root = self.manifest.parent
        self.image_size = 512
        self.include_overlap_mask = False
        self.index = []; self.source_counts = Counter()
        identities = set()
        with self.manifest.open("rb") as stream:
            while True:
                offset = stream.tell(); line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("size_A") != [512, 512] or row.get("size_B") != [512, 512]:
                    raise ValueError("RGB512 manifests require stored 512x512 images on both sides")
                dataset = row.get("source_dataset")
                original = row.get("source_relative_path") or row.get("source_path")
                if not dataset or not original:
                    raise ValueError("RGB512 row lacks its original source identity")
                if row.get("split") != self.split_root.name:
                    raise ValueError("Manifest row split disagrees with its directory")
                pair = str(row["pair_id"])
                if pair in identities:
                    raise ValueError(f"Duplicate RGB512 pair ID: {pair}")
                identities.add(pair)
                group = f"source:{dataset}:{Path(original).as_posix()}"
                self.index.append(_IndexEntry(offset, pair, group, group))
                self.source_counts[dataset] += 1
                if max_pairs is not None and len(self.index) >= max_pairs:
                    break
