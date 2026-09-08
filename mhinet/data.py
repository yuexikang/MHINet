"""Manifest-backed pair loading and leakage-safe grouping helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .geometry import pixel_homography_to_normalized


_KOREA_COORDINATE = re.compile(r"(?P<lat>\d{2}\.\d{6})(?P<lon>\d{3}\.\d+)")


def load_rgb_bicubic(path: str | Path, size: int = 784) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (size, size), resample=Image.Resampling.BICUBIC
        )
        array = np.asarray(resized, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def parse_geo_region(record: dict[str, Any], degrees: float = 0.01) -> str:
    """Return a stable coarse geographic cell, or an external-test namespace."""

    source = str(record.get("source", record.get("parent_image_A", "")))
    match = _KOREA_COORDINATE.search(Path(source).stem)
    if match is None:
        row = record.get("input_pair_row_index")
        if row is not None:
            return f"external_test:{int(row)}"
        raise ValueError(f"Cannot derive geographic group from {source!r}")
    latitude = float(match.group("lat"))
    longitude = float(match.group("lon"))
    lat_bin = math.floor(latitude / degrees)
    lon_bin = math.floor(longitude / degrees)
    return f"geo_{degrees:g}:{lat_bin}:{lon_bin}"


def parent_group(record: dict[str, Any]) -> str:
    if record.get("input_pair_row_index") is not None:
        return f"test_pair:{int(record['input_pair_row_index'])}"
    left = str(record.get("parent_image_A", ""))
    right = str(record.get("parent_image_B", left))
    return "parent:" + "|".join(sorted({left, right}))


def iter_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number} is not an object")
            yield record


def grouping_audit(data_root: str | Path, degrees: float = 0.01) -> dict[str, Any]:
    """Audit parent and geographic cells and define a deterministic safe train set.

    Validation is held fixed.  Any training pair whose 0.01-degree cell occurs in
    validation is excluded, so selection data has no coarse geographic overlap.
    The existing test set belongs to a separate paired-image domain and is never
    used to select exclusions or configurations.
    """

    root = Path(data_root).resolve()
    regions: dict[str, set[str]] = {}
    parents: dict[str, set[str]] = {}
    rows: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_regions: set[str] = set()
        split_parents: set[str] = set()
        count = 0
        for record in iter_manifest(root / split / "pairs.jsonl"):
            count += 1
            split_regions.add(parse_geo_region(record, degrees))
            split_parents.add(parent_group(record))
        rows[split] = count
        regions[split] = split_regions
        parents[split] = split_parents
    conflicting_regions = regions["train"] & regions["val"]
    safe_train_pairs = 0
    excluded_train_pairs = 0
    for record in iter_manifest(root / "train/pairs.jsonl"):
        if parse_geo_region(record, degrees) in conflicting_regions:
            excluded_train_pairs += 1
        else:
            safe_train_pairs += 1
    return {
        "policy": {
            "parent_group": "parent_image_A/B; test=input_pair_row_index",
            "geo_cell_degrees": degrees,
            "selection_rule": "hold val fixed; exclude train cells present in val",
            "test_role": "sealed evaluation only; never used for exclusions or selection",
        },
        "source_rows": rows,
        "unique_parent_groups": {key: len(value) for key, value in parents.items()},
        "unique_geo_groups": {key: len(value) for key, value in regions.items()},
        "train_val_parent_overlap": sorted(parents["train"] & parents["val"]),
        "train_val_geo_overlap": sorted(conflicting_regions),
        "safe_train_pairs": safe_train_pairs,
        "excluded_train_pairs": excluded_train_pairs,
    }


@dataclass(frozen=True)
class _IndexEntry:
    offset: int
    pair_id: str
    parent_group: str
    geo_group: str


class HomographyPairDataset(Dataset[dict[str, Any]]):
    """Lazy JSONL dataset; only byte offsets and grouping fields stay in memory."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        image_size: int = 784,
        max_pairs: int | None = None,
        exclude_geo_groups: Iterable[str] = (),
        geo_cell_degrees: float = 0.01,
    ) -> None:
        super().__init__()
        self.manifest = Path(manifest).resolve()
        self.split_root = self.manifest.parent
        self.image_size = int(image_size)
        excluded = set(exclude_geo_groups)
        self.index: list[_IndexEntry] = []
        with self.manifest.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                geo = parse_geo_region(record, geo_cell_degrees)
                if geo in excluded:
                    continue
                self.index.append(
                    _IndexEntry(
                        offset=offset,
                        pair_id=str(record["pair_id"]),
                        parent_group=parent_group(record),
                        geo_group=geo,
                    )
                )
                if max_pairs is not None and len(self.index) >= max_pairs:
                    break

    def __len__(self) -> int:
        return len(self.index)

    def _read_record(self, offset: int) -> dict[str, Any]:
        with self.manifest.open("rb") as stream:
            stream.seek(offset)
            return json.loads(stream.readline())

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.index[index]
        record = self._read_record(entry.offset)
        image_a = load_rgb_bicubic(
            self.split_root / record["image_A"], self.image_size
        )
        image_b = load_rgb_bicubic(
            self.split_root / record["image_B"], self.image_size
        )
        native_h = torch.tensor(record["H_A_to_B"], dtype=torch.float64)
        width_a, height_a = (int(value) for value in record["size_A"])
        width_b, height_b = (int(value) for value in record["size_B"])
        normalized_h = pixel_homography_to_normalized(
            native_h, (height_a, width_a), (height_b, width_b)
        ).float()
        return {
            "images": torch.stack((image_a, image_b), dim=0),
            "H_gt_norm": normalized_h,
            "pair_id": entry.pair_id,
            "parent_group": entry.parent_group,
            "geo_group": entry.geo_group,
            "size_A": torch.tensor(record["size_A"], dtype=torch.float32),
            "size_B": torch.tensor(record["size_B"], dtype=torch.float32),
        }
