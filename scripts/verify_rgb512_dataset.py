"""Verify MHINet_RGB_512 manifests, PNG headers, geometry, and decoded samples."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
from typing import Any, Iterable

import cv2
import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_header(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as stream:
        signature = stream.read(8)
        length = struct.unpack(">I", stream.read(4))[0]
        kind = stream.read(4)
        payload = stream.read(length)
    if signature != PNG_SIGNATURE or kind != b"IHDR" or len(payload) != 13:
        raise ValueError(f"Invalid PNG header: {path}")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
    return width, height, bit_depth, color_type


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (matrix @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise AssertionError("Invalid homogeneous denominator")
    return projected[:, :2] / projected[:, 2:3]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _check_row(split_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    if row["size_A"] != [512, 512] or row["size_B"] != [512, 512]:
        raise AssertionError(f"Unexpected output size: {row['pair_id']}")
    if row["A_policy"] != "complete_source_image_no_radiometric_augmentation":
        raise AssertionError(f"Invalid A policy: {row['pair_id']}")
    if row["augmentation_mode"] not in {"weak", "strong"}:
        raise AssertionError(f"Invalid augmentation mode: {row['pair_id']}")
    if not 2.0 <= float(row["area_ratio_A_to_B"]) <= 4.0:
        raise AssertionError(f"Invalid area ratio: {row['pair_id']}")
    if min(float(value) for value in row["overlap_fractions"]) < 0.20:
        raise AssertionError(f"Invalid overlap: {row['pair_id']}")
    if float(row["max_reprojection_error"]) > 1e-6:
        raise AssertionError(f"Stored reprojection error too large: {row['pair_id']}")

    transform_a = np.asarray(row["T_source_to_A"], dtype=np.float64)
    transform_b = np.asarray(row["T_source_to_B"], dtype=np.float64)
    homography = np.asarray(row["H_A_to_B"], dtype=np.float64)
    inverse = np.asarray(row["H_B_to_A"], dtype=np.float64)
    expected = transform_b @ np.linalg.inv(transform_a)
    expected /= expected[2, 2]
    if not np.allclose(homography, expected, atol=1e-9, rtol=1e-9):
        raise AssertionError(f"Homography composition mismatch: {row['pair_id']}")
    if not np.allclose(homography @ inverse / (homography @ inverse)[2, 2], np.eye(3), atol=1e-9):
        raise AssertionError(f"Homography inverse mismatch: {row['pair_id']}")
    corner_a = np.asarray(row["corner_A"], dtype=np.float64)
    corner_b = np.asarray(row["corner_B"], dtype=np.float64)
    reprojection = np.linalg.norm(_project(homography, corner_a) - corner_b, axis=1).max()
    if reprojection > 1e-6:
        raise AssertionError(f"Reprojection mismatch: {row['pair_id']} {reprojection}")

    files = {
        "image_A": (3, 2),
        "image_B": (3, 2),
        "mask_A_overlap": (0, 0),
        "mask_B_overlap": (0, 0),
    }
    paths = []
    for key, _ in files.items():
        path = split_root / row[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append((key, path))
    return {"paths": paths, "reprojection": float(reprojection)}


def _check_header(item: tuple[str, Path]) -> None:
    key, path = item
    width, height, depth, color_type = _png_header(path)
    if (width, height, depth) != (512, 512, 8):
        raise AssertionError(f"Unexpected PNG header for {path}: {(width, height, depth, color_type)}")
    if key.startswith("image_") and color_type != 2:
        raise AssertionError(f"Expected RGB PNG: {path} color_type={color_type}")
    if key.startswith("mask_") and color_type != 0:
        raise AssertionError(f"Expected grayscale PNG: {path} color_type={color_type}")


def _decode_sample(split_root: Path, row: dict[str, Any]) -> None:
    image_a = cv2.imread(str(split_root / row["image_A"]), cv2.IMREAD_UNCHANGED)
    image_b = cv2.imread(str(split_root / row["image_B"]), cv2.IMREAD_UNCHANGED)
    mask_a = cv2.imread(str(split_root / row["mask_A_overlap"]), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(str(split_root / row["mask_B_overlap"]), cv2.IMREAD_GRAYSCALE)
    if image_a is None or image_b is None or mask_a is None or mask_b is None:
        raise OSError(f"Full decode failed: {row['pair_id']}")
    if image_a.shape != (512, 512, 3) or image_b.shape != (512, 512, 3):
        raise AssertionError(f"Decoded RGB shape mismatch: {row['pair_id']}")
    if mask_a.shape != (512, 512) or mask_b.shape != (512, 512):
        raise AssertionError(f"Decoded mask shape mismatch: {row['pair_id']}")
    measured = [float((mask_a > 0).mean()), float((mask_b > 0).mean())]
    if not np.allclose(measured, row["overlap_fractions"], atol=1.0 / (512 * 512)):
        raise AssertionError(f"Overlap mask mismatch: {row['pair_id']}")


def _representative_rows(rows: list[dict[str, Any]], per_group: int = 3) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in sorted({row["source_dataset"] for row in rows}):
        for mode in ("weak", "strong"):
            matches = [row for row in rows if row["source_dataset"] == dataset and row["augmentation_mode"] == mode]
            selected.extend(matches[:per_group])
    return selected


def verify_dataset(root: Path, *, full_decode: bool = False) -> dict[str, Any]:
    root = root.resolve()
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary["status"] not in {"generated_pending_verification", "completed"}:
        raise AssertionError(f"Dataset is not ready for verification: {summary['status']}")

    pair_ids: set[str] = set()
    source_paths: dict[str, str] = {}
    counts: dict[str, int] = {}
    all_headers: list[tuple[str, Path]] = []
    max_reprojection = 0.0
    decoded_count = 0
    for split in ("train", "val", "test"):
        rows = _read_rows(root / split / "pairs.jsonl")
        counts[split] = len(rows)
        if len(rows) != int(summary["planned_pair_counts"][split]):
            raise AssertionError(f"Manifest count mismatch for {split}")
        for row in rows:
            if row["split"] != split:
                raise AssertionError(f"Split field mismatch: {row['pair_id']}")
            if row["pair_id"] in pair_ids:
                raise AssertionError(f"Duplicate pair id: {row['pair_id']}")
            pair_ids.add(row["pair_id"])
            previous = source_paths.setdefault(row["source_path"], split)
            if previous != split:
                raise AssertionError(f"Source leakage across splits: {row['source_path']}")
            result = _check_row(root / split, row)
            all_headers.extend(result["paths"])
            max_reprojection = max(max_reprojection, result["reprojection"])
        sample = rows if full_decode else _representative_rows(rows)
        for row in sample:
            _decode_sample(root / split, row)
            decoded_count += 1

    with ThreadPoolExecutor(max_workers=64) as executor:
        for _ in executor.map(_check_header, all_headers, chunksize=64):
            pass
    return {
        "status": "passed",
        "pair_counts": counts,
        "unique_pair_ids": len(pair_ids),
        "unique_source_paths": len(source_paths),
        "png_headers_checked": len(all_headers),
        "pairs_fully_decoded": decoded_count,
        "maximum_reprojection_error_pixels": max_reprojection,
        "source_leakage_across_splits": 0,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-decode", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify_dataset(args.root, full_decode=args.full_decode)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
