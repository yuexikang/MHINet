"""Build a geometry-only GoogleEarth cross-temporal validation set.

Image A is the complete ``target`` image mapped to 512x512. Image B is the
matching ``source`` image after the same single-quadrilateral geometry used by
``generate_rgb512.py``. No synthetic radiometric operation is applied.

The GoogleEarth evaluation images are treated as nominally co-registered. The
stored homography therefore describes the added synthetic geometry relative to
that nominal identity registration; it does not claim to correct small native
cross-temporal registration residuals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm

from mhinet.dataio.generate_rgb512 import (
    GLOBAL_SEED,
    OUTPUT_SIZE,
    _write_png,
    derive_seed,
    project_points,
    read_source_rgb,
    sample_geometry,
)


DEFAULT_INPUT = Path("/home/disk1/Data/datasets/GoogleEarth/evaluation_data")
DEFAULT_OUTPUT = Path("/home/disk1/Data/datasets/Test_Data/GoogleEarth")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class TemporalRecord:
    pair_id: str
    row_index: int
    source_path: str
    target_path: str
    source_relative_path: str
    target_relative_path: str
    geometry_seed: int
    csv_control_points: tuple[tuple[float, float], ...]
    csv_affine_parameters: tuple[float, float, float, float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_records(input_dir: Path, seed: int) -> list[TemporalRecord]:
    csv_path = input_dir / "test_pairs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    records: list[TemporalRecord] = []
    seen_ids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            source_relative = row["Source"].strip()
            target_relative = row["Target"].strip()
            source_path = (input_dir / source_relative).resolve()
            target_path = (input_dir / target_relative).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if not target_path.is_file():
                raise FileNotFoundError(target_path)
            source_number = Path(source_relative).stem.removeprefix("source_")
            target_number = Path(target_relative).stem.removeprefix("target_")
            if source_number != target_number:
                raise ValueError(
                    f"Cross-temporal index mismatch in CSV row {row_index}: "
                    f"{source_relative} vs {target_relative}"
                )
            pair_id = f"googleearth_eval__{source_number}"
            if pair_id in seen_ids:
                raise ValueError(f"Duplicate pair id: {pair_id}")
            seen_ids.add(pair_id)
            points = tuple(
                (float(row[f"XC{index}"]), float(row[f"YC{index}"]))
                for index in range(1, 21)
            )
            affine = tuple(
                float(row[key]) for key in ("A11", "A12", "tx", "A21", "A22", "ty")
            )
            records.append(
                TemporalRecord(
                    pair_id=pair_id,
                    row_index=row_index,
                    source_path=str(source_path),
                    target_path=str(target_path),
                    source_relative_path=source_relative,
                    target_relative_path=target_relative,
                    geometry_seed=derive_seed(seed, "googleearth_temporal_val", pair_id, "geometry"),
                    csv_control_points=points,
                    csv_affine_parameters=affine,
                )
            )
    if not records:
        raise RuntimeError(f"No pairs found in {csv_path}")
    return records


def _generate_worker(payload: tuple[TemporalRecord, str]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    record, split_root_text = payload
    split_root = Path(split_root_text)
    source, source_channels, source_dtype = read_source_rgb(record.source_path)
    target, target_channels, target_dtype = read_source_rgb(record.target_path)
    source_height, source_width = source.shape[:2]
    target_height, target_width = target.shape[:2]
    if (source_width, source_height) != (target_width, target_height):
        raise ValueError(
            f"Native dimensions differ for {record.pair_id}: "
            f"source={(source_width, source_height)} target={(target_width, target_height)}"
        )

    geometry = sample_geometry(
        np.random.default_rng(record.geometry_seed), source_width, source_height
    )
    image_a = cv2.warpPerspective(
        target,
        geometry["T_source_to_A"],
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    image_b = cv2.warpPerspective(
        source,
        geometry["T_source_to_B"],
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    paths = {
        "image_A": f"images/{record.pair_id}_A.png",
        "image_B": f"images/{record.pair_id}_B.png",
        "mask_A_overlap": f"masks/{record.pair_id}_A_overlap.png",
        "mask_B_overlap": f"masks/{record.pair_id}_B_overlap.png",
    }
    _write_png(split_root / paths["image_A"], image_a)
    _write_png(split_root / paths["image_B"], image_b)
    _write_png(split_root / paths["mask_A_overlap"], geometry["mask_A_overlap"])
    _write_png(split_root / paths["mask_B_overlap"], geometry["mask_B_overlap"])

    row: dict[str, Any] = {
        "pair_id": record.pair_id,
        "split": "val",
        "source_dataset": "googleearth_evaluation_cross_temporal",
        "csv_row_index": record.row_index,
        "target_path": record.target_path,
        "source_path": record.source_path,
        "target_relative_path": record.target_relative_path,
        "source_relative_path": record.source_relative_path,
        "target_image_size": [target_width, target_height],
        "source_image_size": [source_width, source_height],
        "target_channels": target_channels,
        "source_channels": source_channels,
        "target_dtype": target_dtype,
        "source_dtype": source_dtype,
        "size_A": [OUTPUT_SIZE, OUTPUT_SIZE],
        "size_B": [OUTPUT_SIZE, OUTPUT_SIZE],
        "A_temporal_role": "target",
        "B_temporal_role": "source",
        "A_policy": "complete_target_image_no_radiometric_augmentation",
        "B_policy": "corresponding_source_single_quadrilateral_no_radiometric_augmentation",
        "radiometric_augmentation": "none",
        "native_registration_model": "nominal_identity_from_dataset_pairing",
        "homography_label_scope": "added_synthetic_geometry_relative_to_nominal_native_coregistration",
        "geometry_seed": record.geometry_seed,
        "source_csv_control_points": [list(point) for point in record.csv_control_points],
        "source_csv_affine_parameters": list(record.csv_affine_parameters),
        "source_csv_geometry_used": False,
        "source_csv_geometry_note": (
            "Preserved for provenance only; it is not the target-to-source native registration."
        ),
        **paths,
    }
    for key in (
        "quad_A_source",
        "quad_B_source",
        "T_source_to_A",
        "T_source_to_B",
        "H_A_to_B",
        "H_B_to_A",
        "corner_A",
        "corner_B",
    ):
        row[key] = geometry[key].tolist()
    for key in (
        "area_ratio_A_to_B",
        "equivalent_side_ratio",
        "overlap_fractions",
        "geometry_stability",
        "sampling_attempts",
        "max_reprojection_error",
    ):
        row[key] = geometry[key]
    return row


def _png_header(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as stream:
        signature = stream.read(8)
        length = struct.unpack(">I", stream.read(4))[0]
        kind = stream.read(4)
        payload = stream.read(length)
    if signature != PNG_SIGNATURE or kind != b"IHDR" or len(payload) != 13:
        raise ValueError(f"Invalid PNG header: {path}")
    width, height, depth, color_type = struct.unpack(">IIBB", payload[:10])
    return width, height, depth, color_type


def _verify_header(item: tuple[Path, int]) -> None:
    path, expected_color_type = item
    width, height, depth, color_type = _png_header(path)
    if (width, height, depth, color_type) != (
        OUTPUT_SIZE,
        OUTPUT_SIZE,
        8,
        expected_color_type,
    ):
        raise AssertionError(
            f"Unexpected PNG header {path}: {(width, height, depth, color_type)}"
        )


def _recompute_pixels(split_root: Path, row: dict[str, Any]) -> None:
    source, _, _ = read_source_rgb(row["source_path"])
    target, _, _ = read_source_rgb(row["target_path"])
    expected_a = cv2.warpPerspective(
        target,
        np.asarray(row["T_source_to_A"], dtype=np.float64),
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    expected_b = cv2.warpPerspective(
        source,
        np.asarray(row["T_source_to_B"], dtype=np.float64),
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    actual_a = cv2.imread(str(split_root / row["image_A"]), cv2.IMREAD_COLOR)
    actual_b = cv2.imread(str(split_root / row["image_B"]), cv2.IMREAD_COLOR)
    if actual_a is None or actual_b is None:
        raise OSError(f"Failed to decode generated pair {row['pair_id']}")
    actual_a = cv2.cvtColor(actual_a, cv2.COLOR_BGR2RGB)
    actual_b = cv2.cvtColor(actual_b, cv2.COLOR_BGR2RGB)
    if not np.array_equal(actual_a, expected_a):
        raise AssertionError(f"A pixel reconstruction mismatch: {row['pair_id']}")
    if not np.array_equal(actual_b, expected_b):
        raise AssertionError(
            f"B pixel reconstruction mismatch (possible radiometric change): {row['pair_id']}"
        )


def verify_dataset(root: Path, *, recompute_all_pixels: bool = False) -> dict[str, Any]:
    root = root.resolve()
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    manifest = root / "val" / "pairs.jsonl"
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != int(summary["pair_count"]):
        raise AssertionError("Manifest count does not match dataset summary")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise AssertionError("Duplicate pair ids")

    headers: list[tuple[Path, int]] = []
    max_reprojection = 0.0
    minimum_overlap = 1.0
    for row in rows:
        if row["radiometric_augmentation"] != "none":
            raise AssertionError(f"Unexpected radiometric operation: {row['pair_id']}")
        if row["size_A"] != [OUTPUT_SIZE, OUTPUT_SIZE] or row["size_B"] != [OUTPUT_SIZE, OUTPUT_SIZE]:
            raise AssertionError(f"Unexpected size: {row['pair_id']}")
        transform_a = np.asarray(row["T_source_to_A"], dtype=np.float64)
        transform_b = np.asarray(row["T_source_to_B"], dtype=np.float64)
        homography = np.asarray(row["H_A_to_B"], dtype=np.float64)
        inverse = np.asarray(row["H_B_to_A"], dtype=np.float64)
        expected = transform_b @ np.linalg.inv(transform_a)
        expected /= expected[2, 2]
        if not np.allclose(homography, expected, atol=1e-9, rtol=1e-9):
            raise AssertionError(f"Homography composition mismatch: {row['pair_id']}")
        identity = homography @ inverse
        identity /= identity[2, 2]
        if not np.allclose(identity, np.eye(3), atol=1e-9, rtol=1e-9):
            raise AssertionError(f"Homography inverse mismatch: {row['pair_id']}")
        corner_a = np.asarray(row["corner_A"], dtype=np.float64)
        corner_b = np.asarray(row["corner_B"], dtype=np.float64)
        error = float(
            np.linalg.norm(project_points(homography, corner_a) - corner_b, axis=1).max()
        )
        if error > 1e-6:
            raise AssertionError(f"Reprojection mismatch: {row['pair_id']} {error}")
        max_reprojection = max(max_reprojection, error)
        minimum_overlap = min(minimum_overlap, *(float(x) for x in row["overlap_fractions"]))
        if not 2.0 <= float(row["area_ratio_A_to_B"]) <= 4.0:
            raise AssertionError(f"Area-ratio mismatch: {row['pair_id']}")
        if min(row["overlap_fractions"]) < 0.20:
            raise AssertionError(f"Overlap below 20%: {row['pair_id']}")
        headers.extend(
            [
                (root / "val" / row["image_A"], 2),
                (root / "val" / row["image_B"], 2),
                (root / "val" / row["mask_A_overlap"], 0),
                (root / "val" / row["mask_B_overlap"], 0),
            ]
        )
    with ThreadPoolExecutor(max_workers=min(64, max(1, os.cpu_count() or 1))) as executor:
        list(executor.map(_verify_header, headers, chunksize=32))

    pixel_rows = rows if recompute_all_pixels else rows[:: max(1, len(rows) // 10)][:10]
    for row in pixel_rows:
        _recompute_pixels(root / "val", row)
    return {
        "status": "passed",
        "pair_count": len(rows),
        "png_headers_checked": len(headers),
        "pairs_pixel_recomputed": len(pixel_rows),
        "maximum_reprojection_error_pixels": max_reprojection,
        "minimum_overlap_fraction": minimum_overlap,
        "radiometric_augmentation": "none",
    }


def _configuration(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "output_format": "PNG",
        "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
        "temporal_roles": {"A": "target", "B": "source"},
        "pairing": "target_NNN.jpg with source_NNN.jpg from test_pairs.csv",
        "native_registration_model": "nominal identity from dataset pairing",
        "label_scope": "synthetic geometry relative to nominal native co-registration",
        "geometry": {
            "implementation": "mhinet.dataio.generate_rgb512.sample_geometry",
            "B": "one quadrant-sampled convex quadrilateral",
            "area_ratio_A_to_B": [2.0, 4.0],
            "minimum_overlap_each": 0.20,
            "minimum_edge_fraction": 0.15,
            "angle_degrees": [25.0, 155.0],
            "solver": "float64 normalized DLT",
        },
        "radiometric": {"A": "none", "B": "none"},
        "csv_geometry": "preserved in manifest for provenance; not used",
    }


def generate_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    workers: int,
    max_pairs: int | None = None,
    allow_empty_output_dir: bool = False,
) -> tuple[Path, Path]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not allow_empty_output_dir or not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.inprogress")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging directory: {staging}")

    all_records = discover_records(input_dir, seed)
    records = all_records if max_pairs is None else all_records[:max_pairs]
    if not records:
        raise ValueError("No records selected")
    summary: dict[str, Any] = {
        "name": output_dir.name,
        "version": "GoogleEarth_cross_temporal_geometry_only_512_v1",
        "status": "generating",
        "dry_run": max_pairs is not None,
        "input_dir": str(input_dir),
        "input_csv": str(input_dir / "test_pairs.csv"),
        "input_csv_sha256": _sha256(input_dir / "test_pairs.csv"),
        "discovered_pair_count": len(all_records),
        "pair_count": len(records),
        "configuration": _configuration(seed),
    }
    staging.mkdir(parents=True)
    try:
        split_root = staging / "val"
        (split_root / "images").mkdir(parents=True)
        (split_root / "masks").mkdir()
        shutil.copy2(input_dir / "test_pairs.csv", staging / "source_test_pairs.csv")
        with (staging / "source_pair_manifest.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        {
                            "pair_id": record.pair_id,
                            "csv_row_index": record.row_index,
                            "target_path": record.target_path,
                            "source_path": record.source_path,
                            "geometry_seed": record.geometry_seed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        payloads = [(record, str(split_root)) for record in records]
        with (split_root / "pairs.jsonl").open("w", encoding="utf-8") as stream:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(_generate_worker, payloads, chunksize=1)
                for row in tqdm(iterator, total=len(payloads), desc="Generating val", unit="pair"):
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary["status"] = "generated_pending_verification"
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return staging, output_dir
    except Exception:
        summary["status"] = "failed"
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


def finalize_dataset(
    staging: Path, output: Path, verification: dict[str, Any], *, allow_empty_output_dir: bool
) -> None:
    summary_path = staging / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["verification"] = verification
    summary["status"] = "completed"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        if not allow_empty_output_dir or not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"Output ceased to be empty before finalization: {output}")
        output.rmdir()
    os.replace(staging, output)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--allow-empty-output-dir",
        action="store_true",
        help="Allow replacing an already-existing but empty output directory.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    staging, output = generate_dataset(
        args.input_dir,
        args.output_dir,
        seed=args.seed,
        workers=args.workers,
        max_pairs=args.max_pairs,
        allow_empty_output_dir=args.allow_empty_output_dir,
    )
    # This validation set is deliberately small enough to prove the absence of
    # synthetic radiometric changes by reconstructing every output pair.
    verification = verify_dataset(staging, recompute_all_pixels=True)
    finalize_dataset(
        staging,
        output,
        verification,
        allow_empty_output_dir=args.allow_empty_output_dir,
    )
    print(
        json.dumps(
            {"output": str(output), "verification": verification},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
