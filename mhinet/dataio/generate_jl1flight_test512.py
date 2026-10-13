"""Repackage JL1Flight test pairs at 512x512 without added augmentation.

The native ``b_*_H_0to1.txt`` matrix maps image ``a`` to image ``b``. Both
images are resampled from 640x640 to 512x512 with the same endpoint-preserving
projective transform, and the native homography is conjugated into the output
pixel coordinate system. No extra geometric or radiometric transform is added.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm

from mhinet.dataio.generate_rgb512 import OUTPUT_SIZE, _write_png, project_points, read_source_rgb


DEFAULT_INPUT = Path("/home/disk1/Data/datasets/jl1flight/test/affine_pairs_test")
DEFAULT_OUTPUT = Path("/home/disk1/Data/datasets/Test_Data/JL1Flight")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PAIR_PATTERN = re.compile(r"^b_(?P<base>.+)_t(?P<group>[0-4])\.png$")


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    base_id: str
    transform_group: str
    image_a_path: str
    image_b_path: str
    gt_path: str


def _natural_base_key(base_id: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in base_id.split("_"))
    except ValueError as error:
        raise ValueError(f"Unexpected JL1Flight base id: {base_id}") from error


def discover_pairs(input_dir: Path) -> list[PairRecord]:
    input_dir = input_dir.resolve()
    a_images = {path.stem[2:]: path for path in input_dir.glob("a_*.png")}
    records: list[PairRecord] = []
    seen: set[str] = set()
    for image_b in input_dir.glob("b_*_t?.png"):
        match = PAIR_PATTERN.fullmatch(image_b.name)
        if match is None:
            continue
        base_id = match.group("base")
        group = f"t{match.group('group')}"
        image_a = a_images.get(base_id)
        if image_a is None:
            raise FileNotFoundError(input_dir / f"a_{base_id}.png")
        gt_path = image_b.with_name(f"{image_b.stem}_H_0to1.txt")
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        pair_id = f"jl1flight__{base_id}__{group}"
        if pair_id in seen:
            raise ValueError(f"Duplicate pair id: {pair_id}")
        seen.add(pair_id)
        records.append(
            PairRecord(
                pair_id=pair_id,
                base_id=base_id,
                transform_group=group,
                image_a_path=str(image_a.resolve()),
                image_b_path=str(image_b.resolve()),
                gt_path=str(gt_path.resolve()),
            )
        )
    records.sort(key=lambda row: (*_natural_base_key(row.base_id), int(row.transform_group[1:])))
    if not records:
        raise RuntimeError(f"No JL1Flight test pairs found in {input_dir}")
    group_counts = {
        group: sum(record.transform_group == group for record in records)
        for group in ("t0", "t1", "t2", "t3", "t4")
    }
    if len(a_images) != 351 or len(records) != 1755 or set(group_counts.values()) != {351}:
        raise ValueError(
            f"Unexpected JL1Flight inventory: A={len(a_images)}, pairs={len(records)}, "
            f"groups={group_counts}"
        )
    return records


def _endpoint_resampling(width: int, height: int) -> np.ndarray:
    if width < 2 or height < 2:
        raise ValueError(f"Image too small to resample: {(width, height)}")
    return np.array(
        [
            [(OUTPUT_SIZE - 1.0) / (width - 1.0), 0.0, 0.0],
            [0.0, (OUTPUT_SIZE - 1.0) / (height - 1.0), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _load_homography(path: str) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid homography: {path}")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(f"Singular homography: {path}")
    matrix /= matrix[2, 2]
    return matrix


def _resize_image(source: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        source,
        transform,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _resize_a_worker(payload: tuple[str, str]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    source_path, output_path = payload
    image, channels, dtype = read_source_rgb(source_path)
    height, width = image.shape[:2]
    if (width, height) != (640, 640):
        raise ValueError(f"Unexpected JL1Flight A size: {source_path} {(width, height)}")
    transform = _endpoint_resampling(width, height)
    _write_png(Path(output_path), _resize_image(image, transform))
    return {
        "source_path": source_path,
        "source_size": [width, height],
        "source_channels": channels,
        "source_dtype": dtype,
        "T_native_A_to_output_A": transform.tolist(),
    }


def _generate_pair_worker(payload: tuple[PairRecord, str, dict[str, Any]]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    record, split_root_text, a_metadata = payload
    split_root = Path(split_root_text)
    image_b, b_channels, b_dtype = read_source_rgb(record.image_b_path)
    height_b, width_b = image_b.shape[:2]
    width_a, height_a = (int(value) for value in a_metadata["source_size"])
    if (width_b, height_b) != (640, 640):
        raise ValueError(
            f"Unexpected JL1Flight B size: {record.image_b_path} {(width_b, height_b)}"
        )
    transform_a = np.asarray(a_metadata["T_native_A_to_output_A"], dtype=np.float64)
    transform_b = _endpoint_resampling(width_b, height_b)
    native_h = _load_homography(record.gt_path)
    output_h = transform_b @ native_h @ np.linalg.inv(transform_a)
    output_h /= output_h[2, 2]
    output_inverse = np.linalg.inv(output_h)
    output_inverse /= output_inverse[2, 2]

    image_b_output = _resize_image(image_b, transform_b)
    one = np.full((OUTPUT_SIZE, OUTPUT_SIZE), 255, dtype=np.uint8)
    mask_a = cv2.warpPerspective(
        one,
        output_inverse,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_b = cv2.warpPerspective(
        one,
        output_h,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    image_a_relative = f"images/a/jl1flight__{record.base_id}_A.png"
    image_b_relative = f"images/b/{record.pair_id}_B.png"
    mask_a_relative = f"masks/{record.pair_id}_A_overlap.png"
    mask_b_relative = f"masks/{record.pair_id}_B_overlap.png"
    _write_png(split_root / image_b_relative, image_b_output)
    _write_png(split_root / mask_a_relative, mask_a)
    _write_png(split_root / mask_b_relative, mask_b)

    native_corners = np.array(
        [[0.0, 0.0], [width_a - 1.0, 0.0], [width_a - 1.0, height_a - 1.0], [0.0, height_a - 1.0]],
        dtype=np.float64,
    )
    output_corners_a = project_points(transform_a, native_corners)
    native_corners_b = project_points(native_h, native_corners)
    output_corners_b = project_points(transform_b, native_corners_b)
    reprojection_error = float(
        np.linalg.norm(project_points(output_h, output_corners_a) - output_corners_b, axis=1).max()
    )
    return {
        "pair_id": record.pair_id,
        "split": "test",
        "source_dataset": "jl1flight_test",
        "source_path": record.image_a_path,
        "source_relative_path": Path(record.image_a_path).name,
        "base_id": record.base_id,
        "transform_group": record.transform_group,
        "image_A": image_a_relative,
        "image_B": image_b_relative,
        "mask_A_overlap": mask_a_relative,
        "mask_B_overlap": mask_b_relative,
        "native_image_A": record.image_a_path,
        "native_image_B": record.image_b_path,
        "native_gt_path": record.gt_path,
        "native_gt_direction": "A_to_B_0to1",
        "native_size_A": [width_a, height_a],
        "native_size_B": [width_b, height_b],
        "size_A": [OUTPUT_SIZE, OUTPUT_SIZE],
        "size_B": [OUTPUT_SIZE, OUTPUT_SIZE],
        "A_temporal_role": "a",
        "B_temporal_role": "b",
        "A_policy": "native_a_endpoint_resampling_only",
        "B_policy": "native_b_endpoint_resampling_only",
        "geometry_augmentation": "none",
        "radiometric_augmentation": "none",
        "label_provenance": "native_b_H_0to1_conjugated_into_512_pixel_coordinates",
        "source_channels_A": a_metadata["source_channels"],
        "source_channels_B": b_channels,
        "source_dtype_A": a_metadata["source_dtype"],
        "source_dtype_B": b_dtype,
        "T_native_A_to_output_A": transform_a.tolist(),
        "T_native_B_to_output_B": transform_b.tolist(),
        "H_native_A_to_B": native_h.tolist(),
        "H_A_to_B": output_h.tolist(),
        "H_B_to_A": output_inverse.tolist(),
        "corner_A": output_corners_a.tolist(),
        "corner_B": output_corners_b.tolist(),
        "overlap_fractions": [float((mask_a > 0).mean()), float((mask_b > 0).mean())],
        "max_reprojection_error": reprojection_error,
    }


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
    path, expected_color = item
    actual = _png_header(path)
    expected = (OUTPUT_SIZE, OUTPUT_SIZE, 8, expected_color)
    if actual != expected:
        raise AssertionError(f"Unexpected PNG header {path}: {actual}, expected {expected}")


def _recompute_image(native_path: str, transform: np.ndarray, output_path: Path) -> None:
    native, _, _ = read_source_rgb(native_path)
    expected = _resize_image(native, transform)
    actual = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if actual is None:
        raise OSError(output_path)
    actual = cv2.cvtColor(actual, cv2.COLOR_BGR2RGB)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"Pixel reconstruction mismatch: {output_path}")


def _recompute_masks(split_root: Path, row: dict[str, Any]) -> None:
    homography = np.asarray(row["H_A_to_B"], dtype=np.float64)
    inverse = np.asarray(row["H_B_to_A"], dtype=np.float64)
    one = np.full((OUTPUT_SIZE, OUTPUT_SIZE), 255, dtype=np.uint8)
    expected_a = cv2.warpPerspective(
        one,
        inverse,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    expected_b = cv2.warpPerspective(
        one,
        homography,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    actual_a = cv2.imread(str(split_root / row["mask_A_overlap"]), cv2.IMREAD_GRAYSCALE)
    actual_b = cv2.imread(str(split_root / row["mask_B_overlap"]), cv2.IMREAD_GRAYSCALE)
    if not np.array_equal(actual_a, expected_a) or not np.array_equal(actual_b, expected_b):
        raise AssertionError(f"Overlap-mask reconstruction mismatch: {row['pair_id']}")
    measured = [float((actual_a > 0).mean()), float((actual_b > 0).mean())]
    if not np.allclose(measured, row["overlap_fractions"], atol=0.0, rtol=0.0):
        raise AssertionError(f"Stored overlap fraction mismatch: {row['pair_id']}")


def verify_dataset(root: Path, *, recompute_all_pixels: bool = True) -> dict[str, Any]:
    root = root.resolve()
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    with (root / "test" / "pairs.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != int(summary["pair_count"]):
        raise AssertionError("Manifest and summary counts differ")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise AssertionError("Duplicate pair ids")
    groups = {group: sum(row["transform_group"] == group for row in rows) for group in ("t0", "t1", "t2", "t3", "t4")}
    if not summary["dry_run"] and groups != {group: 351 for group in groups}:
        raise AssertionError(f"Unexpected transform groups: {groups}")

    headers: set[tuple[Path, int]] = set()
    minimum_overlap = 1.0
    maximum_error = 0.0
    for row in rows:
        if row["geometry_augmentation"] != "none" or row["radiometric_augmentation"] != "none":
            raise AssertionError(f"Unexpected augmentation: {row['pair_id']}")
        transform_a = np.asarray(row["T_native_A_to_output_A"], dtype=np.float64)
        transform_b = np.asarray(row["T_native_B_to_output_B"], dtype=np.float64)
        native_h = _load_homography(row["native_gt_path"])
        output_h = np.asarray(row["H_A_to_B"], dtype=np.float64)
        expected = transform_b @ native_h @ np.linalg.inv(transform_a)
        expected /= expected[2, 2]
        if not np.allclose(output_h, expected, atol=1e-10, rtol=1e-10):
            raise AssertionError(f"GT resampling mismatch: {row['pair_id']}")
        inverse = np.asarray(row["H_B_to_A"], dtype=np.float64)
        identity = output_h @ inverse
        identity /= identity[2, 2]
        if not np.allclose(identity, np.eye(3), atol=1e-10, rtol=1e-10):
            raise AssertionError(f"GT inverse mismatch: {row['pair_id']}")
        error = float(
            np.linalg.norm(
                project_points(output_h, np.asarray(row["corner_A"], dtype=np.float64))
                - np.asarray(row["corner_B"], dtype=np.float64),
                axis=1,
            ).max()
        )
        if error > 1e-6:
            raise AssertionError(f"Corner reprojection mismatch: {row['pair_id']} {error}")
        maximum_error = max(maximum_error, error)
        minimum_overlap = min(minimum_overlap, *(float(value) for value in row["overlap_fractions"]))
        headers.update(
            {
                (root / "test" / row["image_A"], 2),
                (root / "test" / row["image_B"], 2),
                (root / "test" / row["mask_A_overlap"], 0),
                (root / "test" / row["mask_B_overlap"], 0),
            }
        )
    with ThreadPoolExecutor(max_workers=min(64, max(1, os.cpu_count() or 1))) as executor:
        list(executor.map(_verify_header, headers, chunksize=32))

    a_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        a_rows.setdefault(row["base_id"], row)
    pixel_rows = rows if recompute_all_pixels else rows[:: max(1, len(rows) // 10)][:10]
    pixel_a_rows = list(a_rows.values()) if recompute_all_pixels else list(a_rows.values())[:10]
    for row in pixel_a_rows:
        _recompute_image(
            row["native_image_A"],
            np.asarray(row["T_native_A_to_output_A"], dtype=np.float64),
            root / "test" / row["image_A"],
        )
    for row in pixel_rows:
        _recompute_image(
            row["native_image_B"],
            np.asarray(row["T_native_B_to_output_B"], dtype=np.float64),
            root / "test" / row["image_B"],
        )
        _recompute_masks(root / "test", row)
    return {
        "status": "passed",
        "pair_count": len(rows),
        "unique_A_images": len(a_rows),
        "unique_B_images": len(rows),
        "transform_group_counts": groups,
        "png_headers_checked": len(headers),
        "A_images_pixel_recomputed": len(pixel_a_rows),
        "B_images_pixel_recomputed": len(pixel_rows),
        "pair_masks_pixel_recomputed": len(pixel_rows),
        "maximum_reprojection_error_pixels": maximum_error,
        "minimum_overlap_fraction": minimum_overlap,
        "geometry_augmentation": "none",
        "radiometric_augmentation": "none",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    workers: int,
    max_pairs: int | None = None,
) -> tuple[Path, Path]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.inprogress")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite staging output: {staging}")
    all_records = discover_pairs(input_dir)
    records = all_records if max_pairs is None else all_records[:max_pairs]
    if not records:
        raise ValueError("No pairs selected")
    selected_a = {record.base_id: record.image_a_path for record in records}
    group_counts = {
        group: sum(record.transform_group == group for record in records)
        for group in ("t0", "t1", "t2", "t3", "t4")
    }
    summary: dict[str, Any] = {
        "name": output_dir.name,
        "version": "JL1Flight_native_test_512_v1",
        "status": "generating",
        "dry_run": max_pairs is not None,
        "input_dir": str(input_dir),
        "discovered_pair_count": len(all_records),
        "pair_count": len(records),
        "unique_A_images": len(selected_a),
        "transform_group_counts": group_counts,
        "configuration": {
            "output_format": "PNG",
            "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "pairing": "a_<base>.png to b_<base>_t0..t4.png",
            "native_gt": "b_<base>_tN_H_0to1.txt; direction A to B",
            "resampling": "endpoint-preserving projective resampling from native 640x640 to 512x512",
            "homography_conversion": "H_512 = T_B_native_to_512 @ H_native @ inverse(T_A_native_to_512)",
            "geometry_augmentation": "none",
            "radiometric_augmentation": "none",
            "shared_A_storage": "one output A PNG referenced by its five pair rows",
        },
    }
    staging.mkdir(parents=True)
    try:
        split_root = staging / "test"
        (split_root / "images" / "a").mkdir(parents=True)
        (split_root / "images" / "b").mkdir(parents=True)
        (split_root / "masks").mkdir()
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        a_payloads = [
            (source_path, str(split_root / "images" / "a" / f"jl1flight__{base_id}_A.png"))
            for base_id, source_path in sorted(selected_a.items(), key=lambda item: _natural_base_key(item[0]))
        ]
        a_metadata: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_resize_a_worker, a_payloads, chunksize=1)
            for item in tqdm(iterator, total=len(a_payloads), desc="Resampling A", unit="image"):
                a_metadata[item["source_path"]] = item

        payloads = [(record, str(split_root), a_metadata[record.image_a_path]) for record in records]
        with (split_root / "pairs.jsonl").open("w", encoding="utf-8") as stream:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(_generate_pair_worker, payloads, chunksize=1)
                for row in tqdm(iterator, total=len(payloads), desc="Resampling B", unit="pair"):
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary["manifest_sha256"] = _sha256(split_root / "pairs.jsonl")
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


def finalize_dataset(staging: Path, output: Path, verification: dict[str, Any]) -> None:
    summary_path = staging / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["verification"] = verification
    summary["status"] = "completed"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, output)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    staging, output = generate_dataset(
        args.input_dir,
        args.output_dir,
        workers=args.workers,
        max_pairs=args.max_pairs,
    )
    verification = verify_dataset(staging, recompute_all_pixels=True)
    finalize_dataset(staging, output, verification)
    print(json.dumps({"output": str(output), "verification": verification}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
