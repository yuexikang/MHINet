"""Generate deterministic 512x512 RGB homography pairs from heterogeneous imagery.

Image A is the complete source image mapped once to 512x512. Image B is one
quadrilateral sampled from the same source, rectified once to 512x512, and
photometrically augmented. The exact A-to-B homography is composed from the
two source-to-output transforms; it is never estimated from image content.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm

from mhinet.dataio.stable_geometry import check_geometry


NOTION_SPEC_URL = "https://app.notion.com/p/75072d4563224fa39948764e72137803?pvs=204"
OUTPUT_SIZE = 512
GLOBAL_SEED = 42
DATASET_SPECS = {
    "whu": (Path("/home/disk1/Data/datasets/WHU-Building/train/image"), "*.tif"),
    "googleearth_current": (Path("/home/disk1/Data/datasets/GoogleEarth/training_data/current"), "**/*.jpg"),
    "flair_rgb": (Path("/home/disk1/Data/datasets/FLAIR-HUB/extracted"), "**/*.tif"),
}


@dataclass(frozen=True)
class SourceRecord:
    source_dataset: str
    source_path: str
    source_relative_path: str
    pair_id: str
    split: str
    pair_seed: int


def derive_seed(global_seed: int, *parts: object) -> int:
    payload = "\0".join([str(global_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _pair_id(dataset: str, relative_path: str) -> str:
    digest = hashlib.blake2b(relative_path.encode("utf-8"), digest_size=8).hexdigest()
    return f"{dataset}__{digest}"


def discover_sources() -> dict[str, list[tuple[Path, str]]]:
    discovered: dict[str, list[tuple[Path, str]]] = {}
    for dataset, (root, pattern) in DATASET_SPECS.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths = sorted(path for path in root.glob(pattern) if path.is_file())
        if not paths:
            raise RuntimeError(f"No source images discovered for {dataset}: {root}/{pattern}")
        discovered[dataset] = [(path, path.relative_to(root).as_posix()) for path in paths]
    return discovered


def _largest_remainder(total: int, sizes: dict[str, int]) -> dict[str, int]:
    exact = {key: total * size / sum(sizes.values()) for key, size in sizes.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def split_sources(
    discovered: dict[str, list[tuple[Path, str]]], seed: int
) -> dict[str, list[SourceRecord]]:
    sizes = {key: len(value) for key, value in discovered.items()}
    total = sum(sizes.values())
    val_counts = _largest_remainder(round(total * 0.1), sizes)
    test_counts = _largest_remainder(round(total * 0.1), sizes)
    splits: dict[str, list[SourceRecord]] = {"train": [], "val": [], "test": []}
    seen_ids: set[str] = set()
    for dataset, paths in discovered.items():
        shuffled = list(paths)
        random.Random(derive_seed(seed, "split", dataset)).shuffle(shuffled)
        n_val = val_counts[dataset]
        n_test = test_counts[dataset]
        n_train = len(shuffled) - n_val - n_test
        sections = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
        for split, members in sections.items():
            for path, relative in members:
                pair_id = _pair_id(dataset, relative)
                if pair_id in seen_ids:
                    raise ValueError(f"Duplicate pair id: {pair_id}")
                seen_ids.add(pair_id)
                splits[split].append(
                    SourceRecord(
                        source_dataset=dataset,
                        source_path=str(path),
                        source_relative_path=relative,
                        pair_id=pair_id,
                        split=split,
                        pair_seed=derive_seed(seed, "pair", dataset, relative),
                    )
                )
    for split in splits:
        splits[split].sort(key=lambda row: row.pair_id)
    return splits


def select_dry_run(
    splits: dict[str, list[SourceRecord]], per_source_per_split: int
) -> dict[str, list[SourceRecord]]:
    selected: dict[str, list[SourceRecord]] = {"train": [], "val": [], "test": []}
    for split, rows in splits.items():
        for dataset in DATASET_SPECS:
            matches = [row for row in rows if row.source_dataset == dataset]
            selected[split].extend(matches[:per_source_per_split])
        selected[split].sort(key=lambda row: row.pair_id)
    return selected


def normalized_dlt(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != (4, 2) or target.shape != (4, 2):
        raise ValueError("normalized_dlt expects two (4,2) arrays")

    def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = points.mean(axis=0)
        shifted = points - center
        mean_distance = np.linalg.norm(shifted, axis=1).mean()
        if mean_distance <= 1e-12:
            raise ValueError("Degenerate point configuration")
        scale = math.sqrt(2.0) / mean_distance
        matrix = np.array(
            [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        homogeneous = np.column_stack((points, np.ones(4)))
        normalized = (matrix @ homogeneous.T).T[:, :2]
        return normalized, matrix

    src, src_matrix = normalize(source)
    dst, dst_matrix = normalize(target)
    system = []
    for (x, y), (u, v) in zip(src, dst):
        system.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        system.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, right = np.linalg.svd(np.asarray(system, dtype=np.float64))
    normalized_h = right[-1].reshape(3, 3)
    matrix = np.linalg.inv(dst_matrix) @ normalized_h @ src_matrix
    if abs(matrix[2, 2]) < 1e-12:
        matrix /= np.max(np.abs(matrix))
    else:
        matrix /= matrix[2, 2]
    if not np.isfinite(matrix).all():
        raise FloatingPointError("DLT produced a non-finite homography")
    return matrix


def project_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((np.asarray(points, dtype=np.float64), np.ones(len(points))))
    projected = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise FloatingPointError("Point projected to infinity")
    return projected[:, :2] / projected[:, 2:3]


def _quad_is_valid(quad: np.ndarray, width: int, height: int) -> bool:
    quad32 = np.asarray(quad, dtype=np.float32)
    if not cv2.isContourConvex(quad32):
        return False
    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 0.15 * min(width, height):
        return False
    previous = np.roll(quad, 1, axis=0) - quad
    following = np.roll(quad, -1, axis=0) - quad
    cosine = np.sum(previous * following, axis=1) / (
        np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return bool(angles.min() >= 25.0 and angles.max() <= 155.0)


def sample_geometry(
    rng: np.random.Generator, width: int, height: int, max_attempts: int = 20000
) -> dict[str, Any]:
    quad_a = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    rectangle = np.array(
        [[0.0, 0.0], [OUTPUT_SIZE - 1.0, 0.0], [OUTPUT_SIZE - 1.0, OUTPUT_SIZE - 1.0], [0.0, OUTPUT_SIZE - 1.0]],
        dtype=np.float64,
    )
    area_a = float((width - 1) * (height - 1))
    bases = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    scale = np.array([width - 1.0, height - 1.0])
    for attempt in range(1, max_attempts + 1):
        quad_b = (bases + 0.5 * rng.random((4, 2))) * scale
        if not _quad_is_valid(quad_b, width, height):
            continue
        area_b = float(cv2.contourArea(quad_b.astype(np.float32)))
        if area_b <= 0:
            continue
        area_ratio = area_a / area_b
        if not 2.0 <= area_ratio <= 4.0:
            continue
        transform_a = normalized_dlt(quad_a, rectangle)
        transform_b = normalized_dlt(quad_b, rectangle)
        if np.linalg.cond(transform_b) > 1e10:
            continue
        homography = transform_b @ np.linalg.inv(transform_a)
        homography /= homography[2, 2]
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            continue
        inverse /= inverse[2, 2]
        accepted, stability = check_geometry(
            homography, (OUTPUT_SIZE, OUTPUT_SIZE), (OUTPUT_SIZE, OUTPUT_SIZE)
        )
        if not accepted:
            continue
        mask_b = np.full((OUTPUT_SIZE, OUTPUT_SIZE), 255, dtype=np.uint8)
        mask_a = cv2.warpPerspective(
            mask_b,
            inverse,
            (OUTPUT_SIZE, OUTPUT_SIZE),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlap_a = float((mask_a > 0).mean())
        overlap_b = 1.0
        if min(overlap_a, overlap_b) < 0.20:
            continue
        corner_a = project_points(transform_a, quad_b)
        corner_b = rectangle.copy()
        error = np.linalg.norm(project_points(homography, corner_a) - corner_b, axis=1).max()
        if not np.isfinite(error) or error > 1e-6:
            continue
        return {
            "quad_A_source": quad_a,
            "quad_B_source": quad_b,
            "T_source_to_A": transform_a,
            "T_source_to_B": transform_b,
            "H_A_to_B": homography,
            "H_B_to_A": inverse,
            "corner_A": corner_a,
            "corner_B": corner_b,
            "area_ratio_A_to_B": area_ratio,
            "equivalent_side_ratio": math.sqrt(area_ratio),
            "overlap_fractions": [overlap_a, overlap_b],
            "mask_A_overlap": mask_a,
            "mask_B_overlap": mask_b,
            "geometry_stability": stability,
            "sampling_attempts": attempt,
            "max_reprojection_error": float(error),
        }
    raise RuntimeError(f"Geometry sampling exhausted after {max_attempts} attempts")


def _clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _brightness_contrast(
    image: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    mean = image.mean(axis=(0, 1), keepdims=True)
    return _clip(mean + contrast * (image - mean) + brightness)


def _spectral_mix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return _clip(np.einsum("...c,dc->...d", image, matrix, optimize=True))


def l0_gradient_smoothing(
    image: np.ndarray, regularization: float, *, kappa: float = 2.0, beta_max: float = 1e5
) -> np.ndarray:
    """Deterministic periodic-boundary L0 gradient minimization using NumPy FFT."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("L0 smoothing expects an HxWx3 image")
    if not regularization > 0:
        raise ValueError("regularization must be positive")
    height, width = source.shape[:2]
    fy = (2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(height) / height))[:, None]
    fx = (2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(width) / width))[None, :]
    spectrum_penalty = (fy + fx)[..., None]
    source_fft = np.fft.fft2(source, axes=(0, 1))
    result = source.astype(np.float64, copy=True)
    beta = 2.0 * regularization
    while beta < beta_max:
        horizontal = np.roll(result, -1, axis=1) - result
        vertical = np.roll(result, -1, axis=0) - result
        magnitude = np.sum(horizontal * horizontal + vertical * vertical, axis=2)
        small = magnitude < regularization / beta
        horizontal[small] = 0.0
        vertical[small] = 0.0
        divergence = (
            np.roll(horizontal, 1, axis=1)
            - horizontal
            + np.roll(vertical, 1, axis=0)
            - vertical
        )
        numerator = source_fft + beta * np.fft.fft2(divergence, axes=(0, 1))
        result = np.fft.ifft2(
            numerator / (1.0 + beta * spectrum_penalty), axes=(0, 1)
        ).real
        beta *= kappa
    return _clip(result)


def _weak_augmentation(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    brightness = float(rng.uniform(-0.03, 0.03))
    contrast = float(rng.uniform(0.90, 1.10))
    result = _brightness_contrast(image, brightness, contrast)
    gamma_applied = bool(rng.random() < 0.5)
    gamma = float(rng.uniform(0.90, 1.10)) if gamma_applied else 1.0
    if gamma_applied:
        result = _clip(result**gamma)
    spectral_applied = bool(rng.random() < 0.5)
    matrix = np.eye(3, dtype=np.float64)
    if spectral_applied:
        matrix = rng.uniform(-0.02, 0.02, (3, 3))
        np.fill_diagonal(matrix, rng.uniform(0.95, 1.05, 3))
        result = _spectral_mix(result, matrix)
    return result, {
        "brightness": brightness,
        "contrast": contrast,
        "gamma_applied": gamma_applied,
        "gamma": gamma,
        "spectral_applied": spectral_applied,
        "spectral_matrix": matrix.tolist(),
    }


def _strong_augmentation(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    brightness = float(rng.uniform(-0.08, 0.08))
    contrast = float(rng.uniform(0.75, 1.30))
    gamma = float(rng.uniform(0.70, 1.40))
    result = _brightness_contrast(image, brightness, contrast)
    result = _clip(result**gamma)

    matrix = rng.uniform(-0.05, 0.05, (3, 3))
    np.fill_diagonal(matrix, rng.uniform(0.80, 1.20, 3))
    result = _spectral_mix(result, matrix)

    atmosphere_choice = str(rng.choice(["illumination", "haze", "none"], p=[0.4, 0.2, 0.4]))
    atmosphere: dict[str, Any] = {"kind": atmosphere_choice}
    if atmosphere_choice == "illumination":
        field_4x4 = rng.uniform(0.80, 1.20, (4, 4)).astype(np.float32)
        field = cv2.resize(field_4x4, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        result = _clip(result * field[..., None])
        atmosphere["field_4x4"] = field_4x4.tolist()
    elif atmosphere_choice == "haze":
        transmission = float(rng.uniform(0.75, 1.0))
        atmospheric_light = float(rng.uniform(0.85, 1.0))
        result = _clip(transmission * result + (1.0 - transmission) * atmospheric_light)
        atmosphere.update(transmission=transmission, atmospheric_light=atmospheric_light)

    texture_choice = str(
        rng.choice(["blur", "sharpen", "l0", "noise", "none"], p=[0.30, 0.15, 0.15, 0.10, 0.30])
    )
    texture: dict[str, Any] = {"kind": texture_choice}
    if texture_choice == "blur":
        sigma = float(rng.uniform(0.2, 1.2))
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)
        texture["sigma"] = sigma
    elif texture_choice == "sharpen":
        amount = float(rng.uniform(0.2, 0.7))
        sigma = float(rng.uniform(0.6, 1.2))
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)
        result = _clip(result + amount * (result - blurred))
        texture.update(amount=amount, sigma=sigma)
    elif texture_choice == "l0":
        regularization = float(rng.uniform(2e-4, 1e-3))
        result = l0_gradient_smoothing(result, regularization)
        texture["lambda"] = regularization
        texture["implementation"] = "deterministic_numpy_fft"
    elif texture_choice == "noise":
        sigma = float(rng.uniform(0.0, 0.02))
        result = _clip(result + rng.normal(0.0, sigma, result.shape).astype(np.float32))
        texture["sigma"] = sigma

    return _clip(result), {
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "spectral_matrix": matrix.tolist(),
        "atmosphere": atmosphere,
        "texture": texture,
    }


def augment_b(image: np.ndarray, seed: int) -> tuple[np.ndarray, str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    mode = "weak" if rng.random() < 0.60 else "strong"
    before_saturation = float(np.mean(np.any((image <= 0.0) | (image >= 1.0), axis=2)))
    attempts = 0
    while True:
        attempts += 1
        if mode == "weak":
            result, parameters = _weak_augmentation(image, rng)
        else:
            result, parameters = _strong_augmentation(image, rng)
        after_saturation = float(np.mean(np.any((result <= 0.0) | (result >= 1.0), axis=2)))
        if after_saturation <= 0.50 and after_saturation - before_saturation <= 0.15:
            break
        if attempts >= 16:
            raise RuntimeError("Radiometric parameter resampling exhausted due to saturation")
    parameters["resampling_attempts"] = attempts
    parameters["saturation_fraction_before"] = before_saturation
    parameters["saturation_fraction_after"] = after_saturation
    return result, mode, parameters


def read_source_rgb(path: str) -> tuple[np.ndarray, int, str]:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Failed to decode source image: {path}")
    if raw.ndim == 2:
        raw = np.repeat(raw[..., None], 3, axis=2)
    channels = int(raw.shape[2])
    if channels < 3:
        raise ValueError(f"Source has fewer than three channels: {path}")
    raw = raw[..., :3]
    if raw.dtype == np.uint8:
        rgb = raw[..., ::-1].copy()
        dtype_name = "uint8"
    elif np.issubdtype(raw.dtype, np.integer):
        maximum = float(np.iinfo(raw.dtype).max)
        rgb = np.rint(raw[..., ::-1].astype(np.float32) / maximum * 255.0).astype(np.uint8)
        dtype_name = str(raw.dtype)
    else:
        rgb = np.rint(np.clip(raw[..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
        dtype_name = str(raw.dtype)
    return rgb, channels, dtype_name


def _write_png(path: Path, rgb_or_gray: np.ndarray) -> None:
    if rgb_or_gray.ndim == 3:
        data = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2BGR)
    else:
        data = rgb_or_gray
    if not cv2.imwrite(str(path), data, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(path)


def _generate_worker(payload: tuple[SourceRecord, str]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    record, split_root_text = payload
    split_root = Path(split_root_text)
    source, source_channels, source_dtype = read_source_rgb(record.source_path)
    height, width = source.shape[:2]
    geometry_seed = derive_seed(record.pair_seed, "geometry")
    augmentation_seed = derive_seed(record.pair_seed, "radiometric")
    geometry = sample_geometry(np.random.default_rng(geometry_seed), width, height)

    image_a = cv2.warpPerspective(
        source,
        geometry["T_source_to_A"],
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    image_b_geometry = cv2.warpPerspective(
        source,
        geometry["T_source_to_B"],
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    image_b_float = image_b_geometry.astype(np.float32) / 255.0
    image_b_augmented, augmentation_mode, augmentation_parameters = augment_b(
        image_b_float, augmentation_seed
    )
    image_b = np.rint(image_b_augmented * 255.0).astype(np.uint8)

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
        "split": record.split,
        "source_dataset": record.source_dataset,
        "source_path": record.source_path,
        "source_relative_path": record.source_relative_path,
        "source_image_size": [width, height],
        "source_channels": source_channels,
        "source_dtype": source_dtype,
        "flair_channel_policy": "first_three_RGB_drop_NIR" if record.source_dataset == "flair_rgb" else None,
        "size_A": [OUTPUT_SIZE, OUTPUT_SIZE],
        "size_B": [OUTPUT_SIZE, OUTPUT_SIZE],
        "A_policy": "complete_source_image_no_radiometric_augmentation",
        "B_policy": "single_quadrilateral_then_radiometric_augmentation",
        "pair_seed": record.pair_seed,
        "geometry_seed": geometry_seed,
        "augmentation_seed": augmentation_seed,
        "augmentation_mode": augmentation_mode,
        "augmentation_parameters": augmentation_parameters,
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


def _serialize_source(record: SourceRecord) -> dict[str, Any]:
    return {
        "pair_id": record.pair_id,
        "split": record.split,
        "source_dataset": record.source_dataset,
        "source_path": record.source_path,
        "source_relative_path": record.source_relative_path,
        "pair_seed": record.pair_seed,
    }


def _configuration(seed: int) -> dict[str, Any]:
    return {
        "notion_spec_url": NOTION_SPEC_URL,
        "seed": seed,
        "output_format": "PNG",
        "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
        "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "split_policy": "source-stratified deterministic random source-image split before generation",
        "pair_policy": "A=complete source image; B=one quadrant-sampled convex crop from same image",
        "geometry": {
            "area_ratio_A_to_B": [2.0, 4.0],
            "equivalent_side_ratio": [math.sqrt(2.0), 2.0],
            "minimum_overlap_each": 0.20,
            "minimum_edge_fraction": 0.15,
            "angle_degrees": [25.0, 155.0],
            "stable_geometry": True,
            "solver": "float64 normalized DLT",
        },
        "radiometric": {
            "A": "none",
            "B_mode_probabilities": {"weak": 0.60, "strong": 0.40},
            "weak_spectral_probability": 0.50,
            "l0_implementation": "deterministic NumPy FFT; no additional dependency",
            "order": [
                "brightness_contrast",
                "gamma",
                "spectral_mixing",
                "illumination_or_haze",
                "texture_group",
                "clip",
            ],
        },
    }


def generate_dataset(
    output_dir: Path,
    *,
    seed: int,
    workers: int,
    dry_run_per_source_per_split: int | None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.inprogress")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging directory: {staging}")

    discovered = discover_sources()
    full_splits = split_sources(discovered, seed)
    splits = (
        select_dry_run(full_splits, dry_run_per_source_per_split)
        if dry_run_per_source_per_split is not None
        else full_splits
    )
    counts = {split: len(rows) for split, rows in splits.items()}
    source_counts = {
        split: {dataset: sum(row.source_dataset == dataset for row in rows) for dataset in DATASET_SPECS}
        for split, rows in splits.items()
    }
    summary: dict[str, Any] = {
        "name": output_dir.name,
        "version": "MHINet_RGB_512_v1",
        "status": "generating",
        "dry_run": dry_run_per_source_per_split is not None,
        "configuration": _configuration(seed),
        "discovered_source_counts": {key: len(value) for key, value in discovered.items()},
        "planned_pair_counts": counts,
        "planned_source_counts": source_counts,
    }

    staging.mkdir(parents=True)
    try:
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (staging / "source_split_manifest.jsonl").open("w", encoding="utf-8") as stream:
            for split in ("train", "val", "test"):
                for row in splits[split]:
                    stream.write(json.dumps(_serialize_source(row), ensure_ascii=False) + "\n")

        realized_modes = {split: {"weak": 0, "strong": 0} for split in splits}
        texture_counts = {split: {} for split in splits}
        atmosphere_counts = {split: {} for split in splits}
        for split in ("train", "val", "test"):
            split_root = staging / split
            (split_root / "images").mkdir(parents=True)
            (split_root / "masks").mkdir()
            payloads = [(row, str(split_root)) for row in splits[split]]
            with (split_root / "pairs.jsonl").open("w", encoding="utf-8") as stream:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    iterator = executor.map(_generate_worker, payloads, chunksize=1)
                    for row in tqdm(iterator, total=len(payloads), desc=f"Generating {split}", unit="pair"):
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        mode = row["augmentation_mode"]
                        realized_modes[split][mode] += 1
                        parameters = row["augmentation_parameters"]
                        if mode == "strong":
                            texture = parameters["texture"]["kind"]
                            atmosphere = parameters["atmosphere"]["kind"]
                            texture_counts[split][texture] = texture_counts[split].get(texture, 0) + 1
                            atmosphere_counts[split][atmosphere] = atmosphere_counts[split].get(atmosphere, 0) + 1
                        stream.flush()

        summary["realized_pair_counts"] = counts
        summary["realized_augmentation_modes"] = realized_modes
        summary["realized_strong_texture_choices"] = texture_counts
        summary["realized_strong_atmosphere_choices"] = atmosphere_counts
        summary["status"] = "generated_pending_verification"
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {"staging": staging, "output": output_dir, "summary": summary}
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument(
        "--dry-run-per-source-per-split",
        type=int,
        help="Generate only this many sources from each dataset in each split.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.dry_run_per_source_per_split is not None and args.dry_run_per_source_per_split < 1:
        parser.error("--dry-run-per-source-per-split must be positive")

    result = generate_dataset(
        args.output_dir,
        seed=args.seed,
        workers=args.workers,
        dry_run_per_source_per_split=args.dry_run_per_source_per_split,
    )
    from scripts.verify_rgb512_dataset import verify_dataset

    verification = verify_dataset(
        result["staging"],
        full_decode=args.dry_run_per_source_per_split is not None,
    )
    finalize_dataset(result["staging"], result["output"], verification)
    print(json.dumps({"output": str(result["output"]), "verification": verification}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
