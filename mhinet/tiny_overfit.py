"""P4 TINY-S/TINY-8 learnability checks on 32 fixed diagnostic samples.

The diagnostic deliberately differs from formal training in one documented
way: H0 is constructed from the exact target homography plus a small,
deterministic corner residual that is guaranteed to lie inside the active
local-search range. Raw image pairs and frozen CGMDP multi-scale matching
descriptors come from the real training manifest. The registered memorization
diagnostic uses 32 controlled H0 residual conditions for one exact pair; the
stricter ``distinct_pairs`` mode uses 32 unique pairs. No validation or test
item is used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Iterable, Mapping

import torch

from .checkpointing import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from .config import RuntimePaths, sha256_file
from .data import HomographyPairDataset, iter_manifest, parse_geo_region
from .geometry import (
    four_point_dlt,
    image_corners,
    normalized_to_pixel,
    pixel_delta_to_normalized,
    safe_project_points,
)
from .losses import sequence_corner_l1
from .metrics import homography_trajectory_metrics
from .model import MHINet, build_model
from .modules import SCALE_SPECS, build_multiscale_modules


TARGET_HW = (784, 784)
ALL_SCALES = (8, 4, 2, 1)
TINY_PROGRESS_FORMAT = "mhinet.tiny_progress"
TINY_PROGRESS_VERSION = 1


@dataclass
class CachedTinySample:
    """One exact labelled pair with frozen pyramid tensors held on CPU."""

    pair_id: str
    parent_group: str
    geo_group: str
    H_gt_norm: torch.Tensor
    pyramid: dict[int, torch.Tensor]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_experiments(value: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Parse CLI experiment names into iterator scale schedules."""

    aliases: dict[str, tuple[int, ...]] = {
        "D8": (8,),
        "D4": (4,),
        "D2": (2,),
        "D1": (1,),
        "8": (8,),
        "4": (4,),
        "2": (2,),
        "1": (1,),
        "TINY-8": ALL_SCALES,
        "EIGHT": ALL_SCALES,
    }
    tokens = [token.strip().upper() for token in value.split(",") if token.strip()]
    if not tokens or tokens == ["ALL"]:
        tokens = ["D8", "D4", "D2", "D1", "TINY-8"]
    parsed: list[tuple[str, tuple[int, ...]]] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in aliases:
            raise ValueError(
                f"Unknown tiny experiment {token!r}; use D8,D4,D2,D1,TINY-8 or all"
            )
        scales = aliases[token]
        name = "TINY-8" if scales == ALL_SCALES else f"TINY-S-D{scales[0]}"
        if name not in seen:
            parsed.append((name, scales))
            seen.add(name)
    return tuple(parsed)


def _requested_provider_prefix(
    experiments: Iterable[tuple[str, tuple[int, ...]]],
) -> tuple[int, ...]:
    requested = {scale for _name, scales in experiments for scale in scales}
    finest_index = max(ALL_SCALES.index(scale) for scale in requested)
    return ALL_SCALES[: finest_index + 1]


def _safe_tiny_dataset(runtime: RuntimePaths, sample_count: int) -> tuple[HomographyPairDataset, list[str]]:
    validation_manifest = runtime.data_root / "val/pairs.jsonl"
    validation_regions = {
        parse_geo_region(record) for record in iter_manifest(validation_manifest)
    }
    dataset = HomographyPairDataset(
        runtime.data_root / "train/pairs.jsonl",
        max_pairs=sample_count,
        exclude_geo_groups=validation_regions,
    )
    if len(dataset) != sample_count:
        raise RuntimeError(
            f"Requested {sample_count} leakage-safe tiny pairs, found {len(dataset)}"
        )
    return dataset, sorted(validation_regions)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def cache_exact_training_features(
    model: MHINet,
    runtime: RuntimePaths,
    *,
    sample_count: int,
    stored_scales: tuple[int, ...],
    provider_prefix: tuple[int, ...],
) -> tuple[list[CachedTinySample], dict[str, Any]]:
    """Extract every selected pair exactly once through shared DINO/MVT."""

    device = torch.device(runtime.device)
    dataset, validation_regions = _safe_tiny_dataset(runtime, sample_count)
    model.set_training_phase("heads")
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    cache: list[CachedTinySample] = []
    aggregate_calls = {"dino": 0, "mvt": 0, "vgg": 0, "dedode_full_pyramid": 0}
    feature_shapes: dict[int, list[int]] = {}
    feature_dtypes: dict[int, str] = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            images = sample["images"].unsqueeze(0).to(device)
            shared = model.feature_provider(
                images,
                pyramid_scales=provider_prefix,
                compute_stage1=False,
            )
            pyramid: dict[int, torch.Tensor] = {}
            for scale in stored_scales:
                # The CPU clone converts any inference-mode storage returned by
                # the frozen extractor into an ordinary tensor safe for later
                # autograd use by trainable downstream modules.
                cached = shared["pyramid"][scale].detach().to("cpu").clone()
                pyramid[scale] = cached
                feature_shapes.setdefault(scale, list(cached.shape))
                feature_dtypes.setdefault(scale, str(cached.dtype))
            for key in aggregate_calls:
                aggregate_calls[key] += int(shared["call_counts"].get(key, 0))
            cache.append(
                CachedTinySample(
                    pair_id=str(sample["pair_id"]),
                    parent_group=str(sample["parent_group"]),
                    geo_group=str(sample["geo_group"]),
                    H_gt_norm=sample["H_gt_norm"].detach().cpu().clone(),
                    pyramid=pyramid,
                )
            )
            del images, shared, pyramid
            if (index + 1) % 4 == 0:
                print(
                    f"cached {index + 1}/{len(dataset)} exact training pairs",
                    file=sys.stderr,
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    cache_bytes = sum(
        _tensor_bytes(value)
        for sample in cache
        for value in sample.pyramid.values()
    )
    report = {
        "source": "leakage-safe train split; exact single-image homography pairs",
        "manifest": str(runtime.data_root / "train/pairs.jsonl"),
        "manifest_sha256": sha256_file(runtime.data_root / "train/pairs.jsonl"),
        "sample_count": len(cache),
        "pair_ids": [sample.pair_id for sample in cache],
        "parent_groups": [sample.parent_group for sample in cache],
        "geo_groups": [sample.geo_group for sample in cache],
        "validation_geo_groups_used_only_for_exclusion": len(validation_regions),
        "test_used": False,
        "provider_prefix_computed": provider_prefix,
        "stored_scales": stored_scales,
        "feature_shapes": feature_shapes,
        "feature_dtypes": feature_dtypes,
        "cache_bytes_cpu": cache_bytes,
        "shared_call_counts_total": aggregate_calls,
        "shared_once_per_pair": all(
            aggregate_calls[name] == len(cache) for name in ("dino", "mvt")
        ),
        "elapsed_seconds": elapsed,
        "pairs_per_second": len(cache) / elapsed,
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
    }
    return cache, report


def controlled_h0_from_ground_truth(
    H_gt_norm: torch.Tensor,
    *,
    sample_index: int,
    max_abs_residual_px: float,
    seed: int,
    residual_profile: str = "translation",
    target_hw: tuple[int, int] = TARGET_HW,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a legal H0 with a deterministic, locally observable residual."""

    if H_gt_norm.shape != (3, 3):
        raise ValueError(f"H_gt_norm must be 3x3, got {tuple(H_gt_norm.shape)}")
    if max_abs_residual_px <= 0:
        raise ValueError("max_abs_residual_px must be positive")
    if residual_profile not in {"translation", "projective"}:
        raise ValueError("residual_profile must be translation or projective")
    H_gt = H_gt_norm.detach().cpu().float()
    corners = image_corners(target_hw, dtype=torch.float32)
    target_corners, valid, _ = safe_project_points(H_gt, corners)
    if not bool(valid.all()):
        raise FloatingPointError("Tiny target homography has invalid image corners")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 100_003 + int(sample_index) * 97 + 17)
    translation = torch.randn((1, 2), generator=generator)
    translation = translation / translation.abs().amax().clamp_min(1e-6)
    corner_jitter = (
        torch.randn((4, 2), generator=generator) * 0.20
        if residual_profile == "projective"
        else torch.zeros((4, 2))
    )
    pattern = translation + corner_jitter
    pattern = pattern / pattern.abs().amax().clamp_min(1e-6)
    # Use most of the declared local range.  At D8/D4 this yields roughly
    # 1.4--2 target feature pixels: large enough for integer-window correlation
    # channels to expose direction, while remaining at most half of each
    # decoder's per-update input-pixel bound.
    amplitude_fraction = 0.70 + 0.30 * ((sample_index % 7) / 6.0)
    residual_px = pattern * (float(max_abs_residual_px) * amplitude_fraction)
    H0_corners = target_corners - pixel_delta_to_normalized(residual_px, target_hw)
    H0 = four_point_dlt(corners, H0_corners).detach()

    reconstructed, reconstructed_valid, _ = safe_project_points(H0, corners)
    if not bool(reconstructed_valid.all()):
        raise FloatingPointError("Controlled tiny H0 failed corner projection")
    actual_residual_px = normalized_to_pixel(target_corners, target_hw) - normalized_to_pixel(
        reconstructed, target_hw
    )
    if float(actual_residual_px.abs().max().item()) > max_abs_residual_px + 1e-3:
        raise AssertionError("Constructed tiny residual exceeded its declared local range")
    return H0, actual_residual_px


def _statistics(values: torch.Tensor) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"mean": float("inf"), "median": float("inf"), "p90": float("inf"), "max": float("inf")}
    return {
        "mean": float(finite.mean().item()),
        "median": float(finite.median().item()),
        "p90": float(torch.quantile(finite, 0.9).item()),
        "max": float(finite.max().item()),
    }


def _history_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep interval history compact; full per-sample rows remain at endpoints."""

    return {key: value for key, value in metrics.items() if key != "per_pair"}


class _ParameterAverager:
    """Online equal-weight average used only as an explicit tiny diagnostic readout."""

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        parameter_names: list[str] | None = None,
    ) -> None:
        self.parameters = parameters
        self.parameter_names = (
            [f"parameter_{index}" for index in range(len(parameters))]
            if parameter_names is None
            else list(parameter_names)
        )
        if len(self.parameter_names) != len(self.parameters):
            raise ValueError("Parameter-average name count mismatch")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("Parameter-average names must be unique")
        self.average = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in parameters
        ]
        self.count = 0

    @torch.no_grad()
    def update(self) -> None:
        self.count += 1
        weight = 1.0 / self.count
        for average, parameter in zip(self.average, self.parameters):
            average.lerp_(parameter.detach().float(), weight)

    @torch.no_grad()
    def apply(self) -> list[torch.Tensor]:
        if self.count == 0:
            raise RuntimeError("Cannot apply an empty parameter average")
        originals = [parameter.detach().clone() for parameter in self.parameters]
        for parameter, average in zip(self.parameters, self.average):
            parameter.copy_(average.to(dtype=parameter.dtype))
        return originals

    @torch.no_grad()
    def restore(self, originals: list[torch.Tensor]) -> None:
        if len(originals) != len(self.parameters):
            raise ValueError("Parameter-average restore length mismatch")
        for parameter, original in zip(self.parameters, originals):
            parameter.copy_(original)

    def state_dict(self) -> dict[str, Any]:
        """Return a weights-only-safe, identity-bound averaging state."""

        return {
            "count": int(self.count),
            "parameter_names": list(self.parameter_names),
            "averages": [value.detach().cpu().clone() for value in self.average],
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Strictly restore FP32 running averages onto the parameter devices."""

        count = int(state.get("count", -1))
        names = state.get("parameter_names")
        averages = state.get("averages")
        if count < 0:
            raise ValueError("Parameter-average count cannot be negative")
        if names != self.parameter_names:
            raise ValueError("Parameter-average names do not match this experiment")
        if not isinstance(averages, list) or len(averages) != len(self.average):
            raise ValueError("Parameter-average tensor count mismatch")
        for index, (saved, destination) in enumerate(zip(averages, self.average)):
            if not isinstance(saved, torch.Tensor):
                raise TypeError(f"Parameter-average entry {index} is not a tensor")
            if saved.dtype != torch.float32:
                raise ValueError(f"Parameter-average entry {index} is not FP32")
            if saved.shape != destination.shape:
                raise ValueError(f"Parameter-average entry {index} shape mismatch")
            if not bool(torch.isfinite(saved).all()):
                raise FloatingPointError(
                    f"Parameter-average entry {index} contains non-finite values"
                )
            destination.copy_(saved.to(device=destination.device))
        self.count = count


def _tiny_criterion_met(
    metrics: Mapping[str, Any],
    *,
    sample_count: int,
    threshold_mace_px: float,
) -> bool:
    return bool(
        sample_count == 32
        and float(metrics["H_final_mace_px"]["mean"]) < threshold_mace_px
        and int(metrics["failed_pairs"]) == 0
        and int(metrics["rejected_updates"]) == 0
    )


def _tiny_progress_signature(
    *,
    name: str,
    active_scales: tuple[int, ...],
    seed: int,
    max_steps: int,
    eval_interval: int,
    threshold_mace_px: float,
    cache: list[CachedTinySample],
    sample_protocol: str,
    residual_profile: str,
    precision: str,
    weight_average_start_step: int | None,
    residual_bound_fraction: float,
    max_residual: float,
    resume_context: Mapping[str, Any] | None,
    condition_index_offset: int = 0,
) -> dict[str, Any]:
    """Bind a progress file to every choice that can change its trajectory."""

    signature = {
        "format": TINY_PROGRESS_FORMAT,
        "version": TINY_PROGRESS_VERSION,
        "diagnostic": name,
        "active_scales": active_scales,
        "seed": int(seed),
        "maximum_steps": int(max_steps),
        "evaluation_interval": int(eval_interval),
        "threshold_mace_px": float(threshold_mace_px),
        "sample_count": len(cache),
        "sample_protocol": sample_protocol,
        "residual_profile": residual_profile,
        "precision": precision,
        "weight_average_start_step": weight_average_start_step,
        "residual_bound_fraction": float(residual_bound_fraction),
        "maximum_declared_abs_residual_px": float(max_residual),
        "target_hw": TARGET_HW,
        "pair_ids": [sample.pair_id for sample in cache],
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_accumulation": 4,
            "gradient_clip_norm": 1.0,
        },
        "resume_context": dict(resume_context or {}),
    }
    # Keep the default signature byte-for-byte compatible with progress files
    # that were launched before the isolated-condition diagnostic existed.
    if condition_index_offset:
        signature["condition_index_offset"] = int(condition_index_offset)
    return signature


def _path_float_token(value: float) -> str:
    return format(float(value), ".6g").replace("-", "m").replace(".", "p")


def _resume_file_identity(
    path: Path,
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    digest = hash_cache.get(resolved)
    if digest is None:
        digest = sha256_file(resolved)
        hash_cache[resolved] = digest
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest,
    }


def _preflight_tiny_progress(
    path: Path,
    expected_signature: Mapping[str, Any],
) -> None:
    """Validate a restricted-load progress header before mutating live state."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Tiny progress checkpoint root must be a dictionary")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Tiny progress checkpoint uses an unsupported format")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("Tiny progress checkpoint uses an unsupported version")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Tiny progress checkpoint metadata is invalid")
    if metadata.get("checkpoint_role") != "tiny_progress":
        raise ValueError("Checkpoint is not a resumable tiny-progress checkpoint")
    if metadata.get("tiny_progress_signature") != expected_signature:
        raise RuntimeError("Tiny progress checkpoint protocol signature mismatch")
    auxiliary = payload.get("auxiliary_state")
    if not isinstance(auxiliary, Mapping):
        raise ValueError("Tiny progress checkpoint has no auxiliary state")
    if auxiliary.get("format") != TINY_PROGRESS_FORMAT:
        raise ValueError("Tiny progress auxiliary format mismatch")
    if int(auxiliary.get("version", -1)) != TINY_PROGRESS_VERSION:
        raise ValueError("Tiny progress auxiliary version mismatch")
    if auxiliary.get("signature") != expected_signature:
        raise RuntimeError("Tiny progress auxiliary signature mismatch")


@torch.no_grad()
def evaluate_tiny_training_set(
    model: MHINet,
    cache: list[CachedTinySample],
    H0_values: list[torch.Tensor],
    *,
    active_scales: tuple[int, ...],
    device: torch.device,
    cnn_autocast_enabled: bool = True,
    preloaded_pyramid: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if not cache:
        raise ValueError("Tiny endpoint cache must not be empty")
    if len(H0_values) != len(cache):
        raise ValueError("Tiny endpoint H0 count must match the cache")
    iterator = model.iterator
    was_training = iterator.training
    iterator.eval()
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        trajectories: list[torch.Tensor] = []
        per_pair: list[dict[str, Any]] = []
        rejected_updates = 0
        failed_pairs = 0
        feature_dtype = torch.bfloat16 if cnn_autocast_enabled else torch.float32
        for index, sample in enumerate(cache):
            pyramid = {
                scale: (
                    preloaded_pyramid[scale]
                    if preloaded_pyramid is not None
                    else sample.pyramid[scale].to(device=device, dtype=feature_dtype)
                )
                for scale in active_scales
            }
            H_gt = sample.H_gt_norm.unsqueeze(0).to(device)
            H0 = H0_values[index].unsqueeze(0).to(device)
            outputs = iterator(
                pyramid,
                H0,
                torch.ones((1,), dtype=torch.bool, device=device),
                active_scales=active_scales,
                cnn_autocast_enabled=cnn_autocast_enabled,
            )
            metrics = homography_trajectory_metrics(outputs, H_gt, target_hw=TARGET_HW)
            trajectory = metrics["trajectory_mace_px"][0].detach().cpu()
            corners = image_corners(
                TARGET_HW,
                normalized=True,
                device=device,
                dtype=torch.float32,
            )
            target_corners, target_valid, _ = safe_project_points(H_gt[0], corners)
            h0_corners, h0_valid, _ = safe_project_points(H0[0], corners)
            if not bool(target_valid.all() & h0_valid.all()):
                raise FloatingPointError(
                    "Tiny endpoint has invalid H0/GT corner projection"
                )
            update_corners, update_valid, _ = safe_project_points(
                outputs["H_updates_norm"][0], corners
            )
            target_corners_px = normalized_to_pixel(target_corners, TARGET_HW)
            h0_residual_px = target_corners_px - normalized_to_pixel(
                h0_corners, TARGET_HW
            )
            update_residual_px = target_corners_px.unsqueeze(0) - normalized_to_pixel(
                update_corners, TARGET_HW
            )
            update_residual_px = torch.where(
                update_valid[..., None],
                update_residual_px,
                torch.full_like(update_residual_px, float("nan")),
            )
            trajectories.append(trajectory)
            rejected = int((~outputs["update_accepted"]).sum().item())
            failed = not bool(outputs["overall_valid"][0].item())
            rejected_updates += rejected
            failed_pairs += int(failed)
            per_pair.append(
                {
                    "diagnostic_sample_index": index,
                    "pair_id": sample.pair_id,
                    "trajectory_mace_px": trajectory.tolist(),
                    "H0_mace_px": float(trajectory[0].item()),
                    "H_updates_mace_px": trajectory[1:].tolist(),
                    "H_final_mace_px": float(trajectory[-1].item()),
                    "update_accepted": outputs["update_accepted"][0].cpu().tolist(),
                    "failure_reason_codes": outputs["failure_reason_codes"][0]
                    .cpu()
                    .tolist(),
                    "decoder_output_finite": outputs["decoder_output_finite"][0]
                    .cpu()
                    .tolist(),
                    "H0_corner_residual_px": h0_residual_px.detach().cpu().tolist(),
                    "H_updates_corner_residual_px": update_residual_px.detach()
                    .cpu()
                    .tolist(),
                    "H_updates_corner_projection_valid": update_valid.detach()
                    .cpu()
                    .tolist(),
                    "delta_px": outputs["delta_px"][0].detach().cpu().tolist(),
                    "tanh_saturation_fraction": outputs[
                        "tanh_saturation_fraction"
                    ][0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "supported_query_count": outputs["supported_query_count"][0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "condition_number": outputs["condition_number"][0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "solve_info": outputs["solve_info"][0].detach().cpu().tolist(),
                    "H0_norm": outputs["H0_norm"][0].detach().cpu().tolist(),
                    "H_updates_norm": outputs["H_updates_norm"][0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "H_final_norm": outputs["H_final_norm"][0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "update_scale_schedule": list(outputs["update_scale_schedule"]),
                    "cnn_precision": outputs["cnn_precision"],
                }
            )
            del (
                pyramid,
                H_gt,
                H0,
                outputs,
                metrics,
                corners,
                target_corners,
                h0_corners,
                update_corners,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        stacked = torch.stack(trajectories)
        final = stacked[:, -1]
        return {
            "trajectory_mean_mace_px": stacked.mean(dim=0).tolist(),
            "trajectory_median_mace_px": stacked.median(dim=0).values.tolist(),
            "H0_mace_px": _statistics(stacked[:, 0]),
            "H_final_mace_px": _statistics(final),
            "failed_pairs": failed_pairs,
            "failure_rate": failed_pairs / len(cache),
            "rejected_updates": rejected_updates,
            "rejected_update_rate": rejected_updates
            / (len(cache) * (stacked.shape[1] - 1)),
            "elapsed_seconds": elapsed,
            "latency_ms_per_pair_single_pass": elapsed * 1000.0 / len(cache),
            "per_pair": per_pair,
        }
    finally:
        iterator.train(was_training)


def _fresh_new_modules(model: MHINet, device: torch.device, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    adapters, decoders = build_multiscale_modules()
    model.iterator.adapters = adapters.to(device)
    model.iterator.decoders = decoders.to(device)


def run_one_tiny_experiment(
    model: MHINet,
    cache: list[CachedTinySample],
    runtime: RuntimePaths,
    *,
    name: str,
    active_scales: tuple[int, ...],
    seed: int,
    max_steps: int,
    eval_interval: int,
    threshold_mace_px: float,
    sample_protocol: str,
    residual_profile: str,
    precision: str,
    weight_average_start_step: int | None,
    overwrite: bool,
    residual_bound_fraction: float = 0.5,
    progress_checkpoint: Path | None = None,
    resume_progress: Path | None = None,
    heartbeat_interval: int = 8,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    resume_context: Mapping[str, Any] | None = None,
    condition_index_offset: int = 0,
) -> dict[str, Any]:
    if not 0.0 < residual_bound_fraction <= 1.0:
        raise ValueError("residual_bound_fraction must be in (0, 1]")
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be positive")
    if condition_index_offset < 0:
        raise ValueError("condition_index_offset must be non-negative")
    device = torch.device(runtime.device)
    _fresh_new_modules(model, device, seed)
    for scale in ALL_SCALES:
        enabled = scale in active_scales
        for parameter in model.adapters[str(scale)].parameters():
            parameter.requires_grad = enabled
        for parameter in model.refinement_decoders[str(scale)].parameters():
            parameter.requires_grad = enabled
    trainable_named_parameters = [
        (f"{module_name}.{scale}.{parameter_name}", parameter)
        for scale in active_scales
        for module_name, module in (
            ("adapters", model.adapters[str(scale)]),
            ("refinement_decoders", model.refinement_decoders[str(scale)]),
        )
        for parameter_name, parameter in module.named_parameters()
    ]
    trainable_parameters = [parameter for _name, parameter in trainable_named_parameters]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3, weight_decay=0.0)
    parameter_averager = (
        None
        if weight_average_start_step is None
        else _ParameterAverager(
            trainable_parameters,
            [parameter_name for parameter_name, _parameter in trainable_named_parameters],
        )
    )
    max_residual = residual_bound_fraction * max(
        SCALE_SPECS[scale].max_delta_px for scale in active_scales
    )
    H0_values: list[torch.Tensor] = []
    actual_residuals: list[torch.Tensor] = []
    for index, sample in enumerate(cache):
        H0, actual = controlled_h0_from_ground_truth(
            sample.H_gt_norm,
            sample_index=condition_index_offset + index,
            max_abs_residual_px=max_residual,
            seed=seed,
            residual_profile=residual_profile,
        )
        H0_values.append(H0)
        actual_residuals.append(actual)

    progress_signature = _tiny_progress_signature(
        name=name,
        active_scales=active_scales,
        seed=seed,
        max_steps=max_steps,
        eval_interval=eval_interval,
        threshold_mace_px=threshold_mace_px,
        cache=cache,
        sample_protocol=sample_protocol,
        residual_profile=residual_profile,
        precision=precision,
        weight_average_start_step=weight_average_start_step,
        residual_bound_fraction=residual_bound_fraction,
        max_residual=max_residual,
        resume_context=resume_context,
        condition_index_offset=condition_index_offset,
    )
    resume_path = (
        None if resume_progress is None else Path(resume_progress).expanduser().resolve()
    )
    progress_path = (
        resume_path
        if progress_checkpoint is None and resume_path is not None
        else (
            None
            if progress_checkpoint is None
            else Path(progress_checkpoint).expanduser().resolve()
        )
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Tiny progress checkpoint does not exist: {resume_path}")
    if (
        resume_path is None
        and progress_path is not None
        and progress_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Refusing to overwrite tiny progress checkpoint {progress_path}; "
            "pass --overwrite or --resume-progress"
        )

    repeated_single_pair = all(sample is cache[0] for sample in cache)
    feature_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    preloaded_pyramid = (
        {
            scale: cache[0].pyramid[scale].to(
                device=device, dtype=feature_dtype
            )
            for scale in active_scales
        }
        if repeated_single_pair
        else None
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gradient_accumulation = 4
    resume_report: dict[str, Any] | None = None
    resume_summary: dict[str, Any] | None = None
    prior_training_elapsed = 0.0
    prior_evaluation_elapsed = 0.0
    prior_checkpoint_io_elapsed = 0.0
    previous_peak_allocated = 0
    previous_peak_reserved = 0
    resume_segments = 0

    if resume_path is None:
        initial = evaluate_tiny_training_set(
            model,
            cache,
            H0_values,
            active_scales=active_scales,
            device=device,
            cnn_autocast_enabled=precision == "bf16",
            preloaded_pyramid=preloaded_pyramid,
        )
        history = [
            {
                "optimizer_step": 0,
                "raw_metrics": _history_metrics(initial),
                "weight_averaged_metrics": None,
                "primary_metrics": _history_metrics(initial),
                "weight_average_count": 0,
            }
        ]
        step = 0
        final_metrics = initial
        final_raw_metrics = initial
        primary_uses_weight_average = False
        last_evaluated_step = 0
        last_loss = float("nan")
        gradient_finite = True
        maximum_preclip_gradient_norm = 0.0
        data_epoch = 0
        data_position = 0
    else:
        _preflight_tiny_progress(resume_path, progress_signature)
        resume_report = load_checkpoint(
            resume_path,
            model=model.iterator,
            optimizer=optimizer,
            map_location=device,
            strict=True,
            restore_rng=True,
        )
        metadata = resume_report.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Tiny progress checkpoint metadata is invalid")
        if metadata.get("checkpoint_role") != "tiny_progress":
            raise ValueError("Checkpoint is not a resumable tiny-progress checkpoint")
        if metadata.get("tiny_progress_signature") != progress_signature:
            raise RuntimeError("Tiny progress checkpoint protocol signature mismatch")
        auxiliary = resume_report.get("auxiliary_state")
        if not isinstance(auxiliary, Mapping):
            raise ValueError("Tiny progress checkpoint has no auxiliary state")
        if auxiliary.get("format") != TINY_PROGRESS_FORMAT:
            raise ValueError("Tiny progress auxiliary format mismatch")
        if int(auxiliary.get("version", -1)) != TINY_PROGRESS_VERSION:
            raise ValueError("Tiny progress auxiliary version mismatch")
        if auxiliary.get("signature") != progress_signature:
            raise RuntimeError("Tiny progress auxiliary signature mismatch")

        step = int(resume_report["optimizer_step"])
        if not 0 <= step <= max_steps:
            raise ValueError("Tiny progress optimizer step is out of range")
        data_progress = resume_report.get("data_progress")
        if not isinstance(data_progress, Mapping):
            raise ValueError("Tiny progress data cursor is invalid")
        if int(data_progress.get("length", -1)) != len(cache):
            raise RuntimeError("Tiny progress sample count changed")
        data_epoch = int(data_progress.get("epoch", -1))
        data_position = int(data_progress.get("position", -1))
        if data_epoch < 0 or not 0 <= data_position <= len(cache):
            raise ValueError("Tiny progress data cursor is out of range")
        if int(data_progress.get("order_seed", -1)) != seed + data_epoch:
            raise RuntimeError("Tiny progress data-order seed mismatch")

        history_value = auxiliary.get("history")
        if not isinstance(history_value, list) or not history_value:
            raise ValueError("Tiny progress history is missing")
        history = list(history_value)
        initial = auxiliary.get("initial")
        final_raw_metrics = auxiliary.get("last_raw_metrics")
        final_metrics = auxiliary.get("last_primary_metrics")
        if not all(
            isinstance(value, Mapping)
            for value in (initial, final_raw_metrics, final_metrics)
        ):
            raise ValueError("Tiny progress endpoint metrics are invalid")
        initial = dict(initial)
        final_raw_metrics = dict(final_raw_metrics)
        final_metrics = dict(final_metrics)
        last_evaluated_step = int(auxiliary.get("last_evaluated_step", -1))
        if not 0 <= last_evaluated_step <= step:
            raise ValueError("Tiny progress last-evaluated step is invalid")
        if int(history[-1].get("optimizer_step", -1)) != last_evaluated_step:
            raise ValueError("Tiny progress history endpoint is inconsistent")
        primary_uses_weight_average = bool(
            auxiliary.get("primary_uses_weight_average", False)
        )
        last_loss = float(auxiliary.get("last_loss", float("nan")))
        gradient_finite = bool(auxiliary.get("gradient_finite", False))
        if not gradient_finite:
            raise FloatingPointError("Tiny progress records non-finite gradients")
        maximum_preclip_gradient_norm = float(
            auxiliary.get("maximum_preclip_gradient_norm", 0.0)
        )
        prior_training_elapsed = float(
            auxiliary.get("training_elapsed_seconds", 0.0)
        )
        prior_evaluation_elapsed = float(
            auxiliary.get("periodic_evaluation_elapsed_seconds", 0.0)
        )
        prior_checkpoint_io_elapsed = float(
            auxiliary.get("progress_checkpoint_io_seconds", 0.0)
        )
        previous_peak_allocated = int(auxiliary.get("peak_allocated_bytes", 0))
        previous_peak_reserved = int(auxiliary.get("peak_reserved_bytes", 0))
        resume_segments = int(auxiliary.get("resume_segments", 0)) + 1

        average_state = auxiliary.get("parameter_average")
        if parameter_averager is None:
            if average_state is not None:
                raise ValueError("Unexpected parameter-average state in progress file")
        else:
            if not isinstance(average_state, Mapping):
                raise ValueError("Tiny progress parameter-average state is missing")
            parameter_averager.load_state_dict(average_state)
            expected_average_count = max(
                0, step - int(weight_average_start_step) + 1
            )
            if parameter_averager.count != expected_average_count:
                raise ValueError("Tiny progress parameter-average count is inconsistent")
        resume_summary = {
            key: value
            for key, value in resume_report.items()
            if key != "auxiliary_state"
        }

    model.iterator.train()
    data_order = list(range(len(cache)))
    random.Random(seed + data_epoch).shuffle(data_order)
    training_started = time.perf_counter()
    checkpoint_io_elapsed = 0.0
    periodic_evaluation_elapsed = prior_evaluation_elapsed
    last_progress_checkpoint: dict[str, Any] | None = None

    def current_training_elapsed() -> float:
        return (
            prior_training_elapsed
            + time.perf_counter()
            - training_started
            - checkpoint_io_elapsed
        )

    def emit_progress(event: str) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "name": name,
                "status": "running",
                "active_scales": active_scales,
                "optimizer_step": step,
                "maximum_steps": max_steps,
                "data_progress": {
                    "epoch": data_epoch,
                    "position": data_position,
                    "length": len(cache),
                },
                "last_evaluated_step": last_evaluated_step,
                "last_primary_mean_mace_px": float(
                    final_metrics["H_final_mace_px"]["mean"]
                ),
                "last_loss": last_loss,
                "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm,
                "weight_average_count": (
                    0 if parameter_averager is None else parameter_averager.count
                ),
                "training_elapsed_seconds": current_training_elapsed(),
                "progress_checkpoint": last_progress_checkpoint,
                "resumed_from": None if resume_path is None else str(resume_path),
            }
        )

    def save_progress(event: str) -> None:
        nonlocal checkpoint_io_elapsed, last_progress_checkpoint
        if progress_path is not None:
            current_peak_allocated = int(
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            )
            current_peak_reserved = int(
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else 0
            )
            checkpoint_started = time.perf_counter()
            checkpoint_report = save_checkpoint(
                progress_path,
                model=model.iterator,
                optimizer=optimizer,
                optimizer_step=step,
                microbatch_progress={"accumulated_valid": 0},
                data_progress={
                    "epoch": data_epoch,
                    "position": data_position,
                    "order_seed": seed + data_epoch,
                    "length": len(cache),
                },
                metadata={
                    "checkpoint_role": "tiny_progress",
                    "checkpoint_contains_weight_average": False,
                    "optimizer_resume_supported": True,
                    "tiny_progress_signature": progress_signature,
                },
                auxiliary_state={
                    "format": TINY_PROGRESS_FORMAT,
                    "version": TINY_PROGRESS_VERSION,
                    "signature": progress_signature,
                    "history": history,
                    "initial": initial,
                    "last_raw_metrics": final_raw_metrics,
                    "last_primary_metrics": final_metrics,
                    "last_evaluated_step": last_evaluated_step,
                    "primary_uses_weight_average": primary_uses_weight_average,
                    "last_loss": last_loss,
                    "gradient_finite": gradient_finite,
                    "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm,
                    "training_elapsed_seconds": current_training_elapsed(),
                    "periodic_evaluation_elapsed_seconds": periodic_evaluation_elapsed,
                    "progress_checkpoint_io_seconds": (
                        prior_checkpoint_io_elapsed + checkpoint_io_elapsed
                    ),
                    "peak_allocated_bytes": max(
                        previous_peak_allocated, current_peak_allocated
                    ),
                    "peak_reserved_bytes": max(
                        previous_peak_reserved, current_peak_reserved
                    ),
                    "resume_segments": resume_segments,
                    "parameter_average": (
                        None
                        if parameter_averager is None
                        else parameter_averager.state_dict()
                    ),
                },
            )
            checkpoint_report["sha256"] = sha256_file(progress_path)
            checkpoint_io_elapsed += time.perf_counter() - checkpoint_started
            last_progress_checkpoint = {
                key: checkpoint_report[key]
                for key in (
                    "path",
                    "format",
                    "version",
                    "bytes",
                    "optimizer_step",
                    "sha256",
                )
            }
        emit_progress(event)

    if resume_path is None:
        save_progress("initial_evaluation")
    else:
        last_progress_checkpoint = {
            "path": str(resume_path),
            "bytes": resume_path.stat().st_size,
            "optimizer_step": step,
            "sha256": sha256_file(resume_path),
        }
        emit_progress("resumed")

    # Cache/load/hash and the initial progress write are setup costs, not
    # optimizer-loop latency.  Preserve the I/O accounting, then start the
    # timed training segment at the exact restored optimizer boundary.
    prior_checkpoint_io_elapsed += checkpoint_io_elapsed
    checkpoint_io_elapsed = 0.0
    training_started = time.perf_counter()
    segment_start_step = step

    passed_at_current_evaluation = (
        resume_path is not None
        and last_evaluated_step == step
        and _tiny_criterion_met(
            final_metrics,
            sample_count=len(cache),
            threshold_mace_px=threshold_mace_px,
        )
    )
    while step < max_steps and not passed_at_current_evaluation:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _microbatch in range(gradient_accumulation):
            if data_position >= len(data_order):
                data_epoch += 1
                data_position = 0
                data_order = list(range(len(cache)))
                random.Random(seed + data_epoch).shuffle(data_order)
            index = data_order[data_position]
            data_position += 1
            sample = cache[index]
            pyramid = {
                scale: (
                    preloaded_pyramid[scale]
                    if preloaded_pyramid is not None
                    else sample.pyramid[scale].to(
                        device=device, dtype=feature_dtype
                    )
                )
                for scale in active_scales
            }
            H_gt = sample.H_gt_norm.unsqueeze(0).to(device)
            H0 = H0_values[index].unsqueeze(0).to(device)
            outputs = model.iterator(
                pyramid,
                H0,
                torch.ones((1,), dtype=torch.bool, device=device),
                active_scales=active_scales,
                cnn_autocast_enabled=precision == "bf16",
            )
            loss_result = sequence_corner_l1(outputs, H_gt, target_hw=TARGET_HW)
            if loss_result["skip_step"]:
                raise RuntimeError("Known-valid tiny sample unexpectedly requested skip")
            loss = loss_result["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Tiny loss became non-finite")
            (loss / gradient_accumulation).backward()
            accumulated_loss += float(loss.detach().item()) / gradient_accumulation
            del pyramid, H_gt, H0, outputs, loss_result, loss
        preclip_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        gradient_finite = True
        maximum_preclip_gradient_norm = max(
            maximum_preclip_gradient_norm,
            float(preclip_norm.detach().item()),
        )
        optimizer.step()
        step += 1
        if (
            parameter_averager is not None
            and step >= int(weight_average_start_step)
        ):
            parameter_averager.update()
        last_loss = accumulated_loss

        evaluation_due = step % eval_interval == 0 or step == max_steps
        if evaluation_due:
            raw_metrics = evaluate_tiny_training_set(
                model,
                cache,
                H0_values,
                active_scales=active_scales,
                device=device,
                cnn_autocast_enabled=precision == "bf16",
                preloaded_pyramid=preloaded_pyramid,
            )
            averaged_metrics = None
            if parameter_averager is not None and parameter_averager.count:
                originals = parameter_averager.apply()
                try:
                    averaged_metrics = evaluate_tiny_training_set(
                        model,
                        cache,
                        H0_values,
                        active_scales=active_scales,
                        device=device,
                        cnn_autocast_enabled=precision == "bf16",
                        preloaded_pyramid=preloaded_pyramid,
                    )
                finally:
                    parameter_averager.restore(originals)
            periodic_evaluation_elapsed += float(raw_metrics.get("elapsed_seconds", 0.0))
            if averaged_metrics is not None:
                periodic_evaluation_elapsed += float(
                    averaged_metrics.get("elapsed_seconds", 0.0)
                )
            final_raw_metrics = raw_metrics
            primary_uses_weight_average = averaged_metrics is not None
            final_metrics = (
                averaged_metrics if averaged_metrics is not None else raw_metrics
            )
            history.append(
                {
                    "optimizer_step": step,
                    "raw_metrics": _history_metrics(raw_metrics),
                    "weight_averaged_metrics": (
                        None
                        if averaged_metrics is None
                        else _history_metrics(averaged_metrics)
                    ),
                    "primary_metrics": _history_metrics(final_metrics),
                    "weight_average_count": (
                        0
                        if parameter_averager is None
                        else parameter_averager.count
                    ),
                }
            )
            last_evaluated_step = step
            passed_at_current_evaluation = _tiny_criterion_met(
                final_metrics,
                sample_count=len(cache),
                threshold_mace_px=threshold_mace_px,
            )
            save_progress("evaluation")
            print(
                f"{name} step={step} final_mean_MACE="
                f"{final_metrics['H_final_mace_px']['mean']:.6f}px "
                f"rejected={final_metrics['rejected_updates']}",
                file=sys.stderr,
                flush=True,
            )
            if passed_at_current_evaluation:
                break
        elif step % heartbeat_interval == 0:
            save_progress("heartbeat")
            print(
                f"{name} heartbeat step={step}/{max_steps} "
                f"last_loss={last_loss:.6f}",
                file=sys.stderr,
                flush=True,
            )

    if history[-1]["optimizer_step"] != step:
        final_raw_metrics = evaluate_tiny_training_set(
            model,
            cache,
            H0_values,
            active_scales=active_scales,
            device=device,
            cnn_autocast_enabled=precision == "bf16",
            preloaded_pyramid=preloaded_pyramid,
        )
        final_metrics = final_raw_metrics
        primary_uses_weight_average = False
        if parameter_averager is not None and parameter_averager.count:
            originals = parameter_averager.apply()
            try:
                final_metrics = evaluate_tiny_training_set(
                    model,
                    cache,
                    H0_values,
                    active_scales=active_scales,
                    device=device,
                    cnn_autocast_enabled=precision == "bf16",
                    preloaded_pyramid=preloaded_pyramid,
                )
            finally:
                parameter_averager.restore(originals)
            primary_uses_weight_average = True
        periodic_evaluation_elapsed += float(
            final_raw_metrics.get("elapsed_seconds", 0.0)
        )
        if primary_uses_weight_average:
            periodic_evaluation_elapsed += float(
                final_metrics.get("elapsed_seconds", 0.0)
            )
        history.append(
            {
                "optimizer_step": step,
                "raw_metrics": _history_metrics(final_raw_metrics),
                "weight_averaged_metrics": (
                    None
                    if parameter_averager is None or parameter_averager.count == 0
                    else _history_metrics(final_metrics)
                ),
                "primary_metrics": _history_metrics(final_metrics),
                "weight_average_count": (
                    0 if parameter_averager is None else parameter_averager.count
                ),
            }
        )
        last_evaluated_step = step
        save_progress("final_evaluation")
    training_elapsed = (
        prior_training_elapsed
        if step == segment_start_step
        else current_training_elapsed()
    )
    criterion_met = (
        step <= 2000
        and gradient_finite
        and _tiny_criterion_met(
            final_metrics,
            sample_count=len(cache),
            threshold_mace_px=threshold_mace_px,
        )
    )
    checkpoint_suffix = (
        "raw"
        if weight_average_start_step is None
        else f"weight-average-from-{weight_average_start_step}"
    )
    condition_suffix = (
        ""
        if condition_index_offset == 0
        else f"_condition-offset-{condition_index_offset}"
    )
    checkpoint_path = (
        runtime.output_root
        / "tiny_overfit"
        / (
            f"{name.lower()}_{sample_protocol.replace('_', '-')}_"
            f"{residual_profile}_{precision}_{checkpoint_suffix}_"
            f"bound-{_path_float_token(residual_bound_fraction)}_"
            f"budget-{max_steps}_seed{seed}{condition_suffix}.pt"
        )
    )
    if checkpoint_path.exists() and not overwrite and resume_path is None:
        raise FileExistsError(
            f"Refusing to overwrite tiny checkpoint {checkpoint_path}; pass --overwrite"
        )
    if primary_uses_weight_average and parameter_averager is not None:
        parameter_averager.apply()
    checkpoint = save_checkpoint(
        checkpoint_path,
        model=model.iterator,
        optimizer=optimizer,
        optimizer_step=step,
        data_progress={
            "epoch": data_epoch,
            "position": data_position,
            "order_seed": seed + data_epoch,
        },
        metadata={
            "checkpoint_role": "tiny_final_evidence",
            "diagnostic": name,
            "active_scales": active_scales,
            "known_H0_residual_injected": True,
            "formal_training_checkpoint": False,
            "sample_count": len(cache),
            "sample_protocol": sample_protocol,
            "residual_profile": residual_profile,
            "precision": precision,
            "weight_average_start_step": weight_average_start_step,
            "residual_bound_fraction": residual_bound_fraction,
            "condition_index_offset": condition_index_offset,
            "maximum_steps": max_steps,
            "evaluation_interval": eval_interval,
            "weight_average_count": (
                0 if parameter_averager is None else parameter_averager.count
            ),
            "checkpoint_contains_weight_average": primary_uses_weight_average,
            "optimizer_resume_supported": not primary_uses_weight_average,
            "pair_ids": [sample.pair_id for sample in cache],
            "tiny_progress_signature": progress_signature,
        },
    )
    checkpoint["sha256"] = sha256_file(checkpoint_path)
    residual_stack = torch.stack(actual_residuals)
    initial_mean = float(initial["H_final_mace_px"]["mean"])
    final_mean = float(final_metrics["H_final_mace_px"]["mean"])
    return {
        "name": name,
        "status": "passed" if criterion_met else "failed",
        "active_scales": active_scales,
        "updates": 2 * len(active_scales),
        "seed": seed,
        "sample_count": len(cache),
        "sample_protocol": sample_protocol,
        "residual_profile": residual_profile,
        "precision": precision,
        "primary_readout": (
            "equal_weight_parameter_average"
            if primary_uses_weight_average
            else "raw_parameters"
        ),
        "weight_averaging": {
            "enabled": weight_average_start_step is not None,
            "start_optimizer_step": weight_average_start_step,
            "averaged_snapshots": (
                0 if parameter_averager is None else parameter_averager.count
            ),
            "applied_to_primary_readout": primary_uses_weight_average,
            "training_optimizer_unchanged": True,
            "formal_training_enabled": False,
        },
        "frozen_feature_residency": (
            "one shared GPU copy for repeated exact image pair"
            if preloaded_pyramid is not None
            else "per-sample CPU-to-device transfer"
        ),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "scheduler": None,
            "batch_size": 1,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": gradient_accumulation,
            "gradient_clip_norm": 1.0,
            "optimizer_steps": step,
            "maximum_steps": max_steps,
        },
        "controlled_H0": {
            "source": "GT plus deterministic diagnostic-only corner residual",
            "profile": residual_profile,
            "formal_training_injection": False,
            "bound_fraction_of_coarsest_active_decoder": residual_bound_fraction,
            "condition_index_offset": condition_index_offset,
            "condition_indices": list(
                range(
                    condition_index_offset,
                    condition_index_offset + len(cache),
                )
            ),
            "maximum_declared_abs_residual_px": max_residual,
            "actual_abs_residual_px": _statistics(residual_stack.abs().flatten()),
        },
        "criterion": {
            "definition": (
                "32-sample primary-readout mean final MACE < threshold, zero "
                "failed pairs, zero rejected updates"
            ),
            "threshold_mace_px": threshold_mace_px,
            "met": criterion_met,
        },
        "initial": initial,
        "raw_final": final_raw_metrics,
        "final": final_metrics,
        "initial_to_final_mean_mace_improvement_px": initial_mean - final_mean,
        "history": history,
        "last_sequence_corner_l1_px": last_loss,
        "gradient_finite": gradient_finite,
        "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm,
        "resume": resume_summary,
        "progress": {
            "checkpoint": last_progress_checkpoint,
            "heartbeat_interval_optimizer_steps": heartbeat_interval,
            "last_evaluated_step": last_evaluated_step,
            "resume_segments": resume_segments,
            "raw_model_and_optimizer_boundary": True,
        },
        "training_elapsed_seconds": training_elapsed,
        "periodic_evaluation_elapsed_seconds": periodic_evaluation_elapsed,
        "progress_checkpoint_io_seconds": (
            prior_checkpoint_io_elapsed + checkpoint_io_elapsed
        ),
        "mean_training_step_seconds_including_periodic_eval": training_elapsed
        / max(step, 1),
        "peak_allocated_bytes": max(
            previous_peak_allocated,
            int(
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            ),
        ),
        "peak_reserved_bytes": max(
            previous_peak_reserved,
            int(
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else 0
            ),
        ),
        "checkpoint": checkpoint,
        "scope_note": (
            "This is a training-set learnability diagnostic with controlled H0; "
            "it is not validation accuracy and must not select a formal configuration."
        ),
    }


def run_tiny_overfit(
    runtime: RuntimePaths,
    *,
    experiments: tuple[tuple[str, tuple[int, ...]], ...],
    sample_count: int = 32,
    seed: int = 0,
    max_steps: int = 2000,
    eval_interval: int = 32,
    threshold_mace_px: float = 0.1,
    sample_protocol: str = "one_pair_residuals",
    residual_profile: str = "translation",
    precision: str = "bf16",
    weight_average_start_step: int | None = None,
    overwrite: bool = False,
    partial_output: Path | None = None,
    residual_bound_fraction: float = 0.5,
    progress_checkpoint: Path | None = None,
    resume_progress: Path | None = None,
    heartbeat_interval: int = 8,
    condition_index_offset: int = 0,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if max_steps <= 0 or max_steps > 2000:
        raise ValueError("max_steps must be in [1, 2000] for protocol-v1.2 tiny")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if threshold_mace_px <= 0:
        raise ValueError("threshold_mace_px must be positive")
    if sample_protocol not in {"one_pair_residuals", "distinct_pairs"}:
        raise ValueError(
            "sample_protocol must be one_pair_residuals or distinct_pairs"
        )
    if residual_profile not in {"translation", "projective"}:
        raise ValueError("residual_profile must be translation or projective")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    if weight_average_start_step is not None and not (
        1 <= int(weight_average_start_step) <= max_steps
    ):
        raise ValueError("weight_average_start_step must be within [1, max_steps]")
    if not 0.0 < residual_bound_fraction <= 1.0:
        raise ValueError("residual_bound_fraction must be in (0, 1]")
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be positive")
    if condition_index_offset < 0:
        raise ValueError("condition_index_offset must be non-negative")
    if (progress_checkpoint is not None or resume_progress is not None) and len(
        experiments
    ) != 1:
        raise ValueError(
            "Tiny progress checkpoint/resume currently requires exactly one experiment"
        )
    resolved_resume = (
        None
        if resume_progress is None
        else Path(resume_progress).expanduser().resolve()
    )
    resolved_progress = (
        resolved_resume
        if progress_checkpoint is None and resolved_resume is not None
        else (
            None
            if progress_checkpoint is None
            else Path(progress_checkpoint).expanduser().resolve()
        )
    )
    if resolved_resume is not None and not resolved_resume.is_file():
        raise FileNotFoundError(
            f"Tiny progress checkpoint does not exist: {resolved_resume}"
        )
    if (
        resolved_resume is None
        and resolved_progress is not None
        and resolved_progress.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Refusing to overwrite tiny progress checkpoint {resolved_progress}; "
            "pass --overwrite or --resume-progress"
        )
    device = torch.device(runtime.device)
    if device.type != "cuda":
        raise RuntimeError("Real full-resolution tiny-overfit currently requires CUDA")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    model, build_report = build_model(runtime)
    for group in build_report["training_parameters"]["groups"].values():
        group.pop("optimizer_parameter_ids", None)
    stored_scales = tuple(
        scale
        for scale in ALL_SCALES
        if any(scale in active for _name, active in experiments)
    )
    provider_prefix = _requested_provider_prefix(experiments)
    unique_pair_count = 1 if sample_protocol == "one_pair_residuals" else sample_count
    unique_cache, cache_report = cache_exact_training_features(
        model,
        runtime,
        sample_count=unique_pair_count,
        stored_scales=stored_scales,
        provider_prefix=provider_prefix,
    )
    cache = (
        [unique_cache[0] for _ in range(sample_count)]
        if sample_protocol == "one_pair_residuals"
        else unique_cache
    )
    cache_report.update(
        {
            "sample_protocol": sample_protocol,
            "unique_image_pair_count": unique_pair_count,
            "diagnostic_sample_count": len(cache),
            "diagnostic_sample_definition": (
                f"one exact image pair with {sample_count} deterministic locally "
                "observable H0 residuals"
                if sample_protocol == "one_pair_residuals"
                else f"{sample_count} distinct exact image pairs with one "
                "deterministic H0 residual each"
            ),
            "condition_index_offset": condition_index_offset,
        }
    )
    hash_cache: dict[Path, str] = {}
    resume_context = {
        "architecture_sha256": build_report["architecture_sha256"],
        "training_manifest_sha256": cache_report["manifest_sha256"],
        "runtime_config": (
            None
            if runtime.source_path is None
            else _resume_file_identity(runtime.source_path, hash_cache)
        ),
        "dino_checkpoint": _resume_file_identity(
            runtime.dino_checkpoint, hash_cache
        ),
        "selected_ghim_checkpoint": _resume_file_identity(
            runtime.selected_stage1_checkpoint, hash_cache
        ),
        "pyramid_checkpoint": _resume_file_identity(
            runtime.pyramid_checkpoint, hash_cache
        ),
    }
    if resolved_resume is not None:
        resume_name, resume_scales = experiments[0]
        resume_max_residual = residual_bound_fraction * max(
            SCALE_SPECS[scale].max_delta_px for scale in resume_scales
        )
        _preflight_tiny_progress(
            resolved_resume,
            _tiny_progress_signature(
                name=resume_name,
                active_scales=resume_scales,
                seed=seed,
                max_steps=max_steps,
                eval_interval=eval_interval,
                threshold_mace_px=threshold_mace_px,
                cache=cache,
                sample_protocol=sample_protocol,
                residual_profile=residual_profile,
                precision=precision,
                weight_average_start_step=weight_average_start_step,
                residual_bound_fraction=residual_bound_fraction,
                max_residual=resume_max_residual,
                resume_context=resume_context,
                condition_index_offset=condition_index_offset,
            ),
        )
    model.feature_provider.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report: dict[str, Any] = {
        "gate": "P4_TINY_S_TINY_8",
        "status": "running",
        "protocol": {
            "training_revision": "1.2",
            "loss": "uniform proposal-corner coordinate L1",
            "FGO": False,
            "extra_losses": False,
            "planar_head": False,
            "precision": (
                "BF16 adapter/CNN autocast, FP32 correlation/geometry"
                if precision == "bf16"
                else "FP32 adapter/CNN/correlation/geometry"
            ),
            "deterministic_algorithms": "enabled_warn_only",
            "tiny_weight_average_start_step": weight_average_start_step,
            "tiny_progress_checkpoint_interval": heartbeat_interval,
            "tiny_progress_checkpoint_role": "raw optimizer boundary",
            "condition_index_offset": condition_index_offset,
        },
        "runtime_config": str(runtime.source_path),
        "device": str(device),
        "build": build_report,
        "cache": cache_report,
        "experiments": [],
    }
    if partial_output is not None:
        _write_json(partial_output, report)
    for name, active_scales in experiments:
        print(f"starting {name} on scales {active_scales}", file=sys.stderr, flush=True)

        def update_partial(progress: dict[str, Any]) -> None:
            report["current_experiment"] = progress
            if partial_output is not None:
                _write_json(partial_output, report)

        result = run_one_tiny_experiment(
            model,
            cache,
            runtime,
            name=name,
            active_scales=active_scales,
            seed=seed,
            max_steps=max_steps,
            eval_interval=eval_interval,
            threshold_mace_px=threshold_mace_px,
            sample_protocol=sample_protocol,
            residual_profile=residual_profile,
            precision=precision,
            weight_average_start_step=weight_average_start_step,
            overwrite=overwrite,
            residual_bound_fraction=residual_bound_fraction,
            progress_checkpoint=resolved_progress,
            resume_progress=resolved_resume,
            heartbeat_interval=heartbeat_interval,
            progress_callback=update_partial,
            resume_context=resume_context,
            condition_index_offset=condition_index_offset,
        )
        report.pop("current_experiment", None)
        report["experiments"].append(result)
        if partial_output is not None:
            _write_json(partial_output, report)
    report["status"] = (
        "passed"
        if report["experiments"]
        and all(item["status"] == "passed" for item in report["experiments"])
        else "failed"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        default="all",
        help="Comma-separated D8,D4,D2,D1,TINY-8, or all",
    )
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=32)
    parser.add_argument("--threshold-mace-px", type=float, default=0.1)
    parser.add_argument(
        "--sample-protocol",
        choices=("one_pair_residuals", "distinct_pairs"),
        default="one_pair_residuals",
        help=(
            "Standard memorization tiny uses 32 H0 residual samples from one exact "
            "pair; distinct_pairs is a stricter optional stress check"
        ),
    )
    parser.add_argument(
        "--residual-profile",
        choices=("translation", "projective"),
        default="translation",
        help="Start with translations; projective adds independent 20%% corner jitter",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Adapter/decoder precision; correlation, DLT, and loss always remain FP32",
    )
    parser.add_argument(
        "--weight-average-start-step",
        type=int,
        help=(
            "Optional tiny-only equal-weight parameter average; the raw endpoint "
            "is still reported and the formal trainer never enables this"
        ),
    )
    parser.add_argument(
        "--residual-bound-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of the coarsest active decoder update bound used for "
            "controlled H0 residuals; the registered D1 diagnostic uses 1.0"
        ),
    )
    parser.add_argument(
        "--condition-index-offset",
        type=int,
        default=0,
        help=(
            "Diagnostic-only starting controlled-H0 condition index; default 0 "
            "is the registered gate and nonzero values isolate a failure mode"
        ),
    )
    parser.add_argument(
        "--progress-checkpoint",
        type=Path,
        help=(
            "Atomic raw-model/optimizer progress checkpoint written every heartbeat; "
            "currently requires exactly one experiment"
        ),
    )
    parser.add_argument(
        "--resume-progress",
        type=Path,
        help=(
            "Resume a strictly matching tiny progress checkpoint; when "
            "--progress-checkpoint is omitted, updates the same file"
        ),
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=8,
        help="Optimizer-step interval for progress checkpoints and compact heartbeats",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    progress_checkpoint = (
        None
        if args.progress_checkpoint is None
        else args.progress_checkpoint.expanduser().resolve()
    )
    resume_progress = (
        None
        if args.resume_progress is None
        else args.resume_progress.expanduser().resolve()
    )
    effective_progress = (
        resume_progress if progress_checkpoint is None else progress_checkpoint
    )
    if effective_progress == output:
        raise ValueError("Tiny JSON output and progress checkpoint must be different files")
    if output.exists() and not args.overwrite and resume_progress is None:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    report = run_tiny_overfit(
        RuntimePaths.from_json(args.runtime),
        experiments=parse_experiments(args.experiments),
        sample_count=args.sample_count,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        threshold_mace_px=args.threshold_mace_px,
        sample_protocol=args.sample_protocol,
        residual_profile=args.residual_profile,
        precision=args.precision,
        weight_average_start_step=args.weight_average_start_step,
        overwrite=args.overwrite,
        partial_output=output,
        residual_bound_fraction=args.residual_bound_fraction,
        progress_checkpoint=progress_checkpoint,
        resume_progress=resume_progress,
        heartbeat_interval=args.heartbeat_interval,
        condition_index_offset=args.condition_index_offset,
    )
    _write_json(output, report)
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
