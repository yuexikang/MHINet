"""P4 TINY-S/TINY-8 learnability checks on 32 fixed diagnostic samples.

The diagnostic deliberately differs from formal training in one documented
way: H0 is constructed from the exact target homography plus a small,
deterministic corner residual that is guaranteed to lie inside the active
local-search range.  Raw image pairs and frozen LoMa/DeDoDe pyramid features
come from the real training manifest.  The registered memorization diagnostic
uses 32 controlled H0 residual conditions for one exact pair; the stricter
``distinct_pairs`` mode uses 32 unique pairs.  No validation or test item is
used.
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
from typing import Any, Iterable, Mapping

import torch

from .checkpointing import save_checkpoint
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
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
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

    def __init__(self, parameters: list[torch.nn.Parameter]) -> None:
        self.parameters = parameters
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
    iterator = model.iterator
    was_training = iterator.training
    iterator.eval()
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
            }
        )
        del pyramid, H_gt, H0, outputs, metrics
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    stacked = torch.stack(trajectories)
    final = stacked[:, -1]
    result = {
        "trajectory_mean_mace_px": stacked.mean(dim=0).tolist(),
        "trajectory_median_mace_px": stacked.median(dim=0).values.tolist(),
        "H0_mace_px": _statistics(stacked[:, 0]),
        "H_final_mace_px": _statistics(final),
        "failed_pairs": failed_pairs,
        "failure_rate": failed_pairs / len(cache),
        "rejected_updates": rejected_updates,
        "rejected_update_rate": rejected_updates / (len(cache) * (stacked.shape[1] - 1)),
        "elapsed_seconds": elapsed,
        "latency_ms_per_pair_single_pass": elapsed * 1000.0 / len(cache),
        "per_pair": per_pair,
    }
    iterator.train(was_training)
    return result


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
) -> dict[str, Any]:
    device = torch.device(runtime.device)
    _fresh_new_modules(model, device, seed)
    for scale in ALL_SCALES:
        enabled = scale in active_scales
        for parameter in model.adapters[str(scale)].parameters():
            parameter.requires_grad = enabled
        for parameter in model.refinement_decoders[str(scale)].parameters():
            parameter.requires_grad = enabled
    trainable_parameters = [
        parameter
        for scale in active_scales
        for module in (model.adapters[str(scale)], model.refinement_decoders[str(scale)])
        for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3, weight_decay=0.0)
    parameter_averager = (
        None
        if weight_average_start_step is None
        else _ParameterAverager(trainable_parameters)
    )
    max_residual = 0.5 * max(SCALE_SPECS[scale].max_delta_px for scale in active_scales)
    H0_values: list[torch.Tensor] = []
    actual_residuals: list[torch.Tensor] = []
    for index, sample in enumerate(cache):
        H0, actual = controlled_h0_from_ground_truth(
            sample.H_gt_norm,
            sample_index=index,
            max_abs_residual_px=max_residual,
            seed=seed,
            residual_profile=residual_profile,
        )
        H0_values.append(H0)
        actual_residuals.append(actual)

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
    model.iterator.train()
    step = 0
    final_metrics = initial
    final_raw_metrics = initial
    primary_uses_weight_average = False
    training_started = time.perf_counter()
    last_loss = float("nan")
    gradient_finite = True
    maximum_preclip_gradient_norm = 0.0
    gradient_accumulation = 4
    data_epoch = 0
    data_position = 0
    data_order = list(range(len(cache)))
    random.Random(seed + data_epoch).shuffle(data_order)
    while step < max_steps:
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
        gradient_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable_parameters
        )
        if not gradient_finite:
            raise FloatingPointError("Tiny trainable gradient became non-finite")
        preclip_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
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

        if step % eval_interval == 0 or step == max_steps:
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
            print(
                f"{name} step={step} final_mean_MACE="
                f"{final_metrics['H_final_mace_px']['mean']:.6f}px "
                f"rejected={final_metrics['rejected_updates']}",
                file=sys.stderr,
                flush=True,
            )
            if (
                final_metrics["H_final_mace_px"]["mean"] < threshold_mace_px
                and final_metrics["failed_pairs"] == 0
                and final_metrics["rejected_updates"] == 0
            ):
                break

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
    training_elapsed = time.perf_counter() - training_started
    criterion_met = (
        len(cache) == 32
        and step <= 2000
        and final_metrics["H_final_mace_px"]["mean"] < threshold_mace_px
        and final_metrics["failed_pairs"] == 0
        and final_metrics["rejected_updates"] == 0
        and gradient_finite
    )
    checkpoint_suffix = (
        "raw"
        if weight_average_start_step is None
        else f"weight-average-from-{weight_average_start_step}"
    )
    checkpoint_path = (
        runtime.output_root
        / "tiny_overfit"
        / (
            f"{name.lower()}_{sample_protocol.replace('_', '-')}_"
            f"{residual_profile}_{precision}_{checkpoint_suffix}_seed{seed}.pt"
        )
    )
    if checkpoint_path.exists() and not overwrite:
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
            "diagnostic": name,
            "active_scales": active_scales,
            "known_H0_residual_injected": True,
            "formal_training_checkpoint": False,
            "sample_count": len(cache),
            "sample_protocol": sample_protocol,
            "residual_profile": residual_profile,
            "precision": precision,
            "weight_average_start_step": weight_average_start_step,
            "weight_average_count": (
                0 if parameter_averager is None else parameter_averager.count
            ),
            "checkpoint_contains_weight_average": primary_uses_weight_average,
            "optimizer_resume_supported": not primary_uses_weight_average,
            "pair_ids": [sample.pair_id for sample in cache],
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
        "training_elapsed_seconds": training_elapsed,
        "mean_training_step_seconds_including_periodic_eval": training_elapsed
        / max(step, 1),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
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
                "one exact image pair with 32 deterministic locally observable H0 residuals"
                if sample_protocol == "one_pair_residuals"
                else "32 distinct exact image pairs with one deterministic H0 residual each"
            ),
        }
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
        )
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
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
    )
    _write_json(output, report)
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
