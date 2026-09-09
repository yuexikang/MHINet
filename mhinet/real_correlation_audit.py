"""Audit real pyramid correlation signal against exact homography offsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import RuntimePaths
from .correlation import candidate_offsets, h_guided_local_correlation
from .data import HomographyPairDataset
from .geometry import (
    normalized_homography_to_pixel,
    pixel_grid,
    safe_project_points,
)
from .model import build_model
from .modules import PyramidAdapter, SCALE_SPECS
from .tiny_overfit import controlled_h0_from_ground_truth


RADII = {8: 4, 4: 4, 2: 3, 1: 2}


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float()
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.9).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _bf16_signal_diagnostics(
    selected_scores: torch.Tensor,
    selected_valid: torch.Tensor,
    nearest_indices: torch.Tensor,
    *,
    center_index: int,
) -> dict[str, Any]:
    """Measure information lost if FP32 correlation reaches a BF16 CNN input.

    This is a round-trip diagnostic only.  It does not change the production
    correlation or decoder precision policy.
    """

    if selected_scores.ndim != 2 or selected_valid.shape != selected_scores.shape:
        raise ValueError("selected scores/valid mask must have matching NxK shapes")
    if nearest_indices.shape != selected_scores.shape[:1]:
        raise ValueError("nearest_indices must have shape N")
    if not 0 <= int(center_index) < selected_scores.shape[1]:
        raise ValueError("center_index is outside the candidate dimension")
    if not bool(selected_valid.any(dim=1).all()):
        raise ValueError("every selected query must have at least one valid candidate")

    quantized = selected_scores.to(torch.bfloat16).float()
    gather_index = nearest_indices[:, None]
    nearest_raw = selected_scores.gather(1, gather_index).squeeze(1)
    nearest_quantized = quantized.gather(1, gather_index).squeeze(1)
    center_raw = selected_scores[:, center_index]
    center_quantized = quantized[:, center_index]
    raw_margin = nearest_raw - center_raw
    quantized_margin = nearest_quantized - center_quantized
    raw_positive = raw_margin > 0

    raw_masked = selected_scores.masked_fill(~selected_valid, float("-inf"))
    quantized_masked = quantized.masked_fill(~selected_valid, float("-inf"))
    raw_argmax = raw_masked.argmax(dim=1)
    quantized_argmax = quantized_masked.argmax(dim=1)
    quantized_rank = (
        (quantized_masked > nearest_quantized[:, None]).sum(dim=1) + 1
    )
    valid_scores = selected_scores[selected_valid]
    valid_quantized = quantized[selected_valid]

    retained_positive = (
        None
        if not bool(raw_positive.any())
        else float((quantized_margin[raw_positive] > 0).float().mean().item())
    )
    return {
        "emulation": "FP32 correlation -> BF16 -> FP32 round-trip",
        "valid_score_abs_quantization_error": _stats(
            (valid_quantized - valid_scores).abs()
        ),
        "nearest_candidate_abs_quantization_error": _stats(
            (nearest_quantized - nearest_raw).abs()
        ),
        "center_candidate_abs_quantization_error": _stats(
            (center_quantized - center_raw).abs()
        ),
        "nearest_minus_center_raw": _stats(raw_margin),
        "nearest_minus_center_after_bf16": _stats(quantized_margin),
        "nearest_center_equal_after_bf16_fraction": float(
            (quantized_margin == 0).float().mean().item()
        ),
        "raw_positive_margin_fraction": float(raw_positive.float().mean().item()),
        "positive_margin_fraction_after_bf16": float(
            (quantized_margin > 0).float().mean().item()
        ),
        "raw_margin_sign_preserved_fraction": float(
            (torch.sign(quantized_margin) == torch.sign(raw_margin))
            .float()
            .mean()
            .item()
        ),
        "raw_positive_margin_retained_fraction": retained_positive,
        "argmax_candidate_preserved_fraction": float(
            (quantized_argmax == raw_argmax).float().mean().item()
        ),
        "nearest_candidate_rank1_after_bf16_fraction": float(
            (quantized_rank == 1).float().mean().item()
        ),
    }


@torch.no_grad()
def _audit_features(
    pair_features: torch.Tensor,
    H0: torch.Tensor,
    H_gt: torch.Tensor,
    *,
    radius: int,
) -> dict[str, Any]:
    source, target = pair_features[:, 0], pair_features[:, 1]
    correlation, candidate_valid = h_guided_local_correlation(
        source,
        target,
        H0,
        radius=radius,
        activation_checkpoint=False,
    )
    batch, _channels, height, width = source.shape
    source_points = pixel_grid(
        (height, width), device=source.device, dtype=torch.float32
    ).reshape(1, -1, 2).expand(batch, -1, -1)
    H0_pixel = normalized_homography_to_pixel(H0, (height, width))
    Hgt_pixel = normalized_homography_to_pixel(H_gt, (height, width))
    projected_h0, h0_valid, _ = safe_project_points(H0_pixel, source_points)
    projected_gt, gt_valid, _ = safe_project_points(Hgt_pixel, source_points)
    oracle_offset = projected_gt - projected_h0
    observable = (
        h0_valid
        & gt_valid
        & (projected_gt[..., 0] >= 0)
        & (projected_gt[..., 0] <= width - 1)
        & (projected_gt[..., 1] >= 0)
        & (projected_gt[..., 1] <= height - 1)
        & (oracle_offset.abs() <= radius).all(dim=-1)
    )
    offsets = candidate_offsets(radius, device=source.device)
    squared_distance = (
        oracle_offset[:, :, None, :] - offsets[None, None]
    ).square().sum(dim=-1)
    nearest = squared_distance.argmin(dim=-1)
    flat_correlation = correlation.flatten(2).transpose(1, 2)
    flat_valid = candidate_valid.flatten(2).transpose(1, 2)
    nearest_valid = flat_valid.gather(2, nearest[..., None]).squeeze(-1)
    selected = observable & nearest_valid
    if not bool(selected.any()):
        raise RuntimeError("No observable valid query for real correlation audit")
    nearest_correlation = flat_correlation.gather(2, nearest[..., None]).squeeze(-1)
    masked_scores = flat_correlation.masked_fill(~flat_valid, float("-inf"))
    argmax_index = masked_scores.argmax(dim=-1)
    argmax_offset = offsets[argmax_index]
    endpoint_error = torch.linalg.vector_norm(argmax_offset - oracle_offset, dim=-1)
    center_index = radius * (2 * radius + 1) + radius
    center_correlation = flat_correlation[..., center_index]
    valid_count = flat_valid.sum(dim=-1).clamp_min(1)
    correct_rank = (
        (masked_scores > nearest_correlation[..., None]).sum(dim=-1) + 1
    )
    selected_endpoint = endpoint_error[selected]
    selected_nearest = nearest_correlation[selected]
    selected_center = center_correlation[selected]
    selected_rank = correct_rank[selected].float()
    selected_scores = flat_correlation[selected]
    selected_valid_candidates = flat_valid[selected]
    selected_nearest_indices = nearest[selected]
    return {
        "feature_shape": list(pair_features.shape),
        "feature_dtype": str(pair_features.dtype),
        "radius": radius,
        "queries": height * width,
        "observable_queries": int(observable.sum().item()),
        "selected_queries": int(selected.sum().item()),
        "selected_fraction": float(selected.float().mean().item()),
        "oracle_offset_feature_px": _stats(
            torch.linalg.vector_norm(oracle_offset[selected], dim=-1)
        ),
        "argmax_endpoint_error_feature_px": _stats(selected_endpoint),
        "argmax_within_0_75_feature_px": float(
            (selected_endpoint <= 0.75).float().mean().item()
        ),
        "argmax_within_1_5_feature_px": float(
            (selected_endpoint <= 1.5).float().mean().item()
        ),
        "nearest_integer_candidate_top1_fraction": float(
            (selected_rank == 1).float().mean().item()
        ),
        "nearest_integer_candidate_rank": _stats(selected_rank),
        "nearest_integer_candidate_correlation": _stats(selected_nearest),
        "center_candidate_correlation": _stats(selected_center),
        "nearest_minus_center_correlation": _stats(
            selected_nearest - selected_center
        ),
        "bf16_decoder_input_roundtrip": _bf16_signal_diagnostics(
            selected_scores,
            selected_valid_candidates,
            selected_nearest_indices,
            center_index=center_index,
        ),
        "valid_candidates_per_query": _stats(valid_count[selected].float()),
    }


def run_audit(
    runtime: RuntimePaths,
    *,
    pair_index: int,
    scale: int,
    seed: int,
    residual_bound_fraction: float = 0.5,
) -> dict[str, Any]:
    if scale not in RADII:
        raise ValueError(f"scale must be one of {tuple(RADII)}, got {scale}")
    if not 0.0 < residual_bound_fraction <= 1.0:
        raise ValueError("residual_bound_fraction must be in (0, 1]")
    device = torch.device(runtime.device)
    dataset = HomographyPairDataset(
        runtime.data_root / "train/pairs.jsonl", max_pairs=pair_index + 1
    )
    sample = dataset[pair_index]
    images = sample["images"].unsqueeze(0).to(device)
    H_gt = sample["H_gt_norm"].unsqueeze(0).to(device)
    model, build = build_model(runtime)
    model.eval()
    prefix = (8, 4, 2, 1)[: (8, 4, 2, 1).index(scale) + 1]
    with torch.no_grad():
        shared = model.feature_provider(
            images, pyramid_scales=prefix, compute_stage1=False
        )
        H0_cpu, residual = controlled_h0_from_ground_truth(
            sample["H_gt_norm"],
            sample_index=0,
            max_abs_residual_px=(
                residual_bound_fraction * SCALE_SPECS[scale].max_delta_px
            ),
            seed=seed,
        )
        H0 = H0_cpu.unsqueeze(0).to(device)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        adapter = PyramidAdapter(SCALE_SPECS[scale].adapter_channels).to(device)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            adapted, adapter_valid = adapter(shared["pyramid"][scale])
        raw = torch.nn.functional.normalize(
            shared["pyramid"][scale].float(), dim=2, eps=1e-6
        )
        adapted_report = _audit_features(
            adapted.float(), H0, H_gt, radius=RADII[scale]
        )
        raw_report = _audit_features(raw, H0, H_gt, radius=RADII[scale])
    return {
        "gate": "P2_real_correlation_signal_audit",
        "status": "diagnostic",
        "pair_id": sample["pair_id"],
        "scale": scale,
        "seed": seed,
        "residual_bound_fraction": residual_bound_fraction,
        "maximum_declared_abs_residual_px": (
            residual_bound_fraction * SCALE_SPECS[scale].max_delta_px
        ),
        "controlled_corner_residual_input_px": residual.tolist(),
        "adapter_valid_points": int(adapter_valid.sum().item()),
        "random_zero_init_adapter": adapted_report,
        "raw_256d_descriptor": raw_report,
        "shared_call_counts": shared["call_counts"],
        "build": build,
        "interpretation": (
            "This measures fixed-feature local retrieval signal only. It is not a "
            "training result or validation metric."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--residual-bound-fraction",
        type=float,
        default=0.5,
        help="Fraction of the selected decoder's input-pixel update bound in (0,1]",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    report = run_audit(
        RuntimePaths.from_json(args.runtime),
        pair_index=args.pair_index,
        scale=args.scale,
        seed=args.seed,
        residual_bound_fraction=args.residual_bound_fraction,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
