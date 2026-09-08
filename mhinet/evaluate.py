"""Reproducible H0 or full MHINet validation/test evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from .checkpointing import load_checkpoint
from .config import RuntimePaths, sha256_file
from .data import HomographyPairDataset
from .geometry import (
    image_corners,
    normalized_grid,
    normalized_to_pixel,
    safe_project_points,
)
from .model import MHINet, build_model


TARGET_HW = (784, 784)


def _finite_summary(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {
            "count": len(values),
            "finite_count": 0,
            "mean": float("inf"),
            "median": float("inf"),
            "p90": float("inf"),
        }
    return {
        "count": len(values),
        "finite_count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "median": float(finite.median().item()),
        "p90": float(torch.quantile(finite, 0.9).item()),
    }


def _trajectory_errors(
    trajectory: torch.Tensor,
    H_gt_norm: torch.Tensor,
    *,
    native_target_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Compute input-784 and native target errors for one trajectory."""

    if trajectory.ndim != 4 or trajectory.shape[0] != 1:
        raise ValueError(f"Expected trajectory 1xTx3x3, got {tuple(trajectory.shape)}")
    time_count = trajectory.shape[1]
    corners = image_corners(
        TARGET_HW, device=trajectory.device, dtype=torch.float32
    ).reshape(1, 1, 4, 2)
    corner_points = corners.expand(1, time_count, -1, -1)
    predicted, predicted_valid, _ = safe_project_points(
        trajectory.reshape(time_count, 3, 3),
        corner_points.reshape(time_count, 4, 2),
    )
    target, target_valid, _ = safe_project_points(H_gt_norm.float(), corners[:, 0])
    predicted = predicted.reshape(1, time_count, 4, 2)
    predicted_valid = predicted_valid.reshape(1, time_count, 4)
    valid = predicted_valid.all(dim=-1) & target_valid.all(dim=-1, keepdim=True)

    predicted_input_px = normalized_to_pixel(predicted, TARGET_HW)
    target_input_px = normalized_to_pixel(target, TARGET_HW)
    mace_input = torch.linalg.vector_norm(
        predicted_input_px - target_input_px[:, None], dim=-1
    ).mean(dim=-1)
    predicted_native_px = normalized_to_pixel(predicted, native_target_hw)
    target_native_px = normalized_to_pixel(target, native_target_hw)
    mace_native = torch.linalg.vector_norm(
        predicted_native_px - target_native_px[:, None], dim=-1
    ).mean(dim=-1)

    grid = normalized_grid(
        (5, 5), device=trajectory.device, dtype=torch.float32
    ).reshape(1, 1, 25, 2)
    grid_points = grid.expand(1, time_count, -1, -1)
    predicted_grid, predicted_grid_valid, _ = safe_project_points(
        trajectory.reshape(time_count, 3, 3),
        grid_points.reshape(time_count, 25, 2),
    )
    target_grid, target_grid_valid, _ = safe_project_points(
        H_gt_norm.float(), grid[:, 0]
    )
    predicted_grid = predicted_grid.reshape(1, time_count, 25, 2)
    predicted_grid_valid = predicted_grid_valid.reshape(1, time_count, 25)
    grid_valid = predicted_grid_valid.all(dim=-1) & target_grid_valid.all(
        dim=-1, keepdim=True
    )
    predicted_grid_input = normalized_to_pixel(predicted_grid, TARGET_HW)
    target_grid_input = normalized_to_pixel(target_grid, TARGET_HW)
    grid_input = torch.linalg.vector_norm(
        predicted_grid_input - target_grid_input[:, None], dim=-1
    ).mean(dim=-1)
    predicted_grid_native = normalized_to_pixel(predicted_grid, native_target_hw)
    target_grid_native = normalized_to_pixel(target_grid, native_target_hw)
    grid_native = torch.linalg.vector_norm(
        predicted_grid_native - target_grid_native[:, None], dim=-1
    ).mean(dim=-1)

    inf_input = torch.full_like(mace_input, float("inf"))
    inf_native = torch.full_like(mace_native, float("inf"))
    return {
        "mace_input_px": torch.where(valid, mace_input, inf_input),
        "mace_native_target_px": torch.where(valid, mace_native, inf_native),
        "grid5_input_px": torch.where(grid_valid, grid_input, inf_input),
        "grid5_native_target_px": torch.where(grid_valid, grid_native, inf_native),
        "geometry_valid": valid,
    }


@torch.no_grad()
def evaluate_model(
    model: MHINet,
    dataset: HomographyPairDataset,
    *,
    device: torch.device,
    h0_only: bool = False,
    active_scales: tuple[int, ...] = (8, 4, 2, 1),
    iterations_per_scale: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate every requested ID, retaining failed rows in the denominator."""

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    total_rejected_updates = 0
    shared_call_violations = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        images = sample["images"].unsqueeze(0).to(device)
        H_gt = sample["H_gt_norm"].unsqueeze(0).to(device)
        pair_started = time.perf_counter()
        if h0_only:
            shared = model.feature_provider(
                images, pyramid_scales=(), compute_stage1=True
            )
            H0 = shared["H0_norm"]
            trajectory = H0[:, None]
            stage1_valid = shared["stage1_valid"]
            overall_valid = stage1_valid.clone()
            accepted: list[bool] = []
            reasons: list[int] = []
            call_counts = shared["call_counts"]
            valid_correspondences = shared["stage1_valid_correspondences"]
        else:
            outputs = model(
                images,
                active_scales=active_scales,
                iterations_per_scale=iterations_per_scale,
            )
            trajectory = torch.cat(
                (outputs["H0_norm"][:, None], outputs["H_updates_norm"]), dim=1
            )
            stage1_valid = outputs["stage1_valid"]
            overall_valid = outputs["overall_valid"]
            accepted = outputs["update_accepted"][0].cpu().tolist()
            reasons = outputs["failure_reason_codes"][0].cpu().tolist()
            call_counts = outputs["shared_call_counts"]
            valid_correspondences = outputs["stage1_valid_correspondences"]
            total_rejected_updates += int((~outputs["update_accepted"]).sum().item())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pair_elapsed_ms = (time.perf_counter() - pair_started) * 1000.0
        shared_call_violations += int(
            call_counts.get("dino") != 1 or call_counts.get("mvt") != 1
        )
        width_b, height_b = (int(value) for value in sample["size_B"].tolist())
        errors = _trajectory_errors(
            trajectory.float(), H_gt, native_target_hw=(height_b, width_b)
        )
        mace_input = errors["mace_input_px"][0].cpu().tolist()
        mace_native = errors["mace_native_target_px"][0].cpu().tolist()
        grid_input = errors["grid5_input_px"][0].cpu().tolist()
        grid_native = errors["grid5_native_target_px"][0].cpu().tolist()
        valid = bool(overall_valid[0].item()) and bool(
            errors["geometry_valid"][0, -1].item()
        )
        rows.append(
            {
                "index": index,
                "pair_id": sample["pair_id"],
                "parent_group": sample["parent_group"],
                "geo_group": sample["geo_group"],
                "size_A_wh": [int(value) for value in sample["size_A"].tolist()],
                "size_B_wh": [width_b, height_b],
                "stage1_valid": bool(stage1_valid[0].item()),
                "stage1_valid_correspondences": int(valid_correspondences[0].item()),
                "overall_valid": valid,
                "trajectory_mace_input_px": mace_input,
                "trajectory_mace_native_target_px": mace_native,
                "trajectory_grid5_input_px": grid_input,
                "trajectory_grid5_native_target_px": grid_native,
                "H0_mace_input_px": mace_input[0],
                "H_updates_mace_input_px": mace_input[1:],
                "H_final_mace_input_px": mace_input[-1],
                "H_final_mace_native_target_px": mace_native[-1],
                "update_accepted": accepted,
                "failure_reason_codes": reasons,
                "shared_call_counts": call_counts,
                "latency_ms_single_pass": pair_elapsed_ms,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"evaluated {index + 1}/{len(dataset)} pairs", flush=True)
        if not h0_only:
            del outputs
        else:
            del shared
        del images, H_gt, trajectory, errors
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    trajectory_length = len(rows[0]["trajectory_mace_input_px"]) if rows else 0
    trajectory_summary = []
    trajectory_native_summary = []
    grid5_input_summary = []
    grid5_native_summary = []
    for update_index in range(trajectory_length):
        trajectory_summary.append(
            _finite_summary(
                [row["trajectory_mace_input_px"][update_index] for row in rows]
            )
        )
        trajectory_native_summary.append(
            _finite_summary(
                [
                    row["trajectory_mace_native_target_px"][update_index]
                    for row in rows
                ]
            )
        )
        grid5_input_summary.append(
            _finite_summary(
                [row["trajectory_grid5_input_px"][update_index] for row in rows]
            )
        )
        grid5_native_summary.append(
            _finite_summary(
                [
                    row["trajectory_grid5_native_target_px"][update_index]
                    for row in rows
                ]
            )
        )
    final_values = [row["H_final_mace_input_px"] for row in rows]
    valid_final_values = [
        row["H_final_mace_input_px"]
        for row in rows
        if row["overall_valid"] and math.isfinite(row["H_final_mace_input_px"])
    ]
    valid_count = sum(row["overall_valid"] for row in rows)
    stage1_failures = sum(not row["stage1_valid"] for row in rows)
    latency_summary = _finite_summary(
        [row["latency_ms_single_pass"] for row in rows]
    )
    success = {
        str(threshold): sum(
            row["overall_valid"] and row["H_final_mace_input_px"] <= threshold
            for row in rows
        )
        / max(len(rows), 1)
        for threshold in (1, 3, 5)
    }
    summary = {
        "mode": "H0_only" if h0_only else "full_MHINet",
        "pairs": len(rows),
        "trajectory": {
            "labels": (
                ["H0"]
                if h0_only
                else ["H0"]
                + [f"H{index}" for index in range(1, trajectory_length)]
            ),
            "mace_input_px": trajectory_summary,
            "mace_native_target_px": trajectory_native_summary,
            "grid5_input_px": grid5_input_summary,
            "grid5_native_target_px": grid5_native_summary,
        },
        "H0_mace_input_px": _finite_summary(
            [row["H0_mace_input_px"] for row in rows]
        ),
        "H_final_mace_input_px_all_finite_geometry": _finite_summary(final_values),
        "H_final_mace_input_px_conditional_valid": _finite_summary(valid_final_values),
        "H_final_mace_native_target_px_conditional_valid": _finite_summary(
            [
                row["H_final_mace_native_target_px"]
                for row in rows
                if row["overall_valid"]
            ]
        ),
        "stage1_failures": stage1_failures,
        "stage1_failure_rate": stage1_failures / max(len(rows), 1),
        "valid_outputs": valid_count,
        "failure_rate": 1.0 - valid_count / max(len(rows), 1),
        "all_pair_success_at_input_px": success,
        "rejected_updates": total_rejected_updates,
        "rejected_update_rate": (
            0.0
            if h0_only
                else total_rejected_updates
                / max(len(rows) * len(active_scales) * iterations_per_scale, 1)
        ),
        "shared_call_count_violations": shared_call_violations,
        "elapsed_seconds": elapsed,
        "latency_ms_per_pair_single_pass": elapsed * 1000.0 / max(len(rows), 1),
        "latency_ms_single_pass_distribution": latency_summary,
        "latency_scope": (
            "per-pair synchronized single pass; this evaluation entry is not the "
            "20-warmup/100-pair formal latency benchmark"
        ),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
    }
    return summary, rows


def write_evaluation(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to write non-empty {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    jsonl_path = output_dir / "pair_metrics.jsonl"
    csv_path = output_dir / "pair_metrics.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    scalar_fields = [
        "index",
        "pair_id",
        "parent_group",
        "geo_group",
        "stage1_valid",
        "stage1_valid_correspondences",
        "overall_valid",
        "H0_mace_input_px",
        "H_final_mace_input_px",
        "H_final_mace_native_target_px",
        "latency_ms_single_pass",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in scalar_fields})
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "pair_jsonl": str(jsonl_path),
        "pair_jsonl_sha256": sha256_file(jsonl_path),
        "pair_csv": str(csv_path),
        "pair_csv_sha256": sha256_file(csv_path),
    }


def _resolve_manifest(runtime: RuntimePaths, split: str) -> Path:
    candidate = Path(split).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if split not in {"train", "val", "test"}:
        raise ValueError("--split must be train, val, test, or a manifest path")
    return runtime.data_root / split / "pairs.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--h0-only", action="store_true")
    parser.add_argument("--active-scales", default="8,4,2,1")
    parser.add_argument("--iterations-per-scale", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    runtime = RuntimePaths.from_json(args.runtime)
    device = torch.device(runtime.device)
    model, build_report = build_model(runtime)
    for group in build_report["training_parameters"]["groups"].values():
        group.pop("optimizer_parameter_ids", None)
    checkpoint_report = None
    if args.checkpoint is not None:
        checkpoint_report = load_checkpoint(
            args.checkpoint,
            model=model,
            map_location=device,
            restore_rng=False,
        )
    manifest = _resolve_manifest(runtime, args.split)
    dataset = HomographyPairDataset(manifest, max_pairs=args.max_pairs)
    active_scales = tuple(
        int(value) for value in args.active_scales.split(",") if value.strip()
    )
    summary, rows = evaluate_model(
        model,
        dataset,
        device=device,
        h0_only=args.h0_only,
        active_scales=active_scales,
        iterations_per_scale=args.iterations_per_scale,
    )
    summary.update(
        {
            "status": "completed",
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "runtime_config": str(runtime.source_path),
            "checkpoint": checkpoint_report,
            "build": build_report,
            "selection_warning": (
                "test must not be used to choose a configuration"
                if manifest.parent.name == "test"
                else None
            ),
        }
    )
    files = write_evaluation(
        args.output_dir, summary, rows, overwrite=args.overwrite
    )
    print(json.dumps({"summary": summary, "files": files}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
