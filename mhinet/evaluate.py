"""Reproducible H0 or full MHINet validation/test evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import json
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


def _conditional_summary(
    values: list[float], conditional_valid: list[bool]
) -> dict[str, float | int]:
    """Summarize eligible values while retaining the all-pair denominator."""

    if len(values) != len(conditional_valid):
        raise ValueError(
            "Metric values and conditional-valid flags must have equal length"
        )
    selected = [
        value for value, is_valid in zip(values, conditional_valid) if is_valid
    ]
    summary = _finite_summary(selected)
    valid_count = sum(bool(value) for value in conditional_valid)
    total_count = len(conditional_valid)
    summary.update(
        {
            "total_pair_count": total_count,
            "conditional_valid_count": valid_count,
            "conditional_invalid_count": total_count - valid_count,
            "conditional_failure_rate": (
                0.0 if total_count == 0 else 1.0 - valid_count / total_count
            ),
        }
    )
    return summary


def _trajectory_conditional_validity(
    *,
    stage1_valid: bool,
    overall_valid: bool,
    geometry_valid: list[bool],
    update_accepted: list[bool],
) -> list[bool]:
    """Return the explicitly defined validity mask for H0 and update states.

    H0 requires a valid GHIM fit and valid projected geometry.  Every MHIR
    state stores the guarded homography, so a rejected proposal leaves a valid
    retained state and must not silently remove a hard pair from trajectory
    statistics.  Proposal rejection is reported separately.  The deployable
    final state additionally follows ``overall_valid``.
    """

    if len(geometry_valid) != len(update_accepted) + 1:
        raise ValueError("Trajectory geometry and update acceptance lengths disagree")
    validity = [bool(stage1_valid and geometry_valid[0])]
    for update_index, _accepted in enumerate(update_accepted):
        geometry_ok = bool(geometry_valid[update_index + 1])
        is_final = update_index == len(update_accepted) - 1
        state_owner_valid = overall_valid if is_final else stage1_valid
        validity.append(bool(state_owner_valid and geometry_ok))
    return validity


def _trajectory_metric_summaries(
    rows: list[dict[str, Any]],
    metric_key: str,
    trajectory_length: int,
    *,
    conditional_validity_key: str = "trajectory_conditional_valid",
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    all_finite_geometry: list[dict[str, float | int]] = []
    conditional_valid: list[dict[str, float | int]] = []
    for trajectory_index in range(trajectory_length):
        values = [row[metric_key][trajectory_index] for row in rows]
        valid = [
            row[conditional_validity_key][trajectory_index] for row in rows
        ]
        all_finite_geometry.append(_finite_summary(values))
        conditional_valid.append(_conditional_summary(values, valid))
    return all_finite_geometry, conditional_valid


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
        "grid5_geometry_valid": grid_valid,
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
    update_scale_schedule = (
        []
        if h0_only
        else [
            scale
            for scale in active_scales
            for _ in range(iterations_per_scale)
        ]
    )
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
            reason_names: list[str] = []
            supported_query_count: list[int] = []
            condition_number: list[float] = []
            solve_info: list[int] = []
            delta_corner_l2_mean_px: list[float] = []
            delta_corner_l2_max_px: list[float] = []
            tanh_saturation_fraction: list[float] = []
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
            reason_lookup = outputs["failure_reason_names"]
            reason_names = [
                str(
                    reason_lookup.get(
                        int(reason),
                        reason_lookup.get(
                            str(int(reason)), f"unknown_reason_{int(reason)}"
                        ),
                    )
                )
                for reason in reasons
            ]
            supported_query_count = (
                outputs["supported_query_count"][0].cpu().tolist()
            )
            condition_number = outputs["condition_number"][0].float().cpu().tolist()
            solve_info = outputs["solve_info"][0].cpu().tolist()
            delta_corner_l2 = torch.linalg.vector_norm(
                outputs["delta_px"][0].float(), dim=-1
            )
            delta_corner_l2_mean_px = delta_corner_l2.mean(dim=-1).cpu().tolist()
            delta_corner_l2_max_px = delta_corner_l2.amax(dim=-1).cpu().tolist()
            tanh_saturation_fraction = (
                outputs["tanh_saturation_fraction"][0].float().cpu().tolist()
            )
            observed_schedule = [
                int(value) for value in outputs["update_scale_schedule"]
            ]
            expected_schedule = update_scale_schedule
            if observed_schedule != expected_schedule:
                raise RuntimeError(
                    "Model update schedule differs from requested evaluation schedule: "
                    f"{observed_schedule} != {expected_schedule}"
                )
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
        geometry_valid = errors["geometry_valid"][0].cpu().tolist()
        grid_geometry_valid = errors["grid5_geometry_valid"][0].cpu().tolist()
        valid = bool(overall_valid[0].item()) and bool(geometry_valid[-1])
        trajectory_conditional_valid = _trajectory_conditional_validity(
            stage1_valid=bool(stage1_valid[0].item()),
            overall_valid=valid,
            geometry_valid=geometry_valid,
            update_accepted=accepted,
        )
        trajectory_grid5_conditional_valid = [
            bool(state_valid and grid_valid)
            for state_valid, grid_valid in zip(
                trajectory_conditional_valid, grid_geometry_valid
            )
        ]
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
                "trajectory_geometry_valid": geometry_valid,
                "trajectory_conditional_valid": trajectory_conditional_valid,
                "trajectory_grid5_geometry_valid": grid_geometry_valid,
                "trajectory_grid5_conditional_valid": (
                    trajectory_grid5_conditional_valid
                ),
                "H0_mace_input_px": mace_input[0],
                "H_updates_mace_input_px": mace_input[1:],
                "H_final_mace_input_px": mace_input[-1],
                "H_final_mace_native_target_px": mace_native[-1],
                "H_final_grid5_input_px": grid_input[-1],
                "H_final_grid5_native_target_px": grid_native[-1],
                "H_final_geometry_valid": bool(geometry_valid[-1]),
                "H_final_conditional_valid": valid,
                "H_final_grid5_geometry_valid": bool(grid_geometry_valid[-1]),
                "H_final_grid5_conditional_valid": bool(
                    valid and grid_geometry_valid[-1]
                ),
                "update_accepted": accepted,
                "failure_reason_codes": reasons,
                "failure_reason_names": reason_names,
                "supported_query_count": supported_query_count,
                "condition_number": condition_number,
                "solve_info": solve_info,
                "delta_corner_l2_mean_px": delta_corner_l2_mean_px,
                "delta_corner_l2_max_px": delta_corner_l2_max_px,
                "tanh_saturation_fraction": tanh_saturation_fraction,
                "window_recall": None,
                "window_recall_status": (
                    "not_applicable_h0_only"
                    if h0_only
                    else "not_computed_missing_gt_window_membership"
                ),
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
    trajectory_length = (
        len(rows[0]["trajectory_mace_input_px"])
        if rows
        else 1 + len(update_scale_schedule)
    )
    trajectory_summary, trajectory_conditional_summary = (
        _trajectory_metric_summaries(
            rows, "trajectory_mace_input_px", trajectory_length
        )
    )
    trajectory_native_summary, trajectory_native_conditional_summary = (
        _trajectory_metric_summaries(
            rows, "trajectory_mace_native_target_px", trajectory_length
        )
    )
    grid5_input_summary, grid5_input_conditional_summary = (
        _trajectory_metric_summaries(
            rows,
            "trajectory_grid5_input_px",
            trajectory_length,
            conditional_validity_key="trajectory_grid5_conditional_valid",
        )
    )
    grid5_native_summary, grid5_native_conditional_summary = (
        _trajectory_metric_summaries(
            rows,
            "trajectory_grid5_native_target_px",
            trajectory_length,
            conditional_validity_key="trajectory_grid5_conditional_valid",
        )
    )
    final_values = [row["H_final_mace_input_px"] for row in rows]
    final_valid = [bool(row["overall_valid"]) for row in rows]
    h0_values = [row["H0_mace_input_px"] for row in rows]
    h0_valid = [
        bool(row["trajectory_conditional_valid"][0]) for row in rows
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
        "active_scales": [] if h0_only else list(active_scales),
        "iterations_per_scale": 0 if h0_only else int(iterations_per_scale),
        "update_scale_schedule": update_scale_schedule,
        "trajectory": {
            "labels": (
                ["H0"]
                if h0_only
                else ["H0"]
                + [f"H{index}" for index in range(1, trajectory_length)]
            ),
            "conditional_validity_policy": {
                "H0": "stage1_valid and geometry_valid",
                "H_updates": (
                    "valid GHIM owner and guarded state geometry; rejected proposals "
                    "retain the previous legal H and remain in the trajectory"
                ),
                "reported_H_final": "overall_valid and geometry_valid",
                "rejected_updates": (
                    "reported separately; a retained prior legal H may remain a valid final output"
                ),
            },
            "mace_input_px_all_finite_geometry": trajectory_summary,
            "mace_input_px_conditional_valid": trajectory_conditional_summary,
            "mace_native_target_px_all_finite_geometry": trajectory_native_summary,
            "mace_native_target_px_conditional_valid": (
                trajectory_native_conditional_summary
            ),
            "grid5_input_px_all_finite_geometry": grid5_input_summary,
            "grid5_input_px_conditional_valid": grid5_input_conditional_summary,
            "grid5_native_target_px_all_finite_geometry": grid5_native_summary,
            "grid5_native_target_px_conditional_valid": (
                grid5_native_conditional_summary
            ),
        },
        "H0_mace_input_px_all_finite_geometry": _finite_summary(h0_values),
        "H0_mace_input_px_conditional_valid": _conditional_summary(
            h0_values, h0_valid
        ),
        "H_final_mace_input_px_all_finite_geometry": _finite_summary(final_values),
        "H_final_mace_input_px_conditional_valid": _conditional_summary(
            final_values, final_valid
        ),
        "H_final_mace_native_target_px_conditional_valid": _conditional_summary(
            [row["H_final_mace_native_target_px"] for row in rows], final_valid
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
        "diagnostic_availability": {
            "window_recall": {
                "available": False,
                "reason": (
                    "The current forward contract does not retain GT-to-window membership; "
                    "no window-recall value is inferred or fabricated."
                ),
            }
        },
    }
    return summary, rows


def _flatten_evaluation_row(
    row: dict[str, Any], labels: list[str], update_schedule: list[int]
) -> dict[str, Any]:
    """Flatten every trajectory state and existing update diagnostic for CSV."""

    trajectory_keys = {
        "mace_input_px": "trajectory_mace_input_px",
        "mace_native_target_px": "trajectory_mace_native_target_px",
        "grid5_input_px": "trajectory_grid5_input_px",
        "grid5_native_target_px": "trajectory_grid5_native_target_px",
        "geometry_valid": "trajectory_geometry_valid",
        "conditional_valid": "trajectory_conditional_valid",
        "grid5_geometry_valid": "trajectory_grid5_geometry_valid",
        "grid5_conditional_valid": "trajectory_grid5_conditional_valid",
    }
    if labels and any(
        len(row[source]) != len(labels) for source in trajectory_keys.values()
    ):
        raise ValueError("Evaluation row trajectory lengths do not match summary labels")
    update_count = max(len(labels) - 1, 0)
    update_keys = {
        "accepted": "update_accepted",
        "failure_reason_code": "failure_reason_codes",
        "failure_reason_name": "failure_reason_names",
        "supported_query_count": "supported_query_count",
        "condition_number": "condition_number",
        "solve_info": "solve_info",
        "delta_corner_l2_mean_px": "delta_corner_l2_mean_px",
        "delta_corner_l2_max_px": "delta_corner_l2_max_px",
        "tanh_saturation_fraction": "tanh_saturation_fraction",
    }
    if update_count:
        if len(update_schedule) != update_count:
            raise ValueError("Update schedule length does not match trajectory labels")
        if any(len(row[source]) != update_count for source in update_keys.values()):
            raise ValueError("Evaluation row update diagnostics have inconsistent lengths")

    flattened: dict[str, Any] = {
        "index": row["index"],
        "pair_id": row["pair_id"],
        "parent_group": row["parent_group"],
        "geo_group": row["geo_group"],
        "stage1_valid": row["stage1_valid"],
        "stage1_valid_correspondences": row["stage1_valid_correspondences"],
        "overall_valid": row["overall_valid"],
        "size_A_wh": "x".join(str(value) for value in row["size_A_wh"]),
        "size_B_wh": "x".join(str(value) for value in row["size_B_wh"]),
        "window_recall_status": row["window_recall_status"],
        "shared_dino_calls": row["shared_call_counts"].get("dino"),
        "shared_mvt_calls": row["shared_call_counts"].get("mvt"),
        "latency_ms_single_pass": row["latency_ms_single_pass"],
    }
    for trajectory_index, label in enumerate(labels):
        for output_name, source_name in trajectory_keys.items():
            flattened[f"{label}_{output_name}"] = row[source_name][trajectory_index]
        if trajectory_index == 0:
            continue
        update_index = trajectory_index - 1
        flattened[f"{label}_update_scale"] = update_schedule[update_index]
        for output_name, source_name in update_keys.items():
            flattened[f"{label}_{output_name}"] = row[source_name][update_index]

    if labels:
        final_label = labels[-1]
        for output_name in trajectory_keys:
            flattened[f"H_final_{output_name}"] = flattened[
                f"{final_label}_{output_name}"
            ]
    return flattened


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
    labels = list(summary.get("trajectory", {}).get("labels", []))
    update_schedule = list(summary.get("update_scale_schedule", []))
    flattened_rows = [
        _flatten_evaluation_row(row, labels, update_schedule) for row in rows
    ]
    scalar_fields = (
        list(flattened_rows[0])
        if flattened_rows
        else [
            "index",
            "pair_id",
            "parent_group",
            "geo_group",
            "stage1_valid",
            "stage1_valid_correspondences",
            "overall_valid",
            "window_recall_status",
            "latency_ms_single_pass",
        ]
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows(flattened_rows)
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
