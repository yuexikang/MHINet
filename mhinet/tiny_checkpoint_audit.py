"""Read-only endpoint audit for a serialized P4 tiny iterator checkpoint.

This command reconstructs the exact controlled training conditions, reloads
only MHIR's iterator weights and records per-condition H/delta/geometry
diagnostics.  It never restores an optimizer or RNG state and cannot mark the
P4 gate as passed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import random
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import torch

from .checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION, load_checkpoint
from .config import RuntimePaths, sha256_file
from .model import build_model
from .modules import SCALE_SPECS
from .tiny_overfit import (
    TARGET_HW,
    _requested_provider_prefix,
    _write_json,
    cache_exact_training_features,
    controlled_h0_from_ground_truth,
    evaluate_tiny_training_set,
    parse_experiments,
)


_ACCEPTED_ROLES = {"tiny_progress", "tiny_final_evidence"}
_CANONICAL_EXPERIMENTS = {
    "TINY-S-D8": (8,),
    "TINY-S-D4": (4,),
    "TINY-S-D2": (2,),
    "TINY-S-D1": (1,),
    "TINY-8": (8, 4, 2, 1),
}
_PROTOCOL_FIELDS = (
    "diagnostic",
    "active_scales",
    "sample_count",
    "sample_protocol",
    "residual_profile",
    "precision",
    "seed",
    "residual_bound_fraction",
    "pair_ids",
    "target_hw",
    "maximum_declared_abs_residual_px",
    "condition_index_offset",
)


def _normalized_protocol_value(name: str, value: Any) -> Any:
    if name in {"active_scales", "target_hw"}:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Checkpoint field {name} must be a sequence")
        return tuple(int(item) for item in value)
    if name == "pair_ids":
        if not isinstance(value, (list, tuple)):
            raise TypeError("Checkpoint field pair_ids must be a sequence")
        return tuple(str(item) for item in value)
    if name in {"sample_count", "seed", "condition_index_offset"}:
        return int(value)
    if name in {"residual_bound_fraction", "maximum_declared_abs_residual_px"}:
        return float(value)
    if name in {"diagnostic", "sample_protocol", "residual_profile", "precision"}:
        return str(value)
    return value


def _protocol_values_equal(name: str, left: Any, right: Any) -> bool:
    left = _normalized_protocol_value(name, left)
    right = _normalized_protocol_value(name, right)
    if name in {"residual_bound_fraction", "maximum_declared_abs_residual_px"}:
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _inspect_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Validate the restricted-load header before any model/resource work."""

    if not isinstance(payload, Mapping):
        raise ValueError("Tiny checkpoint root must be a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')!r}")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')!r}")
    model_state = payload.get("model")
    progress = payload.get("progress")
    metadata = payload.get("metadata")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("Tiny checkpoint model state is missing or empty")
    if not all(isinstance(key, str) for key in model_state):
        raise TypeError("Tiny checkpoint model-state keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in model_state.values()):
        raise TypeError("Tiny checkpoint model state must contain tensors only")
    if not isinstance(progress, Mapping) or "optimizer_step" not in progress:
        raise ValueError("Tiny checkpoint progress record is invalid")
    if not isinstance(metadata, Mapping):
        raise ValueError("Tiny checkpoint metadata is invalid")

    role = metadata.get("checkpoint_role")
    legacy = role is None
    if legacy:
        if metadata.get("known_H0_residual_injected") is not True:
            raise ValueError("Unlabelled checkpoint is not a recognized legacy tiny artifact")
        if metadata.get("formal_training_checkpoint") is not False:
            raise ValueError("Formal training checkpoints cannot be audited as tiny evidence")
        role = "legacy_tiny_final_evidence"
    elif role not in _ACCEPTED_ROLES:
        raise ValueError(f"Unsupported tiny checkpoint role: {role!r}")
    if metadata.get("formal_training_checkpoint") is True:
        raise ValueError("Formal training checkpoints cannot be audited as tiny evidence")

    signature = metadata.get("tiny_progress_signature")
    if signature is not None and not isinstance(signature, Mapping):
        raise ValueError("tiny_progress_signature must be a mapping")
    if role == "tiny_progress" and not isinstance(signature, Mapping):
        raise ValueError("A tiny_progress checkpoint must carry its protocol signature")
    return {
        "role": role,
        "legacy": legacy,
        "optimizer_step": int(progress["optimizer_step"]),
        "metadata": dict(metadata),
        "signature": {} if signature is None else dict(signature),
        "model_state": model_state,
    }


def _resolve_protocol(
    header: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve signature/metadata/CLI values while failing on every conflict."""

    metadata = header["metadata"]
    signature = header["signature"]
    if not isinstance(metadata, Mapping) or not isinstance(signature, Mapping):
        raise TypeError("Checkpoint header metadata/signature must be mappings")
    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name in _PROTOCOL_FIELDS:
        candidates: list[tuple[str, Any]] = []
        if name in signature:
            candidates.append(("tiny_progress_signature", signature[name]))
        if name in metadata:
            candidates.append(("metadata", metadata[name]))
        override = overrides.get(name)
        if override is not None:
            candidates.append(("explicit_cli", override))
        if not candidates:
            continue
        first_source, first_value = candidates[0]
        for source, value in candidates[1:]:
            if not _protocol_values_equal(name, first_value, value):
                raise RuntimeError(
                    f"Tiny checkpoint protocol conflict for {name}: "
                    f"{first_source}={first_value!r}, {source}={value!r}"
                )
        resolved[name] = _normalized_protocol_value(name, first_value)
        sources[name] = "+".join(source for source, _value in candidates)

    required = {
        "diagnostic",
        "active_scales",
        "sample_count",
        "sample_protocol",
        "residual_profile",
        "precision",
        "seed",
        "residual_bound_fraction",
        "pair_ids",
    }
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(
            "Tiny checkpoint protocol is incomplete; provide explicit legacy "
            "arguments for: " + ", ".join(missing)
        )

    expected_name = str(resolved["diagnostic"])
    expected_scales = _CANONICAL_EXPERIMENTS.get(expected_name)
    if expected_scales is None:
        raise ValueError(f"Non-canonical tiny diagnostic name: {resolved['diagnostic']!r}")
    if expected_scales != resolved["active_scales"]:
        raise ValueError("Tiny diagnostic name and active_scales disagree")
    if resolved["sample_count"] <= 0:
        raise ValueError("Tiny sample_count must be positive")
    if len(resolved["pair_ids"]) != resolved["sample_count"]:
        raise ValueError("Tiny checkpoint pair_ids length does not match sample_count")
    if resolved["sample_protocol"] not in {"one_pair_residuals", "distinct_pairs"}:
        raise ValueError("Unsupported tiny sample protocol")
    if resolved["residual_profile"] not in {"translation", "projective"}:
        raise ValueError("Unsupported tiny residual profile")
    if resolved["precision"] not in {"bf16", "fp32"}:
        raise ValueError("Unsupported tiny precision")
    fraction = resolved["residual_bound_fraction"]
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Tiny residual bound fraction must be in (0, 1]")

    target_hw = resolved.get("target_hw", TARGET_HW)
    if tuple(target_hw) != TARGET_HW:
        raise ValueError(f"Tiny target_hw must be {TARGET_HW}, got {target_hw}")
    if "target_hw" not in resolved:
        resolved["target_hw"] = TARGET_HW
        sources["target_hw"] = "protocol_v1.2_default"
    if "condition_index_offset" not in resolved:
        resolved["condition_index_offset"] = 0
        sources["condition_index_offset"] = "protocol_v1.2_default"
    if resolved["condition_index_offset"] < 0:
        raise ValueError("Tiny condition_index_offset must be non-negative")
    maximum_residual = fraction * max(
        SCALE_SPECS[scale].max_delta_px for scale in expected_scales
    )
    recorded_maximum = resolved.get("maximum_declared_abs_residual_px")
    if recorded_maximum is not None and not math.isclose(
        recorded_maximum, maximum_residual, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "Recorded maximum tiny residual disagrees with active scale/bound fraction"
        )
    if recorded_maximum is None:
        sources["maximum_declared_abs_residual_px"] = "derived_from_scale_and_bound"
    resolved["maximum_declared_abs_residual_px"] = maximum_residual
    resolved["field_sources"] = sources
    resolved["legacy_supplemented_fields"] = sorted(
        name for name, source in sources.items() if "explicit_cli" in source
        and name not in signature and name not in metadata
    )
    return resolved


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes and exact storage bytes canonically."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        for token in (name, str(tensor.dtype), repr(tuple(tensor.shape))):
            encoded = token.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _actual_resume_context(
    runtime: RuntimePaths,
    *,
    architecture_sha256: str,
    training_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "architecture_sha256": architecture_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "runtime_config": (
            None if runtime.source_path is None else _file_identity(runtime.source_path)
        ),
        "dino_checkpoint": _file_identity(runtime.dino_checkpoint),
        "selected_ghim_checkpoint": _file_identity(
            runtime.selected_stage1_checkpoint
        ),
        "pyramid_checkpoint": _file_identity(runtime.pyramid_checkpoint),
    }


def _finite_stats(values: torch.Tensor) -> dict[str, float | None]:
    finite = values.detach().float()[torch.isfinite(values)]
    if not finite.numel():
        return {"mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "mean": float(finite.mean().item()),
        "median": float(finite.median().item()),
        "p90": float(torch.quantile(finite, 0.9).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def _summarize_delta_diagnostics(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    rows = endpoint.get("per_pair")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Endpoint audit has no per-pair rows")
    update_count = len(rows[0]["delta_px"])
    summaries: list[dict[str, Any]] = []
    for update_index in range(update_count):
        desired_before = torch.tensor(
            [
                row["H0_corner_residual_px"]
                if update_index == 0
                else row["H_updates_corner_residual_px"][update_index - 1]
                for row in rows
            ],
            dtype=torch.float32,
        )
        desired_after = torch.tensor(
            [row["H_updates_corner_residual_px"][update_index] for row in rows],
            dtype=torch.float32,
        )
        delta = torch.tensor(
            [row["delta_px"][update_index] for row in rows], dtype=torch.float32
        )
        accepted = torch.tensor(
            [row["update_accepted"][update_index] for row in rows], dtype=torch.bool
        )
        flat_desired = desired_before.flatten(1)
        flat_delta = delta.flatten(1)
        cosine_denominator = torch.linalg.vector_norm(
            flat_desired, dim=1
        ) * torch.linalg.vector_norm(flat_delta, dim=1)
        cosine = torch.full_like(cosine_denominator, float("nan"))
        valid_cosine = cosine_denominator > 1e-12
        cosine[valid_cosine] = (
            (flat_desired[valid_cosine] * flat_delta[valid_cosine]).sum(dim=1)
            / cosine_denominator[valid_cosine]
        )
        coordinate_axes: dict[str, Any] = {}
        for coordinate_index, coordinate_name in enumerate(("x", "y")):
            desired_coordinate = desired_before[..., coordinate_index]
            delta_coordinate = delta[..., coordinate_index]
            after_coordinate = desired_after[..., coordinate_index]
            nonzero_target = desired_coordinate.abs() > 1e-6
            sign_agreement = (
                torch.sign(delta_coordinate[nonzero_target])
                == torch.sign(desired_coordinate[nonzero_target])
            ).float()
            coordinate_axes[coordinate_name] = {
                "desired_before_abs_px": _finite_stats(desired_coordinate.abs()),
                "decoder_delta_abs_px": _finite_stats(delta_coordinate.abs()),
                "delta_minus_desired_abs_px": _finite_stats(
                    (delta_coordinate - desired_coordinate).abs()
                ),
                "desired_after_abs_px": _finite_stats(after_coordinate.abs()),
                "delta_target_sign_agreement_fraction": (
                    None
                    if not sign_agreement.numel()
                    else float(sign_agreement.mean().item())
                ),
            }
        summaries.append(
            {
                "update_index": update_index,
                "scale": int(rows[0]["update_scale_schedule"][update_index]),
                "accepted_fraction": float(accepted.float().mean().item()),
                "desired_before_abs_px": _finite_stats(desired_before.abs()),
                "decoder_delta_abs_px": _finite_stats(delta.abs()),
                "delta_minus_desired_abs_px": _finite_stats(
                    (delta - desired_before).abs()
                ),
                "desired_after_abs_px": _finite_stats(desired_after.abs()),
                "corner_l1_improvement_px_per_condition": _finite_stats(
                    desired_before.abs().mean(dim=(1, 2))
                    - desired_after.abs().mean(dim=(1, 2))
                ),
                "improved_condition_fraction": float(
                    (
                        desired_after.abs().mean(dim=(1, 2))
                        < desired_before.abs().mean(dim=(1, 2))
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "delta_to_desired_cosine_per_condition": _finite_stats(cosine),
                "coordinate_axes": coordinate_axes,
                "tanh_saturation_fraction": _finite_stats(
                    torch.tensor(
                        [row["tanh_saturation_fraction"][update_index] for row in rows]
                    )
                ),
                "supported_query_count": _finite_stats(
                    torch.tensor(
                        [row["supported_query_count"][update_index] for row in rows]
                    )
                ),
                "condition_number": _finite_stats(
                    torch.tensor(
                        [row["condition_number"][update_index] for row in rows]
                    )
                ),
            }
        )
    return {"updates": summaries}


def _stat_snapshot(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def run_checkpoint_audit(
    runtime: RuntimePaths,
    *,
    checkpoint_path: Path,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    device = torch.device(runtime.device)
    if device.type != "cuda":
        raise RuntimeError("Real full-resolution tiny checkpoint audit requires CUDA")
    source = checkpoint_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Tiny checkpoint does not exist: {source}")

    before_copy = _stat_snapshot(source)
    with TemporaryDirectory(prefix="mhinet_tiny_checkpoint_audit_") as directory:
        snapshot = Path(directory) / "checkpoint_snapshot.pt"
        shutil.copyfile(source, snapshot)
        snapshot_identity = _file_identity(snapshot)
        after_copy = _stat_snapshot(source)
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        header = _inspect_checkpoint_payload(payload)
        protocol = _resolve_protocol(header, overrides)
        serialized_state_sha256 = _state_dict_sha256(header["model_state"])

        seed = int(protocol["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

        model, build_report = build_model(runtime)
        for group in build_report["training_parameters"]["groups"].values():
            group.pop("optimizer_parameter_ids", None)
        active_scales = tuple(protocol["active_scales"])
        sample_count = int(protocol["sample_count"])
        sample_protocol = str(protocol["sample_protocol"])
        unique_pair_count = 1 if sample_protocol == "one_pair_residuals" else sample_count
        experiments = ((str(protocol["diagnostic"]), active_scales),)
        unique_cache, cache_report = cache_exact_training_features(
            model,
            runtime,
            sample_count=unique_pair_count,
            stored_scales=active_scales,
            provider_prefix=_requested_provider_prefix(experiments),
        )
        cache = (
            [unique_cache[0] for _ in range(sample_count)]
            if sample_protocol == "one_pair_residuals"
            else unique_cache
        )
        cache_pair_ids = tuple(sample.pair_id for sample in cache)
        if cache_pair_ids != tuple(protocol["pair_ids"]):
            raise RuntimeError("Reconstructed training pair IDs do not match checkpoint")

        actual_context = _actual_resume_context(
            runtime,
            architecture_sha256=str(build_report["architecture_sha256"]),
            training_manifest_sha256=str(cache_report["manifest_sha256"]),
        )
        recorded_context = header["signature"].get("resume_context")
        if recorded_context is not None and recorded_context != actual_context:
            raise RuntimeError("Current resource identities differ from checkpoint signature")
        resource_binding = (
            "strict_signature_match"
            if recorded_context is not None
            else "legacy_checkpoint_without_embedded_resource_hashes"
        )

        load_report = load_checkpoint(
            snapshot,
            model=model.iterator,
            map_location="cpu",
            strict=True,
            restore_rng=False,
        )
        load_report.pop("auxiliary_state", None)
        loaded_state_sha256 = _state_dict_sha256(model.iterator.state_dict())
        if loaded_state_sha256 != serialized_state_sha256:
            raise RuntimeError("Strictly loaded iterator state differs from checkpoint tensors")

        H0_values = [
            controlled_h0_from_ground_truth(
                sample.H_gt_norm,
                sample_index=int(protocol["condition_index_offset"]) + index,
                max_abs_residual_px=float(
                    protocol["maximum_declared_abs_residual_px"]
                ),
                seed=seed,
                residual_profile=str(protocol["residual_profile"]),
            )[0]
            for index, sample in enumerate(cache)
        ]
        feature_dtype = (
            torch.bfloat16 if protocol["precision"] == "bf16" else torch.float32
        )
        preloaded_pyramid = (
            {
                scale: cache[0].pyramid[scale].to(
                    device=device, dtype=feature_dtype
                )
                for scale in active_scales
            }
            if sample_protocol == "one_pair_residuals"
            else None
        )
        model.feature_provider.to("cpu")
        model.iterator.requires_grad_(False)
        model.iterator.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        endpoint = evaluate_tiny_training_set(
            model,
            cache,
            H0_values,
            active_scales=active_scales,
            device=device,
            cnn_autocast_enabled=protocol["precision"] == "bf16",
            preloaded_pyramid=preloaded_pyramid,
        )
        post_forward_state_sha256 = _state_dict_sha256(model.iterator.state_dict())
        if post_forward_state_sha256 != loaded_state_sha256:
            raise RuntimeError("Read-only endpoint audit mutated iterator weights")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    checkpoint_contains_average = bool(
        header["metadata"].get("checkpoint_contains_weight_average", False)
    )
    return {
        "audit": "P4_tiny_checkpoint_endpoint",
        "status": "diagnostic_complete",
        "checkpoint": {
            "source_path": str(source),
            "source_stat_before_snapshot": before_copy,
            "source_stat_after_snapshot": after_copy,
            "source_replaced_while_copying": before_copy != after_copy,
            "snapshot_isolated_before_preflight": True,
            "audited_snapshot_bytes": snapshot_identity["bytes"],
            "audited_snapshot_sha256": snapshot_identity["sha256"],
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "role": header["role"],
            "legacy_checkpoint": header["legacy"],
            "optimizer_step": header["optimizer_step"],
            "checkpoint_contains_weight_average": checkpoint_contains_average,
            "serialized_iterator_state_sha256": serialized_state_sha256,
            "loaded_iterator_state_sha256": loaded_state_sha256,
            "post_forward_iterator_state_sha256": post_forward_state_sha256,
        },
        "protocol": protocol,
        "resource_binding": resource_binding,
        "actual_resources": actual_context,
        "build": build_report,
        "cache": cache_report,
        "load": load_report,
        "endpoint": endpoint,
        "delta_diagnostics": _summarize_delta_diagnostics(endpoint),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "safety": {
            "optimizer_constructed": False,
            "optimizer_or_scheduler_restored": False,
            "rng_restored": False,
            "backward_or_optimizer_step_executed": False,
            "iterator_requires_grad": False,
            "test_split_used": False,
        },
        "scope_note": (
            "Read-only endpoint diagnosis on controlled training conditions. It is "
            "not a P4 gate pass, validation result, configuration-selection result, "
            "or formal training checkpoint. Reported endpoint latency includes the "
            "extra diagnostic tensor transfers and is not comparable to profiler output."
        ),
    }


def _optional_experiment(value: str | None) -> tuple[str | None, tuple[int, ...] | None]:
    if value is None:
        return None, None
    parsed = parse_experiments(value)
    if len(parsed) != 1:
        raise ValueError("--experiment must select exactly one tiny experiment")
    return parsed[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument(
        "--sample-protocol", choices=("one_pair_residuals", "distinct_pairs")
    )
    parser.add_argument("--residual-profile", choices=("translation", "projective"))
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--residual-bound-fraction", type=float)
    parser.add_argument("--condition-index-offset", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    diagnostic, active_scales = _optional_experiment(args.experiment)
    overrides = {
        "diagnostic": diagnostic,
        "active_scales": active_scales,
        "sample_count": args.sample_count,
        "sample_protocol": args.sample_protocol,
        "residual_profile": args.residual_profile,
        "precision": args.precision,
        "seed": args.seed,
        "residual_bound_fraction": args.residual_bound_fraction,
        "condition_index_offset": args.condition_index_offset,
    }
    report = run_checkpoint_audit(
        RuntimePaths.from_json(args.runtime),
        checkpoint_path=args.checkpoint,
        overrides=overrides,
    )
    _write_json(output, report)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
