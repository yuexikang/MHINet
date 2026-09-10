"""Merge independently run TINY-S/TINY-6 artifacts into one audited P4 gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from mhinet.engine.checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from mhinet.config import load_architecture_config, sha256_file


REQUIRED_EXPERIMENTS = {
    "TINY-S-D8": (8,),
    "TINY-S-D4": (4,),
    "TINY-S-D2": (2,),
    "TINY-6": (8, 4, 2),
}

REGISTERED_MAX_RESIDUAL_PX = {
    "TINY-S-D8": 16.0,
    "TINY-S-D4": 8.0,
    "TINY-S-D2": 3.0,
    "TINY-6": 16.0,
}

REGISTERED_PROTOCOL = {
    "training_revision": "1.2",
    "loss": "uniform proposal-corner coordinate L1",
    "FGO": False,
    "extra_losses": False,
    "planar_head": False,
    "precision": "BF16 adapter/CNN autocast, FP32 correlation/geometry",
    "tiny_weight_average_start_step": 1536,
    "sample_protocol": "one_pair_residuals",
    "diagnostic_sample_count": 32,
    "test_used": False,
}


# The former D8/D4 exceptions belonged to the superseded 4x4-pool/MLP
# refinement decoder.  No checkpoint from that architecture may satisfy the
# current MCNet-topology gate.
LEGACY_READOUT_CHECKPOINTS: dict[str, str] = {}


def _exact_float(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _metric_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _metric_close(value: Any, expected: float) -> bool:
    converted = _metric_float(value)
    return converted is not None and math.isclose(
        converted, expected, rel_tol=0.0, abs_tol=1e-6
    )


def _summary_statistics(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "p90": float(torch.quantile(tensor, 0.9).item()),
        "max": float(tensor.max().item()),
    }


def _canonical_plain(value: Any) -> Any:
    """Normalize tuple/list differences introduced by JSON serialization."""

    if isinstance(value, Mapping):
        return {str(key): _canonical_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_plain(item) for item in value]
    return value


def _metric_block_errors(
    block: Any,
    *,
    label: str,
    updates: int,
    prefix: str,
) -> tuple[list[str], dict[str, float] | None]:
    """Validate and independently aggregate one 32-condition metric block."""

    errors: list[str] = []
    if not isinstance(block, Mapping):
        return [f"{prefix} {label} metric block is not an object"], None
    rows = block.get("per_pair")
    if not isinstance(rows, list) or len(rows) != 32:
        return [f"{prefix} {label} does not contain 32 per-pair rows"], None

    final_values: list[float] = []
    rejected_updates = 0
    failed_pairs = 0
    for expected_index, row in enumerate(rows):
        row_prefix = f"{prefix} {label} row {expected_index}"
        if not isinstance(row, Mapping):
            errors.append(f"{row_prefix} is not an object")
            continue
        observed_index = row.get("diagnostic_sample_index")
        if type(observed_index) is not int or observed_index != expected_index:
            errors.append(f"{row_prefix} has a wrong diagnostic sample index")

        trajectory = row.get("trajectory_mace_px")
        update_values = row.get("H_updates_mace_px")
        accepted = row.get("update_accepted")
        reasons = row.get("failure_reason_codes")
        if not isinstance(trajectory, list) or len(trajectory) != updates + 1:
            errors.append(f"{row_prefix} has a wrong trajectory length")
            trajectory_values: list[float] = []
        else:
            converted = [_metric_float(value) for value in trajectory]
            if any(value is None for value in converted):
                errors.append(f"{row_prefix} contains a non-finite trajectory value")
                trajectory_values = []
            else:
                trajectory_values = [float(value) for value in converted if value is not None]

        if not isinstance(update_values, list) or len(update_values) != updates:
            errors.append(f"{row_prefix} has a wrong H-update metric length")
        elif trajectory_values and any(
            not _metric_close(value, expected)
            for value, expected in zip(update_values, trajectory_values[1:])
        ):
            errors.append(f"{row_prefix} H-update metrics disagree with its trajectory")

        if not isinstance(accepted, list) or len(accepted) != updates or not all(
            type(value) is bool for value in accepted
        ):
            errors.append(f"{row_prefix} has an invalid accepted-update vector")
        else:
            rejected_updates += sum(not value for value in accepted)

        if not isinstance(reasons, list) or len(reasons) != updates or not all(
            type(value) is int for value in reasons
        ):
            errors.append(f"{row_prefix} has an invalid failure-reason vector")
        elif isinstance(accepted, list) and len(accepted) == updates and all(
            type(value) is bool for value in accepted
        ):
            if any((flag and reason != 0) or (not flag and reason == 0) for flag, reason in zip(accepted, reasons)):
                errors.append(f"{row_prefix} accepted flags disagree with reason codes")

        final_value = _metric_float(row.get("H_final_mace_px"))
        if final_value is None:
            errors.append(f"{row_prefix} has a non-finite final MACE")
            failed_pairs += 1
        else:
            final_values.append(final_value)
            if trajectory_values and not _metric_close(final_value, trajectory_values[-1]):
                errors.append(f"{row_prefix} final MACE disagrees with its trajectory")
        if trajectory_values and not _metric_close(
            row.get("H0_mace_px"), trajectory_values[0]
        ):
            errors.append(f"{row_prefix} H0 MACE disagrees with its trajectory")

    statistics = _summary_statistics(final_values) if len(final_values) == 32 else None
    recorded_statistics = block.get("H_final_mace_px")
    if statistics is None:
        errors.append(f"{prefix} {label} does not have 32 finite final MACE values")
    elif not isinstance(recorded_statistics, Mapping) or any(
        not _metric_close(recorded_statistics.get(key), value)
        for key, value in statistics.items()
    ):
        errors.append(f"{prefix} {label} summary statistics do not match per-pair rows")

    if block.get("failed_pairs") != failed_pairs:
        errors.append(f"{prefix} {label} failed-pair count does not match per-pair rows")
    if block.get("rejected_updates") != rejected_updates:
        errors.append(f"{prefix} {label} rejected-update count does not match per-pair rows")
    return errors, statistics


def _required_checkpoint_metadata_errors(
    metadata: Any,
    *,
    name: str,
    expected_scales: tuple[int, ...],
    prefix: str,
) -> list[str]:
    if not isinstance(metadata, Mapping):
        return [f"{prefix} checkpoint metadata is not an object"]
    errors: list[str] = []
    architecture = load_architecture_config()
    expected_values = {
        "diagnostic": name,
        "known_H0_residual_injected": True,
        "formal_training_checkpoint": False,
        "sample_count": 32,
        "sample_protocol": "one_pair_residuals",
        "residual_profile": "translation",
        "precision": "bf16",
        "architecture_sha256": architecture.sha256,
        "mhir_revision": architecture.raw["mhir_revision"],
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            errors.append(f"{prefix} checkpoint metadata {key!r} is invalid")
    if tuple(metadata.get("active_scales", ())) != expected_scales:
        errors.append(f"{prefix} checkpoint metadata active scales are invalid")
    pair_ids = metadata.get("pair_ids")
    if not isinstance(pair_ids, list) or len(pair_ids) != 32 or not all(
        isinstance(value, str) and value for value in pair_ids
    ):
        errors.append(f"{prefix} checkpoint metadata does not contain 32 pair IDs")
    signature = metadata.get("tiny_progress_signature")
    if not isinstance(signature, Mapping):
        errors.append(f"{prefix} checkpoint has no strict tiny-progress signature")
    else:
        resume_context = signature.get("resume_context")
        expected_architecture_sha = architecture.sha256
        if not isinstance(resume_context, Mapping) or resume_context.get(
            "architecture_sha256"
        ) != expected_architecture_sha:
            errors.append(
                f"{prefix} checkpoint architecture does not match the current MHIR"
            )
    return errors


def _readout_errors(
    experiment: Mapping[str, Any],
    *,
    name: str,
    checkpoint_sha: str | None,
    checkpoint_metadata: Mapping[str, Any],
    optimizer_steps: int | None,
    prefix: str,
) -> tuple[list[str], bool]:
    """Bind the declared primary metric to the exact weights in the checkpoint."""

    errors: list[str] = []
    averaging = experiment.get("weight_averaging")
    if not isinstance(averaging, Mapping):
        return [f"{prefix} weight-averaging record is not an object"], False
    legacy = checkpoint_sha == LEGACY_READOUT_CHECKPOINTS.get(name)
    if optimizer_steps is None:
        expected_count = None
    else:
        expected_count = max(0, optimizer_steps - 1536 + 1)
    count = averaging.get("averaged_snapshots")
    if type(count) is not int or expected_count is None or count != expected_count:
        errors.append(f"{prefix} averaged-snapshot count disagrees with optimizer step")

    expected_averaged = expected_count is not None and expected_count > 0
    expected_primary = (
        "equal_weight_parameter_average" if expected_averaged else "raw_parameters"
    )
    if experiment.get("primary_readout") != expected_primary:
        errors.append(f"{prefix} primary readout disagrees with averaging window")

    applied = averaging.get("applied_to_primary_readout")
    if applied is None and legacy:
        applied = True
    if applied is not expected_averaged:
        errors.append(f"{prefix} primary-readout application flag is invalid")

    metadata_count = checkpoint_metadata.get("weight_average_count")
    if metadata_count != expected_count:
        errors.append(f"{prefix} checkpoint averaging count is invalid")
    if checkpoint_metadata.get("weight_average_start_step") != 1536:
        errors.append(f"{prefix} checkpoint averaging start step is invalid")
    contains_average = checkpoint_metadata.get("checkpoint_contains_weight_average")
    if contains_average is None and legacy:
        contains_average = True
    if contains_average is not expected_averaged:
        errors.append(f"{prefix} checkpoint/readout weight ownership is invalid")
    if checkpoint_metadata.get("optimizer_resume_supported") is not (
        not expected_averaged
    ):
        errors.append(f"{prefix} checkpoint resumability flag is invalid")
    return errors, legacy


def _checkpoint_evidence_errors(
    experiment: Mapping[str, Any],
    *,
    name: str,
    expected_scales: tuple[int, ...],
    optimizer_steps: int | None,
    prefix: str,
) -> tuple[list[str], bool]:
    """Verify the report, on-disk hash and restricted checkpoint payload."""

    errors: list[str] = []
    checkpoint = experiment.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return [f"{prefix} checkpoint evidence is not an object"], False
    checkpoint_sha = checkpoint.get("sha256")
    sha_valid = isinstance(checkpoint_sha, str) and re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha
    ) is not None
    if not sha_valid:
        errors.append(f"{prefix} checkpoint SHA256 is invalid")
        checkpoint_sha = None
    supplied_path = checkpoint.get("path")
    if not isinstance(supplied_path, str) or not supplied_path:
        errors.append(f"{prefix} checkpoint path is missing")
        return errors, False
    checkpoint_path = Path(supplied_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        errors.append(f"{prefix} checkpoint file does not exist: {checkpoint_path}")
        return errors, False

    actual_bytes = checkpoint_path.stat().st_size
    if type(checkpoint.get("bytes")) is not int or checkpoint.get("bytes") != actual_bytes:
        errors.append(f"{prefix} checkpoint byte count does not match the file")
    actual_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != actual_sha:
        errors.append(f"{prefix} checkpoint SHA256 does not match the file")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        errors.append(f"{prefix} checkpoint report format is invalid")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        errors.append(f"{prefix} checkpoint report version is invalid")
    if optimizer_steps is None or checkpoint.get("optimizer_step") != optimizer_steps:
        errors.append(f"{prefix} checkpoint report optimizer step is invalid")

    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as error:  # fail closed on corrupt or unsafe evidence
        errors.append(
            f"{prefix} checkpoint cannot be restricted-loaded: {type(error).__name__}: {error}"
        )
        return errors, False
    if not isinstance(payload, Mapping):
        return errors + [f"{prefix} checkpoint payload is not an object"], False
    if payload.get("format") != CHECKPOINT_FORMAT:
        errors.append(f"{prefix} checkpoint payload format is invalid")
    if payload.get("version") != CHECKPOINT_VERSION:
        errors.append(f"{prefix} checkpoint payload version is invalid")
    progress = payload.get("progress")
    if not isinstance(progress, Mapping) or progress.get("optimizer_step") != optimizer_steps:
        errors.append(f"{prefix} checkpoint payload optimizer step is invalid")

    reported_metadata = checkpoint.get("metadata")
    payload_metadata = payload.get("metadata")
    errors.extend(
        _required_checkpoint_metadata_errors(
            reported_metadata,
            name=name,
            expected_scales=expected_scales,
            prefix=f"{prefix} reported",
        )
    )
    errors.extend(
        _required_checkpoint_metadata_errors(
            payload_metadata,
            name=name,
            expected_scales=expected_scales,
            prefix=f"{prefix} payload",
        )
    )
    if _canonical_plain(reported_metadata) != _canonical_plain(payload_metadata):
        errors.append(f"{prefix} reported metadata differs from checkpoint payload")
    metadata_for_readout = (
        reported_metadata if isinstance(reported_metadata, Mapping) else {}
    )
    readout_errors, legacy = _readout_errors(
        experiment,
        name=name,
        checkpoint_sha=actual_sha,
        checkpoint_metadata=metadata_for_readout,
        optimizer_steps=optimizer_steps,
        prefix=prefix,
    )
    errors.extend(readout_errors)
    return errors, legacy


def registered_experiment_errors(
    experiment: dict[str, Any],
    name: str,
    *,
    source: str,
) -> list[str]:
    """Return all reasons an experiment cannot satisfy the registered P4 gate."""

    errors: list[str] = []
    expected_scales = REQUIRED_EXPERIMENTS[name]
    prefix = f"{source}: {name}"
    if tuple(experiment.get("active_scales", ())) != expected_scales:
        errors.append(f"{prefix} has wrong active scales")
    if experiment.get("updates") != 2 * len(expected_scales):
        errors.append(f"{prefix} does not use two updates per active scale")
    if experiment.get("sample_count") != 32:
        errors.append(f"{prefix} does not contain 32 samples")
    if experiment.get("sample_protocol") != "one_pair_residuals":
        errors.append(f"{prefix} has the wrong sample protocol")
    if experiment.get("residual_profile") != "translation":
        errors.append(f"{prefix} has the wrong residual profile")
    if experiment.get("precision") != "bf16":
        errors.append(f"{prefix} has the wrong precision")
    if experiment.get("status") != "passed":
        errors.append(f"{prefix} is not passed")
    if experiment.get("gradient_finite") is not True:
        errors.append(f"{prefix} does not prove finite gradients")
    if experiment.get("seed") != 0:
        errors.append(f"{prefix} does not use registered seed 0")

    optimizer = experiment.get("optimizer", {})
    optimizer_ok = isinstance(optimizer, Mapping) and (
        optimizer.get("name") == "AdamW"
        and _exact_float(optimizer.get("learning_rate"), 1e-3)
        and _exact_float(optimizer.get("weight_decay"), 0.0)
        and optimizer.get("scheduler") is None
        and optimizer.get("batch_size") == 1
        and optimizer.get("gradient_accumulation") == 4
        and optimizer.get("effective_batch_size") == 4
        and _exact_float(optimizer.get("gradient_clip_norm"), 1.0)
        and optimizer.get("maximum_steps") == 2000
        and isinstance(optimizer.get("optimizer_steps"), int)
        and 1 <= optimizer["optimizer_steps"] <= 2000
    )
    if not optimizer_ok:
        errors.append(f"{prefix} optimizer recipe differs from protocol v1.2")
    optimizer_steps = (
        optimizer.get("optimizer_steps")
        if isinstance(optimizer, Mapping)
        and type(optimizer.get("optimizer_steps")) is int
        else None
    )

    averaging = experiment.get("weight_averaging", {})
    if not isinstance(averaging, Mapping) or not (
        averaging.get("enabled") is True
        and averaging.get("start_optimizer_step") == 1536
        and averaging.get("training_optimizer_unchanged") is True
        and averaging.get("formal_training_enabled") is False
    ):
        errors.append(f"{prefix} has the wrong registered readout policy")
    primary = experiment.get("primary_readout")
    if primary not in {"raw_parameters", "equal_weight_parameter_average"}:
        errors.append(f"{prefix} has an unknown primary readout")
    if primary == "equal_weight_parameter_average" and not (
        isinstance(averaging.get("averaged_snapshots"), int)
        and averaging["averaged_snapshots"] > 0
    ):
        errors.append(f"{prefix} averaged readout has no snapshots")

    controlled_h0 = experiment.get("controlled_H0", {})
    expected_max_residual = REGISTERED_MAX_RESIDUAL_PX[name]
    expected_fraction = 0.5
    if not isinstance(controlled_h0, Mapping) or not (
        controlled_h0.get("profile") == "translation"
        and controlled_h0.get("formal_training_injection") is False
        and _exact_float(
            controlled_h0.get("maximum_declared_abs_residual_px"),
            expected_max_residual,
        )
    ):
        errors.append(f"{prefix} controlled-H0 provenance is invalid")
    else:
        reported_fraction = controlled_h0.get(
            "bound_fraction_of_coarsest_active_decoder"
        )
        if reported_fraction is not None and not _exact_float(
            reported_fraction, expected_fraction
        ):
            errors.append(f"{prefix} controlled-H0 bound fraction is invalid")
        actual_stats = controlled_h0.get("actual_abs_residual_px")
        actual_max = (
            _metric_float(actual_stats.get("max"))
            if isinstance(actual_stats, Mapping)
            else None
        )
        if actual_max is None or not math.isclose(
            actual_max, expected_max_residual, rel_tol=0.0, abs_tol=1e-3
        ):
            errors.append(f"{prefix} controlled-H0 actual bound is invalid")

    criterion = experiment.get("criterion", {})
    threshold = criterion.get("threshold_mace_px")
    if criterion.get("met") is not True or not _exact_float(threshold, 0.1):
        errors.append(f"{prefix} criterion is not the registered <0.1 px gate")
    final_block = experiment.get("final")
    final = final_block if isinstance(final_block, Mapping) else {}
    try:
        mean_mace = float(
            final.get("H_final_mace_px", {}).get("mean", float("inf"))
        )
    except (TypeError, ValueError):
        mean_mace = float("inf")
    if not math.isfinite(mean_mace) or mean_mace >= 0.1:
        errors.append(f"{prefix} metric does not meet threshold")
    if final.get("failed_pairs") != 0 or final.get("rejected_updates") != 0:
        errors.append(f"{prefix} contains a failure or rejected update")

    updates = 2 * len(expected_scales)
    initial_errors, initial_statistics = _metric_block_errors(
        experiment.get("initial"),
        label="initial",
        updates=updates,
        prefix=prefix,
    )
    raw_errors, raw_statistics = _metric_block_errors(
        experiment.get("raw_final"),
        label="raw final",
        updates=updates,
        prefix=prefix,
    )
    final_errors, final_statistics = _metric_block_errors(
        final_block,
        label="primary final",
        updates=updates,
        prefix=prefix,
    )
    errors.extend(initial_errors)
    errors.extend(raw_errors)
    errors.extend(final_errors)
    if (
        initial_statistics is not None
        and final_statistics is not None
        and final_statistics["mean"] >= initial_statistics["mean"]
    ):
        errors.append(f"{prefix} primary final metric does not improve over H0")
    if primary == "raw_parameters" and (
        raw_statistics is None
        or final_statistics is None
        or any(
            not _metric_close(raw_statistics[key], final_statistics[key])
            for key in raw_statistics
        )
    ):
        errors.append(f"{prefix} raw primary metric differs from raw endpoint")

    final_rows = final.get("per_pair") if isinstance(final, Mapping) else None
    checkpoint = experiment.get("checkpoint")
    checkpoint_metadata = (
        checkpoint.get("metadata") if isinstance(checkpoint, Mapping) else None
    )
    if isinstance(final_rows, list) and isinstance(checkpoint_metadata, Mapping):
        final_pair_ids = [
            row.get("pair_id") if isinstance(row, Mapping) else None
            for row in final_rows
        ]
        if (
            len(final_pair_ids) != 32
            or any(not isinstance(value, str) or not value for value in final_pair_ids)
            or len(set(final_pair_ids)) != 1
        ):
            errors.append(f"{prefix} does not contain one fixed pair across 32 conditions")
        if checkpoint_metadata.get("pair_ids") != final_pair_ids:
            errors.append(f"{prefix} checkpoint pair IDs differ from metric rows")

    history = experiment.get("history")
    if not isinstance(history, list) or not history:
        errors.append(f"{prefix} has no optimizer/evaluation history")
    else:
        last_history = history[-1]
        if not isinstance(last_history, Mapping) or last_history.get(
            "optimizer_step"
        ) != optimizer_steps:
            errors.append(f"{prefix} history endpoint disagrees with optimizer step")
        else:
            history_primary = last_history.get("primary_metrics")
            if not isinstance(history_primary, Mapping) or final_statistics is None:
                errors.append(f"{prefix} history has no primary endpoint metrics")
            else:
                history_stats = history_primary.get("H_final_mace_px")
                if not isinstance(history_stats, Mapping) or any(
                    not _metric_close(history_stats.get(key), value)
                    for key, value in final_statistics.items()
                ):
                    errors.append(f"{prefix} history endpoint differs from final metrics")

    checkpoint_errors, _legacy = _checkpoint_evidence_errors(
        experiment,
        name=name,
        expected_scales=expected_scales,
        optimizer_steps=optimizer_steps,
        prefix=prefix,
    )
    errors.extend(checkpoint_errors)
    return errors


def merge_tiny_gate_artifacts(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one tiny artifact is required")
    sources: list[dict[str, Any]] = []
    experiments: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    reference: dict[str, Any] | None = None
    evidence_reference: dict[str, Any] | None = None
    current_architecture = load_architecture_config()
    legacy_readout_compatibility: list[dict[str, str]] = []
    for supplied in paths:
        path = supplied.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": payload.get("status"),
            }
        )
        if payload.get("gate") != "P4_TINY_S_TINY_6":
            errors.append(f"{path}: wrong gate identifier")
        if payload.get("status") != "passed":
            errors.append(f"{path}: source artifact is not passed")
        protocol = payload.get("protocol", {})
        cache = payload.get("cache", {})
        comparable = {
            "training_revision": protocol.get("training_revision"),
            "loss": protocol.get("loss"),
            "FGO": protocol.get("FGO"),
            "extra_losses": protocol.get("extra_losses"),
            "planar_head": protocol.get("planar_head"),
            "precision": protocol.get("precision"),
            "tiny_weight_average_start_step": protocol.get(
                "tiny_weight_average_start_step"
            ),
            "sample_protocol": cache.get("sample_protocol"),
            "diagnostic_sample_count": cache.get("diagnostic_sample_count"),
            "test_used": cache.get("test_used"),
        }
        build = payload.get("build", {})
        provider = build.get("provider", {}) if isinstance(build, Mapping) else {}
        evidence_comparable = {
            "manifest": cache.get("manifest"),
            "manifest_sha256": cache.get("manifest_sha256"),
            "unique_image_pair_count": cache.get("unique_image_pair_count"),
            "pair_ids": cache.get("pair_ids"),
            "parent_groups": cache.get("parent_groups"),
            "geo_groups": cache.get("geo_groups"),
            "architecture_sha256": (
                build.get("architecture_sha256")
                if isinstance(build, Mapping)
                else None
            ),
            "loma_checkpoint": (
                provider.get("loma_checkpoint")
                if isinstance(provider, Mapping)
                else None
            ),
            "dino_checkpoint": (
                provider.get("dino_checkpoint")
                if isinstance(provider, Mapping)
                else None
            ),
            "selected_ghim_checkpoint_legacy_stage1_key": (
                provider.get("selected_stage1_checkpoint")
                if isinstance(provider, Mapping)
                else None
            ),
        }
        if evidence_comparable["architecture_sha256"] != current_architecture.sha256:
            errors.append(
                f"{path}: architecture SHA256 does not match the current "
                "MCNet-topology MHIR"
            )
        if reference is None:
            reference = comparable
        elif comparable != reference:
            errors.append(f"{path}: protocol/sample definition differs from first input")
        if evidence_reference is None:
            evidence_reference = evidence_comparable
        elif evidence_comparable != evidence_reference:
            errors.append(f"{path}: data/resource evidence differs from first input")
        if cache.get("shared_once_per_pair") is not True:
            errors.append(f"{path}: shared DINO/MVT once-per-pair evidence is missing")
        for experiment in payload.get("experiments", []):
            name = experiment.get("name")
            if name not in REQUIRED_EXPERIMENTS:
                errors.append(f"{path}: unexpected experiment {name!r}")
                continue
            if name in experiments:
                errors.append(f"duplicate experiment {name!r}")
                continue
            errors.extend(
                registered_experiment_errors(
                    experiment, str(name), source=str(path)
                )
            )
            checkpoint = experiment.get("checkpoint")
            checkpoint_sha = (
                checkpoint.get("sha256")
                if isinstance(checkpoint, Mapping)
                else None
            )
            if checkpoint_sha == LEGACY_READOUT_CHECKPOINTS.get(str(name)):
                legacy_readout_compatibility.append(
                    {
                        "experiment": str(name),
                        "checkpoint_sha256": str(checkpoint_sha),
                        "compatibility": (
                            "missing readout flags accepted only for this independently "
                            "reloaded and metric-verified immutable checkpoint"
                        ),
                    }
                )
            experiments[str(name)] = experiment

    missing = sorted(set(REQUIRED_EXPERIMENTS) - set(experiments))
    if missing:
        errors.append(f"missing experiments: {missing}")
    if reference is not None:
        if reference != REGISTERED_PROTOCOL:
            errors.append("merged protocol does not match the registered tiny recipe")
    return {
        "gate": "P4_TINY_S_TINY_6",
        "status": "passed" if not errors else "failed",
        "protocol": reference,
        "data_and_resource_evidence": evidence_reference,
        "current_architecture": {
            "path": str(current_architecture.path),
            "sha256": current_architecture.sha256,
            "mhir_revision": current_architecture.raw.get("mhir_revision"),
        },
        "sources": sources,
        "experiments": [
            experiments[name]
            for name in REQUIRED_EXPERIMENTS
            if name in experiments
        ],
        "errors": errors,
        "legacy_readout_compatibility": legacy_readout_compatibility,
        "scope_note": (
            "This merged artifact is a fixed-train-set learnability gate, not "
            "validation accuracy. Averaged and raw readouts remain explicit in "
            "each experiment."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    report = merge_tiny_gate_artifacts(args.inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
