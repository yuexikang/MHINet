"""Merge independently run TINY-S/TINY-8 artifacts into one audited P4 gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import sha256_file


REQUIRED_EXPERIMENTS = {
    "TINY-S-D8": (8,),
    "TINY-S-D4": (4,),
    "TINY-S-D2": (2,),
    "TINY-S-D1": (1,),
    "TINY-8": (8, 4, 2, 1),
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


def _exact_float(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


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

    optimizer = experiment.get("optimizer", {})
    optimizer_ok = (
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

    averaging = experiment.get("weight_averaging", {})
    if not (
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
    if not (
        controlled_h0.get("profile") == "translation"
        and controlled_h0.get("formal_training_injection") is False
    ):
        errors.append(f"{prefix} controlled-H0 provenance is invalid")

    criterion = experiment.get("criterion", {})
    threshold = criterion.get("threshold_mace_px")
    if criterion.get("met") is not True or not _exact_float(threshold, 0.1):
        errors.append(f"{prefix} criterion is not the registered <0.1 px gate")
    final = experiment.get("final", {})
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

    checkpoint = experiment.get("checkpoint", {})
    checkpoint_sha = checkpoint.get("sha256")
    checkpoint_metadata = checkpoint.get("metadata", {})
    if not (
        isinstance(checkpoint_sha, str)
        and len(checkpoint_sha) == 64
        and checkpoint_metadata.get("formal_training_checkpoint") is False
        and checkpoint_metadata.get("sample_count") == 32
        and checkpoint_metadata.get("sample_protocol") == "one_pair_residuals"
        and checkpoint_metadata.get("residual_profile") == "translation"
        and checkpoint_metadata.get("precision") == "bf16"
    ):
        errors.append(f"{prefix} checkpoint evidence is incomplete")
    return errors


def merge_tiny_gate_artifacts(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one tiny artifact is required")
    sources: list[dict[str, Any]] = []
    experiments: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    reference: dict[str, Any] | None = None
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
        if payload.get("gate") != "P4_TINY_S_TINY_8":
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
        if reference is None:
            reference = comparable
        elif comparable != reference:
            errors.append(f"{path}: protocol/sample definition differs from first input")
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
            experiments[str(name)] = experiment

    missing = sorted(set(REQUIRED_EXPERIMENTS) - set(experiments))
    if missing:
        errors.append(f"missing experiments: {missing}")
    if reference is not None:
        if reference != REGISTERED_PROTOCOL:
            errors.append("merged protocol does not match the registered tiny recipe")
    return {
        "gate": "P4_TINY_S_TINY_8",
        "status": "passed" if not errors else "failed",
        "protocol": reference,
        "sources": sources,
        "experiments": [
            experiments[name]
            for name in REQUIRED_EXPERIMENTS
            if name in experiments
        ],
        "errors": errors,
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
