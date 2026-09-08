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
        for experiment in payload.get("experiments", []):
            name = experiment.get("name")
            if name not in REQUIRED_EXPERIMENTS:
                errors.append(f"{path}: unexpected experiment {name!r}")
                continue
            if name in experiments:
                errors.append(f"duplicate experiment {name!r}")
                continue
            expected_scales = REQUIRED_EXPERIMENTS[name]
            if tuple(experiment.get("active_scales", ())) != expected_scales:
                errors.append(f"{path}: {name} has wrong active scales")
            if experiment.get("sample_count") != 32:
                errors.append(f"{path}: {name} does not contain 32 samples")
            if experiment.get("status") != "passed":
                errors.append(f"{path}: {name} is not passed")
            if not experiment.get("criterion", {}).get("met", False):
                errors.append(f"{path}: {name} criterion is not met")
            final = experiment.get("final", {})
            threshold = float(
                experiment.get("criterion", {}).get(
                    "threshold_mace_px", float("nan")
                )
            )
            mean_mace = float(
                final.get("H_final_mace_px", {}).get("mean", float("inf"))
            )
            if (
                not math.isfinite(threshold)
                or threshold <= 0
                or threshold > 0.1
                or not math.isfinite(mean_mace)
                or mean_mace >= threshold
            ):
                errors.append(f"{path}: {name} metric does not meet threshold")
            if final.get("failed_pairs") != 0 or final.get("rejected_updates") != 0:
                errors.append(f"{path}: {name} contains a failure or rejected update")
            experiments[str(name)] = experiment

    missing = sorted(set(REQUIRED_EXPERIMENTS) - set(experiments))
    if missing:
        errors.append(f"missing experiments: {missing}")
    if reference is not None:
        expected_reference = {
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
        if reference != expected_reference:
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
