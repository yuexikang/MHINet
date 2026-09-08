"""Actual 784x784 MHINet forward/backward memory and timing profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch import nn

from .config import RuntimePaths
from .data import HomographyPairDataset
from .losses import sequence_corner_l1
from .model import build_model


def _grad_stats(parameters: Iterable[nn.Parameter]) -> dict[str, Any]:
    values = list(parameters)
    gradients = [parameter.grad for parameter in values if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    squared = sum(
        float(gradient.detach().float().square().sum().item())
        for gradient in gradients
    )
    return {
        "parameters": sum(parameter.numel() for parameter in values),
        "gradient_tensors": len(gradients),
        "finite": finite,
        "norm": math.sqrt(squared),
        "nonzero_elements": sum(
            int(torch.count_nonzero(gradient.detach()).item())
            for gradient in gradients
        ),
    }


def run_profile(
    runtime: RuntimePaths,
    *,
    profile: str,
    pair_index: int,
    optimizer_steps: int,
) -> dict[str, Any]:
    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be positive")
    device = torch.device(runtime.device)
    if device.type != "cuda":
        raise RuntimeError("The actual 784 memory profile requires CUDA")
    free_before, total_memory = torch.cuda.mem_get_info(device)
    dataset = HomographyPairDataset(
        runtime.data_root / "val/pairs.jsonl", max_pairs=pair_index + 1
    )
    sample = dataset[pair_index]
    images = sample["images"].unsqueeze(0).to(device)
    H_gt = sample["H_gt_norm"].unsqueeze(0).to(device)
    model, build = build_model(runtime)
    parameter_report = model.set_training_phase(profile)
    model.train()
    optimizer = torch.optim.AdamW(
        model.optimizer_group_spec(),
        weight_decay=float(model.architecture.raw["training"]["weight_decay"]),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, Any]] = []
    for step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        forward_started = time.perf_counter()
        outputs = model(images)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        loss_result = sequence_corner_l1(outputs, H_gt)
        if loss_result["skip_step"]:
            raise RuntimeError("Selected profile pair has invalid Stage1")
        backward_started = time.perf_counter()
        loss_result["loss"].backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        gradients = {
            name: _grad_stats(parameters)
            for name, parameters in model._all_parameter_groups().items()
        }
        optimizer.step()
        records.append(
            {
                "step_before_optimizer": step,
                "loss": float(loss_result["loss"].detach().item()),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "shared_call_counts": outputs["shared_call_counts"],
                "H_updates_require_grad": bool(
                    outputs["H_updates_norm"].requires_grad
                ),
                "proposal_Q_requires_grad": bool(
                    outputs["proposal_Q_norm"].requires_grad
                ),
                "accepted_updates": int(outputs["update_accepted"].sum().item()),
                "gradients": gradients,
                "memory_allocated_bytes_after_backward": int(
                    torch.cuda.memory_allocated(device)
                ),
                "memory_reserved_bytes_after_backward": int(
                    torch.cuda.memory_reserved(device)
                ),
            }
        )
        del outputs, loss_result
    groups_finite = all(
        group["finite"] for record in records for group in record["gradients"].values()
    )
    frozen_ok = all(
        record["gradients"][name]["gradient_tensors"] == 0
        for record in records
        for name in ("dino", "stage1_head_parameters")
    )
    calls_ok = all(
        record["shared_call_counts"].get("dino") == 1
        and record["shared_call_counts"].get("mvt") == 1
        for record in records
    )
    return {
        "gate": "P4_actual_784_profile",
        "status": "passed" if groups_finite and frozen_ok and calls_ok else "failed",
        "profile": profile,
        "pair_id": sample["pair_id"],
        "optimizer_steps": optimizer_steps,
        "device": str(device),
        "physical_device_name": torch.cuda.get_device_name(device),
        "free_memory_before_build_bytes": int(free_before),
        "total_device_memory_bytes": int(total_memory),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "records": records,
        "assertions": {
            "all_recorded_gradients_finite": groups_finite,
            "dino_and_stage1_head_grad_none": frozen_ok,
            "shared_dino_mvt_once_per_step": calls_ok,
        },
        "training_parameters": parameter_report,
        "build": build,
        "note": (
            "Single-pair contended-server engineering profile; timings are not a "
            "production benchmark. Full joint is not inferred from a frozen-heads run."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--profile", default="heads")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--optimizer-steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    try:
        report = run_profile(
            RuntimePaths.from_json(args.runtime),
            profile=args.profile,
            pair_index=args.pair_index,
            optimizer_steps=args.optimizer_steps,
        )
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        report = {
            "gate": "P4_actual_784_profile",
            "status": "failed",
            "profile": args.profile,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "cuda_memory_allocated_bytes": int(
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            ),
            "cuda_memory_reserved_bytes": int(
                torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
