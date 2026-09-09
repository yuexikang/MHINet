"""Real-resource P3 zero-initialization and six-round mainline smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from .config import RuntimePaths
from .data import HomographyPairDataset
from .geometry import image_corners, normalized_to_pixel, safe_project_points
from .model import build_model
from .modules import EXPECTED_NEW_PARAMETERS, count_new_parameters


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


@torch.no_grad()
def run_zero_init_smoke(runtime: RuntimePaths, pair_index: int = 0) -> dict[str, Any]:
    device = torch.device(runtime.device)
    dataset = HomographyPairDataset(
        runtime.data_root / "val/pairs.jsonl", max_pairs=pair_index + 1
    )
    sample = dataset[pair_index]
    images = sample["images"].unsqueeze(0).to(device)
    H_gt = sample["H_gt_norm"].unsqueeze(0).to(device)
    model, build_report = build_model(runtime)
    for group in build_report["training_parameters"]["groups"].values():
        group.pop("optimizer_parameter_ids", None)
    model.eval()
    new_parameters = count_new_parameters(model.adapters, model.refinement_decoders)
    output_projection_nonzero = {
        scale: {
            "weight": int(
                torch.count_nonzero(
                    model.refinement_decoders[scale].out_conv.weight
                )
            ),
            "bias": int(
                torch.count_nonzero(
                    model.refinement_decoders[scale].out_conv.bias
                )
            ),
        }
        for scale in ("8", "4", "2", "1")
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    outputs = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    corners = image_corners((784, 784), device=device).unsqueeze(0)
    h0_corners, h0_valid, _ = safe_project_points(outputs["H0_norm"], corners)
    final_corners, final_valid, _ = safe_project_points(
        outputs["H_final_norm"], corners
    )
    difference_px = torch.linalg.vector_norm(
        normalized_to_pixel(final_corners, (784, 784))
        - normalized_to_pixel(h0_corners, (784, 784)),
        dim=-1,
    )
    max_noop_error = float(difference_px.max().item())
    passed = (
        new_parameters == EXPECTED_NEW_PARAMETERS
        and not any(
            values[part]
            for values in output_projection_nonzero.values()
            for part in ("weight", "bias")
        )
        and bool(h0_valid.all())
        and bool(final_valid.all())
        and max_noop_error < 1e-3
        and bool(torch.isfinite(outputs["H_updates_norm"]).all())
        and outputs["shared_call_counts"].get("dino") == 1
        and outputs["shared_call_counts"].get("mvt") == 1
        and outputs["shared_call_counts"].get("dedode_scale1") == 0
        and outputs["H_updates_norm"].shape[1] == 6
        and outputs["update_scale_schedule"] == (8, 8, 4, 4, 2, 2)
    )
    return {
        "gate": "P3_real_zero_init_six_round",
        "status": "passed" if passed else "failed",
        "pair_id": sample["pair_id"],
        "device": str(device),
        "new_parameters": new_parameters,
        "expected_new_parameters": EXPECTED_NEW_PARAMETERS,
        "output_projection_nonzero_elements": output_projection_nonzero,
        "H0_to_Hfinal_max_corner_projection_error_px": max_noop_error,
        "H0_to_Hfinal_mean_corner_projection_error_px": float(
            difference_px.mean().item()
        ),
        "stage1_valid": bool(outputs["stage1_valid"][0].item()),
        "overall_valid": bool(outputs["overall_valid"][0].item()),
        "accepted_updates": int(outputs["update_accepted"].sum().item()),
        "failure_reason_codes": _jsonable(outputs["failure_reason_codes"]),
        "supported_query_count": _jsonable(outputs["supported_query_count"]),
        "condition_number": _jsonable(outputs["condition_number"]),
        "max_abs_delta_px": float(outputs["delta_px"].abs().max().item()),
        "shared_call_counts": outputs["shared_call_counts"],
        "active_scales": list(outputs["active_scales"]),
        "update_scale_schedule": list(outputs["update_scale_schedule"]),
        "forward_ms_single_unwarmed": elapsed_ms,
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "build": build_report,
        "note": (
            "No-grad engineering fixed-point smoke on one validation pair. "
            "It is not an accuracy result, a latency benchmark, or training validation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    runtime = RuntimePaths.from_json(args.runtime)
    report = run_zero_init_smoke(runtime, args.pair_index)
    encoded = json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
