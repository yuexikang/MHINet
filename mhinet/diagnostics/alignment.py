"""P0/P1 live alignment against the pinned legacy shared implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from mhinet.config import RuntimePaths
from mhinet.dataio.data import HomographyPairDataset
from mhinet.models.feature_provider import build_feature_provider, make_loma_importable


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    if actual.shape != expected.shape:
        return {
            "shape_equal": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    delta = (actual.float() - expected.float()).abs()
    return {
        "shape_equal": True,
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "exact_fraction": float((delta == 0).float().mean().item()),
    }


@torch.no_grad()
def run_alignment(runtime: RuntimePaths, pair_index: int = 0) -> dict[str, object]:
    make_loma_importable(runtime.loma_root, runtime.loretta_source_root)
    from experiments.loretta_stage1_h.fitter import (
        MatchabilityWeightedHomographyFitter,
    )
    from experiments.stage1_dedode_pyramid_hroi_v1.pyramid_descriptor import (
        decode_cumulative_pyramid,
    )

    device = torch.device(runtime.device)
    dataset = HomographyPairDataset(runtime.data_root / "train/pairs.jsonl")
    sample = dataset[pair_index]
    images = sample["images"].unsqueeze(0).to(device)
    provider, build_report = build_feature_provider(runtime)
    provider = provider.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = provider(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Legal-sample equality of the safety-refactored Stage1 solve.
    safe_stage1 = output["stage1"]
    safe_fitter = provider.stage1_head.homography_fitter
    legacy_fitter = MatchabilityWeightedHomographyFitter(
        matchability_threshold=safe_fitter.matchability_threshold,
        minimum_correspondences=safe_fitter.minimum_correspondences,
        ridge=safe_fitter.ridge,
    ).to(device)
    legacy_h, legacy_ok, legacy_counts = legacy_fitter(
        safe_stage1["coarse_warp"], safe_stage1["coarse_matchability"]
    )

    # Re-run only the pinned legacy D8/D2 decode on the same context/module.
    flat_images = images.squeeze(0)
    vgg_features, vgg_sizes = provider.vgg(flat_images)
    legacy_pyramid = decode_cumulative_pyramid(
        provider.descriptor,
        vgg_features,
        vgg_sizes,
        output["context"],
        debug_continue_scale1=False,
    )
    d8 = output["pyramid"][8].squeeze(0)
    d2 = output["pyramid"][2].squeeze(0)
    comparisons = {
        "safe_vs_legacy_H0": _difference(output["H0_norm"], legacy_h),
        "new_vs_legacy_D8": _difference(d8, legacy_pyramid.d8),
        "new_vs_legacy_D2": _difference(d2, legacy_pyramid.d2),
    }
    passed = (
        bool(output["stage1_valid"].all())
        and bool(legacy_ok.all())
        and all(
            item.get("shape_equal") is True and float(item.get("max_abs", 1.0)) <= 2e-3
            for item in comparisons.values()
        )
        and output["call_counts"] == {
            "dino": 1,
            "mvt": 1,
            "vgg": 1,
            "dedode_decode_calls": 1,
            "dedode_steps": 4,
            "dedode_scale1": 0,
        }
    )
    return {
        "gate": "P0_P1_live_alignment",
        "status": "passed" if passed else "failed",
        "pair_id": sample["pair_id"],
        "split": "train",
        "test_used": False,
        "device": str(device),
        "shapes": {
            "images": list(images.shape),
            "pair_descriptors": list(output["pair_descriptors"].shape),
            "context": list(output["context"].shape),
            "H0": list(output["H0_norm"].shape),
            **{
                f"D{scale}": list(output["pyramid"][scale].shape)
                for scale in (8, 4, 2)
            },
        },
        "stage1_valid": bool(output["stage1_valid"][0].item()),
        "stage1_valid_correspondences": int(
            output["stage1_valid_correspondences"][0].item()
        ),
        "legacy_valid_correspondences": int(legacy_counts[0].item()),
        "comparisons": comparisons,
        "call_counts": output["call_counts"],
        "ownership": provider.ownership_report,
        "build": build_report,
        "forward_ms_single_unwarmed": elapsed_ms,
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "note": "Single unwarmed engineering alignment; not a latency benchmark or training result.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    runtime = RuntimePaths.from_json(args.runtime)
    report = run_alignment(runtime, pair_index=args.pair_index)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
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
