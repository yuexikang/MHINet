"""P4 real-resource gradient audit for zero-init and joint feature branches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from mhinet.config import RuntimePaths
from mhinet.dataio.data import HomographyPairDataset
from mhinet.ops.geometry import image_corners, safe_project_points
from mhinet.engine.losses import sequence_corner_l1
from mhinet.models.model import build_model


MAINLINE_SCALE_KEYS = ("8", "4", "2")


def _grad_stats(parameters: Iterable[nn.Parameter]) -> dict[str, Any]:
    parameter_list = list(parameters)
    gradients = [parameter.grad for parameter in parameter_list if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    squared_norm = sum(
        float(gradient.detach().float().square().sum().item()) for gradient in gradients
    )
    nonzero = sum(int(torch.count_nonzero(gradient.detach()).item()) for gradient in gradients)
    return {
        "parameter_tensors": len(parameter_list),
        "gradient_tensors": len(gradients),
        "finite": finite,
        "norm": math.sqrt(squared_norm),
        "nonzero_elements": nonzero,
    }


def _decoder_upstream_parameters(decoder: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in decoder.named_parameters()
        if not name.startswith("out_conv.")
    ]


def _new_layer_gradient_snapshot(model: nn.Module) -> dict[str, Any]:
    return {
        "adapters": {
            scale: _grad_stats(model.adapters[scale].parameters())
            for scale in MAINLINE_SCALE_KEYS
        },
        "decoder_out_conv": {
            scale: _grad_stats(model.refinement_decoders[scale].out_conv.parameters())
            for scale in MAINLINE_SCALE_KEYS
        },
        "decoder_upstream": {
            scale: _grad_stats(
                _decoder_upstream_parameters(model.refinement_decoders[scale])
            )
            for scale in MAINLINE_SCALE_KEYS
        },
    }


def _module_grad_none(module: nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def run_gradient_audit(runtime: RuntimePaths, pair_index: int = 0) -> dict[str, Any]:
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
    model.train()
    model.set_training_phase("heads")

    # Frozen-provider caching is permitted for the fixed-input heads diagnostic.
    with torch.no_grad():
        shared = model.feature_provider(images)
    cached_pyramid = {scale: value.detach() for scale, value in shared["pyramid"].items()}
    cached_h0 = shared["H0_norm"].detach()
    cached_stage1_valid = shared["stage1_valid"].detach()
    del shared
    model.feature_provider.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.AdamW(
        [
            parameter
            for scale in MAINLINE_SCALE_KEYS
            for module in (model.adapters[scale], model.refinement_decoders[scale])
            for parameter in module.parameters()
        ],
        lr=1e-3,
        weight_decay=0.0,
    )
    heads_steps: list[dict[str, Any]] = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = model.iterator(cached_pyramid, cached_h0, cached_stage1_valid)
        loss_result = sequence_corner_l1(outputs, H_gt)
        if loss_result["skip_step"]:
            raise RuntimeError("Selected gradient-audit pair has invalid Stage1")
        loss = loss_result["loss"]
        loss.backward()
        snapshot = _new_layer_gradient_snapshot(model)
        snapshot.update(
            {
                "step_before_optimizer": step,
                "loss": float(loss.detach().item()),
                "all_update_H_requires_grad": bool(outputs["H_updates_norm"].requires_grad),
                "all_proposals_require_grad": bool(outputs["proposal_Q_norm"].requires_grad),
            }
        )
        heads_steps.append(snapshot)
        optimizer.step()
        del outputs, loss_result, loss

    # A final-only diagnostic must reach every earlier decoder through H/T state.
    optimizer.zero_grad(set_to_none=True)
    final_outputs = model.iterator(cached_pyramid, cached_h0, cached_stage1_valid)
    corners = image_corners((784, 784), device=device).unsqueeze(0)
    predicted_final, final_valid, _ = safe_project_points(
        final_outputs["H_final_norm"], corners
    )
    target_final, target_valid, _ = safe_project_points(H_gt, corners)
    if not bool(final_valid.all() and target_valid.all()):
        raise RuntimeError("Final-only gradient diagnostic encountered invalid geometry")
    final_only_loss = (predicted_final - target_final).abs().mean()
    final_only_loss.backward()
    final_only_decoder_grad = {
        scale: _grad_stats(model.refinement_decoders[scale].parameters())
        for scale in MAINLINE_SCALE_KEYS
    }
    heads_peak = int(
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    del final_outputs, final_only_loss, predicted_final, target_final
    del cached_pyramid, cached_h0, cached_stage1_valid
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Restore the provider and audit each MVT route independently.  The H0-only
    # route skips VGG/DeDoDe; the feature-only route skips Stage1 and stops at D8.
    model.feature_provider.to(device)
    model.set_training_phase("joint")
    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    h0_branch = model.feature_provider(
        images, pyramid_scales=(), compute_stage1=True
    )
    h0_corners, h0_valid, _ = safe_project_points(h0_branch["H0_norm"], corners)
    gt_corners, gt_valid, _ = safe_project_points(H_gt, corners)
    if not bool(h0_valid.all() and gt_valid.all()):
        raise RuntimeError("H0 branch has invalid corner geometry")
    h0_branch_loss = (h0_corners - gt_corners).abs().mean()
    h0_branch_loss.backward()
    provider_groups = model.feature_provider.parameter_groups()
    h0_branch_report = {
        "loss": float(h0_branch_loss.detach().item()),
        "mvt": _grad_stats(provider_groups["mvt"]),
        "vgg": _grad_stats(provider_groups["vgg"]),
        "dedode": _grad_stats(provider_groups["dedode"]),
        "dino_grad_all_none": _module_grad_none(model.feature_provider.dino),
        "stage1_head_grad_all_none": _module_grad_none(
            model.feature_provider.stage1_head
        ),
        "call_counts": h0_branch["call_counts"],
    }
    del h0_branch, h0_branch_loss, h0_corners, gt_corners
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    feature_branch = model.feature_provider(
        images, pyramid_scales=(8,), compute_stage1=False
    )
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        adapted, _adapter_valid = model.adapters["8"](
            feature_branch["pyramid"][8]
        )
    # Deliberately use fixed GT H to isolate MVT->pyramid from MVT->H0.
    correlation, _candidate_valid = model.iterator.correlations["8"](
        adapted[:, 0], adapted[:, 1], H_gt.detach()
    )
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        feature_delta = model.refinement_decoders["8"](correlation)
    feature_branch_loss = feature_delta.float().sum()
    feature_branch_loss.backward()
    feature_branch_report = {
        "diagnostic_loss": float(feature_branch_loss.detach().item()),
        "mvt": _grad_stats(provider_groups["mvt"]),
        "vgg": _grad_stats(provider_groups["vgg"]),
        "dedode": _grad_stats(provider_groups["dedode"]),
        "adapter_D8": _grad_stats(model.adapters["8"].parameters()),
        "decoder_D8": _grad_stats(model.refinement_decoders["8"].parameters()),
        "dino_grad_all_none": _module_grad_none(model.feature_provider.dino),
        "stage1_head_grad_all_none": _module_grad_none(
            model.feature_provider.stage1_head
        ),
        "call_counts": feature_branch["call_counts"],
        "H0_detach_note": (
            "H_gt.detach() is used only in this diagnostic to isolate the pyramid "
            "route; production joint forward does not detach H0/H/T."
        ),
    }
    joint_branch_peak = int(
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    first = heads_steps[0]
    second = heads_steps[1]
    first_out_conv_nonzero = all(
        first["decoder_out_conv"][scale]["norm"] > 0
        for scale in MAINLINE_SCALE_KEYS
    )
    first_upstream_zero = all(
        first["decoder_upstream"][scale]["norm"] == 0
        and first["adapters"][scale]["norm"] == 0
        for scale in MAINLINE_SCALE_KEYS
    )
    second_upstream_nonzero = all(
        second["decoder_upstream"][scale]["norm"] > 0
        and second["adapters"][scale]["norm"] > 0
        for scale in MAINLINE_SCALE_KEYS
    )
    final_reaches_all = all(
        final_only_decoder_grad[scale]["norm"] > 0 for scale in MAINLINE_SCALE_KEYS
    )
    dormant_d1_ok = all(
        parameter.grad is None and not parameter.requires_grad
        for module in (model.adapters["1"], model.refinement_decoders["1"])
        for parameter in module.parameters()
    )
    h0_mvt_ok = h0_branch_report["mvt"]["norm"] > 0
    feature_mvt_ok = feature_branch_report["mvt"]["norm"] > 0
    feature_vgg_ok = feature_branch_report["vgg"]["norm"] > 0
    feature_dedode_ok = feature_branch_report["dedode"]["norm"] > 0
    frozen_ok = all(
        report["dino_grad_all_none"] and report["stage1_head_grad_all_none"]
        for report in (h0_branch_report, feature_branch_report)
    )
    passed = all(
        (
            first_out_conv_nonzero,
            first_upstream_zero,
            second_upstream_nonzero,
            final_reaches_all,
            h0_mvt_ok,
            feature_mvt_ok,
            feature_vgg_ok,
            feature_dedode_ok,
            frozen_ok,
            dormant_d1_ok,
        )
    )
    return {
        "gate": "P4_gradient_audit",
        "status": "passed" if passed else "failed",
        "pair_id": sample["pair_id"],
        "device": str(device),
        "build": build_report,
        "heads_cached_feature_two_steps": heads_steps,
        "final_only_decoder_gradients": final_only_decoder_grad,
        "mvt_H0_route": h0_branch_report,
        "mvt_pyramid_route": feature_branch_report,
        "assertions": {
            "step0_all_out_conv_nonzero": first_out_conv_nonzero,
            "step0_all_upstream_zero_expected": first_upstream_zero,
            "step1_all_decoder_and_adapter_upstream_nonzero": second_upstream_nonzero,
            "final_only_reaches_all_scales": final_reaches_all,
            "mvt_H0_route_nonzero": h0_mvt_ok,
            "mvt_pyramid_route_nonzero": feature_mvt_ok,
            "vgg_pyramid_route_nonzero": feature_vgg_ok,
            "dedode_pyramid_route_nonzero": feature_dedode_ok,
            "dino_and_stage1_head_parameter_grads_none": frozen_ok,
            "d1_registered_but_inactive_grad_none": dormant_d1_ok,
        },
        "peak_allocated_bytes": {
            "heads_cached_six_round": heads_peak,
            "joint_branch_diagnostics": joint_branch_peak,
        },
        "scope_note": (
            "This audits full cached-feature six-round heads backward and the two "
            "real MVT routes separately. A full 784 joint six-round backward is a "
            "separate smoke/profile gate."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = run_gradient_audit(
        RuntimePaths.from_json(args.runtime), pair_index=args.pair_index
    )
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
