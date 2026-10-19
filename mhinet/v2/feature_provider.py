"""512-only shared features with a homography-free D16 locator for MHI v2.

The adapter operates on the existing Stage1 head. It registers no modules or
parameters, so a v1 shared state dictionary retains exactly the same keys.
Coordinates follow ``grid_sample(align_corners=False)``; flow is A-to-B target
minus the A cell center, in normalized image coordinates, with channel order xy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mhinet.models.feature_provider import build_feature_provider
from mhinet.pretraining.model import SharedDescriptorNetwork


GEOMETRY_MODES = ("flow_d16", "none", "h0")


def normalized_d16_grid(batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return BCHW xy cell centers for the fixed 32x32 D16 grid."""
    centers = torch.linspace(-1 + 1 / 32, 1 - 1 / 32, 32, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack((xx, yy)).unsqueeze(0).expand(batch, -1, -1, -1)


class FlowStage1Adapter:
    """The legacy positional matching/decoder path, without its H fitter.

    This intentionally is not an ``nn.Module`` and owns no reference to the
    head. The original head remains registered only as ``stage1_head``.
    """

    @staticmethod
    def forward(
        head: nn.Module,
        contextualized: torch.Tensor,
        *,
        position_dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        if contextualized.ndim != 5 or contextualized.shape[1] != 2:
            raise ValueError("Shared MVT output must be Bx2xHxWxC")
        if contextualized.shape[-1] != head.localization_dim:
            raise ValueError(f"Expected {head.localization_dim} MVT channels")
        batch, _, height, width, _ = contextualized.shape
        if position_dtype is None:
            position_dtype = contextualized.dtype
        # Same cell centers and positional projection as Stage1HeadFromSharedMVT.
        ys = torch.linspace(-1 + 1 / height, 1 - 1 / height, height,
                            device=contextualized.device, dtype=position_dtype)
        xs = torch.linspace(-1 + 1 / width, 1 - 1 / width, width,
                            device=contextualized.device, dtype=position_dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        projection = F.linear(
            grid.reshape(batch, height * width, 2),
            (head.position_scale * head.position_frequencies).to(
                device=contextualized.device, dtype=position_dtype),
        ).reshape(batch, height, width, -1)
        positions = torch.cat((projection.sin(), projection.cos()), dim=-1)

        # Exact legacy locator calculation, kept local to avoid depending on the
        # external Stage1 module's homography forward or inference wrappers.
        reference = F.normalize(contextualized[:, 0].float(), dim=-1)
        sensed = F.normalize(contextualized[:, 1].float(), dim=-1)
        similarities = torch.einsum("bhwd,bijd->bhwij", reference, sensed)
        logits = similarities.reshape(batch, height * width, height * width) / head.temperature
        logits = torch.cat((logits, head.no_match_logit.to(logits).expand(
            batch, height * width, 1)), dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        candidates = positions.reshape(batch, height * width, -1)
        no_match = head.no_match_embedding.to(candidates).reshape(1, 1, -1)
        candidates = torch.cat((candidates, no_match.expand(batch, 1, -1)), dim=1)
        embeddings = torch.einsum("bnm,bmd->bnd", probabilities, candidates).reshape(
            batch, height, width, -1)
        warp, matchability = head.match_embedding_decoder(embeddings)
        return {
            "coarse_warp": warp,
            "coarse_matchability": matchability,
            "match_probabilities": probabilities,
            "H_A_to_B_norm": None,
            "fit_succeeded": None,
            "valid_correspondences": None,
        }


def _flow_fields(stage1: dict[str, Any] | None) -> dict[str, torch.Tensor | None]:
    keys = ("flow16_target_norm", "flow16_norm", "flow16_matchability", "flow16_valid")
    if stage1 is None:
        return dict.fromkeys(keys)
    target = stage1["coarse_warp"].float()
    matchability = stage1["coarse_matchability"].float()
    if target.ndim != 4 or tuple(target.shape[1:]) != (2, 32, 32):
        raise ValueError(f"D16 target must be Bx2x32x32, got {tuple(target.shape)}")
    if tuple(matchability.shape) != (target.shape[0], 1, 32, 32):
        raise ValueError("D16 matchability must be Bx1x32x32")
    source = normalized_d16_grid(target.shape[0], device=target.device, dtype=target.dtype)
    # The decoder already emits probabilities: applying sigmoid again would
    # silently turn an invalid 0 probability into a seemingly usable 0.5.
    valid = (torch.isfinite(target).all(dim=1, keepdim=True)
             & torch.isfinite(matchability) & (matchability >= 0) & (matchability <= 1)
             & (target.abs() <= 1).all(dim=1, keepdim=True))
    return dict(zip(keys, (target, target - source, matchability, valid)))


class V2SharedDescriptorNetwork(SharedDescriptorNetwork):
    """Original shared topology with selectable v2 geometry computation."""

    def __init__(self, descriptor: nn.Module, stage1_head: nn.Module,
                 *, geometry_mode: str = "flow_d16") -> None:
        if geometry_mode not in GEOMETRY_MODES:
            raise ValueError(f"geometry_mode must be one of {GEOMETRY_MODES}")
        super().__init__(descriptor, stage1_head)
        self.geometry_mode = geometry_mode
        self.lora_enabled = False

    def forward(self, images: torch.Tensor, *, pyramid_scales: tuple[int, ...] = (8, 4, 2),
                compute_stage1: bool = True, compute_flow: bool | None = None) -> dict[str, Any]:
        if images.ndim != 5 or images.shape[0] < 1 or tuple(images.shape[1:]) != (2, 3, 512, 512):
            raise ValueError(f"MHI v2 requires Bx2x3x512x512, got {tuple(images.shape)}")
        legacy_h0 = self.geometry_mode == "h0" and compute_stage1
        result = super().forward(images, pyramid_scales=pyramid_scales,
                                 compute_stage1=legacy_h0)
        stage1 = result["stage1"]
        # A flow-mode caller may disable H computation using the legacy argument;
        # its required Flow0 output must still be present.
        need_flow = self.geometry_mode == "flow_d16" or compute_flow is True
        if stage1 is None and need_flow:
            stage1 = FlowStage1Adapter.forward(
                self.stage1_head, result["context"],
                position_dtype=result["pair_descriptors"].dtype)
            result["stage1"] = stage1
        result.update(_flow_fields(stage1))
        result["geometry_mode"] = self.geometry_mode
        result["call_counts"] = {**result["call_counts"],
                                 "flow_head": int(stage1 is not None),
                                 "homography_fitter": int(legacy_h0)}
        return result


def build_v2_shared(runtime, geometry_mode: str = "flow_d16") -> tuple[V2SharedDescriptorNetwork, dict[str, Any]]:
    """Build pinned shared components without changing any old parameter keys."""
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f"geometry_mode must be one of {GEOMETRY_MODES}")
    original, report = build_feature_provider(runtime)
    model = V2SharedDescriptorNetwork(original.descriptor, original.stage1_head,
                                      geometry_mode=geometry_mode).configure(lora=False)
    model = model.to(runtime.device)
    report = {**report, "geometry_mode": geometry_mode, "input_size": 512,
              "flow_grid_hw": [32, 32], "flow_coordinate_order": "xy",
              "flow_align_corners": False,
              "stage1_parameter_keys_preserved": True,
              "homography_fitter_enabled": geometry_mode == "h0"}
    return model, report


__all__ = ["FlowStage1Adapter", "V2SharedDescriptorNetwork", "build_v2_shared",
           "normalized_d16_grid"]
