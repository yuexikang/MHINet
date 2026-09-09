"""Shared GHIM and CGMDP feature provider for MHINet.

This module deliberately calls the reusable LoMa components below their legacy
``inference_mode`` wrappers. DINOv3 remains frozen; the frozen GHIM homography
head still participates in autograd with respect to its MVT input. CGMDP
combines MVT context, VGG features, and a DeDoDe-style cumulative decoder. The
mainline stops after D2 and emits D8/D4/D2; D1 remains an explicit opt-in path.
These are fused matching descriptors, not DeDoDe-only features.
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.torch_version import TorchVersion

from .config import RuntimePaths


def make_loma_importable(loma_root: str | Path, loretta_root: str | Path) -> None:
    """Expose the pinned source trees without copying or modifying them."""

    roots = (
        Path(loma_root).resolve(),
        Path(loma_root).resolve() / "src",
        Path(loretta_root).resolve() / "src",
    )
    for root in reversed(roots):
        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)


def _safe_load(path: Path) -> Mapping[str, Any]:
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


class SafeMatchabilityWeightedHomographyFitter(nn.Module):
    """GHIM weighted fit with invalid samples isolated before autograd solve.

    The normal equation, threshold, tiny clamped row weights, and ridge exactly
    match the legacy Stage1 fitter on eligible samples. Eligibility is inspected
    without gradients; ``solve_ex`` is invoked only for that subset.
    """

    def __init__(
        self,
        *,
        matchability_threshold: float = 0.3,
        minimum_correspondences: int = 10,
        ridge: float = 1e-4,
    ) -> None:
        super().__init__()
        self.matchability_threshold = float(matchability_threshold)
        self.minimum_correspondences = int(minimum_correspondences)
        self.ridge = float(ridge)

    def forward(
        self, coarse_warp: torch.Tensor, coarse_matchability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from experiments.loretta_stage1_h.geometry import normalized_grid

        if coarse_warp.ndim != 4 or coarse_warp.shape[1] != 2:
            raise ValueError(f"Expected coarse warp [B,2,H,W], got {tuple(coarse_warp.shape)}")
        expected_matchability = (coarse_warp.shape[0], 1, *coarse_warp.shape[2:])
        if coarse_matchability.shape != expected_matchability:
            raise ValueError("Coarse warp and matchability shapes are incompatible")
        batch, _, height, width = coarse_warp.shape
        sensed = coarse_warp.float().permute(0, 2, 3, 1).reshape(batch, -1, 2)
        reference = normalized_grid(
            batch, height, width, device=coarse_warp.device, dtype=torch.float32
        ).reshape(batch, -1, 2)
        raw_weights = coarse_matchability.float().reshape(batch, -1)
        weights = torch.where(
            raw_weights > self.matchability_threshold,
            raw_weights,
            torch.zeros_like(raw_weights),
        )
        valid_counts = (weights > 0).sum(dim=1)
        x, y = reference.unbind(dim=-1)
        u, v = sensed.unbind(dim=-1)
        ones, zeros = torch.ones_like(x), torch.zeros_like(x)
        row_u = torch.stack(
            (x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=-1
        )
        row_v = torch.stack(
            (zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=-1
        )
        design = torch.stack((row_u, row_v), dim=2).reshape(batch, -1, 8)
        targets = torch.stack((u, v), dim=2).reshape(batch, -1)
        sqrt_weights = torch.sqrt(weights.clamp_min(1e-8))
        row_weights = torch.stack((sqrt_weights, sqrt_weights), dim=2).reshape(batch, -1)
        weighted_design = design * row_weights.unsqueeze(-1)
        weighted_targets = targets * row_weights
        normal_matrix = weighted_design.transpose(1, 2) @ weighted_design
        normal_target = (
            weighted_design.transpose(1, 2) @ weighted_targets.unsqueeze(-1)
        ).squeeze(-1)
        eye = torch.eye(8, device=coarse_warp.device, dtype=torch.float32).unsqueeze(0)
        normal_matrix = normal_matrix + self.ridge * eye

        with torch.no_grad():
            finite = torch.isfinite(normal_matrix).all(dim=(1, 2)) & torch.isfinite(
                normal_target
            ).all(dim=1)
            eligible = finite & (valid_counts >= self.minimum_correspondences)
            # Rank is evaluated only for finite matrices; no factorization sees NaN.
            finite_indices = torch.nonzero(eligible, as_tuple=False).flatten()
            rank_ok = torch.zeros_like(eligible)
            if finite_indices.numel():
                ranks = torch.linalg.matrix_rank(normal_matrix[finite_indices])
                rank_ok[finite_indices] = ranks == 8
            eligible &= rank_ok

        identity = torch.eye(
            3, device=coarse_warp.device, dtype=torch.float32
        ).expand(batch, -1, -1).clone()
        succeeded = torch.zeros(batch, device=coarse_warp.device, dtype=torch.bool)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if not eligible_indices.numel():
            return identity, succeeded, valid_counts
        solution, info = torch.linalg.solve_ex(
            normal_matrix[eligible_indices], normal_target[eligible_indices].unsqueeze(-1)
        )
        solution = solution.squeeze(-1)
        solved_ok = (info == 0) & torch.isfinite(solution).all(dim=1)
        good_indices = eligible_indices[solved_ok]
        if good_indices.numel():
            good_solution = solution[solved_ok]
            rows = torch.stack(
                (
                    good_solution[:, 0:3],
                    good_solution[:, 3:6],
                    torch.cat(
                        (good_solution[:, 6:8], torch.ones_like(good_solution[:, :1])),
                        dim=1,
                    ),
                ),
                dim=1,
            )
            identity[good_indices] = rows
            succeeded[good_indices] = True
        return identity, succeeded, valid_counts


class SharedFeatureProvider(nn.Module):
    """Run shared GHIM/CGMDP and emit H0, MVT context, and matching descriptors."""

    scales = (8, 4, 2, 1)

    def __init__(self, descriptor: nn.Module, stage1_head: nn.Module) -> None:
        super().__init__()
        self.descriptor = descriptor
        # Legacy compatibility alias: ``stage1_head`` is the GHIM homography head.
        self.stage1_head = stage1_head
        self._profile = "heads"
        self._call_counts: dict[str, int] = {}
        self.set_training_groups(mvt=False, vgg=False, dedode=False)

    @property
    def shared_encoder(self) -> nn.Module:
        return self.descriptor.encoder.frozen_dinov3

    @property
    def dino(self) -> nn.Module:
        return self.shared_encoder.model

    @property
    def mvt(self) -> nn.Module:
        return self.shared_encoder.multi_view_transformer

    @property
    def vgg(self) -> nn.Module:
        return self.descriptor.encoder.vgg

    @property
    def dedode(self) -> nn.Module:
        return self.descriptor.decoder

    @property
    def ownership_report(self) -> dict[str, Any]:
        modules = list(self.modules())
        return {
            "dino_wrapper_instances": sum(
                type(module).__name__ == "FrozenDINOv3L11L17Stage1" for module in modules
            ),
            "mvt_object_references": sum(module is self.mvt for module in modules),
            "stage1_head_contains_dino": int(
                hasattr(self.stage1_head, "dinov3_feature_extractor")
            ),
            "stage1_head_contains_mvt": int(
                hasattr(self.stage1_head, "multi_view_transformer")
            ),
            "decoder_scales": list(self.dedode.scales),
        }

    @property
    def last_call_counts(self) -> dict[str, int]:
        return dict(self._call_counts)

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(enabled)

    def set_training_groups(self, *, mvt: bool, vgg: bool, dedode: bool) -> None:
        self._set_requires_grad(self.dino, False)
        self._set_requires_grad(self.stage1_head, False)
        self._set_requires_grad(self.mvt, mvt)
        self._set_requires_grad(self.vgg, vgg)
        self._set_requires_grad(self.dedode, dedode)
        self._mvt_trainable = bool(mvt)
        self._vgg_trainable = bool(vgg)
        self._dedode_trainable = bool(dedode)
        self._restore_modes(self.training)

    def _restore_modes(self, mode: bool) -> None:
        self.dino.eval()
        self.stage1_head.eval()
        self.mvt.train(bool(mode and self._mvt_trainable))
        self.vgg.train(bool(mode and self._vgg_trainable))
        # Protocol v1.2 fixes VGG running statistics while allowing BN affine grads.
        for module in self.vgg.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.dedode.train(bool(mode and self._dedode_trainable))

    def train(self, mode: bool = True) -> "SharedFeatureProvider":
        # Some reused wrappers override train(); repair each independently afterward.
        super().train(mode)
        self._restore_modes(mode)
        return self

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "mvt": list(self.mvt.parameters()),
            "vgg": list(self.vgg.parameters()),
            "dedode": list(self.dedode.parameters()),
            "dino": list(self.dino.parameters()),
            "stage1_head_parameters": list(self.stage1_head.parameters()),
        }
        ids: dict[int, str] = {}
        for name in ("mvt", "vgg", "dedode", "dino", "stage1_head_parameters"):
            for parameter in groups[name]:
                previous = ids.setdefault(id(parameter), name)
                if previous != name:
                    raise RuntimeError(f"Parameter is registered in both {previous} and {name}")
        return groups

    def _contextualize(self, descriptors: torch.Tensor) -> torch.Tensor:
        self._call_counts["mvt"] += 1
        return self.mvt(descriptors)

    def _decode_pyramid(
        self,
        vgg_features: list[torch.Tensor],
        vgg_sizes: list[tuple[int, int]],
        contextualized: torch.Tensor,
        requested_scales: tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        shared_feature = contextualized.squeeze(0).permute(0, 3, 1, 2).contiguous()
        features = vgg_features + [shared_feature]
        sizes = vgg_sizes + [tuple(shared_feature.shape[-2:])]
        descriptions: torch.Tensor | float = 0.0
        context: torch.Tensor | None = None
        outputs: dict[int, torch.Tensor] = {}
        scale_to_output = {"8": 8, "4": 4, "2": 2, "1": 1}
        expected = ["16", "8", "4", "2", "1"]
        if list(self.dedode.scales) != expected:
            raise AssertionError(f"Unexpected DeDoDe scales: {self.dedode.scales}")
        self._call_counts["dedode_decode_calls"] += 1
        for index, (feature_map, scale) in enumerate(
            zip(reversed(features), self.dedode.scales)
        ):
            self._call_counts["dedode_steps"] += 1
            if scale == "1":
                self._call_counts["dedode_scale1"] += 1
            delta, context = self.dedode(feature_map, scale=scale, context=context)
            descriptions = descriptions + delta
            if not isinstance(descriptions, torch.Tensor):
                raise AssertionError("DeDoDe did not return tensor descriptions")
            if scale in scale_to_output:
                selected_scale = scale_to_output[scale]
                if selected_scale in requested_scales:
                    outputs[selected_scale] = descriptions.unsqueeze(0)
                if selected_scale == requested_scales[-1]:
                    break
            if index < len(self.dedode.scales) - 1:
                next_size = sizes[-(index + 2)]
                descriptions = F.interpolate(
                    descriptions, size=next_size, mode="bilinear", align_corners=False
                )
                if context is None:
                    raise AssertionError("DeDoDe context unexpectedly missing")
                context = F.interpolate(
                    context, size=next_size, mode="bilinear", align_corners=False
                )
        if tuple(outputs) != requested_scales:
            raise AssertionError(f"Missing pyramid outputs: {tuple(outputs)}")
        return outputs

    def forward(
        self,
        images: torch.Tensor,
        *,
        pyramid_scales: tuple[int, ...] = (8, 4, 2),
        compute_stage1: bool = True,
    ) -> dict[str, Any]:
        if images.shape != (1, 2, 3, 784, 784):
            raise ValueError(
                "MHINet v1 provider requires B=1 and shape 1x2x3x784x784, got "
                f"{tuple(images.shape)}"
            )
        flat_images = images.squeeze(0)
        allowed_scale_sets = {(8,), (8, 4), (8, 4, 2), (8, 4, 2, 1)}
        pyramid_scales = tuple(int(scale) for scale in pyramid_scales)
        if pyramid_scales and pyramid_scales not in allowed_scale_sets:
            raise ValueError(
                "pyramid_scales must be a coarse-to-fine prefix of (8,4,2,1)"
            )
        self._call_counts = {
            "dino": 1,
            "mvt": 0,
            "vgg": 0,
            "dedode_decode_calls": 0,
            "dedode_steps": 0,
            "dedode_scale1": 0,
        }
        # The reused extraction routine has no_grad only around DINO, which is required.
        pair_descriptors, token_size = self.shared_encoder.extract_pair_descriptors(
            flat_images
        )
        # Bypass legacy contextualize_pair(): it hard-codes torch.no_grad().
        amp_enabled = pair_descriptors.device.type == "cuda"
        with torch.autocast(
            pair_descriptors.device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            contextualized = self._contextualize(pair_descriptors)
        stage1 = (
            self.stage1_head(contextualized, position_dtype=pair_descriptors.dtype)
            if compute_stage1
            else None
        )

        pyramid: dict[int, torch.Tensor] = {}
        if pyramid_scales:
            self._call_counts["vgg"] += 1
            vgg_context = nullcontext() if self._vgg_trainable else torch.no_grad()
            with vgg_context:
                vgg_features, vgg_sizes = self.vgg(flat_images)
            pyramid_needs_grad = self._dedode_trainable or contextualized.requires_grad
            decoder_context = nullcontext() if pyramid_needs_grad else torch.no_grad()
            with decoder_context:
                pyramid = self._decode_pyramid(
                    vgg_features, vgg_sizes, contextualized, pyramid_scales
                )
        expected_shapes = {
            8: (1, 2, 256, 98, 98),
            4: (1, 2, 256, 196, 196),
            2: (1, 2, 256, 392, 392),
            1: (1, 2, 256, 784, 784),
        }
        for scale in pyramid_scales:
            expected = expected_shapes[scale]
            if tuple(pyramid[scale].shape) != expected:
                raise AssertionError(
                    f"D{scale} shape mismatch: {tuple(pyramid[scale].shape)} != {expected}"
                )
        # ``stage1_*`` return keys are legacy aliases retained for callers and
        # serialized artifacts; architecturally they report GHIM validity/output.
        return {
            "pair_descriptors": pair_descriptors,
            "context": contextualized,
            "token_size": token_size,
            "H0_norm": None if stage1 is None else stage1["H_A_to_B_norm"].float(),
            "stage1_valid": None if stage1 is None else stage1["fit_succeeded"].bool(),
            "stage1_valid_correspondences": (
                None if stage1 is None else stage1["valid_correspondences"]
            ),
            "stage1": stage1,
            "pyramid": pyramid,
            "call_counts": self.last_call_counts,
        }


def _load_selected_locator_if_needed(
    full_stage1: nn.Module, selected: Path, dino_checkpoint: Path
) -> dict[str, Any]:
    if selected.resolve() == dino_checkpoint.resolve():
        return {"source": "official_loretta_checkpoint", "overridden": False}
    payload = _safe_load(selected)
    locator = payload.get("locator")
    if not isinstance(locator, Mapping):
        raise ValueError(
            "A selected Stage1 checkpoint distinct from the DINO checkpoint must "
            "contain a complete 'locator' state mapping"
        )
    full_stage1.locator.load_state_dict(locator, strict=True, assign=True)
    return {
        "source": "stage1_experiment_locator",
        "overridden": True,
        "step": int(payload.get("step", 0)),
        "best_auc": float(payload.get("best_auc", float("nan"))),
        "tensors": len(locator),
    }


def build_feature_provider(runtime: RuntimePaths) -> tuple[SharedFeatureProvider, dict[str, Any]]:
    """Build the shared provider from the exact external source and checkpoints."""

    runtime.validate()
    make_loma_importable(runtime.loma_root, runtime.loretta_source_root)
    from experiments.loma_dinov3_l17_v1.descriptor import DeDoDeDINOv3L17
    from experiments.loretta_stage1_h.model import LoRettaStage1H
    from experiments.stage1_loma_shared_v1.stage1_head import Stage1HeadFromSharedMVT

    official_state = _safe_load(runtime.pyramid_checkpoint)
    descriptor = DeDoDeDINOv3L17(
        runtime.dino_checkpoint,
        fusion="l11_l17_stage1_transformer",
        multi_view_checkpoint=runtime.selected_stage1_checkpoint,
    )
    descriptor_report = descriptor.initialize_from_loma_b(official_state)
    del official_state
    full_stage1, resolved_dino, loaded_keys = LoRettaStage1H.from_pretrained(
        runtime.dino_checkpoint,
        enable_amp=True,
        dinov3_weights="lvd1689m",
    )
    locator_report = _load_selected_locator_if_needed(
        full_stage1,
        runtime.selected_stage1_checkpoint,
        runtime.dino_checkpoint,
    )
    old_fitter = full_stage1.homography_fitter
    full_stage1.homography_fitter = SafeMatchabilityWeightedHomographyFitter(
        matchability_threshold=old_fitter.matchability_threshold,
        minimum_correspondences=old_fitter.minimum_correspondences,
        ridge=old_fitter.ridge,
    )
    stage1_head = Stage1HeadFromSharedMVT(full_stage1)
    del full_stage1
    provider = SharedFeatureProvider(descriptor, stage1_head)
    report = {
        "loma_root": str(runtime.loma_root),
        "loma_checkpoint": str(runtime.pyramid_checkpoint),
        "dino_checkpoint": str(resolved_dino),
        "selected_stage1_checkpoint": str(runtime.selected_stage1_checkpoint),
        "loretta_loaded_keys": loaded_keys,
        "locator": locator_report,
        "descriptor_initialization": descriptor_report,
        "ownership": provider.ownership_report,
        "legacy_training_wrappers_reused": False,
        "safe_stage1_fitter": True,
    }
    gc.collect()
    return provider, report
