"""Multi-scale Homography Iterative Refinement Module (MHIR)."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping

import torch
from torch import nn

from .correlation import HGuidedLocalCorrelation
from .geometry import (
    GUARD_REASON_NAMES,
    count_supported_queries,
    guarded_four_point_dlt,
    image_corners,
    pixel_delta_to_normalized,
    safe_project_points,
)
from .modules import MAINLINE_SCALES, REGISTERED_SCALES, SCALE_SPECS


class IteratorReason(IntEnum):
    """Geometry guard codes plus iterator-level initialization failures."""

    ACCEPTED = 0
    NO_VALID_SUPPORT = 1
    NONFINITE_SYSTEM = 2
    ILL_CONDITIONED_SYSTEM = 3
    SOLVE_FAILED = 4
    NONFINITE_SOLUTION = 5
    INVALID_PROJECTION_DENOMINATOR = 6
    STAGE1_INVALID = 7
    NONFINITE_DECODER_OUTPUT = 8


ITERATOR_REASON_NAMES = {
    **GUARD_REASON_NAMES,
    int(IteratorReason.STAGE1_INVALID): "stage1_invalid",
    int(IteratorReason.NONFINITE_DECODER_OUTPUT): "nonfinite_decoder_output",
}


class MultiScaleHIterator(nn.Module):
    """Run the six-update D8/D4/D2 mainline; retain D1 as explicit opt-in."""

    scales = REGISTERED_SCALES
    mainline_scales = MAINLINE_SCALES

    def __init__(
        self,
        adapters: nn.ModuleDict,
        decoders: nn.ModuleDict,
        *,
        query_chunk_size: int = 1024,
        feature_epsilon: float = 1e-6,
        minimum_supported_queries: int = 16,
        condition_max: float = 1e6,
        projection_denominator_min: float = 1e-4,
        validity_grid_hw: tuple[int, int] = (9, 9),
        target_hw: tuple[int, int] = (784, 784),
    ) -> None:
        super().__init__()
        expected_keys = tuple(str(scale) for scale in self.scales)
        if tuple(adapters.keys()) != expected_keys:
            raise ValueError(f"Adapter keys must be {expected_keys}, got {tuple(adapters.keys())}")
        if tuple(decoders.keys()) != expected_keys:
            raise ValueError(f"Decoder keys must be {expected_keys}, got {tuple(decoders.keys())}")
        self.adapters = adapters
        self.decoders = decoders
        radii = {8: 4, 4: 4, 2: 3, 1: 2}
        self.correlations = nn.ModuleDict(
            {
                str(scale): HGuidedLocalCorrelation(
                    radii[scale],
                    query_chunk_size=query_chunk_size,
                    feature_epsilon=feature_epsilon,
                )
                for scale in self.scales
            }
        )
        self.minimum_supported_queries = int(minimum_supported_queries)
        self.condition_max = float(condition_max)
        self.projection_denominator_min = float(projection_denominator_min)
        self.validity_grid_hw = tuple(int(value) for value in validity_grid_hw)
        self.target_hw = tuple(int(value) for value in target_hw)

    @staticmethod
    def _validate_inputs(
        pyramid: Mapping[int, torch.Tensor],
        H0_norm: torch.Tensor,
        stage1_valid: torch.Tensor,
        scales: tuple[int, ...],
    ) -> int:
        if H0_norm.ndim != 3 or H0_norm.shape[-2:] != (3, 3):
            raise ValueError(f"H0_norm must be Bx3x3, got {tuple(H0_norm.shape)}")
        batch = H0_norm.shape[0]
        if stage1_valid.shape != (batch,):
            raise ValueError(
                f"stage1_valid must have shape {(batch,)}, got {tuple(stage1_valid.shape)}"
            )
        for scale in scales:
            if scale not in pyramid:
                raise KeyError(f"Missing D{scale} pyramid tensor")
            value = pyramid[scale]
            if value.ndim != 5 or value.shape[:3] != (batch, 2, 256):
                raise ValueError(
                    f"D{scale} must be Bx2x256xHxW, got {tuple(value.shape)}"
                )
        return batch

    def forward(
        self,
        pyramid: Mapping[int, torch.Tensor],
        H0_norm: torch.Tensor,
        stage1_valid: torch.Tensor,
        *,
        target_hw: tuple[int, int] | None = None,
        active_scales: tuple[int, ...] | None = None,
        iterations_per_scale: int = 2,
        cnn_autocast_enabled: bool = True,
    ) -> dict[str, Any]:
        selected_scales = (
            self.mainline_scales
            if active_scales is None
            else tuple(int(scale) for scale in active_scales)
        )
        if not selected_scales:
            raise ValueError("active_scales must contain at least one scale")
        ordered_selection = tuple(
            scale for scale in self.scales if scale in selected_scales
        )
        if selected_scales != ordered_selection or len(set(selected_scales)) != len(
            selected_scales
        ):
            raise ValueError(
                "active_scales must be a duplicate-free coarse-to-fine selection "
                f"of {self.scales}, got {selected_scales}"
            )
        iterations_per_scale = int(iterations_per_scale)
        if iterations_per_scale <= 0:
            raise ValueError("iterations_per_scale must be positive")
        batch = self._validate_inputs(
            pyramid, H0_norm, stage1_valid, selected_scales
        )
        selected_target_hw = self.target_hw if target_hw is None else target_hw
        height_b, width_b = (int(value) for value in selected_target_hw)
        if height_b <= 0 or width_b <= 0:
            raise ValueError(f"Invalid target_hw={selected_target_hw}")
        geometry_dtype = torch.float32
        H = H0_norm.to(dtype=geometry_dtype)
        corners = image_corners(
            selected_target_hw,
            normalized=True,
            dtype=geometry_dtype,
            device=H.device,
        ).unsqueeze(0).expand(batch, -1, -1)
        initial_q, initial_projection_valid, _ = safe_project_points(H, corners)
        active = (
            stage1_valid.bool()
            & torch.isfinite(H).all(dim=(1, 2))
            & initial_projection_valid.all(dim=1)
        )
        T = initial_q - corners
        T = torch.where(active[:, None, None], T, torch.zeros_like(T))

        h_updates: list[torch.Tensor] = []
        proposal_qs: list[torch.Tensor] = []
        proposal_ts: list[torch.Tensor] = []
        accepted_updates: list[torch.Tensor] = []
        reason_codes: list[torch.Tensor] = []
        support_counts: list[torch.Tensor] = []
        condition_numbers: list[torch.Tensor] = []
        solve_infos: list[torch.Tensor] = []
        deltas_px: list[torch.Tensor] = []
        decoder_output_finite: list[torch.Tensor] = []
        saturation: list[torch.Tensor] = []
        feature_valid_counts: dict[int, torch.Tensor] = {}

        update_scale_schedule: list[int] = []
        for scale in selected_scales:
            with torch.autocast(
                pyramid[scale].device.type,
                dtype=torch.bfloat16,
                enabled=(
                    cnn_autocast_enabled
                    and pyramid[scale].device.type == "cuda"
                ),
            ):
                pair_features, adapter_valid = self.adapters[str(scale)](
                    pyramid[scale]
                )
            if pair_features.ndim != 5 or pair_features.shape[:2] != (batch, 2):
                raise ValueError(
                    f"D{scale} adapter must return Bx2xCxHxW, got {tuple(pair_features.shape)}"
                )
            source_features = pair_features[:, 0]
            target_features = pair_features[:, 1]
            feature_valid_counts[scale] = adapter_valid.flatten(3).sum(dim=-1)
            for _iteration in range(iterations_per_scale):
                update_scale_schedule.append(scale)
                correlation, candidate_valid = self.correlations[str(scale)](
                    source_features, target_features, H
                )
                # MCNet's correlation decoder consumes the local-search tensor
                # directly.  Invalid candidates are already exactly zero; the
                # boolean mask remains available to the geometry support guard.
                decoder_input = correlation
                expected_channels = SCALE_SPECS[scale].decoder_input_channels
                if decoder_input.shape[1] != expected_channels:
                    raise AssertionError(
                        f"D{scale} decoder input has {decoder_input.shape[1]} channels, "
                        f"expected {expected_channels}"
                    )
                with torch.autocast(
                    decoder_input.device.type,
                    dtype=torch.bfloat16,
                    enabled=(
                        cnn_autocast_enabled
                        and decoder_input.device.type == "cuda"
                    ),
                ):
                    delta_px = self.decoders[str(scale)](decoder_input).float()
                delta_finite = torch.isfinite(delta_px).all(dim=(1, 2))
                decoder_output_finite.append(delta_finite)
                safe_delta_px = torch.where(
                    delta_finite[:, None, None], delta_px, torch.zeros_like(delta_px)
                )
                proposal_T = T + pixel_delta_to_normalized(
                    safe_delta_px, (height_b, width_b)
                )
                proposal_Q = corners + proposal_T
                support = count_supported_queries(candidate_valid)
                support_for_guard = torch.where(active & delta_finite, support, 0)
                guarded = guarded_four_point_dlt(
                    corners,
                    proposal_Q,
                    fallback_homography=H,
                    support_count=support_for_guard,
                    minimum_supported_queries=self.minimum_supported_queries,
                    condition_max=self.condition_max,
                    projection_denominator_min=self.projection_denominator_min,
                    validity_grid_hw=self.validity_grid_hw,
                )
                accepted = guarded.accepted & active & delta_finite
                reason = guarded.reason.clone()
                reason = torch.where(
                    active,
                    reason,
                    torch.full_like(reason, int(IteratorReason.STAGE1_INVALID)),
                )
                reason = torch.where(
                    active & ~delta_finite,
                    torch.full_like(reason, int(IteratorReason.NONFINITE_DECODER_OUTPUT)),
                    reason,
                )
                H = torch.where(accepted[:, None, None], guarded.homography, H)
                T = torch.where(accepted[:, None, None], proposal_T, T)

                h_updates.append(H)
                proposal_qs.append(proposal_Q)
                proposal_ts.append(proposal_T)
                accepted_updates.append(accepted)
                reason_codes.append(reason)
                support_counts.append(support)
                condition_numbers.append(guarded.condition_number)
                solve_infos.append(guarded.solve_info)
                deltas_px.append(safe_delta_px)
                bound = SCALE_SPECS[scale].max_delta_px
                saturation.append((safe_delta_px.abs() >= 0.95 * bound).float().mean(dim=(1, 2)))

        H_updates = torch.stack(h_updates, dim=1)
        proposal_Q = torch.stack(proposal_qs, dim=1)
        accepted_tensor = torch.stack(accepted_updates, dim=1)
        reasons = torch.stack(reason_codes, dim=1)
        final_q, final_projection_valid, _ = safe_project_points(H, corners)
        overall_valid = active & final_projection_valid.all(dim=1) & torch.isfinite(final_q).all(
            dim=(1, 2)
        )
        return {
            "H0_norm": H0_norm.float(),
            "H_updates_norm": H_updates,
            "H_scales_norm": H_updates[
                :, iterations_per_scale - 1 :: iterations_per_scale
            ],
            "H_final_norm": H_updates[:, -1],
            "proposal_Q_norm": proposal_Q,
            "proposal_T_norm": torch.stack(proposal_ts, dim=1),
            "delta_px": torch.stack(deltas_px, dim=1),
            "decoder_output_finite": torch.stack(decoder_output_finite, dim=1),
            "update_accepted": accepted_tensor,
            "failure_reason_codes": reasons,
            "failure_reason_names": ITERATOR_REASON_NAMES,
            "supported_query_count": torch.stack(support_counts, dim=1),
            "condition_number": torch.stack(condition_numbers, dim=1),
            "solve_info": torch.stack(solve_infos, dim=1),
            "tanh_saturation_fraction": torch.stack(saturation, dim=1),
            "stage1_valid": stage1_valid.bool(),
            "overall_valid": overall_valid,
            "refinement_any_accepted": accepted_tensor.any(dim=1),
            "feature_valid_counts": feature_valid_counts,
            "active_scales": selected_scales,
            "iterations_per_scale": iterations_per_scale,
            "update_scale_schedule": tuple(update_scale_schedule),
            "cnn_precision": (
                "bf16_autocast"
                if cnn_autocast_enabled and H.device.type == "cuda"
                else "fp32"
            ),
        }


__all__ = [
    "ITERATOR_REASON_NAMES",
    "IteratorReason",
    "MultiScaleHIterator",
]
