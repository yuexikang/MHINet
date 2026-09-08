"""H-guided dense local correlation for MHINet.

For every source feature pixel ``p`` this operator projects ``p`` with the
current normalized A-to-B homography, adds integer candidate offsets in the
*target feature pixel coordinate system*, samples target descriptors, and
returns the full cosine-similarity window.  It does not detach the homography,
features, or sampling grid.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import (
    normalized_homography_to_pixel,
    pixel_grid,
    pixel_to_normalized,
    safe_project_points,
)


def candidate_offsets(
    radius: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return ``(K,2)`` offsets with dy outer and dx inner, both ascending."""

    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    dy_values = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    dx_values = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    dy, dx = torch.meshgrid(dy_values, dx_values, indexing="ij")
    return torch.stack((dx, dy), dim=-1).reshape(-1, 2)


def _validate_inputs(source: Tensor, target: Tensor, homography: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if source.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"source and target must be (B,C,H,W), got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"source/target batch and channel dimensions must match, got {tuple(source.shape[:2])} "
            f"and {tuple(target.shape[:2])}"
        )
    if not source.is_floating_point() or not target.is_floating_point():
        raise TypeError("source and target features must be floating point")
    if homography.ndim == 2:
        homography = homography.unsqueeze(0)
    if homography.ndim != 3 or homography.shape[-2:] != (3, 3):
        raise ValueError(f"homography must be (3,3) or (B,3,3), got {tuple(homography.shape)}")
    if homography.shape[0] not in (1, source.shape[0]):
        raise ValueError(
            f"homography batch must be 1 or {source.shape[0]}, got {homography.shape[0]}"
        )
    if source.device != target.device or source.device != homography.device:
        raise ValueError(
            f"source, target, and homography must share a device, got "
            f"{source.device}, {target.device}, and {homography.device}"
        )
    return source, target, homography.expand(source.shape[0], -1, -1)


def _compute_dtype(source: Tensor, target: Tensor, homography: Tensor) -> torch.dtype:
    if torch.float64 in (source.dtype, target.dtype, homography.dtype):
        return torch.float64
    # Geometry, sampling, normalization, and dot accumulation are deliberately
    # at least FP32 even when the stored pyramid features are BF16/FP16.
    return torch.float32


def h_guided_local_correlation(
    source: Tensor,
    target: Tensor,
    homography_normalized: Tensor,
    *,
    radius: int,
    query_chunk_size: int | None = 1024,
    feature_epsilon: float = 1e-6,
    projection_denominator_epsilon: float = 1e-8,
    activation_checkpoint: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute MHINet's full-window H-guided local cosine correlation.

    Args:
        source: Source descriptors ``(B,C,H_A,W_A)``.
        target: Target descriptors ``(B,C,H_B,W_B)``.
        homography_normalized: Current normalized A-to-B H, ``(B,3,3)``.
        radius: Local target-pixel radius. Candidate count is ``(2r+1)^2``.
        query_chunk_size: Number of row-major source queries per sampling call.
        feature_epsilon: L2 norm validity/normalization epsilon.
        projection_denominator_epsilon: Projection denominator validity limit.

    Returns:
        ``(correlation, candidate_valid)`` with shapes ``(B,K,H_A,W_A)``.
        Invalid entries are exactly zero and the mask is boolean.
    """

    source, target, homography_normalized = _validate_inputs(
        source, target, homography_normalized
    )
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if feature_epsilon <= 0 or projection_denominator_epsilon <= 0:
        raise ValueError("normalization and projection epsilons must be positive")

    batch, channels, source_height, source_width = source.shape
    _, _, target_height, target_width = target.shape
    query_count = source_height * source_width
    if query_chunk_size is None:
        chunk_size = query_count
    else:
        chunk_size = int(query_chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive or None, got {query_chunk_size}")
        chunk_size = min(chunk_size, query_count)

    dtype = _compute_dtype(source, target, homography_normalized)
    source_compute = source.to(dtype=dtype)
    target_compute = target.to(dtype=dtype)
    homography_compute = homography_normalized.to(dtype=dtype)

    source_finite = torch.isfinite(source_compute).all(dim=1)
    safe_source = torch.where(source_finite[:, None], source_compute, torch.zeros_like(source_compute))
    source_norm = torch.linalg.vector_norm(safe_source, dim=1)
    source_valid = source_finite & torch.isfinite(source_norm) & (source_norm > feature_epsilon)
    source_unit = safe_source / source_norm.clamp_min(feature_epsilon)[:, None]
    source_flat = source_unit.flatten(2)
    source_valid_flat = source_valid.flatten(1)

    target_finite = torch.isfinite(target_compute).all(dim=1, keepdim=True)
    safe_target = torch.where(target_finite, target_compute, torch.zeros_like(target_compute))
    target_finite_float = target_finite.to(dtype=dtype)

    source_positions = pixel_grid(
        (source_height, source_width), dtype=dtype, device=source.device
    ).reshape(1, query_count, 2)
    source_positions = source_positions.expand(batch, -1, -1)
    homography_pixel = normalized_homography_to_pixel(
        homography_compute,
        (source_height, source_width),
        (target_height, target_width),
    )
    projected, projection_valid, _ = safe_project_points(
        homography_pixel,
        source_positions,
        denominator_epsilon=projection_denominator_epsilon,
    )
    offsets = candidate_offsets(radius, dtype=dtype, device=source.device)
    candidate_count = offsets.shape[0]

    def compute_chunk(
        source_chunk: Tensor,
        source_valid_chunk: Tensor,
        sampling_target_tensor: Tensor,
        sampling_grid_chunk: Tensor,
        in_bounds_chunk: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Features and their finite-support mask use the same grid.  Sampling
        # them together removes one CUDA grid_sample launch per fixed query
        # chunk without changing chunk size/order or either autograd path.
        sampled_with_finite = F.grid_sample(
            sampling_target_tensor,
            sampling_grid_chunk,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled_with_finite[:, :-1]
        sampled_finite_weight = sampled_with_finite[:, -1]
        sampled_finite = sampled_finite_weight >= (1.0 - 1e-6)
        sampled_norm = torch.linalg.vector_norm(sampled, dim=1)
        sampled_valid = (
            sampled_finite
            & torch.isfinite(sampled).all(dim=1)
            & torch.isfinite(sampled_norm)
            & (sampled_norm > feature_epsilon)
        )
        sampled_unit = sampled / sampled_norm.clamp_min(feature_epsilon)[:, None]

        correlation = torch.einsum("bcq,bcqk->bqk", source_chunk, sampled_unit)
        valid = in_bounds_chunk & source_valid_chunk[:, :, None] & sampled_valid
        correlation = torch.where(valid, correlation, torch.zeros_like(correlation))
        return correlation, valid

    def compute_all_chunks(
        source_tensor: Tensor,
        source_valid_tensor: Tensor,
        target_tensor: Tensor,
        target_finite_tensor: Tensor,
        projected_tensor: Tensor,
        projection_valid_tensor: Tensor,
        offset_tensor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the fixed 1024-query sampling chunks under one checkpoint.

        A checkpoint per small chunk creates hundreds of Python checkpoint
        contexts at D2/D1.  Checkpointing the complete chunk loop retains the
        exact sampling chunk size and tensor order while recomputing the same
        loop once during backward.
        """

        sampling_target = torch.cat((target_tensor, target_finite_tensor), dim=1)
        # Candidate geometry is independent of the query chunk.  Building the
        # complete grid once inside this checkpointed region avoids repeating
        # thousands of small pointwise CUDA launches at D2/D1.  Sampling itself
        # remains in the contract's fixed row-major query chunks.
        candidate_px = (
            projected_tensor[:, :, None, :] + offset_tensor[None, None, :, :]
        )
        candidate_finite = torch.isfinite(candidate_px).all(dim=-1)
        in_bounds = (
            candidate_finite
            & projection_valid_tensor[:, :, None]
            & (candidate_px[..., 0] >= 0.0)
            & (candidate_px[..., 0] <= target_width - 1.0)
            & (candidate_px[..., 1] >= 0.0)
            & (candidate_px[..., 1] <= target_height - 1.0)
        )
        # Keep grid_sample away from huge/invalid coordinates while preserving
        # the continuous H -> grid path for every geometrically valid entry.
        safe_candidate_px = torch.where(
            in_bounds[..., None], candidate_px, torch.zeros_like(candidate_px)
        )
        sampling_grid = pixel_to_normalized(
            safe_candidate_px, (target_height, target_width)
        )
        correlation_chunks: list[Tensor] = []
        valid_chunks: list[Tensor] = []
        for start in range(0, query_count, chunk_size):
            stop = min(start + chunk_size, query_count)
            correlation, valid = compute_chunk(
                source_tensor[:, :, start:stop],
                source_valid_tensor[:, start:stop],
                sampling_target,
                sampling_grid[:, start:stop],
                in_bounds[:, start:stop],
            )
            correlation_chunks.append(correlation)
            valid_chunks.append(valid)
        return torch.cat(correlation_chunks, dim=1), torch.cat(valid_chunks, dim=1)

    checkpoint_inputs = (
        source_flat,
        source_valid_flat,
        safe_target,
        target_finite_float,
        projected,
        projection_valid,
        offsets,
    )
    checkpoint_needed = activation_checkpoint and torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in checkpoint_inputs
        if tensor.is_floating_point()
    )
    if checkpoint_needed:
        correlation_flat, valid_flat = checkpoint(
            compute_all_chunks,
            *checkpoint_inputs,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    else:
        correlation_flat, valid_flat = compute_all_chunks(*checkpoint_inputs)
    correlation_map = correlation_flat.permute(0, 2, 1).reshape(
        batch, candidate_count, source_height, source_width
    )
    valid_map = valid_flat.permute(0, 2, 1).reshape(
        batch, candidate_count, source_height, source_width
    )
    return correlation_map, valid_map


class HGuidedLocalCorrelation(nn.Module):
    """Parameter-free module wrapper for :func:`h_guided_local_correlation`."""

    def __init__(
        self,
        radius: int,
        *,
        query_chunk_size: int | None = 1024,
        feature_epsilon: float = 1e-6,
        projection_denominator_epsilon: float = 1e-8,
        activation_checkpoint_training: bool = True,
    ) -> None:
        super().__init__()
        if int(radius) < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if query_chunk_size is not None and int(query_chunk_size) <= 0:
            raise ValueError("query_chunk_size must be positive or None")
        if feature_epsilon <= 0 or projection_denominator_epsilon <= 0:
            raise ValueError("normalization and projection epsilons must be positive")
        self.radius = int(radius)
        self.query_chunk_size = None if query_chunk_size is None else int(query_chunk_size)
        self.feature_epsilon = float(feature_epsilon)
        self.projection_denominator_epsilon = float(projection_denominator_epsilon)
        self.activation_checkpoint_training = bool(activation_checkpoint_training)

    @property
    def candidate_count(self) -> int:
        return (2 * self.radius + 1) ** 2

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        homography_normalized: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return h_guided_local_correlation(
            source,
            target,
            homography_normalized,
            radius=self.radius,
            query_chunk_size=self.query_chunk_size,
            feature_epsilon=self.feature_epsilon,
            projection_denominator_epsilon=self.projection_denominator_epsilon,
            activation_checkpoint=(self.training and self.activation_checkpoint_training),
        )

    def extra_repr(self) -> str:
        return (
            f"radius={self.radius}, candidate_count={self.candidate_count}, "
            f"query_chunk_size={self.query_chunk_size}, feature_epsilon={self.feature_epsilon}, "
            f"activation_checkpoint_training={self.activation_checkpoint_training}"
        )


__all__ = [
    "HGuidedLocalCorrelation",
    "candidate_offsets",
    "h_guided_local_correlation",
]
