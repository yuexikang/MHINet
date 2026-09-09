"""Trainable feature adapters and MCNet-style correlation decoders for MHINet.

This module intentionally contains no iteration or homography logic.  It only
implements the new parameterized layers specified by the v1.2 design package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


PYRAMID_INPUT_CHANNELS: Final = 256
DECODER_HIDDEN_CHANNELS: Final = 64
CORNER_GRID_SIZE: Final = (2, 2)
EXPECTED_NEW_PARAMETERS: Final = 1_176_712
EXPECTED_MAINLINE_TRAINABLE_NEW_PARAMETERS: Final = 833_222
MAINLINE_SCALES: Final = (8, 4, 2)
REGISTERED_SCALES: Final = (8, 4, 2, 1)


@dataclass(frozen=True)
class ScaleModuleSpec:
    """Fixed v1 architecture values for one pyramid scale."""

    scale: int
    spatial_size: int
    adapter_channels: int
    correlation_candidates: int
    decoder_input_channels: int
    down_blocks: int
    max_delta_px: float


SCALE_SPECS: Final[dict[int, ScaleModuleSpec]] = {
    8: ScaleModuleSpec(
        scale=8,
        spatial_size=98,
        adapter_channels=64,
        correlation_candidates=81,
        decoder_input_channels=81,
        down_blocks=6,
        max_delta_px=32.0,
    ),
    4: ScaleModuleSpec(
        scale=4,
        spatial_size=196,
        adapter_channels=64,
        correlation_candidates=81,
        decoder_input_channels=81,
        down_blocks=7,
        max_delta_px=16.0,
    ),
    2: ScaleModuleSpec(
        scale=2,
        spatial_size=392,
        adapter_channels=32,
        correlation_candidates=49,
        decoder_input_channels=49,
        down_blocks=8,
        max_delta_px=6.0,
    ),
    1: ScaleModuleSpec(
        scale=1,
        spatial_size=784,
        adapter_channels=32,
        correlation_candidates=25,
        decoder_input_channels=25,
        down_blocks=9,
        max_delta_px=2.0,
    ),
}


def _init_kaiming(module: nn.Conv2d | nn.Linear) -> None:
    """Apply the common v1 initialization to a Conv/Linear layer."""

    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class PyramidAdapter(nn.Module):
    """Shared A/B 1x1 projection followed by point-wise L2 normalization.

    Inputs may be ordinary ``N,C,H,W`` tensors or carry additional leading
    dimensions, notably the contract shape ``B,2,C,H,W``.  The returned valid
    mask has the same leading/spatial dimensions and one channel.  A point is
    valid only when its projected norm is finite and strictly above ``eps``.
    Near-zero finite points remain zero (or near zero) after division by eps;
    non-finite input is not silently replaced with a finite value.
    """

    def __init__(
        self,
        out_channels: int,
        *,
        in_channels: int = PYRAMID_INPUT_CHANNELS,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Adapter channel counts must be positive")
        if eps <= 0:
            raise ValueError("Adapter normalization epsilon must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.eps = float(eps)
        self.projection = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )
        _init_kaiming(self.projection)

    def forward(self, descriptors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim < 4:
            raise ValueError(
                "Pyramid descriptors must have shape (..., C, H, W), got "
                f"{tuple(descriptors.shape)}"
            )
        channel_axis = descriptors.ndim - 3
        if descriptors.shape[channel_axis] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} descriptor channels, got "
                f"{descriptors.shape[channel_axis]}"
            )

        leading_shape = descriptors.shape[:-3]
        height, width = descriptors.shape[-2:]
        flattened = descriptors.reshape(-1, self.in_channels, height, width)
        projected = self.projection(flattened)
        # ``meta`` tensors are used by the exact full-resolution contract test
        # and intentionally carry no readable storage.  Keep the fail-fast
        # runtime check on every materialized CPU/GPU tensor without forcing a
        # data-dependent scalar read from a shape-only tensor.
        if projected.device.type != "meta" and not bool(torch.isfinite(projected).all()):
            raise FloatingPointError("Pyramid adapter received or produced non-finite values")
        norm = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
        valid = torch.isfinite(norm) & (norm > self.eps)
        normalized = projected / norm.clamp_min(self.eps)

        feature_shape = (*leading_shape, self.out_channels, height, width)
        mask_shape = (*leading_shape, 1, height, width)
        return normalized.reshape(feature_shape), valid.reshape(mask_shape)


class DownBlock(nn.Module):
    """MCNet correlation-decoder block adapted to non-power-of-two grids.

    MCNet uses ``Conv3x3 -> GroupNorm -> ReLU -> MaxPool2``.  MHINet's
    98/196/392/784 descriptor sizes are not exact powers of two, so ceil-mode
    pooling retains the boundary evidence and reaches a strict 2x2 corner grid
    after 6/7/8/9 blocks respectively.
    """

    def __init__(self, channels: int = DECODER_HIDDEN_CHANNELS) -> None:
        super().__init__()
        if channels <= 0 or channels % 8 != 0:
            raise ValueError("DownBlock channels must be positive and divisible by 8")
        self.channels = int(channels)
        self.conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.norm = nn.GroupNorm(8, self.channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_kaiming(self.conv)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"DownBlock expects N,{self.channels},H,W, got {tuple(inputs.shape)}"
            )
        return self.pool(self.activation(self.norm(self.conv(inputs))))


class ResidualGeometryDecoder(nn.Module):
    """MCNet-style correlation decoder with a direct 2x2 corner topology.

    The learnable path deliberately mirrors MCNet's official decoder:
    correlation-only 1x1 input projection, repeated Conv/GN/ReLU/MaxPool
    blocks, then a two-channel 1x1 output on a 2x2 grid.  The grid is flattened
    row-major after moving x/y to the last axis, giving TL/TR/BL/BR corner
    order.  MHINet retains two safety adaptations: an all-zero final projection
    and a scale-specific tanh bound in input-image pixels.
    """

    def __init__(
        self,
        input_channels: int,
        down_blocks: int,
        max_delta_px: float,
        *,
        input_spatial_size: int | None = None,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("Decoder input_channels must be positive")
        if down_blocks < 0:
            raise ValueError("Decoder down_blocks cannot be negative")
        if max_delta_px <= 0:
            raise ValueError("Decoder residual bound must be positive")

        self.input_channels = int(input_channels)
        self.num_down_blocks = int(down_blocks)
        self.max_delta_px = float(max_delta_px)
        self.input_spatial_size = (
            None if input_spatial_size is None else int(input_spatial_size)
        )
        if self.input_spatial_size is not None and self.input_spatial_size <= 0:
            raise ValueError("Decoder input_spatial_size must be positive")

        self.in_conv = nn.Conv2d(
            self.input_channels,
            DECODER_HIDDEN_CHANNELS,
            kernel_size=1,
            bias=True,
        )
        self.layers = nn.ModuleList(
            DownBlock(DECODER_HIDDEN_CHANNELS) for _ in range(self.num_down_blocks)
        )
        self.out_conv = nn.Conv2d(
            DECODER_HIDDEN_CHANNELS,
            2,
            kernel_size=1,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_kaiming(self.in_conv)
        # DownBlock constructors own their initialization.  Reinitializing here
        # keeps reset_parameters() semantically complete when called explicitly.
        for block in self.layers:
            block.reset_parameters()
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
        if decoder_input.ndim != 4 or decoder_input.shape[1] != self.input_channels:
            raise ValueError(
                f"Decoder expects N,{self.input_channels},H,W, got "
                f"{tuple(decoder_input.shape)}"
            )
        if self.input_spatial_size is not None and decoder_input.shape[-2:] != (
            self.input_spatial_size,
            self.input_spatial_size,
        ):
            raise ValueError(
                "Decoder production input has the wrong spatial size: expected "
                f"{self.input_spatial_size}x{self.input_spatial_size}, got "
                f"{tuple(decoder_input.shape[-2:])}"
            )
        output = self.in_conv(decoder_input)
        for block in self.layers:
            output = block(output)
        if output.shape[-2:] != CORNER_GRID_SIZE:
            raise ValueError(
                "MCNet-style pooling path must terminate at 2x2; got "
                f"{tuple(output.shape[-2:])}. Check the input size and block count."
            )
        logits_grid = self.out_conv(output)
        if logits_grid.shape[-2:] != CORNER_GRID_SIZE:
            raise AssertionError(
                f"Corner logits must be 2x2, got {tuple(logits_grid.shape[-2:])}"
            )
        logits = logits_grid.permute(0, 2, 3, 1).contiguous().reshape(-1, 4, 2)
        return torch.tanh(logits) * self.max_delta_px


def count_new_parameters(*modules: nn.Module) -> int:
    """Count unique parameters across the supplied new-layer containers."""

    seen: set[int] = set()
    count = 0
    for module in modules:
        for parameter in module.parameters():
            identifier = id(parameter)
            if identifier not in seen:
                seen.add(identifier)
                count += parameter.numel()
    return count


def build_multiscale_modules() -> tuple[nn.ModuleDict, nn.ModuleDict]:
    """Build the four independent adapters and scale-specific decoders.

    A/B sharing is achieved by calling a scale's single adapter twice.  The
    caller likewise invokes a scale's single decoder for both iterations, so
    no duplicate per-iteration modules are created here.
    """

    adapters = nn.ModuleDict(
        {
            str(scale): PyramidAdapter(spec.adapter_channels)
            for scale, spec in SCALE_SPECS.items()
        }
    )
    decoders = nn.ModuleDict(
        {
            str(scale): ResidualGeometryDecoder(
                input_channels=spec.decoder_input_channels,
                down_blocks=spec.down_blocks,
                max_delta_px=spec.max_delta_px,
                input_spatial_size=spec.spatial_size,
            )
            for scale, spec in SCALE_SPECS.items()
        }
    )
    actual = count_new_parameters(adapters, decoders)
    if actual != EXPECTED_NEW_PARAMETERS:
        raise AssertionError(
            f"MHINet new parameter count mismatch: {actual} != "
            f"{EXPECTED_NEW_PARAMETERS}"
        )
    return adapters, decoders


__all__ = [
    "CORNER_GRID_SIZE",
    "DECODER_HIDDEN_CHANNELS",
    "EXPECTED_MAINLINE_TRAINABLE_NEW_PARAMETERS",
    "EXPECTED_NEW_PARAMETERS",
    "MAINLINE_SCALES",
    "PYRAMID_INPUT_CHANNELS",
    "REGISTERED_SCALES",
    "SCALE_SPECS",
    "DownBlock",
    "PyramidAdapter",
    "ResidualGeometryDecoder",
    "ScaleModuleSpec",
    "build_multiscale_modules",
    "count_new_parameters",
]
