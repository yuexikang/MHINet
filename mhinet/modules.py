"""Trainable feature adapters and residual geometry decoders for MHINet v1.

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
ADAPTIVE_POOL_SIZE: Final = (4, 4)
EXPECTED_NEW_PARAMETERS: Final = 2_544_160


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
        decoder_input_channels=166,
        down_blocks=3,
        max_delta_px=32.0,
    ),
    4: ScaleModuleSpec(
        scale=4,
        spatial_size=196,
        adapter_channels=64,
        correlation_candidates=81,
        decoder_input_channels=166,
        down_blocks=4,
        max_delta_px=16.0,
    ),
    2: ScaleModuleSpec(
        scale=2,
        spatial_size=392,
        adapter_channels=32,
        correlation_candidates=49,
        decoder_input_channels=102,
        down_blocks=5,
        max_delta_px=6.0,
    ),
    1: ScaleModuleSpec(
        scale=1,
        spatial_size=784,
        adapter_channels=32,
        correlation_candidates=25,
        decoder_input_channels=54,
        down_blocks=6,
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
    """The fixed 64-channel residual stride-2 block from the v1 design."""

    def __init__(self, channels: int = DECODER_HIDDEN_CHANNELS) -> None:
        super().__init__()
        if channels <= 0 or channels % 8 != 0:
            raise ValueError("DownBlock channels must be positive and divisible by 8")
        self.channels = int(channels)
        self.conv1 = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(8, self.channels)
        self.activation1 = nn.GELU()
        self.conv2 = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(8, self.channels)
        self.skip = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            stride=2,
            bias=False,
        )
        self.activation2 = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.conv1, self.conv2, self.skip):
            _init_kaiming(layer)
        for norm in (self.norm1, self.norm2):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"DownBlock expects N,{self.channels},H,W, got {tuple(inputs.shape)}"
            )
        residual = self.skip(inputs)
        output = self.activation1(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation2(output + residual)


class ResidualGeometryDecoder(nn.Module):
    """Decode a dense correlation tensor into four bounded corner residuals."""

    def __init__(
        self,
        input_channels: int,
        down_blocks: int,
        max_delta_px: float,
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

        self.stem_conv = nn.Conv2d(
            self.input_channels,
            DECODER_HIDDEN_CHANNELS,
            kernel_size=1,
            bias=False,
        )
        self.stem_norm = nn.GroupNorm(8, DECODER_HIDDEN_CHANNELS)
        self.stem_activation = nn.GELU()
        self.down_blocks = nn.ModuleList(
            DownBlock(DECODER_HIDDEN_CHANNELS) for _ in range(self.num_down_blocks)
        )
        self.pool = nn.AdaptiveAvgPool2d(ADAPTIVE_POOL_SIZE)
        self.fc1 = nn.Linear(DECODER_HIDDEN_CHANNELS * 4 * 4, 256, bias=True)
        self.fc1_activation = nn.GELU()
        self.fc2 = nn.Linear(256, 8, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_kaiming(self.stem_conv)
        nn.init.ones_(self.stem_norm.weight)
        nn.init.zeros_(self.stem_norm.bias)
        # DownBlock constructors own their initialization.  Reinitializing here
        # keeps reset_parameters() semantically complete when called explicitly.
        for block in self.down_blocks:
            block.reset_parameters()
        _init_kaiming(self.fc1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
        if decoder_input.ndim != 4 or decoder_input.shape[1] != self.input_channels:
            raise ValueError(
                f"Decoder expects N,{self.input_channels},H,W, got "
                f"{tuple(decoder_input.shape)}"
            )
        output = self.stem_activation(self.stem_norm(self.stem_conv(decoder_input)))
        for block in self.down_blocks:
            output = block(output)
        output = self.pool(output).flatten(start_dim=1)
        output = self.fc1_activation(self.fc1(output))
        logits = self.fc2(output).reshape(-1, 4, 2)
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
    "ADAPTIVE_POOL_SIZE",
    "DECODER_HIDDEN_CHANNELS",
    "EXPECTED_NEW_PARAMETERS",
    "PYRAMID_INPUT_CHANNELS",
    "SCALE_SPECS",
    "DownBlock",
    "PyramidAdapter",
    "ResidualGeometryDecoder",
    "ScaleModuleSpec",
    "build_multiscale_modules",
    "count_new_parameters",
]
