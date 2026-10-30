"""Validated CGMDP D2 depth extension; descriptor dimensions stay unchanged."""
import torch
from torch import nn


class SpatialResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 64, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class DeepD2(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.extra = nn.Sequential(SpatialResidual(), SpatialResidual())

    def forward(self, features):
        with torch.autocast(features.device.type, enabled=self.base.amp,
                            dtype=self.base.amp_dtype):
            x0 = self.base.block1(features)
            x = (self.base.hidden_blocks(x0) + x0) / 1.4
            return self.base.out_conv(self.extra(x))


def configure_d2(shared, refinement='none'):
    if refinement == 'residual_deep':
        base = shared.dedode.layers['2']
        shared.dedode.layers['2'] = DeepD2(base).to(base.out_conv.weight)
    elif refinement != 'none':
        raise ValueError(f'Unsupported D2 refinement: {refinement}')
