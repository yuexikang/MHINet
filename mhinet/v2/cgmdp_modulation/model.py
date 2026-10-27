"""Modulate accumulated descriptions or decoder increments, leaving context intact."""
import torch
from torch import nn
import torch.nn.functional as F
from mhinet.v2.d2_modulation.model import LocalPairModulation


class GlobalPairModulation(LocalPairModulation):
    def forward(self, feature):
        # One A/B pair per batch item; no correspondence, overlap mask or GT is needed.
        b2, channels, h, w = feature.shape
        values = feature.reshape(b2//2, 2, channels, h*w).permute(0, 1, 3, 2).float()
        a, b = values.unbind(1)
        valid = torch.ones(a.shape[:2], device=a.device, dtype=torch.bool)
        a, b = super().forward(a, b, valid, valid)
        return torch.stack((a, b), 1).permute(0, 1, 3, 2).reshape_as(feature).to(feature.dtype)


class ModulatedCGMDP(nn.Module):
    """Frozen base; only three full-map conditioning MLPs are trained.

    The descriptor sum and decoder context are separate paths. Context/delta
    computation stays frozen; accumulated descriptions retain modulation grads
    through the ordinary bilinear upsampling/addition chain.
    """
    def __init__(self, base, mode):
        super().__init__()
        if mode not in ('none', 'cumulative', 'delta'): raise ValueError(mode)
        self.base = base.requires_grad_(False).eval()
        self.mode = mode
        self.modulators = nn.ModuleDict({s: GlobalPairModulation('pair') for s in ('8', '4', '2')})

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            shared = self.base(images, pyramid_scales=())
            features, sizes = self.base.vgg(images.flatten(0, 1))
            context16 = shared['context'].flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        features = features+[context16]; sizes = sizes+[context16.shape[-2:]]
        descriptions, context, pyramid = 0., None, {}
        for index, (feature, scale) in enumerate(zip(reversed(features), self.base.dedode.scales)):
            with torch.no_grad():
                delta, context = self.base.dedode(feature, scale=scale, context=context)
            if scale in self.modulators and self.mode == 'delta':
                delta = self.modulators[scale](delta)
            descriptions = descriptions+delta
            if scale in self.modulators and self.mode == 'cumulative':
                descriptions = self.modulators[scale](descriptions)
            if scale in self.modulators:
                pyramid[int(scale)] = descriptions.reshape(len(images), 2, *descriptions.shape[1:])
            if scale == '2': break
            next_size = sizes[-index-2]
            descriptions = F.interpolate(descriptions, size=next_size, mode='bilinear', align_corners=False)
            context = F.interpolate(context, size=next_size, mode='bilinear', align_corners=False)
        shared['pyramid'] = pyramid
        return shared
