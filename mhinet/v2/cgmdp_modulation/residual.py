"""Bounded delta-only adaptation; frozen decoder increments are computed once per pair."""
import torch
from torch import nn
import torch.nn.functional as F
from mhinet.v2.d2_modulation.model import LocalPairModulation


def channel_parameters(module, feature):
    means = feature.float().mean((-2, -1)).reshape(-1, 2, feature.shape[1])
    own = means.flatten(0, 1)
    other = means.flip(1).flatten(0, 1)
    return module.condition(torch.cat((own, other, (own-other).abs()), -1)).chunk(2, -1)


def parameter_stats(feature, gamma, beta):
    norm = feature.float().norm(dim=1).mean((-2, -1)).clamp_min(1e-8)
    return dict(gamma_abs=float(gamma.detach().abs().mean()),
                gamma_max=float(gamma.detach().abs().max()),
                beta_ratio=float((beta.detach().norm(dim=-1)/norm).mean()))


class BoundedDelta(LocalPairModulation):
    def __init__(self, channel_rms, config):
        super().__init__('pair')
        self.register_buffer('beta_scale', torch.tensor(channel_rms)*config['beta_rms_fraction'])
        self.gamma_bound = config['gamma_bound']
        self.strength = config['residual_strength']

    def forward(self, feature):
        raw_gamma, raw_beta = channel_parameters(self, feature)
        gamma = self.gamma_bound*raw_gamma.tanh()
        beta = self.beta_scale*raw_beta.tanh()
        output = feature + self.strength*(gamma[..., None, None]*feature+beta[..., None, None])
        stats = parameter_stats(feature, gamma, beta)
        stats['lambda'] = self.strength
        return output.to(feature.dtype), stats


class ResidualCGMDP(nn.Module):
    def __init__(self, base, scales, calibration, config):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.modulators = nn.ModuleDict({str(s): BoundedDelta(calibration[str(s)]['channel_rms'], config) for s in scales})

    def train(self, mode=True):
        super().train(mode); self.base.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        shared = self.base(images, pyramid_scales=())
        features, sizes = self.base.vgg(images.flatten(0, 1))
        context16 = shared['context'].flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        features = features+[context16]
        context, deltas = None, {}
        for index, (feature, scale) in enumerate(zip(reversed(features), self.base.dedode.scales)):
            delta, context = self.base.dedode(feature, scale=scale, context=context)
            deltas[scale] = delta
            if scale == '2': break
            context = F.interpolate(context, size=features[-index-2].shape[-2:], mode='bilinear', align_corners=False)
        return shared, deltas

    def decode(self, shared, deltas):
        description, baseline = None, None
        pyramid, original, stats, drift = {}, {}, {}, []
        for scale, delta in deltas.items():
            if description is None:
                description, baseline = delta, delta
            else:
                description = F.interpolate(description, size=delta.shape[-2:], mode='bilinear', align_corners=False)
                baseline = F.interpolate(baseline, size=delta.shape[-2:], mode='bilinear', align_corners=False)+delta
                if scale in self.modulators:
                    changed, stats[scale] = self.modulators[scale](delta)
                    description = description+changed
                else: description = description+delta
            if scale in ('8', '4', '2'):
                ratio = (description.float()-baseline.float()).square().mean()/baseline.float().square().mean().clamp_min(1e-8)
                drift.append(ratio)
                stats.setdefault(scale, {})['drift'] = float(ratio.detach())
                pyramid[int(scale)] = description.reshape(-1, 2, *description.shape[1:])
                original[int(scale)] = baseline.reshape(-1, 2, *baseline.shape[1:])
        return dict(shared, pyramid=pyramid), dict(shared, pyramid=original), sum(drift)/len(drift), stats
