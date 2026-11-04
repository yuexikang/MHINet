"""QRRU local coordinate refinement after Pair-D2, in native image pixels."""
import torch
from torch import nn

from mhinet.downstream.qrru import ChannelNorm
from mhinet.downstream.supervision import offsets, sample_uv


class LocalRefinement(nn.Module):
    """Predict four bounded control offsets and apply their mean to the D2 point."""
    def __init__(self, channels=256, radius=1.5, chunk_size=512):
        super().__init__()
        self.radius, self.chunk_size = radius, chunk_size
        # Keep the trained parameter layout; the zero context preserves its input channels.
        self.project = nn.Conv2d(channels, 48, 1)
        self.encoder = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), ChannelNorm(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 32, 1), nn.GELU())
        self.gate = nn.Conv2d(64, 32, 1)
        self.down = nn.Conv2d(32, 32, 2, stride=2)
        self.head = nn.Sequential(nn.Conv2d(32, 32, 1), nn.GELU(), nn.Conv2d(32, 3, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, points_a, points_b):
        """features: 2×256×H2×W2; xy points and result: N×2 original pixels."""
        if not len(points_a):
            return points_b + self.head[-1].weight.sum()*0
        projected = self.project(features.float())
        au, bu = (points_a+.5)/2-.5, (points_b+.5)/2-.5
        off = offsets(au.device)
        updates = []
        for start in range(0, len(au), self.chunk_size):
            stop = start+self.chunk_size
            wa = sample_uv(projected[0], au[start:stop, None, None]+off).permute(0, 3, 1, 2)
            wb = sample_uv(projected[1], bu[start:stop, None, None]+off).permute(0, 3, 1, 2)
            context = wa.new_zeros((len(wa), 32, 4, 4))
            z = self.encoder(torch.cat((context, wa, wb), 1))
            gate = torch.sigmoid(self.gate(torch.cat((context, z), 1)))
            out = self.head(self.down(gate*z))
            controls = torch.sigmoid(out[:, 2:3])*self.radius*torch.tanh(out[:, :2])
            updates.append(controls.mean((-2, -1))*2)
        return points_b + torch.cat(updates)


def local_refinement_loss(before, after, target, valid, harm_weight=1.):
    """Mean endpoint error plus a penalty for worsening, both in original pixels."""
    e0 = (before.detach()[valid]-target[valid]).norm(dim=-1)
    e1 = (after[valid]-target[valid]).norm(dim=-1)
    if not len(e1):
        zero = after.sum()*0
        return zero, dict(localization=zero, harm=zero)
    localization, harm = e1.mean(), (e1-e0).relu().mean()
    return localization+harm_weight*harm, dict(localization=localization, harm=harm)
