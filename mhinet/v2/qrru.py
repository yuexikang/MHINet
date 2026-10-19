"""Version-two QRRU with optional zero-initialized discrepancy/DGFO branches."""
import torch
import torch.nn.functional as F
from torch import nn

from mhinet.downstream.qrru import QRRU
from mhinet.downstream.supervision import offsets, interpolate_controls, sample_uv
from .dgfo import LocalDGFO


def _inside_samples(points, hw):
    """Conservative validity in D2 pixel-center coordinates."""
    h, w = hw
    return (torch.isfinite(points).all(-1) & (points >= 0).all(-1)
            & (points <= points.new_tensor([w-1, h-1])).all(-1))


class QRRUV2(QRRU):
    """Preserve all v1 parameter names, shapes, and control-point geometry.

    Both adapters start at zero. Disabling the branches registers no extra
    parameters and directly calls the original implementation. No H0 or GT
    geometry enters this module.
    """

    def __init__(self, channels=256, iterations=4, radius=1.5, *,
                 use_discrepancy=False, use_dgfo=False, dgfo_radius=1, dgfo_dim=32):
        if use_dgfo and not use_discrepancy:
            raise ValueError("DGFO requires use_discrepancy=True")
        super().__init__(channels, iterations, radius)
        self.use_discrepancy, self.use_dgfo = use_discrepancy, use_dgfo
        if use_discrepancy:
            self.diff_encoder = nn.Sequential(nn.Conv2d(48, 32, 1), nn.GroupNorm(8, 32), nn.GELU())
            self.diff_adapter = nn.Conv2d(32, 32, 1)
            nn.init.zeros_(self.diff_adapter.weight)
            nn.init.zeros_(self.diff_adapter.bias)
        if use_dgfo:
            self.dgfo = LocalDGFO(32, dgfo_dim, dgfo_radius)
            # Bias must stay absent: a zero propagation residual has zero effect
            # even after training, in particular during the initial iteration.
            self.flow_adapter = nn.Conv2d(2, 32, 1, bias=False)
            nn.init.zeros_(self.flow_adapter.weight)

    def forward(self, feature_a, feature_b, points_a_uv, points_b_uv, *, projected=False):
        if not self.use_discrepancy and len(points_a_uv):
            return super().forward(feature_a, feature_b, points_a_uv, points_b_uv, projected=projected)
        if not projected:
            feature_a = self.project(feature_a[None])[0]
            feature_b = self.project(feature_b[None])[0]
        n = len(points_a_uv)
        if n == 0:
            # Empty matches are a legal inference result and remain connected to
            # input features if a caller accumulates an empty training loss.
            zero = (feature_a.sum()+feature_b.sum())*0
            empty = feature_a.new_empty
            return dict(flows=empty((self.iterations, 0, 2, 2, 2))+zero,
                        centers=empty((self.iterations, 0, 2))+zero,
                        grids=empty((self.iterations, 0, 4, 4, 2))+zero,
                        gates=empty((self.iterations, 0, 1, 2, 2))+zero)
        off = offsets(points_a_uv.device)
        ga = points_a_uv[:, None, None, :]+off
        gb0 = points_b_uv[:, None, None, :]+off
        wa = sample_uv(feature_a, ga).permute(0, 3, 1, 2)
        source_valid = _inside_samples(ga, feature_a.shape[-2:])
        h = wa.new_zeros((n, 32, 4, 4))
        flow = wa.new_zeros((n, 2, 2, 2))
        flows, centers, grids, gates = [], [], [], []
        for _ in range(self.iterations):
            # The controls lie at +/-1 whereas samples lie at +/-1.5. This
            # polynomial extrapolation must not be replaced by image resizing.
            local_flow = interpolate_controls(flow, off)
            gb = gb0+local_flow
            wb = sample_uv(feature_b, gb).permute(0, 3, 1, 2)
            z = self.encoder(torch.cat((h, wa, wb), 1))
            valid = source_valid & _inside_samples(gb, feature_b.shape[-2:])
            discrepancy = (F.normalize(wa, dim=1)-F.normalize(wb, dim=1)).abs()
            discrepancy = discrepancy * valid[:, None]
            representation = self.diff_encoder(discrepancy) * valid[:, None]
            z = z+self.diff_adapter(representation)*valid[:, None]
            if self.use_dgfo:
                flow_chw = local_flow.permute(0, 3, 1, 2)
                propagated = self.dgfo(representation, flow_chw, valid)
                z = z+self.flow_adapter(propagated-flow_chw)
            g = torch.sigmoid(self.gate(torch.cat((h, z), 1)))
            h = (1-g)*h+g*z
            out = self.head(self.down(h))
            flow = flow+torch.sigmoid(out[:, 2:3])*self.radius*torch.tanh(out[:, :2])
            flows.append(flow)
            gates.append(torch.sigmoid(out[:, 2:3]))
            centers.append(points_b_uv+flow.mean((-2, -1)))
            grids.append(gb0+interpolate_controls(flow, off))
        return dict(flows=torch.stack(flows), centers=torch.stack(centers),
                    grids=torch.stack(grids), gates=torch.stack(gates))
