"""Quadrant recurrent refinement. This module has no H0 input or H0 mask."""
import torch
from torch import nn
from .supervision import sample_uv, offsets, interpolate_controls


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self,x):
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class QRRU(nn.Module):
    def __init__(self, channels=256, iterations=4, radius=1.5):
        super().__init__()
        if iterations<1 or radius<=0:
            raise ValueError('Invalid QRRU iteration/radius')
        self.iterations,self.radius = iterations,radius
        self.project = nn.Conv2d(channels,48,1)
        self.encoder = nn.Sequential(nn.Conv2d(128,64,3,padding=1),ChannelNorm(64),nn.GELU(),
            nn.Conv2d(64,64,3,padding=1,groups=64),nn.Conv2d(64,32,1),nn.GELU())
        self.gate = nn.Conv2d(64,32,1)
        self.down = nn.Conv2d(32,32,2,stride=2)
        self.head = nn.Sequential(nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,3,1))
        # Start from the supplied correspondence; audit upstream gradients after step two.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, feature_a, feature_b, points_a_uv, points_b_uv, *, projected=False):
        if not projected:
            feature_a = self.project(feature_a[None])[0]
            feature_b = self.project(feature_b[None])[0]
        n = len(points_a_uv)
        off = offsets(points_a_uv.device)
        ga = points_a_uv[:,None,None,:]+off
        gb0 = points_b_uv[:,None,None,:]+off
        wa = sample_uv(feature_a,ga).permute(0,3,1,2)
        h = wa.new_zeros((n,32,4,4))
        flow = wa.new_zeros((n,2,2,2))
        flows,centers,grids,gates = [],[],[],[]
        for _ in range(self.iterations):
            gb = gb0+interpolate_controls(flow,off)
            wb = sample_uv(feature_b,gb).permute(0,3,1,2)
            z = self.encoder(torch.cat((h,wa,wb),1))
            g = torch.sigmoid(self.gate(torch.cat((h,z),1)))
            h = (1-g)*h+g*z
            out = self.head(self.down(h))
            flow = flow+torch.sigmoid(out[:,2:3])*self.radius*torch.tanh(out[:,:2])
            flows.append(flow)
            gates.append(torch.sigmoid(out[:,2:3]))
            centers.append(points_b_uv+flow.mean((-2,-1)))
            grids.append(gb0+interpolate_controls(flow,off))
        return dict(flows=torch.stack(flows),centers=torch.stack(centers),grids=torch.stack(grids),gates=torch.stack(gates))
