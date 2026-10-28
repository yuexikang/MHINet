"""Equal-context modulation, followed by 16-query/64-candidate D2 matching."""
import torch
from torch import nn
import torch.nn.functional as F
from mhinet.v2.flow_d8_d2.matching import select_local_parents
from mhinet.v2.joint_single.path import coarse, sample, decode


def center_indices(device):
    axis = torch.arange(2, 6, device=device)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return (y * 8 + x).flatten()


class LocalPairModulation(nn.Module):
    """A/B share weights and simultaneously use ORIGINAL masked context means."""
    def __init__(self, mode='pair', channels=256, hidden=64):
        super().__init__()
        if mode not in ('none', 'self', 'pair'):
            raise ValueError(mode)
        self.mode = mode
        self.condition = nn.Sequential(nn.Linear(3*channels, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 2*channels))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    @staticmethod
    def mean(features, valid):
        return (features * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)

    def forward(self, a, b, valid_a, valid_b):
        if self.mode == 'none':
            return a, b
        sa, sb = self.mean(a, valid_a), self.mean(b, valid_b)
        def adapt(x, own, other):
            if self.mode == 'self':
                other = own
            z = torch.cat((own, other, (own-other).abs()), -1)
            gamma, beta = self.condition(z).chunk(2, -1)
            return x + gamma[:, None]*x + beta[:, None]
        return adapt(a, sa, sb), adapt(b, sb, sa)


@torch.no_grad()
def extract_windows(shared, matcher, max_parents=128):
    """Fixed legacy source-parent selection; continuous frozen D8 target centers.

    No geometry labels enter selection, feature sampling, or context masks.
    """
    selected = select_local_parents(shared, matcher, max_parents)
    r = coarse(shared, matcher, selected)
    axis = (torch.arange(8, device=selected.device)-3.5)*2
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    delta = torch.stack((x, y), -1).reshape(1, 64, 2)
    ca, cb = selected[:, None]+delta, r['p8'][:, None]+delta
    d2 = shared['pyramid'][2][0].float()
    index = center_indices(selected.device)
    source = ca[:, index]
    return dict(a=sample(d2[0], ca), b=sample(d2[1], cb),
                valid_a=(ca >= 0).all(-1) & (ca <= 511).all(-1),
                valid_b=(cb >= 0).all(-1) & (cb <= 511).all(-1),
                source=source, candidates=cb,
                flow=source+sample(shared['flow16_norm'][0], source)*256,
                score8=r['score8'])


def match_windows(adapter, windows, temperature):
    a, b = adapter(windows['a'], windows['b'], windows['valid_a'], windows['valid_b'])
    index = center_indices(a.device)
    a, b = F.normalize(a[:, index], dim=-1), F.normalize(b, dim=-1)
    scores = a @ b.transpose(-1, -2) / temperature
    valid = windows['valid_a'][:, index, None] & windows['valid_b'][:, None, :]
    scores = scores.masked_fill(~valid, -1e4)
    logp = scores.log_softmax(-1)+scores.log_softmax(-2)
    candidates = windows['candidates'][:, None].expand(-1, 16, -1, -1).reshape(-1, 64, 2)
    point, confidence, _, _ = decode(logp.reshape(-1, 64), candidates, 8)
    return dict(points_b=point.reshape(-1, 16, 2), logp=logp,
                confidence=confidence.reshape(-1, 16), valid=valid)


@torch.no_grad()
def infer(shared, matcher, adapter):
    """Drop-in experimental inference without labels, thresholds, or QRRU."""
    windows = extract_windows(shared, matcher)
    result = match_windows(adapter, windows, matcher.tau_f)
    return dict(points_a=windows['source'].reshape(-1, 2),
                points_b=result['points_b'].reshape(-1, 2),
                confidence=(result['confidence']*windows['score8'][:, None]).sqrt().flatten())


def load_adapter(checkpoint, device='cuda:0'):
    """Load a trained adapter for infer(); use the same frozen matcher checkpoint."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    adapter = LocalPairModulation(state['config']['mode']).to(device)
    adapter.load_state_dict(state['adapter'])
    return adapter.requires_grad_(False).eval()
