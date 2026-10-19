"""Discrepancy-guided local flow propagation on the existing QRRU patch."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class LocalDGFO(nn.Module):
    """Attend to valid local flow values; retain the input at empty queries.

    ``valid_mask`` describes real source AND warped-target samples. Unfolding
    that mask with zero padding additionally excludes positions outside the
    patch. Returned attention, when requested, has shape [N, K, H, W].
    """

    def __init__(self, channels=32, dim=32, radius=1):
        super().__init__()
        if channels < 1 or dim < 1 or radius != 1:
            raise ValueError("DGFO requires positive channels/dim and radius=1")
        self.channels, self.dim, self.radius = channels, dim, radius
        self.query = nn.Conv2d(channels, dim, 1)
        self.key = nn.Conv2d(channels, dim, 1)

    def forward(self, discrepancy, local_flow, valid_mask=None, *, return_attention=False):
        if discrepancy.ndim != 4 or discrepancy.shape[1] != self.channels:
            raise ValueError("discrepancy must have shape [N, channels, H, W]")
        n, _, h, w = discrepancy.shape
        if local_flow.shape != (n, 2, h, w):
            raise ValueError("local_flow must have shape [N, 2, H, W]")
        if valid_mask is None:
            valid = torch.ones((n, h, w), dtype=torch.bool, device=local_flow.device)
        else:
            if valid_mask.shape == (n, 1, h, w):
                valid_mask = valid_mask[:, 0]
            if valid_mask.shape != (n, h, w) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [N, H, W] or [N, 1, H, W]")
            valid = valid_mask
        if n == 0:
            attention = local_flow.new_empty((0, 9, h, w))
            return (local_flow, attention) if return_attention else local_flow
        valid = valid & torch.isfinite(discrepancy).all(1) & torch.isfinite(local_flow).all(1)
        safe_features = torch.where(valid[:, None], discrepancy, torch.zeros_like(discrepancy))
        q = self.query(safe_features).flatten(2)[:, :, None]
        k = F.unfold(self.key(safe_features), 3, padding=1).reshape(n, self.dim, 9, h*w)
        logits = (q.float()*k.float()).sum(1) / math.sqrt(self.dim)
        neighbors = F.unfold(valid[:, None].float(), 3, padding=1).bool()
        eligible = neighbors & valid.flatten(1)[:, None]
        # A finite sentinel, explicit zeroing, and renormalization keep empty
        # neighborhoods finite while giving them exactly zero attention.
        weights = torch.softmax(logits.masked_fill(~eligible, torch.finfo(logits.dtype).min), 1)
        weights = weights * eligible
        weights = weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        safe_flow = torch.where(valid[:, None], local_flow, torch.zeros_like(local_flow))
        values = F.unfold(safe_flow, 3, padding=1).reshape(n, 2, 9, h*w)
        propagated = (values*weights[:, None].to(values.dtype)).sum(2).reshape(n, 2, h, w)
        has_neighbors = eligible.any(1).reshape(n, 1, h, w)
        propagated = torch.where(has_neighbors, propagated, local_flow)
        if return_attention:
            return propagated, weights.reshape(n, 9, h, w)
        return propagated
