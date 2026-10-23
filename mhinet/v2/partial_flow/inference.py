"""Additive output API. U0 dense predictions remain available unchanged."""
import torch
from .labels import MATCH, NO_MATCH, UNKNOWN
from .nomatch_head import query_features
from mhinet.v2.feature_provider import normalized_d16_grid


def conditional_output(target_norm, no_match_prob, low, high):
    if not low < high:
        raise ValueError('calibrated low must be below high')
    batch = target_norm.shape[0]
    p = no_match_prob.reshape(batch,32,32)
    state = torch.full_like(p, UNKNOWN, dtype=torch.long)
    decision_p = p.double()  # preserve calibrated float64 tie boundaries
    state[decision_p <= low] = MATCH
    state[decision_p >= high] = NO_MATCH
    indices = (state.flatten() == MATCH).nonzero().flatten()
    target = ((target_norm.permute(0,2,3,1).reshape(-1,2)+1)*256-.5)[indices]
    source = ((normalized_d16_grid(batch,device=p.device,dtype=target.dtype)
               .permute(0,2,3,1).reshape(-1,2)+1)*256-.5)[indices]
    # Global indices encode batch*1024+row*32+column; no batch identity is lost.
    return dict(state=state, no_match_prob=p[:,None], source_xy=source,
                target_xy=target, flow_xy=target-source, accepted_indices=indices)


@torch.no_grad()
def predict(u0, head, images, thresholds):
    dense = u0(images,pyramid_scales=())
    features, _ = query_features(dense,u0.stage1_head.no_match_logit)
    output = conditional_output(dense['flow16_target_norm'],head(features).sigmoid(),*thresholds)
    return output, dense
