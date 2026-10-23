"""Independent query classifier; all U0 inputs are detached."""
import torch
from torch import nn
import torch.nn.functional as F


class NoMatchHead(nn.Sequential):
    def __init__(self, input_dim=1029):
        super().__init__(nn.Linear(input_dim,128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128,1))

    def forward(self, features):
        return super().forward(features.detach()).squeeze(-1)


def query_features(result, no_match_logit):
    p = result['stage1']['match_probabilities'].detach().float()
    spatial, dust = p[..., :-1], p[..., -1]
    top = spatial.topk(2, dim=-1).values
    log_top = top.clamp_min(1e-30).log()
    logits_max = no_match_logit.detach().float() + log_top[...,0] - dust.clamp_min(1e-30).log()
    gap = log_top[...,0]-log_top[...,1]
    conditional = spatial / spatial.sum(-1,keepdim=True).clamp_min(1e-30)
    entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(-1)
    mat = result['flow16_matchability'].detach().float().flatten(1)
    f = result['context'][:,0].detach().float().flatten(1,2)
    # Fixed L2 normalization; no fitted feature statistics and no calibration leak.
    features = torch.cat((F.normalize(f,dim=-1), torch.stack((dust,mat,logits_max,gap,entropy),-1)), -1)
    return features, dict(p_dustbin=dust, matchability=mat, p_spatial_max=top[...,0],
        dustbin_logit_margin=dust.clamp_min(1e-30).log()-log_top[...,0],
        spatial_entropy=entropy, top1_top2_margin=gap, dustbin_top1=dust>top[...,0])
