"""Cycle in original pixels, sampling the reverse absolute coordinate map."""
import torch
import torch.nn.functional as F
from mhinet.v2.feature_provider import normalized_d16_grid


def cycle_consistency(forward, backward, sigma_px=16.):
    ab=forward['flow16_target_norm'].float();ba=backward['flow16_target_norm'].float()
    finite=torch.isfinite(ab).all(1,keepdim=True)
    grid=torch.where(torch.isfinite(ab),ab,torch.zeros_like(ab)).permute(0,2,3,1)
    reverse_valid=backward['flow16_valid'] & torch.isfinite(ba).all(1,keepdim=True)
    def sample(value):
        return F.grid_sample(value,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    back=sample(torch.where(reverse_valid,ba,torch.zeros_like(ba)))
    # Absolute maps cannot be extrapolated beyond the outer D16 centres.
    interior=(grid.abs()<=31/32).all(-1)[:,None]
    valid=forward['flow16_valid'] & finite & interior & (sample(reverse_valid.float())>=1-1e-6)
    valid &= torch.isfinite(back).all(1,keepdim=True)
    source=normalized_d16_grid(len(ab),device=ab.device,dtype=ab.dtype)
    error=(back-source).norm(dim=1,keepdim=True)*256
    error=error.masked_fill(~valid,float('inf'))
    reliability=torch.exp(-.5*(error/sigma_px).square())
    reverse_mat=sample(torch.nan_to_num(backward['flow16_matchability'].float())).masked_fill(~valid,0.)
    return dict(error_px=error,valid=valid,reliability=reliability,reverse_matchability=reverse_mat,
                boundary_or_outside=~interior,returned_target=back)


def distribution_features(result):
    p=result['stage1']['match_probabilities'][...,:1024].float()
    p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)/torch.log(p.new_tensor(1024.))
    top=p.topk(2,-1).values
    return torch.stack((entropy,top[...,0]-top[...,1]),1).reshape(-1,2,32,32)


def reliability_features(forward,backward,cycle):
    a=distribution_features(forward);b=distribution_features(backward)
    grid=torch.nan_to_num(forward['flow16_target_norm']).permute(0,2,3,1)
    b=F.grid_sample(b,grid,align_corners=False).masked_fill(~cycle['valid'],0.)
    return torch.cat((forward['flow16_matchability'].float(),cycle['reverse_matchability'],
        (cycle['error_px']/32).clamp(0,1),cycle['valid'].float(),a,b),1).detach()


def modulate(shared,cycle,variant):
    """Only change the reward weight; never remove D8 candidates."""
    m=shared['flow16_matchability'];r=cycle['reliability'];result=dict(shared)
    if variant=='B2':result['flow16_matchability']=m*r
    elif variant=='B3':result['flow16_matchability']=r
    elif variant=='B4_uniform':
        valid=shared['flow16_valid']
        alpha=(m*r)[valid].sum()/m[valid].sum().clamp_min(1e-12)
        result['flow16_matchability']=m*alpha
    return result
