"""D16 positive regions, explicit ignore queries, and controlled absent-image negatives."""
import math
import torch
import torch.nn.functional as F
from mhinet.v2.losses import flow_head_loss
from mhinet.downstream.supervision import valid_points


def grid(device):
    y,x=torch.meshgrid((torch.arange(32,device=device)+.5)*16-.5,
                       (torch.arange(32,device=device)+.5)*16-.5,indexing='ij')
    return torch.stack((x,y),-1).reshape(-1,2)


def targets(gt):
    """No inferred negatives from geographic H. D16 cells cover full image edges."""
    xy=grid(gt.H.device)
    destination,known,visible=gt.query((xy+.5)/256-1)
    target=(destination+1)*256-.5
    candidate_valid=valid_points(gt.mask_b,(xy+.5)/256-1)
    if gt.exact:
        # Original subpixel four-neighbour soft labels, including edge clamping.
        q=(target+.5)/16-.5;low=q.floor();fraction=q-low
        weights=target.new_zeros((1024,1024))
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            index=(low[:,1].long()+dy).clamp(0,31)*32+(low[:,0].long()+dx).clamp(0,31)
            w=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
            weights.scatter_add_(1,index[:,None],w[:,None])
        weights*=candidate_valid[None]
        region=weights>0
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    else:
        # A candidate represents its 16px cell footprint. An uncertainty box can
        # intersect several cells even when no cell centre is inside that box.
        region=((xy[None]-target[:,None]).abs() <= 8+gt.tolerance_px).all(-1)
        region &= candidate_valid[None]
        weights=None
    positive=known & visible & region.any(-1)
    legacy_positive=known & visible & (destination.abs()<=31/32).all(-1)
    return dict(source=xy,target=target,region=region,weights=weights,positive=positive,
                legacy_positive=legacy_positive,unknown=~positive,
                boundary_legacy_negative=positive & ~legacy_positive)


def correspondence_loss(stage,gt,label,config):
    if label=='U':
        return flow_head_loss(stage,geometry=gt)
    geo,old=flow_head_loss(stage,geometry=gt,mat_weight=0.,cls_weight=0.)
    t=targets(gt);positive=t['positive'];region=t['region'][positive]
    logp=stage['match_probabilities'][0].float().clamp_min(1e-30).log()
    values=logp[positive,:1024]
    zero=stage['coarse_warp'].sum()*0
    if bool(positive.any()):
        pos_values=values.masked_fill(~region,-torch.inf)
        if gt.exact:
            position=-(values*t['weights'][positive]).sum(-1).mean()
        else:
            position=-torch.logsumexp(pos_values,-1).mean()
        negatives=values.masked_fill(region,-torch.inf).topk(config['hard_topk'],-1).values
        hard=F.softplus(torch.logsumexp(negatives,-1)-math.log(config['hard_topk'])
                        -pos_values.max(-1).values+config['hard_margin']).mean()
        mat=.5*(-stage['coarse_matchability'][0,0].flatten()[positive].clamp(1e-6,1-1e-6).log()).mean()
    else:
        position=hard=mat=zero
    total=geo+config['position_weight']*position+config['hard_weight']*hard+config['mat_weight']*mat
    return total,dict(old,position=float(position.detach()),hard=float(hard.detach()),
                      positive_queries=int(positive.sum()),negative_queries=0,ignore_queries=int(t['unknown'].sum()))


def absent_image_loss(stage,source_mask,config):
    """B has been completely erased: no observable B correspondence exists.

    This is a controlled negative, not inferred real occlusion from approximate H.
    """
    xy=grid(stage['coarse_warp'].device)
    valid=valid_points(source_mask,(xy+.5)/256-1)
    p=stage['match_probabilities'][0,valid,-1].float().clamp_min(1e-30)
    mat=stage['coarse_matchability'][0,0].flatten()[valid].float().clamp(1e-6,1-1e-6)
    if not len(p):return stage['coarse_warp'].sum()*0
    return config['position_weight']*(-p.log().mean())+config['mat_weight']*.5*(-torch.log1p(-mat).mean())


@torch.no_grad()
def evaluate(stage,gt):
    t=targets(gt);keep=t['positive'];region=t['region'][keep]
    pred=(stage['coarse_warp'][0].float().permute(1,2,0).reshape(-1,2)+1)*256-.5
    epe=(pred-t['target']).norm(dim=-1)[keep]
    prob=stage['match_probabilities'][0,:1024,:1024].float()[keep]
    pos_best=prob.masked_fill(~region,-1).max(-1).values
    negative_best=prob.masked_fill(region,-1).max(-1).values
    rank=(prob>pos_best[:,None]).sum(-1)+1
    result=dict(points=len(epe),epe=float(epe.mean()) if len(epe) else None,
                hard_confusion=float((negative_best>pos_best).float().mean()) if len(epe) else None,
                ignore_queries=int(t['unknown'].sum()),real_negative_queries=0,
                legacy_border_false_negative=int(t['boundary_legacy_negative'].sum()))
    for threshold in (1,3,5,10):result[f'flow_p{threshold}']=float((epe<=threshold).float().mean()) if len(epe) else None
    for threshold in (1,5,10):result[f'rank_at{threshold}']=float((rank<=threshold).float().mean()) if len(epe) else None
    for threshold in (8,16,32):result[f'epe_gt{threshold}']=float((epe>threshold).float().mean()) if len(epe) else None
    # Keep the previous centre-bound metric to expose denominator changes.
    old=(pred-t['target']).norm(dim=-1)[t['legacy_positive']]
    result['legacy_epe']=float(old.mean()) if len(old) else None
    return result


@torch.no_grad()
def absent_image_metrics(stage,mask):
    xy=grid(stage['coarse_warp'].device);valid=valid_points(mask,(xy+.5)/256-1)
    p=stage['match_probabilities'][0,valid].float();mat=stage['coarse_matchability'][0,0].flatten()[valid]
    return dict(dustbin_top1=float((p.argmax(-1)==1024).float().mean()),
                dustbin_probability=float(p[:,-1].mean()),matchability_false_positive=float((mat>=.5).float().mean()))


@torch.no_grad()
def diagnostic(stage,gt):
    t=targets(gt);keep=t['positive'];pred=(stage['coarse_warp'][0].float().permute(1,2,0).reshape(-1,2)+1)*256-.5
    error=(pred-t['target']).norm(dim=-1);error=error.masked_fill(~keep,-1)
    selected=error.topk(min(8,int(keep.sum()))).indices
    p=stage['match_probabilities'][0].float();records=[]
    for i in selected.tolist():
        prob,index=p[i,:1024].topk(10);positive=t['region'][i]
        best=p[i,:1024].masked_fill(~positive,-1).max()
        records.append(dict(query_index=i,source_xy=t['source'][i].tolist(),reference_xy=t['target'][i].tolist(),
            predicted_xy=pred[i].tolist(),epe=float(error[i]),reference_region_rank=int((p[i,:1024]>best).sum()+1),
            dustbin_probability=float(p[i,-1]),top10=[dict(index=int(j),xy=t['source'][j].tolist(),probability=float(v),
                in_positive_region=bool(positive[j])) for v,j in zip(prob,index)]))
    return records,t,pred
