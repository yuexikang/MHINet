"""Deterministic pixel-space B rectangles and conservative three-state labels."""
import math
import random
import zlib
import torch
import torch.nn.functional as F
from mhinet.v2.d16_experiment.supervision import targets


def label_rectangle(gt, rect, buffer=2.):
    t = targets(gt)
    # Pixel-edge coordinates; support includes D16 half-cell + H uncertainty + buffer.
    center = t['target']+.5
    radius = 8 + gt.tolerance_px + buffer
    low, high = center-radius, center+radius
    inside = (low>=0).all(-1) & (high<=512).all(-1)
    # Require every pixel touched by this support to be originally valid.
    mask = gt.mask_b.reshape(512,512) if gt.mask_b is not None else center.new_ones(512,512)
    invalid = F.pad((mask<.5).float().cumsum(0).cumsum(1),(1,0,1,0))
    lo = low.floor().long().clamp(0,512); hi = high.ceil().long().clamp(0,512)
    bad = invalid[hi[:,1],hi[:,0]]-invalid[lo[:,1],hi[:,0]]-invalid[hi[:,1],lo[:,0]]+invalid[lo[:,1],lo[:,0]]
    reliable = t['positive'] & inside & (bad==0)
    x0,y0,x1,y1 = rect
    negative = reliable & (low[:,0]>=x0) & (low[:,1]>=y0) & (high[:,0]<=x1) & (high[:,1]<=y1)
    disjoint = (high[:,0]<=x0) | (high[:,1]<=y0) | (low[:,0]>=x1) | (low[:,1]>=y1)
    positive = reliable & disjoint
    t.update(positive=positive,negative=negative,ignore=~(positive|negative),
             boundary=reliable & ~(positive|negative),original_positive=targets(gt)['positive'])
    return t


def make_occlusion(images, gt, pair_id, epoch, seed, *, area=None, fill=None, max_attempts=8, min_negatives=4):
    # images are RGB float [0,1], before the encoder's normalization: 0/1 == RGB 0/255.
    rng = random.Random(zlib.crc32(f'{seed}:{pair_id}:{epoch}'.encode()))
    if area is None:
        stratum = rng.randrange(3)
        area = rng.uniform(.05+.05*stratum,.10+.05*stratum)
    if fill is None:
        fill = rng.choice((0.,1.))
    base = targets(gt);eligible=base['target'][base['positive']].detach().cpu()
    for attempt in range(max_attempts):
        ratio = math.exp(rng.uniform(math.log(.5),math.log(2)))
        width = min(512,round(math.sqrt(area*512*512*ratio)))
        height = min(512,round(math.sqrt(area*512*512/ratio)))
        if len(eligible) and rng.random()<.5:
            xy=eligible[rng.randrange(len(eligible))]
            x0=round(float(xy[0])+.5-width/2);y0=round(float(xy[1])+.5-height/2)
            x0=max(0,min(512-width,x0));y0=max(0,min(512-height,y0))
        else:
            x0=rng.randrange(513-width);y0=rng.randrange(513-height)
        rect=(x0,y0,x0+width,y0+height)
        labels=label_rectangle(gt,rect)
        if int(labels['negative'].sum())>=min_negatives:break
    skipped = int(labels['negative'].sum())<min_negatives
    if skipped:
        labels['positive']=torch.zeros_like(labels['positive']);labels['negative']=torch.zeros_like(labels['negative'])
        labels['ignore']=torch.ones_like(labels['ignore'])
    modified=images.clone();modified[:,1,:,y0:y0+height,x0:x0+width]=fill
    record=dict(rect=list(rect),requested_area=area,actual_area=width*height/(512*512),fill_rgb=round(fill*255),
        attempts=attempt+1,skipped=skipped,reason='fewer_than_4_trusted_negative_queries' if skipped else None,
        positive=int(labels['positive'].sum()),negative=int(labels['negative'].sum()),ignore=int(labels['ignore'].sum()))
    return modified,labels,record


def augmented_losses(stage, gt, labels, config):
    p=stage['match_probabilities'][0].float().clamp_min(1e-30)
    mat=stage['coarse_matchability'][0,0].float().flatten().clamp(1e-6,1-1e-6)
    warp=stage['coarse_warp'][0].float().permute(1,2,0).reshape(-1,2)
    positive,negative=labels['positive'],labels['negative']
    zero=(warp.sum()+p.sum()+mat.sum())*0
    occ=keep=geo=match=zero
    if negative.any():
        occ=-p[negative,-1].log().mean()-config['lambda_m']*torch.log1p(-mat[negative]).mean()
    if positive.any():
        keep=-(p[positive,:1024]*labels['region'][positive]).sum(-1).clamp_min(1e-30).log().mean()
        reference=(labels['target'][positive]+.5)/256-1
        residual=warp[positive]-reference
        residual=residual.sign()*(residual.abs()-gt.tolerance_px/256).clamp_min(0)
        geo=torch.log1p(.5*(residual/.1).square()).sum(-1).mean()
        match=-mat[positive].log().mean()
    return dict(occ=config['lambda_occ']*occ,keep=config['lambda_keep']*keep,
                keep_geo=config['lambda_keep_geo']*geo,keep_match=config['lambda_keep_match']*match)
