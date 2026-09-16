"""MHINet round stream and H0/H6 sufficiency scoring, at input-pixel scale."""
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from mhinet.ops.geometry import safe_project_points
from .config import PairAFSSV2Config
from .controller import PairAFSSController


@torch.no_grad()
def score_pair(outputs, gt, overlap, model_size=784):
    coarse=outputs['ghim_outputs']
    warp=coarse['coarse_warp'].float()
    b,_,h,w=warp.shape
    y=(torch.arange(h,device=warp.device)+.5)*2/h-1
    x=(torch.arange(w,device=warp.device)+.5)*2/w-1
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    grid=torch.stack([xx,yy],-1).reshape(1,-1,2).expand(b,-1,-1)
    target,valid,_=safe_project_points(gt.float(),grid)
    visible=(valid & (target.abs()<=1).all(-1)
             & (F.interpolate(overlap.float(),(h,w),mode='nearest')[:,0].reshape(b,-1)>.5))
    error=(warp.permute(0,2,3,1).reshape(b,-1,2)-target).norm(dim=-1)*(model_size/2)
    positive=coarse['coarse_matchability'][:,0].reshape(b,-1)>=.3
    correct=visible & torch.isfinite(error) & (error<=1)
    tp=(positive&correct).sum(1).float()
    precision=tp/positive.sum(1).clamp_min(1)
    recall=tp/visible.sum(1).clamp_min(1)
    thresholds=torch.arange(.5,3.01,.5,device=warp.device)
    aucs=[]
    for H in (outputs['H0_norm'],outputs['H_updates_norm'][:,-1]):
        points,ok,_=safe_project_points(H.float(),grid)
        e=(points-target).norm(dim=-1)*(model_size/2)
        hit=(e[...,None]<=thresholds)&ok[...,None]&visible[...,None]&torch.isfinite(e)[...,None]
        pck=hit.sum(1).float()/visible.sum(1).clamp_min(1)[:,None]
        aucs.append(torch.trapezoid(pck,thresholds,dim=1)/2.5)
    fit=outputs['stage1_valid'].bool() & visible.any(1)
    auc=torch.minimum(*aucs)
    score=torch.minimum(torch.minimum(precision,recall),auc)
    score=torch.where(fit & torch.isfinite(score),score,torch.zeros_like(score))
    return dict(score=score.cpu(),precision_geo=precision.cpu(),recall_geo=recall.cpu(),
                homography_auc=auc.cpu(),fit_succeeded=fit.cpu())


class RoundStream:
    def __init__(self, pair_ids, seed, batch, rounds, strategy, manifest_hash, saved=None):
        self.length=len(pair_ids);self.batch=batch;self.strategy=strategy;self.manifest_hash=manifest_hash
        config=PairAFSSV2Config(seed=seed,model_size=784,warmup_rounds=2 if strategy=='afss_v2' else rounds)
        self.controller=PairAFSSController(pair_ids,config,
            baseline_max_steps=rounds*math.ceil(self.length/batch),baseline_steps_per_round=math.ceil(self.length/batch))
        if saved:
            if saved['manifest_hash']!=manifest_hash or saved['strategy']!=strategy or saved['batch']!=batch:
                raise ValueError('Round stream manifest/strategy/batch changed')
            self.controller.load_state_dict(saved['controller'])

    @property
    def done(self):
        c=self.controller
        return not c.has_training_remaining() or (c.round_complete() and c.current_round+1>=math.ceil(c.total_rounds))

    def prepare(self,refresh):
        c=self.controller
        if c.round_complete():c.finish_round()
        if not c.has_training_remaining():return
        if not c.selected_indices and self.strategy=='afss_v2' and c.refresh_due():refresh(self)
        c.prepare_round(pad_to_multiple=self.batch)

    def next(self):
        c=self.controller
        if c.consumed_global>=len(c.selected_indices):raise RuntimeError('Round boundary crossed within optimizer step')
        index=c.selected_indices[c.consumed_global];c.advance(1)
        return index

    def state_dict(self):
        return {'length':self.length,'strategy':self.strategy,'batch':self.batch,
                'manifest_hash':self.manifest_hash,'controller':self.controller.state_dict()}

    def log_state(self):
        c=self.controller
        return {'round':c.current_round,'position':c.consumed_global,'selected':len(c.selected_indices),
                'unique':len(c.selected_unique_indices),'padding':c.padding_count,
                'baseline_equivalent_step':c.baseline_equivalent_step(),
                'counts':c.state.counts(),'selected_by_state':c.selected_by_state}


@torch.no_grad()
def refresh_scores(stream, model, dataset, device, path):
    started=time.perf_counter();model.eval()
    fields={k:torch.zeros(len(dataset),dtype=torch.bool if k=='fit_succeeded' else torch.float32)
            for k in ('score','precision_geo','recall_geo','homography_auc','fit_succeeded')}
    for i in tqdm(range(len(dataset)),desc='AFSS train-only score refresh',unit='pair'):
        sample=dataset[i]
        output=model(sample['images'][None].to(device))
        values=score_pair(output,sample['H_gt_norm'][None].to(device),sample['mask_A_overlap'][None].to(device))
        for key,value in values.items():fields[key][i]=value[0]
        del output
    c=stream.controller
    c.state.update_scores(round_id=c.current_round,**fields,scored=torch.ones(len(dataset),dtype=torch.bool),config=c.config)
    with Path(path).open('a') as f:
        f.write(json.dumps({'round':c.current_round,'pairs':len(dataset),'seconds':time.perf_counter()-started,
            'thresholds':c.state.thresholds(c.config),'counts':c.state.counts(),
            'score_min':float(fields['score'].min()),'score_max':float(fields['score'].max()),
            'scoring':'min(P,R,H0-AUC,H6-AUC);784 input px;train only'})+'\n')
