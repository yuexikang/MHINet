"""Explicit second-stage ownership and strict first-stage loading."""
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch import nn
from mhinet.pretraining.model import build_shared_network,load_shared
from mhinet.config import sha256_file
from mhinet.engine.ghim_losses import ghim_loss
from mhinet.ops.geometry import safe_project_points
from .semidense import SemidenseMatcher,SemidenseConfig


class SemidenseSystem(nn.Module):
    def __init__(self,shared,matcher,*,h0_loss_weight=0.,h0_loss_weights=None):
        super().__init__()
        self.shared=shared
        self.matcher=matcher
        self.configure_h0_supervision(h0_loss_weight,h0_loss_weights)
        shared.set_training_groups(mvt=False,vgg=True,dedode=True,ghim_head=False)

    def configure_h0_supervision(self,weight=0.,weights=None):
        """Attach independent GHIM supervision without changing matcher priors.

        Zero preserves the historical Lc/Lf/Lq protocol.  The normalized GHIM
        target and robust scale are independent of the selected input size.
        """
        weight=float(weight)
        if not math.isfinite(weight) or weight<0:
            raise ValueError('h0_loss_weight must be finite and nonnegative')
        defaults=dict(mat=.01,cls=1e-4,H=.05,robust_scale=.1)
        supplied=dict(weights or {})
        unknown=set(supplied)-set(defaults)
        if unknown:
            raise ValueError(f'Unknown h0_loss_weights: {sorted(unknown)}')
        defaults.update(supplied)
        defaults={key:float(value) for key,value in defaults.items()}
        for key,value in defaults.items():
            if not math.isfinite(value) or value<0 or (key=='robust_scale' and value==0):
                raise ValueError(f'Invalid h0_loss_weights.{key}')
        self.h0_loss_weight=weight
        self.h0_loss_weights=defaults

    @staticmethod
    @torch.no_grad()
    def _h0_supervised_points(outputs,H_gt,mask):
        """Count the GT-visible GHIM cells used by its geometry term."""
        _,_,height,width=outputs['coarse_warp'].shape
        device=H_gt.device
        y=(torch.arange(height,device=device).float()+.5)*2/height-1
        x=(torch.arange(width,device=device).float()+.5)*2/width-1
        yy,xx=torch.meshgrid(y,x,indexing='ij')
        grid=torch.stack((xx,yy),-1).reshape(1,-1,2)
        target,valid,_=safe_project_points(H_gt.float(),grid)
        matchable=valid.reshape(1,height,width)&(target.reshape(1,height,width,2).abs()<=1).all(-1)
        matchable&=F.interpolate(mask.float(),(height,width),mode='nearest')[:,0]>.5
        return int(matchable.sum())

    def forward(self,images,H_gt=None,mask_A_overlap=None,mask_B_overlap=None):
        size=self.matcher.config.input_size
        if images.ndim!=5 or images.shape[1:]!=(2,3,size,size):
            raise ValueError(f'Configured input_size={size}, got {tuple(images.shape)}')
        shared=self.shared(images)
        if H_gt is None:
            return shared
        if mask_A_overlap is None or mask_B_overlap is None:
            raise ValueError('Training forward requires both overlap masks')
        loss,records=self.matcher.training_losses(shared,H_gt,mask_A_overlap,mask_B_overlap)
        if self.h0_loss_weight==0:
            return loss,records
        stage1=shared.get('stage1')
        if stage1 is None:
            raise ValueError('H0 supervision requires differentiable shared stage1 outputs')
        if len(records)!=len(H_gt):
            raise ValueError('H0 supervision requires one loss record per image pair')
        weights=self.h0_loss_weights
        h0_losses=[];records=[dict(record) for record in records]
        # Per-pair evaluation gives equal pair weighting and traceable records.
        # No H0 gradient is introduced into the detached fine matching lattice.
        keys=('coarse_warp','coarse_matchability','match_probabilities','H_A_to_B_norm','fit_succeeded')
        for index,record in enumerate(records):
            outputs={key:stage1[key][index:index+1] for key in keys}
            target=H_gt[index:index+1];mask=mask_A_overlap[index:index+1]
            result=ghim_loss(outputs,target,mask,mat_weight=weights['mat'],
                cls_weight=weights['cls'],h_weight=weights['H'],robust_scale=weights['robust_scale'])
            h0_losses.append(result['total'])
            record.update(lh0=float(result['total'].detach()),
                h0_geo=float(result['geo'].detach()),h0_mat=float(result['mat'].detach()),
                h0_cls=float(result['cls'].detach()),h0_h=float(result['H'].detach()),
                h0_supervised_points=self._h0_supervised_points(outputs,target,mask),
                h0_fit_valid=bool(outputs['fit_succeeded'][0]),
                h0_valid_H_pairs=int(result['valid_H_pairs']))
        return loss+self.h0_loss_weight*torch.stack(h0_losses).mean(),records

    def optimizer_groups(self,shared_lr=1e-5,head_lr=1e-4):
        return [dict(name='cgmdp',params=[p for p in self.shared.parameters() if p.requires_grad],lr=shared_lr),
                dict(name='semidense_qrru',params=list(self.matcher.parameters()),lr=head_lr)]


def load_first_stage(runtime,path,config):
    """Accept shared export, shared training checkpoint, or full MHINet checkpoint.

    Full MHINet extraction explicitly selects feature_provider; no matcher weights
    or optimizer state migrate. Missing shared keys always fail strictly.
    """
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if payload.get('format')=='mhinet.shared-descriptor.v1':
        if payload.get('lora'):raise ValueError('Mainline requires frozen non-LoRA DINO')
        shared=load_shared(runtime,path)
    elif payload.get('format')=='mhinet.training' and isinstance(payload.get('model'),dict):
        if payload.get('metadata',{}).get('config',{}).get('lora',False):
            raise ValueError('LoRA checkpoint is outside this protocol')
        expected=payload.get('metadata',{}).get('dino_sha256')
        if expected and expected!=sha256_file(runtime.dino_checkpoint):raise ValueError('DINO identity mismatch')
        shared,_=build_shared_network(runtime,lora=False)
        state=payload['model']
        if any(k.startswith('feature_provider.') for k in state):
            state={k[len('feature_provider.'):]:v for k,v in state.items() if k.startswith('feature_provider.')}
        shared.load_state_dict(state,strict=True)
    else:
        raise ValueError('Unrecognized first-stage checkpoint format')
    return SemidenseSystem(shared,SemidenseMatcher(config)).to(runtime.device)
