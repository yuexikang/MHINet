"""Fixed B0 features/probabilities/matchability with an independently trained XY head."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from mhinet.v2.feature_provider import _flow_fields
from mhinet.v2.d16_experiment.supervision import grid


def embeddings(head, result):
    dtype=result['pair_descriptors'].dtype
    xy=(grid(result['context'].device)+.5)/256-1
    projection=F.linear(xy.to(dtype),(head.position_scale*head.position_frequencies).to(dtype))
    pe=torch.cat((projection.sin(),projection.cos()),-1)
    pe=torch.cat((pe,head.no_match_embedding.to(pe)[None]),0)
    return torch.einsum('bnm,md->bnd',result['stage1']['match_probabilities'],pe).reshape(-1,32,32,1024)


class LocalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2052,256),nn.ReLU(),nn.Linear(256,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,result):
        prob=result['stage1']['match_probabilities'][...,:1024].float()
        peak=prob.detach().argmax(-1)
        offset=torch.tensor([(dx,dy) for dy in (-1,0,1) for dx in (-1,0,1)],device=prob.device)
        ix=peak[...,None]%32+offset[:,0];iy=peak[...,None]//32+offset[:,1]
        valid=(ix>=0)&(ix<32)&(iy>=0)&(iy<32)
        index=iy.clamp(0,31)*32+ix.clamp(0,31)
        local=prob.gather(-1,index)*valid
        mass=local.sum(-1,keepdim=True);weights=local/mass.clamp_min(1e-12)
        xy=grid(prob.device)[index];coarse=(weights[...,None]*xy).sum(-2)
        context=result['context'].float().reshape(-1,2,1024,1024)
        source=F.normalize(context[:,0],dim=-1);sensed=F.normalize(context[:,1],dim=-1)
        candidates=sensed[torch.arange(len(prob),device=prob.device)[:,None,None],index]
        target_feature=(weights[...,None]*candidates).sum(-2)
        spatial=prob/prob.sum(-1,keepdim=True).clamp_min(1e-12)
        entropy=-(spatial*spatial.clamp_min(1e-30).log()).sum(-1,keepdim=True)/torch.log(prob.new_tensor(1024.))
        outside=prob.clone();outside.scatter_(-1,index,0.)
        best=prob.max(-1,keepdim=True).values
        summary=torch.cat((mass,best,best-outside.max(-1,keepdim=True).values,entropy),-1)
        delta=8*torch.tanh(self.net(torch.cat((source,target_feature,summary),-1)))
        target=(coarse+delta+.5)/256-1
        return target.reshape(-1,32,32,2).permute(0,3,1,2),dict(local_mass=mass,entropy=entropy,delta_px=delta,coarse_px=coarse)


class CoordinateModel(nn.Module):
    def __init__(self,shared,local):
        super().__init__();self.shared=shared;self.local=local
        shared.requires_grad_(False);shared.eval()
        if local:self.coordinate=LocalResidual().to(next(shared.parameters()).device)
        else:
            self.coordinate=copy.deepcopy(shared.stage1_head.match_embedding_decoder)
            self.coordinate.requires_grad_(True)
            self.coordinate.matchability_head.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode);self.shared.eval();return self

    def forward(self,images,**_):
        with torch.no_grad():result=self.shared(images,pyramid_scales=())
        if self.local:warp,diagnostic=self.coordinate(result)
        else:
            with torch.no_grad():encoded=embeddings(self.shared.stage1_head,result)
            warp=self.coordinate.warp_head(self.coordinate.net(encoded.permute(0,3,1,2).contiguous()))
            diagnostic={}
        stage=dict(result['stage1'],coarse_warp=warp)
        result.update(stage1=stage,coordinate_diagnostic=diagnostic,**_flow_fields(stage))
        return result
