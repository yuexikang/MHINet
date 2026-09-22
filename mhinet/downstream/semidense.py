"""GT-decoupled training and genuinely cascaded semidense inference."""
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from .supervision import (coarse_labels,fine_grids,fine_labels,to_uv,to_norm,
    sample_uv,valid_points,legal_h,qrru_targets)
from .losses import coarse_positive_logp,positive_focal,dual_log_probability,qrru_loss
from .qrru import QRRU
from .loma_reference.matching_utils import normalized_cell_centers,centers_to_native,flat_indices_to_centers
from .loma_reference.overlap_masks import predicted_overlap_masks
from .loma_reference.coarse_matcher import match_d8


@dataclass(frozen=True)
class SemidenseConfig:
    coarse_queries: int = 1024
    fine_windows: int = 64
    qrru_queries: int = 128
    chunk: int = 128
    window_chunk: int = 32
    iterations: int = 4
    radius: float = 1.5
    gamma: float = .8
    noise_std: float = .8
    noise_bound: float = 2.4
    temperature: float = .05
    fine_gt_tolerance: float = 1.5
    max_coarse: int = 6000
    max_matches: int = 12000
    confidence_threshold: float = 0.
    lambda_c: float = 1.
    lambda_f: float = 1.
    lambda_q: float = 1.

    def __post_init__(self):
        for key in ('coarse_queries','fine_windows','qrru_queries','chunk','window_chunk','iterations','max_coarse','max_matches'):
            if getattr(self,key)<1:raise ValueError(f'{key} must be positive')
        for key in ('radius','temperature','fine_gt_tolerance','gamma'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError(key)
        for key in ('noise_std','noise_bound','confidence_threshold','lambda_c','lambda_f','lambda_q'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<0:raise ValueError(key)


def subset(n,cap,device):
    return torch.randperm(n,device=device)[:cap]


class SemidenseMatcher(nn.Module):
    def __init__(self,config=None,channels=256):
        super().__init__()
        self.config=config or SemidenseConfig()
        self.log_tau_c=nn.Parameter(torch.tensor(math.log(self.config.temperature)))
        self.log_tau_f=nn.Parameter(torch.tensor(math.log(self.config.temperature)))
        self.qrru=QRRU(channels,self.config.iterations,self.config.radius)

    @property
    def tau_c(self):return self.log_tau_c.exp().clamp_min(1e-4)

    @property
    def tau_f(self):return self.log_tau_f.exp().clamp_min(1e-4)

    def fine_scores(self,da,db,a,b,va,vb):
        fa=F.normalize(sample_uv(da,to_uv(a,da.shape[-2:])),dim=-1)
        fb=F.normalize(sample_uv(db,to_uv(b,db.shape[-2:])),dim=-1)
        return dual_log_probability(fa@fb.transpose(-1,-2)/self.tau_f,va[:,:,None]&vb[:,None,:])

    def training_losses(self,shared,H_gt,mask_a,mask_b):
        """Labels are explicit; never called implicitly by forward/infer."""
        c=self.config
        totals=[]; records=[]
        for batch in range(len(H_gt)):
            d8=shared['pyramid'][8][batch].float(); d2=shared['pyramid'][2][batch].float()
            H=H_gt[batch].float();ma=mask_a[batch];mb=mask_b[batch]
            hc=d8.shape[-2:]; hf=d2.shape[-2:]
            i,j=coarse_labels(H,ma,mb,hc)
            take=subset(len(i),c.coarse_queries,i.device)
            a=F.normalize(d8[0].flatten(1).T,dim=-1);b=F.normalize(d8[1].flatten(1).T,dim=-1)
            lc=positive_focal(coarse_positive_logp(a,b,self.tau_c,i[take],j[take],c.chunk))
            lf=d2.sum()*0; fine_count=0; fine_tokens=0
            H0=shared['H0_norm'][batch].detach().float()
            good_h=bool(shared['stage1_valid'][batch]) and legal_h(H0)
            if good_h and len(i):
                take=subset(len(i),c.fine_windows,i.device)
                with torch.no_grad():
                    ga,gb=fine_grids(i[take],j[take],H0,hc,hf)
                    mi,si,ti,va,vb=fine_labels(ga,gb,H,ma,mb,hf,c.fine_gt_tolerance)
                fine_count=len(mi);fine_tokens=ga.shape[0]*ga.shape[1]
                # Per-window rematerialization retains features without every correlation activation.
                for start in range(0,len(ga),c.window_chunk):
                    end=start+c.window_chunk
                    selected=(mi>=start)&(mi<end)
                    if not bool(selected.any()):continue
                    lp=checkpoint(self.fine_scores,d2[0],d2[1],ga[start:end],gb[start:end],
                                  va[start:end],vb[start:end],use_reentrant=False)
                    values=lp[mi[selected]-start,si[selected],ti[selected]]
                    lf=lf+positive_focal(values)*len(values)/fine_count
            # Independent GT fine centers, including when H0 is invalid.
            fi,fj=coarse_labels(H,ma,mb,hf)
            take=subset(len(fi),c.qrru_queries,fi.device)
            grid=normalized_cell_centers(*hf,device=d2.device).reshape(-1,2)
            with torch.no_grad():
                from .loma_reference.matching_utils import project_normalized
                pa=to_uv(grid[fi[take]],hf)
                truth,_=project_normalized(grid[fi[take]],H)
                noise=torch.randn_like(pa)*c.noise_std
                pb=to_uv(truth,hf)+noise.clamp(-c.noise_bound,c.noise_bound)
                target,valid,center,center_valid=qrru_targets(pa,pb,H,ma,mb,hf)
            lq=d2.sum()*0; control_value=lq.detach();center_value=lq.detach()
            q_epe=[0.]*(c.iterations+1);q_update_max=0.
            if len(pa):
                projected=self.qrru.project(d2)
                control_count=int(valid.sum()); center_count=int(center_valid.sum())
                for start in range(0,len(pa),c.window_chunk):
                    end=start+c.window_chunk
                    def refine(fa,fb,aa,bb):
                        result=self.qrru(fa,fb,aa,bb,projected=True)
                        return result['flows'],result['centers']
                    flows,centers=checkpoint(refine,projected[0],projected[1],pa[start:end],pb[start:end],use_reentrant=False)
                    with torch.no_grad():
                        cv=center_valid[start:end]
                        if bool(cv.any()):
                            trajectory=torch.cat((pb[start:end][None],centers),0)
                            errors=(trajectory-center[start:end][None]).norm(dim=-1)[:,cv].sum(-1)
                            for t,value in enumerate(errors):q_epe[t]+=float(value)/max(1,center_count)
                        q_update_max=max(q_update_max,float(flows.detach().abs().max()))
                    # Separate normalization for controls and centers across window chunks.
                    vs=valid[start:end];cs=center_valid[start:end]
                    weights=[int(vs.sum())/max(1,control_count),int(cs.sum())/max(1,center_count)]
                    def term(x,y,m):
                        err=torch.sqrt((x-y).square()+1e-6)-.001
                        return err[m].mean() if bool(m.any()) else err.sum()*0
                    cl=lq*0;ce=lq*0
                    for t in range(c.iterations):
                        weight=c.gamma**(c.iterations-1-t)
                        cl=cl+weight*term(flows[t].permute(0,2,3,1),target[start:end].permute(0,2,3,1),vs)
                        ce=ce+weight*term(centers[t],center[start:end],cs)
                    lq=lq+weights[0]*cl+weights[1]*ce
                    control_value=control_value+weights[0]*cl.detach();center_value=center_value+weights[1]*ce.detach()
            total=c.lambda_c*lc+c.lambda_f*lf+c.lambda_q*lq
            totals.append(total)
            records.append(dict(lc=float(lc.detach()),lf=float(lf.detach()),lq=float(lq.detach()),
                q_control=float(control_value),q_center=float(center_value),coarse_positive=min(len(i),c.coarse_queries),
                fine_positive=fine_count,fine_tokens=fine_tokens,fine_coverage=fine_count/max(1,fine_tokens),
                qrru_queries=len(pa),qrru_epe_d2_by_iteration=q_epe,qrru_max_control_d2=q_update_max,h0_valid=good_h))
        return torch.stack(totals).mean(),records

    @torch.no_grad()
    def infer(self,shared,sizes=None,*,diagnostics=False):
        """No GT argument. Returns one variable-length match set per image pair."""
        results=[];c=self.config
        for batch in range(len(shared['H0_norm'])):
            d8=shared['pyramid'][8][batch].float();d2=shared['pyramid'][2][batch].float()
            H=shared['H0_norm'][batch].float()
            size=sizes[batch] if sizes is not None else ((784,784),(784,784))
            empty=d2.new_empty((0,2))
            if not bool(shared['stage1_valid'][batch]) or not legal_h(H):
                results.append(dict(points_a=empty,points_b=empty,confidence=empty[:,0],failure_reason='invalid_h0'));continue
            if d8.shape != (2,256,98,98) or d2.shape != (2,256,392,392):
                raise ValueError('Inference requires registered 784/D8/D2 feature contract')
            masks=predicted_overlap_masks(H[None],98)
            coarse=match_d8(d8[0],d8[1],masks.a,masks.b,temperature=float(self.tau_c),
                threshold=c.confidence_threshold,max_coarse=c.max_coarse,policy='mutual',
                chunk_rows=c.chunk,force_chunked=True)
            mask2=predicted_overlap_masks(H[None],392)
            matches=[];fine_trace=[];qrru_trace=[]
            probe_parents=set(torch.linspace(0,max(0,len(coarse.source_flat)-1),4).long().tolist())
            for start in range(0,len(coarse.source_flat),c.window_chunk):
                end=start+c.window_chunk
                ga,gb=fine_grids(coarse.source_flat[start:end],coarse.target_flat[start:end],H,(98,98),(392,392))
                va=valid_points(mask2.a,ga);vb=valid_points(mask2.b,gb)
                lp=self.fine_scores(d2[0],d2[1],ga,gb,va,vb)
                if diagnostics and len(fine_trace)<4:
                    for k in range(len(ga)):
                        if start+k not in probe_parents:continue
                        fine_trace.append(dict(parent=start+k,a=ga[k].cpu(),b=gb[k].cpu(),
                            logp=lp[k].cpu(),valid_a=va[k].cpu(),valid_b=vb[k].cpu()))
                best=lp.argmax(-1);reverse=lp.argmax(-2)
                ids=torch.arange(16,device=d2.device)[None].expand_as(best)
                valid=va & vb.gather(1,best) & (reverse.gather(1,best)==ids)
                m,s=valid.nonzero(as_tuple=True);t=best[m,s]
                score=(lp[m,s,t].exp()*coarse.confidence[start:end][m]).sqrt()
                keep=score>=c.confidence_threshold
                matches.append((ga[m[keep],s[keep]],gb[m[keep],t[keep]],score[keep],torch.stack((m[keep]+start,s[keep],t[keep]),-1)))
            if not matches or sum(len(x[0]) for x in matches)==0:
                results.append(dict(points_a=empty,points_b=empty,confidence=empty[:,0],failure_reason='no_fine_matches'));continue
            a,b,score,lineage=[torch.cat([x[k] for x in matches]) for k in range(4)]
            order=score.argsort(descending=True,stable=True)[:c.max_matches]
            a,b,score,lineage=a[order],b[order],score[order],lineage[order]
            projected=self.qrru.project(d2)
            refined=[];out_of_bounds=0
            for start in range(0,len(a),c.window_chunk):
                end=start+c.window_chunk
                qr=self.qrru(projected[0],projected[1],to_uv(a[start:end],(392,392)),to_uv(b[start:end],(392,392)),projected=True)
                if diagnostics and start==0:
                    qrru_trace.append(dict(a=a[start:end][:8].cpu(),b=b[start:end][:8].cpu(),lineage=lineage[start:end][:8].cpu(),
                        **{k:v[:,:8].cpu() for k,v in qr.items()}))
                refined.append(to_norm(qr['centers'][-1],(392,392)))
                grid=qr['grids'][-1]
                out_of_bounds+=int(((grid<0)|(grid>391)).any(-1).sum())
            out=torch.cat(refined)
            valid=torch.isfinite(out).all(-1)&(out>=-1).all(-1)&(out<=1).all(-1)
            results.append(dict(points_a=centers_to_native(a[valid],size[0]),points_b=centers_to_native(out[valid],size[1]),
                fine_points_b=centers_to_native(b[valid],size[1]),confidence=score[valid],
                fine_all_a=centers_to_native(a,size[0]),fine_all_b=centers_to_native(b,size[1]),
                coarse_a=centers_to_native(flat_indices_to_centers(coarse.source_flat,98,98),size[0]),
                coarse_b=centers_to_native(flat_indices_to_centers(coarse.target_flat,98,98),size[1]),
                qrru_outside_samples=out_of_bounds,rejected_qrru=int((~valid).sum()),
                failure_reason='none' if bool(valid.any()) else 'invalid_qrru_output'))
            if diagnostics:
                results[-1]['diagnostics']=dict(coarse_source=coarse.source_flat.cpu(),coarse_target=coarse.target_flat.cpu(),
                    coarse_confidence=coarse.confidence.cpu(),coarse_counts=coarse.diagnostics,
                    mask_a=masks.a.cpu(),mask_b=masks.b.cpu(),fine=fine_trace,qrru=qrru_trace,
                    fine_before_cap=sum(len(x[0]) for x in matches),fine_after_cap=len(a),lineage=lineage.cpu(),
                    qrru_valid=valid.cpu())
        return results
