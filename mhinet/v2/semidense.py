"""Independent 512-only v2 matcher, retaining v1 checkpoint parameter names."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mhinet.downstream.semidense import SemidenseMatcher, subset
from mhinet.downstream.losses import coarse_positive_logp, positive_focal
from mhinet.downstream.loma_reference.matching_utils import (
    normalized_cell_centers, centers_to_native, flat_indices_to_centers,
)
from .config import SemidenseConfigV2
from .qrru import QRRUV2
from .coarse_matcher import match_d8, flow_prior_from_shared
from .supervision import (direct_d8_fine_grids, coarse_labels, fine_labels, to_uv,
    to_norm, valid_points, qrru_targets)
from mhinet.downstream.supervision import sample_uv
from mhinet.downstream.losses import dual_log_probability


_MATCH_FIELDS = ('points_a', 'points_b', 'confidence', 'fine_points_b', 'lineage')


def cap_output_matches(result, cap):
    """Stable confidence cap applied to every per-match output after validity.

    Diagnostic candidate arrays are deliberately retained under their explicit
    pre-cap names. They describe the QRRU input population, not returned pairs.
    """
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError('max_output_matches must be a positive integer')
    count = len(result['confidence'])
    keep = result['confidence'].argsort(descending=True, stable=True)[:cap]
    output = dict(result)
    for key in _MATCH_FIELDS:
        if key in output:
            if len(output[key]) != count:
                raise ValueError(f'Per-match field {key} does not align with confidence')
            output[key] = output[key][keep]
    if 'diagnostics' in output:
        diagnostic = dict(output['diagnostics'])
        diagnostic['output_before_cap'] = count
        diagnostic['output_after_cap'] = len(keep)
        diagnostic['output_cap'] = cap
        if 'lineage' in diagnostic:
            diagnostic['precap_lineage'] = diagnostic.pop('lineage')
        diagnostic['output_selected_precap_indices'] = keep.detach().cpu()
        output['diagnostics'] = diagnostic
    output['output_before_cap'] = count
    output['output_cap'] = cap
    return output


class SemidenseMatcherV2(SemidenseMatcher):
    def __init__(self, config=None, channels=256):
        config = config or SemidenseConfigV2()
        super().__init__(config, channels)
        self.channels = channels
        self.qrru = QRRUV2(channels, config.iterations, config.radius,
            use_discrepancy=config.use_discrepancy, use_dgfo=config.use_dgfo,
            dgfo_radius=config.dgfo_radius, dgfo_dim=config.dgfo_dim)

    def fine_scores(self, da, db, a, b, va, vb):
        # Sampling and normalization stay FP32; only inference correlation is optional AMP.
        fa = F.normalize(sample_uv(da, to_uv(a, da.shape[-2:])), dim=-1)
        fb = F.normalize(sample_uv(db, to_uv(b, db.shape[-2:])), dim=-1)
        precision = self.config.fine_precision
        amp = not torch.is_grad_enabled() and da.is_cuda and precision != 'fp32'
        dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
        with torch.autocast(da.device.type, dtype=dtype, enabled=amp):
            scores = fa @ fb.transpose(-1, -2)
        return dual_log_probability(scores.float()/self.tau_f, va[:, :, None] & vb[:, None, :])

    def training_losses(self,shared,H_gt,mask_a,mask_b):
        """Labels are explicit; never called implicitly by forward/infer."""
        if self.config.geometry_mode == 'h0':
            return super().training_losses(shared,H_gt,mask_a,mask_b)
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
            good_h = True  # Fine supervision has no predicted-geometry gate.
            if len(i):
                take=subset(len(i),c.fine_windows,i.device)
                with torch.no_grad():
                    ga,gb=direct_d8_fine_grids(i[take],j[take],hc,hf)
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
                from mhinet.downstream.loma_reference.matching_utils import project_normalized
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
                qrru_queries=len(pa),qrru_epe_d2_by_iteration=q_epe,qrru_max_control_d2=q_update_max,h0_valid=None,fine_prior='direct_d8'))
        return torch.stack(totals).mean(),records

    @torch.no_grad()
    def infer(self, shared, sizes=None, *, diagnostics=False, refine=True):
        """No GT/H0 is read for none/Flow mode; batch comes from the pyramid."""
        c = self.config
        if c.geometry_mode == 'h0':
            if not refine:
                raise ValueError('refine=False is available for no-H v2 candidate generation')
            return [cap_output_matches(item, c.max_output_matches)
                    for item in super().infer(shared, sizes, diagnostics=diagnostics)]
        hc, hf = (64, 64), (256, 256)
        d8_batch, d2_batch = shared['pyramid'][8], shared['pyramid'][2]
        if d8_batch.ndim != 5 or d2_batch.ndim != 5 or d8_batch.shape[0] != d2_batch.shape[0]:
            raise ValueError('Expected matching Bx2xCxHxW D8/D2 pyramids')
        results = []
        for batch in range(d8_batch.shape[0]):
            d8, d2 = d8_batch[batch].float(), d2_batch[batch].float()
            if d8.shape != (2, self.channels, *hc) or d2.shape != (2, self.channels, *hf):
                raise ValueError('V2 inference requires the registered 512/D8/D2 feature contract')
            size = sizes[batch] if sizes is not None else ((512, 512), (512, 512))
            empty = d2.new_empty((0, 2))
            prior = None
            if c.geometry_mode == 'flow_d16' and c.flow_guidance_enabled and c.flow_lambda != 0:
                prior = flow_prior_from_shared(shared, batch, hc,
                    min_matchability=c.flow_min_matchability)
            coarse = match_d8(d8[0], d8[1], temperature=float(self.tau_c),
                threshold=c.confidence_threshold, max_coarse=c.max_coarse,
                policy='mutual', chunk_rows=c.chunk, force_chunked=c.coarse_mode == 'chunked',
                flow_prior=prior, flow_lambda=c.flow_lambda, flow_sigma_d8=c.flow_sigma_d8)
            fine_matches, fine_trace, qrru_trace = [], [], []
            parent_probes = set(torch.linspace(0, max(0, len(coarse.source_flat)-1), 4).long().tolist())
            # Boundary validity is geometric only; no GT/predicted-overlap mask.
            support = torch.ones(hf, device=d2.device, dtype=torch.bool)
            for start in range(0, len(coarse.source_flat), c.inference_window_chunk):
                end = start + c.inference_window_chunk
                ga, gb = direct_d8_fine_grids(coarse.source_flat[start:end],
                    coarse.target_flat[start:end], hc, hf)
                va, vb = valid_points(support, ga), valid_points(support, gb)
                lp = self.fine_scores(d2[0], d2[1], ga, gb, va, vb)
                if diagnostics and len(fine_trace) < 4:
                    for k in range(len(ga)):
                        if start + k in parent_probes:
                            fine_trace.append(dict(parent=start+k, a=ga[k].cpu(), b=gb[k].cpu(),
                                logp=lp[k].cpu(), valid_a=va[k].cpu(), valid_b=vb[k].cpu()))
                best, reverse = lp.argmax(-1), lp.argmax(-2)
                ids = torch.arange(16, device=d2.device)[None].expand_as(best)
                valid = va & vb.gather(1, best) & (reverse.gather(1, best) == ids)
                m, s = valid.nonzero(as_tuple=True)
                t = best[m, s]
                score = (lp[m, s, t].exp() * coarse.confidence[start:end][m]).sqrt()
                keep = score >= c.confidence_threshold
                lineage = torch.stack((m[keep]+start, s[keep], t[keep]), -1)
                fine_matches.append((ga[m[keep], s[keep]], gb[m[keep], t[keep]], score[keep], lineage))
            fine_count = sum(len(item[0]) for item in fine_matches)
            if not fine_count:
                output = dict(points_a=empty, points_b=empty.clone(), confidence=empty[:, 0],
                    fine_points_b=empty.clone(), fine_all_a=empty.clone(), fine_all_b=empty.clone(),
                    fine_all_confidence=empty[:, 0], coarse_a=empty.clone(), coarse_b=empty.clone(),
                    lineage=torch.empty((0,3), device=d2.device, dtype=torch.long),
                    failure_reason='no_fine_matches')
                if diagnostics:
                    output['diagnostics'] = dict(coarse_counts=coarse.diagnostics,
                        coarse_source=coarse.source_flat.cpu(), coarse_target=coarse.target_flat.cpu(),
                        fine_before_qrru_budget=0, fine_after_qrru_budget=0, fine=fine_trace, qrru=[])
                results.append(cap_output_matches(output, c.max_output_matches))
                continue
            a, b, score, lineage = [torch.cat([item[k] for item in fine_matches]) for k in range(4)]
            order = score.argsort(descending=True, stable=True)[:c.max_matches]
            a, b, score, lineage = a[order], b[order], score[order], lineage[order]
            candidate_fields = dict(fine_all_a=centers_to_native(a, size[0]),
                fine_all_b=centers_to_native(b, size[1]), fine_all_confidence=score,
                coarse_a=centers_to_native(flat_indices_to_centers(coarse.source_flat, *hc), size[0]),
                coarse_b=centers_to_native(flat_indices_to_centers(coarse.target_flat, *hc), size[1]))
            if not refine:
                output = dict(points_a=candidate_fields['fine_all_a'],
                    points_b=candidate_fields['fine_all_b'], fine_points_b=candidate_fields['fine_all_b'],
                    confidence=score, lineage=lineage, failure_reason='none', **candidate_fields)
                if diagnostics:
                    output['diagnostics'] = dict(coarse_counts=coarse.diagnostics,
                        fine_before_qrru_budget=fine_count, fine_after_qrru_budget=len(a),
                        precap_lineage=lineage.cpu(), fine=fine_trace, qrru=[])
                results.append(cap_output_matches(output, c.max_output_matches))
                continue
            projected = self.qrru.project(d2)
            refined, outside = [], d2.new_zeros((), dtype=torch.long)
            accepted = 0
            for start in range(0, len(a), c.inference_window_chunk):
                end = start + c.inference_window_chunk
                qr = self.qrru(projected[0], projected[1], to_uv(a[start:end], hf),
                    to_uv(b[start:end], hf), projected=True, history=diagnostics,
                    iterations=c.inference_iterations)
                if diagnostics and start == 0:
                    qrru_trace.append(dict(a=a[start:end][:8].cpu(), b=b[start:end][:8].cpu(),
                        lineage=lineage[start:end][:8].cpu(),
                        **{key: value[:, :8].cpu() for key, value in qr.items()}))
                points = to_norm(qr['centers'][-1], hf)
                refined.append(points)
                grid = qr['grids'][-1]
                outside += ((grid < 0) | (grid > grid.new_tensor([hf[1]-1, hf[0]-1]))).any(-1).sum()
                # Confidence order is fixed and QRRU treats each match independently.
                # Later candidates cannot displace an already valid output prefix.
                if c.qrru_early_stop and not diagnostics:
                    accepted += int((torch.isfinite(points).all(-1) & (points.abs() <= 1).all(-1)).sum())
                    if accepted >= c.max_output_matches:
                        break
            out = torch.cat(refined)
            candidate_count = len(a)
            a, b, score, lineage = a[:len(out)], b[:len(out)], score[:len(out)], lineage[:len(out)]
            valid = torch.isfinite(out).all(-1) & (out >= -1).all(-1) & (out <= 1).all(-1)
            output = dict(points_a=centers_to_native(a[valid], size[0]),
                points_b=centers_to_native(out[valid], size[1]),
                fine_points_b=centers_to_native(b[valid], size[1]), confidence=score[valid],
                lineage=lineage[valid], qrru_outside_samples=int(outside),
                qrru_processed=len(out), qrru_candidates=candidate_count,
                qrru_stopped_early=len(out) < candidate_count,
                rejected_qrru=int((~valid).sum()), **candidate_fields,
                failure_reason='none' if bool(valid.any()) else 'invalid_qrru_output')
            if diagnostics:
                output['diagnostics'] = dict(coarse_source=coarse.source_flat.cpu(),
                    coarse_target=coarse.target_flat.cpu(), coarse_confidence=coarse.confidence.cpu(),
                    coarse_counts=coarse.diagnostics, fine=fine_trace, qrru=qrru_trace,
                    fine_before_qrru_budget=fine_count, fine_after_qrru_budget=len(a),
                    precap_fine_all_a=centers_to_native(a, size[0]).cpu(),
                    precap_fine_all_b=centers_to_native(b, size[1]).cpu(),
                    precap_lineage=lineage.cpu(), qrru_valid=valid.cpu())
            results.append(cap_output_matches(output, c.max_output_matches))
        return results
