"""Inference-only LoMa D8→D2 consumer; owns no GHIM/CGMDP modules."""
import torch
from .loma_reference.configs import registered_config
from .loma_reference.coarse_matcher import match_d8
from .loma_reference.fine_matcher import match_d2
from .loma_reference.warped_fine_matcher import match_d2_h_warped_sym4
from .loma_reference.matching_utils import flat_indices_to_centers, centers_to_native
from .loma_reference.overlap_masks import predicted_overlap_masks


class HGuidedDenseDownstream:
    """Zero-shot reference path, not a differentiable trainable dense head.

    Input H0 is predicted, align_corners=False normalized A→B; no GT accepted.
    Shared network runs outside this class once. D4 need not be consumed here.
    """
    def __init__(self, result_id='C3-P3', **overrides):
        if result_id not in ('C3-P3','C3-P6'):
            raise ValueError('Only registered non-oracle P3/P6 enabled in migration phase 1')
        self.config=registered_config(result_id,**overrides)

    @torch.no_grad()
    def __call__(self, shared, sizes=((784,784),(784,784))):
        H=shared['H0_norm']
        if H is None or H.shape!=(1,3,3):raise ValueError('Require one predicted H0 [1,3,3]')
        empty=H.new_empty((0,2)); conf=H.new_empty((0,))
        def failure(reason):
            return dict(points_a=empty,points_b=empty.clone(),confidence=conf,
                        H0_norm=H,failure_reason=reason,config=self.config.to_dict())
        if not bool(shared['stage1_valid'][0]):return failure('ghim_fit_failed')
        if not bool(torch.isfinite(H).all()):return failure('nonfinite_h0')
        # Isolate invalid matrices before any downstream inverse or projection.
        norm=H.abs().amax()
        if norm==0:return failure('singular_h0')
        s=torch.linalg.svdvals((H/norm).double())[0]
        if s[-1]<=s[0]*1e-10:return failure('ill_conditioned_h0')
        corners=H.new_tensor([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
        z=corners@H[0,2]
        if not bool((z>1e-8).all() or (z<-1e-8).all()):return failure('h0_horizon_crossing')
        pyramid=shared['pyramid']
        for scale,n in ((8,98),(2,392)):
            if pyramid[scale].shape[:2]!=(1,2) or pyramid[scale].shape[-2:]!=(n,n):
                raise ValueError(f'D{scale} must have shape [1,2,C,{n},{n}]')
        d8,d2=pyramid[8][0],pyramid[2][0];c=self.config
        masks8=predicted_overlap_masks(H,98);masks2=predicted_overlap_masks(H,392)
        coarse=match_d8(d8[0],d8[1],masks8.a,masks8.b,
            temperature=c.coarse_temperature,threshold=c.coarse_confidence_threshold,
            max_coarse=c.max_coarse,policy=c.coarse_policy,chunk_rows=c.coarse_chunk_rows,
            force_chunked=c.force_chunked_coarse)
        if c.fine_mode=='h_warped_sym4':
            fine=match_d2_h_warped_sym4(d2[0],d2[1],masks2.a,masks2.b,
                coarse.source_flat,coarse.target_flat,coarse.confidence,H,
                temperature=c.fine_temperature,threshold=c.fine_confidence_threshold,
                max_final_matches=c.max_final_matches,geometry_radius=c.fine_geometry_radius)
            a,b=fine.source_norm,fine.target_norm
        else:
            fine=match_d2(d2[0],d2[1],masks2.a,masks2.b,
                coarse.source_flat,coarse.target_flat,coarse.confidence,H,
                fine_prior=c.fine_prior,window=c.fine_window,temperature=c.fine_temperature,
                threshold=c.fine_confidence_threshold,selection=c.fine_selection,
                local_topk=c.local_topk,max_final_matches=c.max_final_matches)
            a=flat_indices_to_centers(fine.source_flat,392,392)
            b=flat_indices_to_centers(fine.target_flat,392,392)
        return dict(points_a=centers_to_native(a,sizes[0]),points_b=centers_to_native(b,sizes[1]),
            confidence=fine.confidence,H0_norm=H,
            failure_reason='none' if len(a) else 'no_fine_matches',
            config=c.to_dict(),config_sha256=c.sha256(),
            coarse_diagnostics=coarse.diagnostics,fine_diagnostics=fine.diagnostics)
