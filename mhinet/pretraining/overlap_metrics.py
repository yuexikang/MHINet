"""H0 error on GT-defined visible overlap, in resized target-image pixels.

All source pixel centres are considered. Predictions NEVER select the support;
out-of-image predictions count as errors and illegal projections as failures.
"""
import numpy as np
import torch
from mhinet.ops.geometry import safe_project_points
from mhinet.pretraining.loss import grid, sample


@torch.no_grad()
def overlap_projection_error(prediction, truth, mask_a, mask_b, *, fit_valid=True):
    if prediction.shape != (1,3,3) or truth.shape != (1,3,3):
        raise ValueError('Overlap evaluation currently requires batch size one')
    h,w = mask_a.shape[-2:]
    hb,wb = mask_b.shape[-2:]
    points = grid(h,w,truth.device)
    points = points[mask_a.reshape(-1) > .999]
    targets, legal, _ = safe_project_points(truth.float(),points[None])
    targets, legal = targets[0],legal[0]
    bound = targets.new_tensor([1-1/wb,1-1/hb])
    support = legal & (targets.abs()<=bound).all(-1)
    support &= sample(mask_b[0],targets)[:,0] > .999
    points, targets = points[support], targets[support]
    count = len(points)
    predicted, valid, _ = safe_project_points(prediction.float(),points[None])
    valid = valid[0] & bool(fit_valid)
    errors = ((predicted[0,valid]-targets[valid])*targets.new_tensor([wb/2,hb/2])).norm(dim=-1)
    errors = errors[torch.isfinite(errors)]
    finite = len(errors)
    return dict(direction='A_to_B',units='resized_target_pixels',support_pixels=count,
        finite_predictions=finite,invalid_predictions=count-finite,
        fit_valid=bool(fit_valid),mean_px=float(errors.mean()) if finite else None,
        median_px=float(torch.quantile(errors,.5)) if finite else None,
        p90_px=float(torch.quantile(errors,.9)) if finite else None,
        success_counts={str(t):int((errors<=t).sum()) for t in (1,3,5)})


def summarize_overlap(records):
    total=sum(x['support_pixels'] for x in records)
    finite=sum(x['finite_predictions'] for x in records)
    pair_means=[x['mean_px'] for x in records if x['mean_px'] is not None]
    return dict(protocol='H0_visible_overlap_all_source_pixels_A_to_B_v1',pairs=len(records),
        no_support_pairs=sum(x['support_pixels']==0 for x in records),support_pixels=total,
        invalid_predictions=total-finite,
        invalid_projection_rate=(total-finite)/total if total else None,
        failed_fit_pairs=sum(not x['fit_valid'] for x in records),
        pair_mean_px=float(np.mean(pair_means)) if pair_means else None,
        pair_median_px=float(np.median(pair_means)) if pair_means else None,
        pair_p90_px=float(np.quantile(pair_means,.9)) if pair_means else None,
        pixel_mean_px=sum(x['mean_px']*x['finite_predictions'] for x in records
                          if x['mean_px'] is not None)/finite if finite else None,
        point_recall={str(t):sum(x['success_counts'][str(t)] for x in records)/total
                      if total else None for t in (1,3,5)},
        note='Support comes only from GT and both masks; illegal predictions remain in recall denominator. Mean errors are conditional on finite predictions; consult failure counts.')
