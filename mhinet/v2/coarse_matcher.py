"""Global D8 matching with an optional bounded, one-way Flow0 soft reward.

Flow validity controls the reward only. Both images retain their complete D8
candidate grids; a poor or unavailable flow can never suppress a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from mhinet.downstream.loma_reference.coarse_matcher import (
    CoarseMatchResult, _candidate_selection,
)
from mhinet.downstream.loma_reference.matching_utils import (
    l2_normalize_descriptors, normalized_cell_centers,
)


@dataclass(frozen=True)
class FlowPriorD8:
    """A-to-B target coordinates in D8 pixels, with per-source probability."""
    target_uv: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


def flow_prior_from_shared(shared, batch, d8_hw, *, min_matchability=0.):
    """Upsample normalized *displacement*, then add D8 source cell centers.

    Interpolating absolute target coordinates would shift the boundary cells
    even for identity flow. Nonfinite entries are removed before interpolation;
    a D8 query receives no reward if any contributing D16 cell was invalid.
    """
    if 'flow16_norm' not in shared:
        return None
    flow = shared['flow16_norm'][batch].float()
    confidence = shared['flow16_matchability'][batch].float()
    valid = shared['flow16_valid'][batch].bool()
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError('flow16_norm must have shape Bx2xH16xW16')
    if confidence.shape != (1, *flow.shape[-2:]) or valid.shape != confidence.shape:
        raise ValueError('Flow0 probability/validity shape mismatch')
    source = normalized_cell_centers(*flow.shape[-2:], device=flow.device).permute(2, 0, 1)
    target = shared.get('flow16_target_norm')
    target = target[batch].float() if target is not None else source + flow
    valid = valid & torch.isfinite(flow).all(0, keepdim=True)
    valid &= torch.isfinite(target).all(0, keepdim=True) & (target.abs() <= 1).all(0, keepdim=True)
    valid &= torch.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)
    # The threshold applies at the prediction resolution and after interpolation.
    valid &= confidence >= min_matchability
    safe_flow = torch.where(valid.expand_as(flow), flow, torch.zeros_like(flow))
    safe_confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    def up(x):
        return F.interpolate(x[None], size=d8_hw, mode='bilinear', align_corners=False)[0]
    displacement = up(safe_flow).permute(1, 2, 0)
    targets = normalized_cell_centers(*d8_hw, device=flow.device) + displacement
    conf = up(safe_confidence)[0]
    good = up(valid.float())[0] >= 1 - 1e-6
    good &= torch.isfinite(targets).all(-1) & (targets.abs() <= 1).all(-1)
    good &= conf >= min_matchability
    h, w = d8_hw
    uv = (targets + 1) * targets.new_tensor([w, h]) / 2 - .5
    return FlowPriorD8(uv.reshape(-1, 2), conf.reshape(-1), good.reshape(-1))


def geometry_reward(prior, source, target, *, width, strength, sigma):
    """Return a finite reward in [0,strength] for a selected score block."""
    uv = torch.stack((target % width, target // width), -1).float()
    prediction = prior.target_uv[source]
    probability = prior.confidence[source]
    valid = prior.valid[source] & torch.isfinite(prediction).all(-1)
    valid &= torch.isfinite(probability) & (probability >= 0) & (probability <= 1)
    safe_prediction = torch.where(valid[:, None], prediction, torch.zeros_like(prediction))
    distance = (safe_prediction[:, None] - uv[None]).square().sum(-1)
    weight = torch.where(valid, probability, torch.zeros_like(probability))
    return strength * weight[:, None] * torch.exp(-distance / (2 * sigma**2))


def match_d8(descriptor_a, descriptor_b, mask_a=None, mask_b=None, *,
             temperature, threshold, max_coarse, policy='mutual', chunk_rows=128,
             force_chunked=True, flow_prior=None, flow_lambda=3., flow_sigma_d8=4.):
    """FP32 global dual-softmax; full and chunked paths share every score.

    Masks are supported for explicit callers, but v2 Flow/none inference passes
    full support. The helper accepts small feature grids for CPU numerical tests;
    the inference boundary enforces the registered 512/D8/D2 feature contract.
    """
    if descriptor_a.ndim != 3 or descriptor_a.shape != descriptor_b.shape:
        raise ValueError('Expected matching CxHxW D8 descriptors')
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError('temperature must be finite and positive')
    if policy not in ('mutual', 'dual_topk'):
        raise ValueError('policy must be mutual or dual_topk')
    if chunk_rows < 1 or max_coarse < 1:
        raise ValueError('chunk_rows and max_coarse must be positive')
    if not math.isfinite(flow_lambda) or flow_lambda < 0:
        raise ValueError('flow_lambda must be finite and nonnegative')
    if not math.isfinite(flow_sigma_d8) or flow_sigma_d8 <= 0:
        raise ValueError('flow_sigma_d8 must be finite and positive')
    channels, height, width = descriptor_a.shape
    full_a, full_b = mask_a is None, mask_b is None
    if mask_a is None:
        mask_a = torch.ones((height, width), dtype=torch.bool, device=descriptor_a.device)
    if mask_b is None:
        mask_b = torch.ones_like(mask_a)
    if mask_a.shape != (height, width) or mask_b.shape != (height, width):
        raise ValueError('D8 mask shape mismatch')
    source = torch.arange(height*width, device=descriptor_a.device) if full_a else mask_a.flatten().nonzero().flatten()
    target = torch.arange(height*width, device=descriptor_a.device) if full_b else mask_b.flatten().nonzero().flatten()
    rows, cols = len(source), len(target)
    if not rows or not cols:
        empty = torch.empty(0, dtype=torch.long, device=descriptor_a.device)
        return CoarseMatchResult(empty, empty.clone(), descriptor_a.new_empty(0, dtype=torch.float32),
            source, target, empty.clone(), dict(policy=policy, correlation_path='empty_support',
                matrix_rows=rows, matrix_cols=cols, matrix_elements=0, raw_candidates=0,
                thresholded_candidates=0, capped_candidates=0, deduplicated_candidates=0, duplicate_rejections=0))
    fa = l2_normalize_descriptors(descriptor_a.flatten(1).T[source])
    fb = l2_normalize_descriptors(descriptor_b.flatten(1).T[target])
    # An explicit no-op branch preserves the pure-appearance numerical path.
    if flow_lambda == 0:
        flow_prior = None
    if flow_prior is not None:
        if flow_prior.target_uv.shape != (height * width, 2) or flow_prior.confidence.shape != (height * width,) or flow_prior.valid.shape != (height * width,):
            raise ValueError('Flow prior must cover the D8 source grid')
    def logits(start, end):
        x = (fa[start:end] @ fb.T).float().div_(temperature)
        if flow_prior is not None:
            x = x + geometry_reward(flow_prior, source[start:end], target,
                width=width, strength=flow_lambda, sigma=flow_sigma_d8)
        return x
    if not force_chunked:
        x = logits(0, rows)
        row_lse = torch.logsumexp(x, 1)
        col_lse = torch.logsumexp(x, 0)
        row_best = x.argmax(1)
        col_best = x.argmax(0)
        if policy == 'mutual':
            si = torch.arange(rows, device=fa.device)
            si = si[col_best[row_best] == si]
            ti = row_best[si]
            confidence = torch.exp(2*x[si, ti] - row_lse[si] - col_lse[ti])
            raw = len(si)
            thresholded = int((confidence >= threshold).sum())
        else:
            probability = torch.exp(2*x - row_lse[:, None] - col_lse[None])
            raw = rows * cols
            thresholded = int((probability >= threshold).sum())
            confidence, index = torch.topk(probability.flatten(), min(max_coarse, raw), sorted=True)
            si, ti = index // cols, index % cols
        path = 'full_gathered'
    else:
        row_lse = fa.new_empty(rows, dtype=torch.float32)
        row_best = source.new_empty(rows)
        col_lse = fa.new_full((cols,), -torch.inf, dtype=torch.float32)
        col_max = fa.new_full((cols,), -torch.inf, dtype=torch.float32)
        col_best = source.new_zeros(cols)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            x = logits(start, end)
            row_lse[start:end] = torch.logsumexp(x, 1)
            row_best[start:end] = x.argmax(1)
            col_lse = torch.logaddexp(col_lse, torch.logsumexp(x, 0))
            values, indices = x.max(0)
            improve = values > col_max
            col_max = torch.where(improve, values, col_max)
            col_best = torch.where(improve, indices + start, col_best)
        if policy == 'mutual':
            si = torch.arange(rows, device=fa.device)
            si = si[col_best[row_best] == si]
            ti = row_best[si]
            selected = (fa[si] * fb[ti]).sum(-1).float() / temperature
            if flow_prior is not None:
                # Paired scores use the same reward without constructing KxK.
                pred = flow_prior.target_uv[source[si]]
                prob = flow_prior.confidence[source[si]]
                good = flow_prior.valid[source[si]] & torch.isfinite(pred).all(-1)
                good &= torch.isfinite(prob) & (prob >= 0) & (prob <= 1)
                pred = torch.where(good[:, None], pred, torch.zeros_like(pred))
                uv = torch.stack((target[ti] % width, target[ti] // width), -1).float()
                selected = selected + flow_lambda * torch.where(good, prob, torch.zeros_like(prob)) * torch.exp(-(pred-uv).square().sum(-1)/(2*flow_sigma_d8**2))
            confidence = torch.exp(2*selected-row_lse[si]-col_lse[ti])
            raw = len(si)
            thresholded = int((confidence >= threshold).sum())
        else:
            candidates = []
            thresholded = 0
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                x = logits(start, end)
                probability = torch.exp(2*x-row_lse[start:end, None]-col_lse[None])
                thresholded += int((probability >= threshold).sum())
                values, index = torch.topk(probability.flatten(), min(max_coarse, probability.numel()), sorted=False)
                candidates.append((index // cols + start, index % cols, values))
            si, ti, confidence = [torch.cat([part[k] for part in candidates]) for k in range(3)]
            if len(confidence) > max_coarse:
                keep = confidence.topk(max_coarse, sorted=True).indices
                si, ti, confidence = si[keep], ti[keep], confidence[keep]
            raw = rows * cols
        path = f'chunked_gathered_rows_{chunk_rows}'
    result = _candidate_selection(policy=policy, source_local=si, target_local=ti,
        confidence=confidence, gathered_source=source, gathered_target=target,
        threshold=threshold, max_coarse=max_coarse, raw_candidates=raw,
        thresholded_candidates=thresholded, matrix_shape=(rows, cols), path=path)
    object.__setattr__(result, 'prefilter_target_for_source', target[row_best])
    result.diagnostics['flow_guidance'] = flow_prior is not None
    return result
