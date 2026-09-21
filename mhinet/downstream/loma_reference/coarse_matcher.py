"""Gather-first scale-8 cosine correlation and FP32 dual-softmax extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .configs import CoarsePolicy
from .matching_utils import l2_normalize_descriptors, stable_greedy_one_to_one


@dataclass(frozen=True)
class CoarseMatchResult:
    source_flat: torch.Tensor
    target_flat: torch.Tensor
    confidence: torch.Tensor
    gathered_source_flat: torch.Tensor
    gathered_target_flat: torch.Tensor
    prefilter_target_for_source: torch.Tensor
    diagnostics: dict[str, float | int | str]


def _candidate_selection(
    *,
    policy: CoarsePolicy,
    source_local: torch.Tensor,
    target_local: torch.Tensor,
    confidence: torch.Tensor,
    gathered_source: torch.Tensor,
    gathered_target: torch.Tensor,
    threshold: float,
    max_coarse: int,
    raw_candidates: int,
    thresholded_candidates: int,
    matrix_shape: tuple[int, int],
    path: str,
) -> CoarseMatchResult:
    above = confidence >= threshold
    source_local = source_local[above]
    target_local = target_local[above]
    confidence = confidence[above]
    if confidence.numel() > max_coarse:
        keep = torch.argsort(confidence, descending=True, stable=True)[:max_coarse]
        source_local = source_local[keep]
        target_local = target_local[keep]
        confidence = confidence[keep]
    capped = int(confidence.numel())
    if policy == "dual_topk":
        unique = stable_greedy_one_to_one(
            source_local, target_local, confidence, cap=max_coarse
        )
        source_local = source_local[unique]
        target_local = target_local[unique]
        confidence = confidence[unique]
    # Mutual is already one-to-one by construction.
    output_source = gathered_source[source_local]
    output_target = gathered_target[target_local]
    return CoarseMatchResult(
        source_flat=output_source,
        target_flat=output_target,
        confidence=confidence,
        gathered_source_flat=gathered_source,
        gathered_target_flat=gathered_target,
        prefilter_target_for_source=torch.empty(0, dtype=torch.long, device=gathered_source.device),
        diagnostics={
            "policy": policy,
            "correlation_path": path,
            "matrix_rows": matrix_shape[0],
            "matrix_cols": matrix_shape[1],
            "matrix_elements": matrix_shape[0] * matrix_shape[1],
            "raw_candidates": raw_candidates,
            "thresholded_candidates": thresholded_candidates,
            "capped_candidates": capped,
            "deduplicated_candidates": int(confidence.numel()),
            "duplicate_rejections": capped - int(confidence.numel()),
        },
    )


def _full_dual_softmax(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    gathered_a: torch.Tensor,
    gathered_b: torch.Tensor,
    *,
    temperature: float,
    threshold: float,
    max_coarse: int,
    policy: CoarsePolicy,
) -> CoarseMatchResult:
    # Both operands have already been support-gathered. No 9604x9604 tensor exists.
    logits = (features_a @ features_b.T).float().div_(temperature)
    row_argmax = logits.argmax(dim=1)
    col_argmax = logits.argmax(dim=0)
    row_lse = torch.logsumexp(logits, dim=1)
    col_lse = torch.logsumexp(logits, dim=0)
    logits.mul_(2.0).sub_(row_lse[:, None]).sub_(col_lse[None, :]).exp_()

    if policy == "mutual":
        source = torch.arange(features_a.shape[0], device=features_a.device)
        mutual = col_argmax[row_argmax] == source
        source = source[mutual]
        target = row_argmax[mutual]
        confidence = logits[source, target]
        raw_candidates = int(mutual.sum().item())
        thresholded = int((confidence >= threshold).sum().item())
    else:
        flat = logits.reshape(-1)
        raw_candidates = int(flat.numel())
        thresholded = int((flat >= threshold).sum().item())
        candidate_count = min(max_coarse, flat.numel())
        confidence, flat_index = torch.topk(flat, candidate_count, sorted=True)
        source = torch.div(flat_index, features_b.shape[0], rounding_mode="floor")
        target = torch.remainder(flat_index, features_b.shape[0])

    result = _candidate_selection(
        policy=policy,
        source_local=source,
        target_local=target,
        confidence=confidence,
        gathered_source=gathered_a,
        gathered_target=gathered_b,
        threshold=threshold,
        max_coarse=max_coarse,
        raw_candidates=raw_candidates,
        thresholded_candidates=thresholded,
        matrix_shape=(features_a.shape[0], features_b.shape[0]),
        path="full_gathered",
    )
    object.__setattr__(result, "prefilter_target_for_source", gathered_b[row_argmax])
    return result


def _chunked_dual_softmax(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    gathered_a: torch.Tensor,
    gathered_b: torch.Tensor,
    *,
    temperature: float,
    threshold: float,
    max_coarse: int,
    policy: CoarsePolicy,
    chunk_rows: int,
) -> CoarseMatchResult:
    rows, cols = features_a.shape[0], features_b.shape[0]
    row_lse = torch.empty(rows, dtype=torch.float32, device=features_a.device)
    row_argmax = torch.empty(rows, dtype=torch.long, device=features_a.device)
    col_lse = torch.full((cols,), -torch.inf, dtype=torch.float32, device=features_a.device)
    col_max = torch.full((cols,), -torch.inf, dtype=torch.float32, device=features_a.device)
    col_argmax = torch.zeros(cols, dtype=torch.long, device=features_a.device)

    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        logits = (features_a[start:end] @ features_b.T).float().div_(temperature)
        row_lse[start:end] = torch.logsumexp(logits, dim=1)
        row_argmax[start:end] = logits.argmax(dim=1)
        col_lse = torch.logaddexp(col_lse, torch.logsumexp(logits, dim=0))
        values, indices = logits.max(dim=0)
        improve = values > col_max
        col_max[improve] = values[improve]
        col_argmax[improve] = indices[improve] + start

    if policy == "mutual":
        source = torch.arange(rows, device=features_a.device)
        mutual = col_argmax[row_argmax] == source
        source = source[mutual]
        target = row_argmax[mutual]
        selected_logits = (features_a[source] * features_b[target]).sum(dim=-1).float() / temperature
        confidence = torch.exp(2.0 * selected_logits - row_lse[source] - col_lse[target])
        raw_candidates = int(mutual.sum().item())
        thresholded = int((confidence >= threshold).sum().item())
    else:
        block_sources: list[torch.Tensor] = []
        block_targets: list[torch.Tensor] = []
        block_confidences: list[torch.Tensor] = []
        thresholded = 0
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            logits = (features_a[start:end] @ features_b.T).float().div_(temperature)
            confidence_block = torch.exp(
                2.0 * logits - row_lse[start:end, None] - col_lse[None, :]
            )
            thresholded += int((confidence_block >= threshold).sum().item())
            count = min(max_coarse, confidence_block.numel())
            values, flat = torch.topk(confidence_block.reshape(-1), count, sorted=False)
            block_sources.append(torch.div(flat, cols, rounding_mode="floor") + start)
            block_targets.append(torch.remainder(flat, cols))
            block_confidences.append(values)
        source = torch.cat(block_sources)
        target = torch.cat(block_targets)
        confidence = torch.cat(block_confidences)
        if confidence.numel() > max_coarse:
            keep = torch.topk(confidence, max_coarse, sorted=True).indices
            source, target, confidence = source[keep], target[keep], confidence[keep]
        raw_candidates = rows * cols

    result = _candidate_selection(
        policy=policy,
        source_local=source,
        target_local=target,
        confidence=confidence,
        gathered_source=gathered_a,
        gathered_target=gathered_b,
        threshold=threshold,
        max_coarse=max_coarse,
        raw_candidates=raw_candidates,
        thresholded_candidates=thresholded,
        matrix_shape=(rows, cols),
        path=f"chunked_gathered_rows_{chunk_rows}",
    )
    object.__setattr__(result, "prefilter_target_for_source", gathered_b[row_argmax])
    return result


def match_d8(
    descriptor_a: torch.Tensor,
    descriptor_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    *,
    temperature: float,
    threshold: float,
    max_coarse: int,
    policy: CoarsePolicy,
    chunk_rows: int,
    force_chunked: bool,
) -> CoarseMatchResult:
    """Gather overlap rows first, then correlate only the gathered tensors."""

    if descriptor_a.shape != descriptor_b.shape or descriptor_a.ndim != 3:
        raise ValueError("Expected matching CxHxW D8 tensors")
    channels, height, width = descriptor_a.shape
    if channels != 256 or (height, width) != (98, 98):
        raise ValueError(f"D8 contract is 256x98x98, got {tuple(descriptor_a.shape)}")
    if mask_a.shape != (height, width) or mask_b.shape != (height, width):
        raise ValueError("D8 mask shape mismatch")
    flat_a = descriptor_a.permute(1, 2, 0).reshape(-1, channels)
    flat_b = descriptor_b.permute(1, 2, 0).reshape(-1, channels)
    gathered_a = torch.nonzero(mask_a.reshape(-1), as_tuple=False).flatten()
    gathered_b = torch.nonzero(mask_b.reshape(-1), as_tuple=False).flatten()
    if not gathered_a.numel() or not gathered_b.numel():
        empty = torch.empty(0, dtype=torch.long, device=descriptor_a.device)
        return CoarseMatchResult(
            source_flat=empty,
            target_flat=empty.clone(),
            confidence=torch.empty(0, dtype=torch.float32, device=descriptor_a.device),
            gathered_source_flat=gathered_a,
            gathered_target_flat=gathered_b,
            prefilter_target_for_source=empty.clone(),
            diagnostics={
                "policy": policy,
                "correlation_path": "empty_support",
                "matrix_rows": int(gathered_a.numel()),
                "matrix_cols": int(gathered_b.numel()),
                "matrix_elements": 0,
                "raw_candidates": 0,
                "thresholded_candidates": 0,
                "capped_candidates": 0,
                "deduplicated_candidates": 0,
                "duplicate_rejections": 0,
            },
        )
    # The gather happens before normalization and before either GEMM path.
    features_a = l2_normalize_descriptors(flat_a[gathered_a])
    features_b = l2_normalize_descriptors(flat_b[gathered_b])
    common = dict(
        temperature=temperature,
        threshold=threshold,
        max_coarse=max_coarse,
        policy=policy,
    )
    if force_chunked:
        return _chunked_dual_softmax(
            features_a,
            features_b,
            gathered_a,
            gathered_b,
            chunk_rows=chunk_rows,
            **common,
        )
    return _full_dual_softmax(features_a, features_b, gathered_a, gathered_b, **common)
