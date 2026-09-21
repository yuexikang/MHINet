from __future__ import annotations

import pytest
import torch

from mhinet.downstream.loma_reference.coarse_matcher import match_d8


@pytest.mark.parametrize("policy", ["mutual", "dual_topk"])
def test_chunked_and_full_gathered_paths_agree(policy: str) -> None:
    generator = torch.Generator().manual_seed(17)
    a = torch.randn((256, 98, 98), generator=generator)
    b = torch.randn((256, 98, 98), generator=generator)
    mask_a = torch.zeros((98, 98), dtype=torch.bool)
    mask_b = torch.zeros((98, 98), dtype=torch.bool)
    mask_a[10:14, 20:25] = True
    mask_b[30:35, 40:46] = True
    common = dict(
        temperature=0.05,
        threshold=0.0,
        max_coarse=40,
        policy=policy,
        chunk_rows=7,
    )
    full = match_d8(a, b, mask_a, mask_b, force_chunked=False, **common)
    chunked = match_d8(a, b, mask_a, mask_b, force_chunked=True, **common)
    torch.testing.assert_close(chunked.source_flat, full.source_flat)
    torch.testing.assert_close(chunked.target_flat, full.target_flat)
    torch.testing.assert_close(chunked.confidence, full.confidence, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        chunked.prefilter_target_for_source, full.prefilter_target_for_source
    )
    assert full.diagnostics["matrix_rows"] == int(mask_a.sum())
    assert full.diagnostics["matrix_cols"] == int(mask_b.sum())


def test_empty_support_never_correlates_full_grid() -> None:
    descriptor = torch.zeros((256, 98, 98))
    empty = torch.zeros((98, 98), dtype=torch.bool)
    result = match_d8(
        descriptor,
        descriptor,
        empty,
        empty,
        temperature=0.05,
        threshold=0,
        max_coarse=10,
        policy="mutual",
        chunk_rows=4,
        force_chunked=True,
    )
    assert not result.source_flat.numel()
    assert result.diagnostics["matrix_elements"] == 0
