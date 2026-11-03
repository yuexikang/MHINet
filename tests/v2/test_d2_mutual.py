"""Reciprocity, invalid support, and stable output lineage for Pair-D2."""
import torch
from mhinet.v2.pair_matching import local
from mhinet.v2 import pipeline
from types import SimpleNamespace


def fixture_windows():
    logp = torch.full((2, 16, 64), -10.)
    index = torch.arange(16)
    logp[:, index, index] = 9
    # Two queries compete for one target, with a deterministic tie on parent 1.
    logp[:, 1, 1] = -10
    logp[0, 1, 0] = 8
    logp[1, 1, 0] = 9
    source = torch.arange(64).reshape(2, 16, 2).float()+20
    target = source+1
    valid = torch.ones_like(logp, dtype=torch.bool)
    valid[:, 2] = False
    target[:, 3, 0] = float('nan')
    target[:, 4, 0] = 512
    source[:, 5, 0] = -1
    windows = dict(source=source, parents=torch.tensor([[3.5,3.5],[11.5,3.5]]),
                   valid_a=torch.ones(2,64,dtype=torch.bool),
                   valid_b=torch.ones(2,64,dtype=torch.bool), score8=torch.ones(2))
    fine = dict(points_b=target, logp=logp, valid=valid, confidence=torch.ones(2,16))
    keep = torch.ones(2,16,dtype=torch.bool)
    keep[:,1:6] = False
    return windows, fine, keep.flatten()


def test_reciprocity_collisions_ties_and_invalid_rows():
    windows, fine, expected = fixture_windows()
    torch.testing.assert_close(local.mutual_filter_mask(windows, fine), expected)


def test_empty_parent_set():
    windows = dict(source=torch.empty(0,16,2))
    fine = dict(points_b=torch.empty(0,16,2), logp=torch.empty(0,16,64),
                valid=torch.empty(0,16,64,dtype=torch.bool))
    assert local.mutual_filter_mask(windows,fine).shape == (0,)


def test_both_entries_filter_all_output_fields_in_order(monkeypatch):
    windows, fine, expected = fixture_windows()
    calls = []
    def match(*args):
        calls.append(1)
        return fine
    for module in (local,pipeline):
        monkeypatch.setattr(module,'extract_windows',lambda *args: windows)
        monkeypatch.setattr(module,'match_windows',match)
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor=torch.nn.Parameter(torch.zeros(()))
            self.calls=0
        def forward(self,*args,**kwargs):
            self.calls+=1
            return shared
    shared={key:None for key in ('flow16_norm','flow16_target_norm','flow16_matchability','flow16_valid')}
    shared['call_counts']={}
    windows.update(centers=windows['parents'])
    matcher=SimpleNamespace(tau_f=1.)
    model=pipeline.MHIV2(Shared(),matcher,None)
    baseline=model.match_shared(shared)
    filtered=model.match(torch.zeros(2,3,512,512),mutual=True,diagnostics=True)
    assert model.shared.calls==1 and len(calls)==2
    for key,value in baseline.items():
        torch.testing.assert_close(filtered[key],value[expected])
    torch.testing.assert_close(filtered['diagnostics']['output_mask'],expected)
    # Full windows remain indexed by original parents even after point filtering.
    assert filtered['diagnostics']['candidate_logp'].shape==(2,16,64)
    legacy=local.infer(shared,matcher,None,mutual=True)
    for key,value in legacy.items():
        torch.testing.assert_close(value,filtered[key])
