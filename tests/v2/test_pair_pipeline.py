"""Production contracts and legacy compatibility, without downloading weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from mhinet.v2.pipeline import MHIV2, MatchingTemperatures, load_components, load_mhi_v2, DEFAULT_CONFIG
from mhinet.v2.config import SemidenseConfigV2
from mhinet.v2.pair_matching.local import LocalPairModulation
from mhinet.v2.pair_matching.coarse import select_local_parents,local_coarse
from mhinet.v2.coarse_matcher import flow_prior_from_shared
from mhinet.v2.pair_matching.continuous import source_grid


def features(valid=True):
    torch.manual_seed(1)
    flow=torch.zeros(1,2,32,32)
    axis=torch.linspace(-1+1/32,1-1/32,32)
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    return dict(pyramid={8:torch.randn(1,2,256,64,64),2:torch.randn(1,2,256,256,256)},
                flow16_norm=flow,flow16_target_norm=torch.stack((x,y))[None],
                flow16_matchability=torch.ones(1,1,32,32),flow16_valid=torch.full((1,1,32,32),valid),
                call_counts={'homography_fitter':0})


def test_gt_free_contract_and_lineage():
    shared=features();matcher=MatchingTemperatures(SemidenseConfigV2())
    model=MHIV2(torch.nn.Identity(),matcher,LocalPairModulation(),max_parents=2).eval()
    result=model.match_shared(shared,diagnostics=True)
    assert result['points_a'].shape==result['points_b'].shape==(32,2)
    assert result['confidence'].shape==result['valid'].shape==(32,)
    assert torch.isfinite(result['points_b']).all()
    assert ((result['confidence']>=0)&(result['confidence']<=1)).all()
    assert result['parent'].tolist()==[0]*16+[1]*16
    assert not hasattr(matcher,'qrru') and not hasattr(matcher,'dgfo')
    trace=result['diagnostics']
    assert trace['candidate_logp'].shape==(2,16,64)
    assert trace['windows']['a'].shape==trace['windows']['b'].shape==(2,64,256)
    prior=flow_prior_from_shared(shared,0,(64,64))
    ids,_,_,_=local_coarse(shared['pyramid'][8][0],prior,matcher)
    torch.testing.assert_close(select_local_parents(shared,matcher,2),source_grid(ids.device)[ids[:2]],rtol=0,atol=0)


def test_empty_matches():
    matcher=MatchingTemperatures(SemidenseConfigV2())
    model=MHIV2(torch.nn.Identity(),matcher,LocalPairModulation())
    result=model.match_shared(features(False))
    assert result['points_a'].shape==result['points_b'].shape==(0,2)
    assert result['confidence'].shape==result['valid'].shape==(0,)


def test_component_loading_strict_and_temperatures():
    def linear():return torch.nn.Linear(2,2)
    shared=SimpleNamespace(mvt=linear(),stage1_head=linear(),vgg=linear(),dedode=linear())
    matcher=MatchingTemperatures(SemidenseConfigV2());adapter=LocalPairModulation()
    flow={k:getattr(shared,v).state_dict() for k,v in [('mvt','mvt'),('head','stage1_head')]}
    d2={k:getattr(shared,k).state_dict() for k in ('vgg','dedode')}
    d2.update(adapter=adapter.state_dict(),temperatures={'log_tau_c':torch.tensor(-3.),'log_tau_f':torch.tensor(-4.)})
    load_components(shared,matcher,adapter,flow,d2)
    assert matcher.log_tau_c.item()==-3 and matcher.log_tau_f.item()==-4
    with pytest.raises(RuntimeError):load_components(shared,matcher,adapter,{**flow,'head':{}},d2)


def test_release_disabled_experiments_and_input_shape():
    c=json.loads(DEFAULT_CONFIG.read_text());c['qrru']=True
    with pytest.raises(ValueError,match='Experimental'):load_mhi_v2(c,'cpu')
    model=MHIV2(torch.nn.Identity(),MatchingTemperatures(SemidenseConfigV2()),LocalPairModulation())
    with pytest.raises(ValueError,match='single pair'):model.match(torch.empty(2,3,256,256))
    # Historical H0 configuration and expanded Fine window are still available.
    legacy=SemidenseConfigV2(geometry_mode='h0',fine_prior='h0_residual')
    assert legacy.fine_target_side==8


def test_single_stage_resume_does_not_read_initial_weights(monkeypatch,tmp_path):
    from dataclasses import asdict
    from mhinet.v2.pair_matching import training
    def linear():return torch.nn.Linear(2,2)
    shared=SimpleNamespace(mvt=linear(),stage1_head=linear(),vgg=linear(),dedode=linear())
    monkeypatch.setattr(training,'build_shared',lambda *a:shared)
    adapter=LocalPairModulation()
    matcher=MatchingTemperatures(SemidenseConfigV2())
    for stage in ('flow','d2'):
        checkpoint=tmp_path/f'{stage}.pt'
        torch.save(dict(stage=stage,matcher_config=asdict(matcher.config),
                        **training.snapshot(shared,matcher,adapter,stage)),checkpoint)
        # No initial_locator/downstream/adapter keys at all: these must not be read.
        model,restored,local=training.initialize({'runtime':{}},'cpu',tmp_path,checkpoint)
        assert model is shared
        assert set(restored.state_dict())=={'log_tau_c','log_tau_f'}
        assert local.mode=='pair'
