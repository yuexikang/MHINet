"""Exercise real two-process synchronization and stochastic resume on tiny models."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from mhinet.downstream.train import validate
from mhinet.engine.checkpointing import capture_rng_state, load_checkpoint, save_checkpoint
from mhinet.engine.semidense_runtime import (
    LEGACY_ORCHESTRATION, boundary_actions, rank_zero_work, reconcile_training_log,
    restore_training_rng, resume_expectations, save_training_checkpoint, validation_indices,
    restore_rank_buffers,
)


class TinySystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(3,5),nn.BatchNorm1d(5),nn.Dropout(.3),nn.Linear(5,2))

    def forward(self, images, *unused):
        value=self.layers(images).square().mean()+torch.rand((),device=images.device)*.01
        record={key:float(value.detach()) for key in ('lc','lf','lq','fine_coverage','q_control','q_center')}
        return value,[record]


class TinyPairs(Dataset):
    def __init__(self, fail=False):self.fail=fail
    def __len__(self):return 9
    def __getitem__(self,index):
        if self.fail and index==1:raise ValueError('broken validation pair')
        return dict(images=torch.tensor([index/10, .2, .3]),H_gt_norm=torch.eye(3),
                    mask_A_overlap=torch.ones(1),mask_B_overlap=torch.ones(1),pair_id=f'p{index}')


def _distributed_worker(rank, root, cuda):
    torch.set_num_threads(1)
    device=f'cuda:{rank}' if cuda else 'cpu'
    if cuda:torch.cuda.set_device(rank)
    backend='nccl' if cuda else 'gloo'
    dist.init_process_group(backend,init_method='file://'+str(Path(root)/'rendezvous'),
                            rank=rank,world_size=2,timeout=timedelta(seconds=30))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=30))
    gradient_group=dist.new_group(backend=backend,timeout=timedelta(seconds=2))
    try:
        system=TinySystem().to(device)
        wrap=lambda module:DDP(module,process_group=gradient_group,broadcast_buffers=False,**({'device_ids':[rank]} if cuda else {}))
        model=wrap(system)
        optimizer=torch.optim.AdamW(system.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
        random.seed(37+rank);np.random.seed(37+rank);torch.manual_seed(37+rank)

        def step(model,optimizer,scheduler):
            optimizer.zero_grad()
            loss,_=model(torch.randn(4,3,device=device)+random.random()+float(np.random.rand()))
            loss.backward();optimizer.step();scheduler.step()

        step(model,optimizer,scheduler)
        checkpoint=Path(root)/'latest.pt'
        save_training_checkpoint(checkpoint,model=system,optimizer=optimizer,scheduler=scheduler,
            step=1,cursor=4,epoch=0,metadata={'task':'test'},rank=rank,world_size=2,control_group=control)
        step(model,optimizer,scheduler)
        expected={key:value.clone() for key,value in system.state_dict().items()}
        expected_lr=optimizer.param_groups[0]['lr']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        replacement=TinySystem().to(device)
        replacement_ddp=wrap(replacement)
        replacement_optimizer=torch.optim.AdamW(replacement.parameters(),lr=.5)
        replacement_scheduler=torch.optim.lr_scheduler.StepLR(replacement_optimizer,1,.9)
        load_checkpoint(checkpoint,model=replacement,optimizer=replacement_optimizer,
            scheduler=replacement_scheduler,map_location='cpu',restore_rng=False,checkpoint_payload=payload)
        restore_rank_buffers(payload,replacement,rank)
        assert restore_training_rng(payload,rank=rank,world_size=2,seed=37)=='exact_per_rank'
        step(replacement_ddp,replacement_optimizer,replacement_scheduler)
        for key,value in replacement.state_dict().items():
            torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
        assert replacement_optimizer.param_groups[0]['lr']==expected_lr

        # A diagnostic longer than the gradient group's timeout must not leave
        # an NCCL barrier queued, and must not prevent the next backward pass.
        rank_zero_work(lambda:time.sleep(3),rank=rank,control_group=control)
        step(replacement_ddp,replacement_optimizer,replacement_scheduler)

        before=capture_rng_state(current_cuda_only=True)
        validate(replacement,TinyPairs(),device,Path(root),7,sample_count=7,
                 rank=rank,world_size=2,control_group=control)
        after=capture_rng_state(current_cuda_only=True)
        torch.testing.assert_close(before['torch_cpu'],after['torch_cpu'],rtol=0,atol=0)
        for first,last in zip(before['torch_cuda'],after['torch_cuda']):
            torch.testing.assert_close(first,last,rtol=0,atol=0)
        assert before['python']==after['python']
        assert replacement.training

        def compare_serial():
            validate(replacement,TinyPairs(),device,Path(root),8,sample_count=7)
            parallel=json.loads((Path(root)/'validation/step_0000007.json').read_text())
            serial=json.loads((Path(root)/'validation/step_0000008.json').read_text())
            assert parallel['pairs']==serial['pairs']
            assert parallel['summary']['pairs']==7
            assert len({row['pair_id'] for row in parallel['pairs']})==7
        rank_zero_work(compare_serial,rank=rank,control_group=control)
        with pytest.raises(RuntimeError,match='broken validation pair'):
            validate(replacement,TinyPairs(fail=True),device,Path(root),9,
                     rank=rank,world_size=2,control_group=control)

        def fail():raise ValueError('diagnostic IO error')
        with pytest.raises(RuntimeError,match='diagnostic IO error'):
            rank_zero_work(fail,rank=rank,control_group=control)
        (Path(root)/f'passed_rank_{rank}').write_text('ok')
    finally:
        dist.destroy_process_group()


def test_two_rank_resume_and_validation(tmp_path):
    cuda=os.environ.get('MHINET_TEST_CUDA')=='1'
    if cuda and torch.cuda.device_count()<2:pytest.skip('requires two GPUs')
    mp.spawn(_distributed_worker,args=(str(tmp_path),cuda),nprocs=2,join=True)
    assert all((tmp_path/f'passed_rank_{rank}').exists() for rank in range(2))


def test_resume_compatibility_does_not_relax_model_or_data(tmp_path):
    old=dict(implementation={**LEGACY_ORCHESTRATION,'qrru.py':'model-v1'},config={'lr':.01})
    current=dict(implementation={'train.py':'reviewed-trainer','visualize.py':'reviewed-visuals','qrru.py':'model-v1'},
                 config={'lr':.01},training_runtime_sha256='runtime',checkpoint_io_sha256='io',shared_data_loader_sha256='data')
    model=nn.Linear(2,1);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    path=tmp_path/'legacy.pt'
    save_checkpoint(path,model=model,optimizer=optimizer,optimizer_step=1,metadata=old)
    load_checkpoint(path,model=model,map_location='cpu',restore_rng=False,
                    expected_metadata=resume_expectations(current,old))
    for kind in ('model','data'):
        incompatible=deepcopy(current)
        if kind=='model':incompatible['implementation']['qrru.py']='model-v2'
        else:incompatible['config']['lr']=.02
        with pytest.raises(RuntimeError,match='metadata mismatch'):
            load_checkpoint(path,model=model,map_location='cpu',restore_rng=False,
                            expected_metadata=resume_expectations(incompatible,old))


def test_log_rewind_archives_unsaved_rows_and_partial_tail(tmp_path):
    path=tmp_path/'train.jsonl'
    path.write_text(''.join(json.dumps({'step':i})+'\n' for i in range(1,5))+'{"step":')
    validation=tmp_path/'validation';validation.mkdir()
    (validation/'step_0000004.json').write_text('{}')
    assert reconcile_training_log(tmp_path,2)==3
    assert [json.loads(line)['step'] for line in path.read_text().splitlines()]==[1,2]
    archive=list((tmp_path/'resume_history').glob('*/discarded_train.jsonl'))
    assert len(archive)==1 and '"step": 4' in archive[0].read_text()
    assert not (validation/'step_0000004.json').exists()
    assert reconcile_training_log(tmp_path,2)==0


def test_validation_boundary_is_saved_once_without_periodic_save():
    config=dict(save_every=250,validate_every=5107,visualization={'enabled':True,'every':2000,'curves_every':100})
    assert not any(boundary_actions(5001,25535,config).values())
    assert boundary_actions(5107,25535,config)==dict(save=True,validation=True,snapshot=False,curves=False)
    assert boundary_actions(6000,25535,config)==dict(save=True,validation=False,snapshot=True,curves=True)
    assert validation_indices(9,7)==[0,1,2,4,5,6,8]


def test_legacy_rng_reports_inexact_ddp_resume():
    payload=dict(rng=capture_rng_state(),progress={'optimizer_step':10})
    assert restore_training_rng(payload,rank=1,world_size=2,seed=42)=='legacy_missing_rank_rng'
    first=torch.rand(5)
    restore_training_rng(payload,rank=1,world_size=2,seed=42)
    torch.testing.assert_close(torch.rand(5),first,rtol=0,atol=0)
