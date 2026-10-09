"""Joint GHIM/H0 and GT-decoupled Lc/Lf/Lq training."""
import argparse
from dataclasses import asdict,replace
import fcntl
import json
import math
import os
import random
import time
import atexit
from datetime import timedelta
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.data import SharedPairDataset
from mhinet.engine.checkpointing import load_checkpoint,capture_rng_state,restore_rng_state
from .semidense import SemidenseConfig
from .training import load_first_stage
from mhinet.engine.semidense_runtime import (
    boundary_actions, rank_zero_work, reconcile_training_log, resume_expectations,
    restore_training_rng, save_training_checkpoint, validation_indices,
    restore_rank_buffers, validation_buffers, restore_model_buffers,
)


def batch_loss(system,batch,device):
    tensors={k:v.to(device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
    loss,records=system(tensors['images'],tensors['H_gt_norm'],tensors['mask_A_overlap'],tensors['mask_B_overlap'])
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite semidense loss')
    return loss,records


def configure_h0_training(system,config):
    """Require direct supervision whenever the GHIM head is trainable."""
    frozen=bool(config.get('freeze_h0',True))
    weight=config.get('h0_loss_weight',0. if frozen else 1.)
    system.configure_h0_supervision(weight,config.get('h0_loss_weights'))
    if not frozen and system.h0_loss_weight==0:
        raise ValueError('Trainable GHIM/H0 requires h0_loss_weight > 0')
    return dict(weight=system.h0_loss_weight,**system.h0_loss_weights)


def parameter_gradient_norm(parameters):
    squares=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.


@torch.no_grad()
def validate(system,dataset,device,output,step,*,sample_count=None,rank=0,world_size=1,control_group=None):
    state=capture_rng_state(current_cuda_only=True)
    was_training=system.training
    indices=validation_indices(len(dataset),sample_count)
    local_indices=indices[rank::world_size]
    rows=[];error=None
    saved_buffers=None
    started=time.perf_counter()
    try:
        saved_buffers=validation_buffers(system,rank=rank,control_group=control_group)
        system.eval()
        loader=DataLoader(Subset(dataset,local_indices),batch_size=1,num_workers=0,
                          generator=torch.Generator().manual_seed(0))
        try:
            for number,(index,batch) in enumerate(zip(local_indices,loader),1):
                # Fixed per-pair draws make the metric independent of rank count.
                random.seed(index);np.random.seed(index);torch.manual_seed(index)
                loss,records=batch_loss(system,batch,device)
                rows.append(dict(index=index,pair_id=batch['pair_id'][0],loss=float(loss),**records[0]))
                if rank==0 and (number%128==0 or number==len(local_indices)):
                    print(f'VAL step={step} rank0_pairs={number}/{len(local_indices)}',flush=True)
        except Exception as exc:
            error=f'rank {rank}: {type(exc).__name__}: {exc}'
        shards=[dict(rows=rows,error=error)]
        if control_group is not None:
            import torch.distributed as dist
            shards=[None]*world_size
            dist.all_gather_object(shards,dict(rows=rows,error=error),group=control_group)
        errors=[shard['error'] for shard in shards if shard['error']]
        if errors:raise RuntimeError('Validation failed: '+'; '.join(errors))
        rows=sorted([row for shard in shards for row in shard['rows']],key=lambda row:row['index'])
        if [row['index'] for row in rows]!=indices:raise RuntimeError('Validation coverage mismatch')
        def write_report():
            metric_keys=['lc','lf','lq','fine_coverage','q_control','q_center']
            metric_keys += [key for key in ('lh0','h0_geo','h0_mat','h0_cls','h0_h',
                'h0_supervised_points','h0_fit_valid','h0_valid_H_pairs') if all(key in x for x in rows)]
            report=dict(step=step,pairs=len(rows),loss=float(np.mean([x['loss'] for x in rows])),
                metrics={key:float(np.mean([x[key] for x in rows])) for key in metric_keys},
                h0_valid_rate=float(np.mean([x['h0_valid'] for x in rows])) if all('h0_valid' in x for x in rows) else None,
                validation_total_pairs=len(dataset),validation_indices=indices,
                validation_sample='full' if len(rows)==len(dataset) else 'evenly spaced',
                rng_policy='per_manifest_index_v1',world_size=world_size,seconds=time.perf_counter()-started,
                scope='Independent GHIM/H0 and GT-decoupled matching losses; use evaluate-semidense for cascaded accuracy')
            folder=output/'validation';folder.mkdir(exist_ok=True)
            target=folder/f'step_{step:07d}.json'
            temporary=target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(dict(summary=report,pairs=rows),indent=2));temporary.replace(target)
            print(f'VAL completed step={step} pairs={len(rows)} seconds={report["seconds"]:.1f}',flush=True)
            return report
        return rank_zero_work(write_report,rank=rank,control_group=control_group)
    finally:
        if saved_buffers is not None:restore_model_buffers(system,saved_buffers)
        restore_rng_state(state);system.train(was_training)


def main(argv=None,*,source_loader=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,help='New-run configuration; resumes default to the saved configuration')
    p.add_argument('--checkpoint',type=Path,help='New-run source; resumes default to the recorded source')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--validation-pairs',type=int,help='Evenly spaced validation cohort; 0=full; resumes retain the saved policy')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-steps',type=int,help='Diagnostic stop, preserves configured schedule')
    p.add_argument('--limit-train',type=int);p.add_argument('--limit-val',type=int)
    p.add_argument('--input-size',type=int,choices=(512,784),help='Override matcher input_size for this new training protocol')
    args=p.parse_args(argv)
    world_size=int(os.environ.get('WORLD_SIZE','1'))
    distributed=world_size>1
    rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    control_group=None
    if distributed:
        if not torch.cuda.is_available():raise RuntimeError('Distributed semidense training requires CUDA')
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        # CPU control collectives may span a full validation or plot export.
        # NCCL retains its short failure timeout for actual gradient collectives.
        control_group=dist.new_group(backend='gloo',timeout=timedelta(hours=2))
        atexit.register(dist.destroy_process_group)
    resume_payload=torch.load(args.resume,map_location='cpu',weights_only=True,mmap=True) if args.resume else None
    if not args.resume and (args.config is None or args.checkpoint is None):
        p.error('New runs require --config and --checkpoint')
    config=json.loads(args.config.read_text()) if args.config else deepcopy(resume_payload['metadata']['config'])
    if args.checkpoint is None:args.checkpoint=Path(resume_payload['metadata']['source_checkpoint'])
    if args.validation_pairs is None and args.resume:
        args.validation_pairs=(resume_payload.get('auxiliary_state') or {}).get('execution',{}).get('validation_pairs')
    if args.validation_pairs==0:args.validation_pairs=None
    if args.input_size is not None:config.setdefault('matcher',{})['input_size']=args.input_size
    mc=SemidenseConfig(**config.get('matcher',{}))
    for key in ('epochs','accumulation','save_every','validate_every'):
        if config[key]<1:raise ValueError(key)
    if any(x is not None and x<1 for x in (args.max_steps,args.limit_train,args.limit_val,args.validation_pairs)):raise ValueError('Invalid diagnostic limit')
    device=f'cuda:{local_rank}' if distributed else args.device
    runtime=replace(RuntimePaths.from_json(config['runtime']),device=device)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    seed=config.get('seed',0);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    configured_tier=config.get('tier')
    train=SharedPairDataset(runtime.data_root/'train/pairs.jsonl',tier=configured_tier,max_pairs=args.limit_train,image_size=mc.input_size)
    val=SharedPairDataset(runtime.data_root/'val/pairs.jsonl',tier=configured_tier,max_pairs=args.limit_val,image_size=mc.input_size)
    if not len(train) or not len(val):raise ValueError('Empty dataset')
    if ({x.geo_group for x in train.index}&{x.geo_group for x in val.index}
        or {x.parent_group for x in train.index}&{x.parent_group for x in val.index}):raise ValueError('Train/val group leakage')
    out=args.output.resolve()
    lock=None
    def prepare_output():
        nonlocal lock
        if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True)
        lock=(out/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rank_zero_work(prepare_output,rank=rank,control_group=control_group)
    accumulation=config['accumulation']
    local_epoch_size=math.ceil(len(train)/world_size) if distributed else len(train)
    per_epoch=math.ceil(local_epoch_size/accumulation)
    total_steps=per_epoch*config['epochs'];end=min(total_steps,args.max_steps or total_steps)
    metadata=dict(task='semidense_qrru_v1',config=config,matcher=asdict(mc),
        source_checkpoint=str(args.checkpoint.resolve()),source_sha256=sha256_file(args.checkpoint),
        runtime=asdict(runtime),train_sha256=sha256_file(train.manifest),val_sha256=sha256_file(val.manifest),
        train_pairs=len(train),val_pairs=len(val),total_steps=total_steps,world_size=world_size,
        global_batch_pairs=accumulation*world_size,
        dino_sha256=sha256_file(runtime.dino_checkpoint),
        implementation={x.name:sha256_file(x) for x in Path(__file__).parent.glob('*.py')},
        data_loader_sha256=sha256_file(Path(__file__).resolve().parents[1]/'dataio'/'data.py'),
        shared_data_loader_sha256=sha256_file(Path(__file__).resolve().parents[1]/'pretraining'/'data.py'),
        training_runtime_sha256=sha256_file(Path(__file__).resolve().parents[1]/'engine'/'semidense_runtime.py'),
        checkpoint_io_sha256=sha256_file(Path(__file__).resolve().parents[1]/'engine'/'checkpointing.py'))
    freeze_mvt=bool(config.get('freeze_mvt',True))
    freeze_h0=bool(config.get('freeze_h0',True))
    metadata['freeze_policy']=f"DINOv3 frozen; MVT={'frozen' if freeze_mvt else 'trainable'}; GHIM/H0={'frozen' if freeze_h0 else 'trainable'}; VGG BN running stats fixed; D1 inactive"
    metadata=json.loads(json.dumps(metadata,default=str))
    if args.resume:
        from mhinet.pretraining.model import build_shared_network
        from .training import SemidenseSystem
        from .semidense import SemidenseMatcher
        shared,_=build_shared_network(runtime,lora=False)
        system=SemidenseSystem(shared,SemidenseMatcher(mc)).to(runtime.device)
    else:
        system=(source_loader or load_first_stage)(runtime,args.checkpoint,mc)
    system.shared.set_training_groups(mvt=not freeze_mvt,vgg=True,dedode=True,ghim_head=not freeze_h0)
    metadata['h0_supervision']=configure_h0_training(system,config)
    if system.h0_loss_weight>0:
        from mhinet.engine import ghim_losses
        metadata['ghim_loss_sha256']=sha256_file(Path(ghim_losses.__file__))
    if any(p.requires_grad for p in system.shared.dino.parameters()):
        raise RuntimeError('Freeze policy violated: DINOv3 must remain frozen')
    for name,expected_frozen in (('mvt',freeze_mvt),('stage1_head',freeze_h0)):
        if any(p.requires_grad==expected_frozen for p in getattr(system.shared,name).parameters()):
            raise RuntimeError(f'Freeze policy violated: {name} expected frozen={expected_frozen}')
    if args.resume:
        metadata['initial_matcher_sha256']=resume_payload['metadata']['initial_matcher_sha256']
    else:
        import hashlib
        initial_digest=hashlib.sha256()
        for name,value in sorted(system.matcher.state_dict().items()):
            initial_digest.update(name.encode());initial_digest.update(value.detach().cpu().numpy().tobytes())
        metadata['initial_matcher_sha256']=initial_digest.hexdigest()
    optimizer=torch.optim.AdamW(system.optimizer_groups(config.get('shared_lr',1e-5),config.get('head_lr',1e-4)),weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_steps)
    completed=cursor=epoch=0
    if args.resume:
        state=load_checkpoint(args.resume,model=system,optimizer=optimizer,scheduler=scheduler,
            map_location='cpu',restore_rng=False,checkpoint_payload=resume_payload,
            expected_metadata=resume_expectations(metadata,resume_payload['metadata']))
        completed=state['optimizer_step'];cursor=state['data_progress']['cursor'];epoch=state['data_progress']['epoch']
    train_system=system
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        train_system=DistributedDataParallel(system,device_ids=[local_rank],output_device=local_rank,
            broadcast_buffers=False,find_unused_parameters=True)
    from .visualize import snapshot,update_overview
    if args.resume:restore_rank_buffers(resume_payload,system,rank)
    if not args.resume:
        random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    rng_continuity=(restore_training_rng(resume_payload,rank=rank,world_size=world_size,seed=seed)
                    if args.resume else 'new_run')
    def write_run():
        if args.resume:
            discarded=reconcile_training_log(out,completed)
            event=dict(step=completed,checkpoint=str(args.resume.resolve()),discarded_log_rows=discarded,
                       rng_continuity=rng_continuity,previous_implementation=resume_payload['metadata']['implementation'],
                       current_implementation=metadata['implementation'],time=time.time())
            with (out/'resume_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(f'RESUMED step={completed} epoch={epoch} cursor={cursor} RNG={rng_continuity}',flush=True)
        (out/'run.json').write_text(json.dumps(metadata,indent=2))
        execution=dict(validation_pairs=args.validation_pairs,validation_total_pairs=len(val),
                       validation_indices=validation_indices(len(val),args.validation_pairs),
                       validation_rng_policy='per_manifest_index_v1',world_size=world_size,
                       cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),resume_step=completed,
                       h0_gradient_path=('Independent GHIM geo/mat/cls/H supervision updates H0 and its upstream MVT; matcher prior remains detached'
                           if system.h0_loss_weight>0 else 'Detached prior; GHIM/H0 supervision disabled'),
                       h0_supervision=metadata['h0_supervision'],
                       training_records_scope='rank0 local microbatches')
        (out/'execution.json').write_text(json.dumps(execution,indent=2))
    rank_zero_work(write_run,rank=rank,control_group=control_group)
    del resume_payload
    system.train()
    vis=config.get('visualization',{})
    def save_boundary():
        save_training_checkpoint(out/'latest.pt',model=system,optimizer=optimizer,scheduler=scheduler,
            step=completed,cursor=cursor,epoch=epoch,metadata=metadata,rank=rank,
            world_size=world_size,control_group=control_group,
            execution_settings=dict(validation_pairs=args.validation_pairs))
    if not args.resume and vis.get('enabled'):
        save_boundary()
        rank_zero_work(lambda:snapshot(system,val,runtime.device,out,completed,
            pairs=vis.get('pairs',12),all_channels=True),rank=rank,control_group=control_group)
    frozen_parameters=[(name,p) for name,p in system.shared.named_parameters() if not p.requires_grad]
    trainable_parameters=[p for p in system.parameters() if p.requires_grad]
    while completed<end:
        if cursor==local_epoch_size:epoch+=1;cursor=0
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler=DistributedSampler(train,num_replicas=world_size,rank=rank,shuffle=True,seed=seed,drop_last=False)
            sampler.set_epoch(epoch);order=list(iter(sampler))
        else:
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        if len(order)!=local_epoch_size:raise RuntimeError(f'Unexpected local epoch size {len(order)} != {local_epoch_size}')
        loader=iter(DataLoader(Subset(train,order[cursor:]),batch_size=1,num_workers=0,
            generator=torch.Generator().manual_seed(seed+epoch)))
        while cursor<local_epoch_size and completed<end:
            started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            count=min(accumulation,local_epoch_size-cursor);records=[];micro_losses=[]
            for micro in range(count):
                sync_context=(train_system.no_sync() if distributed and micro<count-1 else nullcontext())
                with sync_context:
                    batch=next(loader)
                    loss,info=batch_loss(train_system,batch,runtime.device)
                    (loss/count).backward()
                micro_losses.append(float(loss.detach()))
                for index,record in enumerate(info):
                    record['pair_id']=batch['pair_id'][index]
                    record['dataset_index']=order[cursor+index]
                records.extend(info);cursor+=1
            frozen_bad=[name for name,param in frozen_parameters if param.grad is not None]
            if frozen_bad:raise RuntimeError(f'Frozen gradients: {frozen_bad[:3]}')
            if not freeze_h0 and all(p.grad is None for p in system.shared.stage1_head.parameters()):
                raise RuntimeError('Trainable GHIM/H0 received no gradient despite enabled supervision')
            group_grads={}
            if rank==0:
                for group in optimizer.param_groups:
                    group_grads[group['name']]=parameter_gradient_norm(group['params'])
                group_grads['ghim_h0']=parameter_gradient_norm(system.shared.stage1_head.parameters())
                group_grads['mvt']=parameter_gradient_norm(system.shared.mvt.parameters())
            grad=torch.nn.utils.clip_grad_norm_(trainable_parameters,1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();completed+=1
            if rank==0:
                row=dict(step=completed,epoch=epoch,cursor=cursor,gradient_norm=float(grad),
                    loss=float(np.mean(micro_losses)),
                    seconds=time.perf_counter()-started,records=records,gradient_groups=group_grads,
                    lr_shared=optimizer.param_groups[0]['lr'],lr_head=optimizer.param_groups[1]['lr'],
                    tau_c=float(system.matcher.tau_c.detach()),tau_f=float(system.matcher.tau_f.detach()),
                    records_scope='rank0_local_microbatches',world_size=world_size,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if runtime.device.startswith('cuda') else 0)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            actions=boundary_actions(completed,end,config)
            if actions['save']:save_boundary()
            if actions['validation']:
                validate(system,val,runtime.device,out,completed,sample_count=args.validation_pairs,
                         rank=rank,world_size=world_size,control_group=control_group)
            if actions['snapshot']:
                rank_zero_work(lambda:snapshot(system,val,runtime.device,out,completed,pairs=vis.get('pairs',12),
                    all_channels=completed==end or completed==math.ceil(total_steps/2/vis.get('every',1000))*vis.get('every',1000)),
                    rank=rank,control_group=control_group)
            elif actions['curves']:
                rank_zero_work(lambda:update_overview(out),rank=rank,control_group=control_group)
    return 0


if __name__=='__main__':raise SystemExit(main())
