"""Independent Lc/Lf/Lq trainer; leaves first-stage training untouched."""
import argparse
from dataclasses import asdict,replace
import fcntl
import json
import math
import os
import random
import time
import atexit
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.data import SharedPairDataset
from mhinet.engine.checkpointing import save_checkpoint,load_checkpoint,capture_rng_state,restore_rng_state
from .semidense import SemidenseConfig
from .training import load_first_stage


def batch_loss(system,batch,device):
    tensors={k:v.to(device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
    loss,records=system(tensors['images'],tensors['H_gt_norm'],tensors['mask_A_overlap'],tensors['mask_B_overlap'])
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite semidense loss')
    return loss,records


@torch.no_grad()
def validate(system,dataset,device,output,step):
    state=capture_rng_state()
    system.eval();torch.manual_seed(0)
    rows=[]
    try:
        for batch in DataLoader(dataset,batch_size=1,num_workers=0):
            loss,records=batch_loss(system,batch,device)
            rows.append(dict(pair_id=batch['pair_id'][0],loss=float(loss),**records[0]))
        report=dict(step=step,pairs=len(rows),loss=float(np.mean([x['loss'] for x in rows])),
            metrics={key:float(np.mean([x[key] for x in rows])) for key in ('lc','lf','lq','fine_coverage','q_control','q_center')},
            scope='GT-decoupled validation losses; use evaluate-semidense for cascaded matching accuracy')
        folder=output/'validation';folder.mkdir(exist_ok=True)
        (folder/f'step_{step:07d}.json').write_text(json.dumps(dict(summary=report,pairs=rows),indent=2))
        return report
    finally:
        restore_rng_state(state);system.train()


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path,help='First-stage source, including when resuming')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-steps',type=int,help='Diagnostic stop, preserves configured schedule')
    p.add_argument('--limit-train',type=int);p.add_argument('--limit-val',type=int)
    p.add_argument('--input-size',type=int,choices=(512,784),help='Override matcher input_size for this new training protocol')
    args=p.parse_args(argv)
    world_size=int(os.environ.get('WORLD_SIZE','1'))
    distributed=world_size>1
    rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if distributed:
        if not torch.cuda.is_available():raise RuntimeError('Distributed semidense training requires CUDA')
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        atexit.register(dist.destroy_process_group)
    config=json.loads(args.config.read_text())
    if args.input_size is not None:config.setdefault('matcher',{})['input_size']=args.input_size
    mc=SemidenseConfig(**config.get('matcher',{}))
    for key in ('epochs','accumulation','save_every','validate_every'):
        if config[key]<1:raise ValueError(key)
    if any(x is not None and x<1 for x in (args.max_steps,args.limit_train,args.limit_val)):raise ValueError('Invalid diagnostic limit')
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
    if rank==0:
        if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True)
        lock=(out/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if distributed:dist.barrier()
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
        data_loader_sha256=sha256_file(Path(__file__).resolve().parents[1]/'dataio'/'data.py'))
    freeze_mvt=bool(config.get('freeze_mvt',True))
    freeze_h0=bool(config.get('freeze_h0',True))
    metadata['freeze_policy']=f"DINOv3 frozen; MVT={'frozen' if freeze_mvt else 'trainable'}; GHIM/H0={'frozen' if freeze_h0 else 'trainable'}; VGG BN running stats fixed; D1 inactive"
    metadata=json.loads(json.dumps(metadata,default=str))
    system=load_first_stage(runtime,args.checkpoint,mc)
    system.shared.set_training_groups(mvt=not freeze_mvt,vgg=True,dedode=True,ghim_head=not freeze_h0)
    if any(p.requires_grad for p in system.shared.dino.parameters()):
        raise RuntimeError('Freeze policy violated: DINOv3 must remain frozen')
    for name,expected_frozen in (('mvt',freeze_mvt),('stage1_head',freeze_h0)):
        if any(p.requires_grad==expected_frozen for p in getattr(system.shared,name).parameters()):
            raise RuntimeError(f'Freeze policy violated: {name} expected frozen={expected_frozen}')
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
            map_location='cpu',expected_metadata=metadata)
        completed=state['optimizer_step'];cursor=state['data_progress']['cursor'];epoch=state['data_progress']['epoch']
    train_system=system
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        train_system=DistributedDataParallel(system,device_ids=[local_rank],output_device=local_rank,
            broadcast_buffers=False,find_unused_parameters=True)
        torch.manual_seed(seed+rank)
    if rank==0:(out/'run.json').write_text(json.dumps(metadata,indent=2))
    if distributed:dist.barrier()
    system.train()
    from .visualize import snapshot,update_overview
    vis=config.get('visualization',{})
    if rank==0 and vis.get('enabled') and not (out/'visualizations'/f'step_{completed:07d}'/'summary.json').exists():
        if completed==0:
            save_checkpoint(out/'latest.pt',model=system,optimizer=optimizer,scheduler=scheduler,
                optimizer_step=0,data_progress=dict(cursor=cursor,epoch=epoch),metadata=metadata)
        snapshot(system,val,runtime.device,out,completed,pairs=vis.get('pairs',12),all_channels=True)
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
            count=min(accumulation,local_epoch_size-cursor);records=[]
            for micro in range(count):
                sync_context=(train_system.no_sync() if distributed and micro<count-1 else __import__('contextlib').nullcontext())
                with sync_context:
                    loss,info=batch_loss(train_system,next(loader),runtime.device)
                    (loss/count).backward()
                records.extend(info);cursor+=1
            frozen_bad=[name for name,param in system.shared.named_parameters() if not param.requires_grad and param.grad is not None]
            if frozen_bad:raise RuntimeError(f'Frozen gradients: {frozen_bad[:3]}')
            group_grads={group['name']:float(torch.stack([p.grad.detach().float().square().sum() for p in group['params'] if p.grad is not None]).sum().sqrt()) for group in optimizer.param_groups}
            grad=torch.nn.utils.clip_grad_norm_([x for x in system.parameters() if x.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();completed+=1
            row=dict(step=completed,epoch=epoch,cursor=cursor,gradient_norm=float(grad),
                seconds=time.perf_counter()-started,records=records,gradient_groups=group_grads,
                lr_shared=optimizer.param_groups[0]['lr'],lr_head=optimizer.param_groups[1]['lr'],
                tau_c=float(system.matcher.tau_c.detach()),tau_f=float(system.matcher.tau_f.detach()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
            if rank==0:
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if rank==0 and (completed%config['save_every']==0 or completed==end):
                save_checkpoint(out/'latest.pt',model=system,optimizer=optimizer,scheduler=scheduler,
                    optimizer_step=completed,data_progress=dict(cursor=cursor,epoch=epoch),metadata=metadata)
            if distributed:dist.barrier()
            if rank==0 and vis.get('enabled') and (completed%vis.get('curves_every',100)==0 or completed==end):
                update_overview(out)
            if rank==0 and (completed%config['validate_every']==0 or completed==end):
                validate(system,val,runtime.device,out,completed)
            if distributed:dist.barrier()
            if rank==0 and vis.get('enabled') and (completed%vis.get('every',1000)==0 or completed==end):
                # Save exact diagnostic state even if intervals do not coincide.
                save_checkpoint(out/'latest.pt',model=system,optimizer=optimizer,scheduler=scheduler,
                    optimizer_step=completed,data_progress=dict(cursor=cursor,epoch=epoch),metadata=metadata)
                snapshot(system,val,runtime.device,out,completed,pairs=vis.get('pairs',12),
                    all_channels=completed==end or completed==math.ceil(total_steps/2/vis.get('every',1000))*vis.get('every',1000))
            if distributed:dist.barrier()
    return 0


if __name__=='__main__':raise SystemExit(main())
