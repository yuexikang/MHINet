"""Independent Lc/Lf/Lq trainer; leaves first-stage training untouched."""
import argparse
from dataclasses import asdict,replace
import fcntl
import json
import math
import os
import random
import time
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
    shared=system(tensors['images'])
    loss,records=system.matcher.training_losses(shared,tensors['H_gt_norm'],tensors['mask_A_overlap'],tensors['mask_B_overlap'])
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
    args=p.parse_args(argv)
    config=json.loads(args.config.read_text());mc=SemidenseConfig(**config.get('matcher',{}))
    for key in ('epochs','accumulation','save_every','validate_every'):
        if config[key]<1:raise ValueError(key)
    if any(x is not None and x<1 for x in (args.max_steps,args.limit_train,args.limit_val)):raise ValueError('Invalid diagnostic limit')
    runtime=replace(RuntimePaths.from_json(config['runtime']),device=args.device)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    seed=config.get('seed',0);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    train=SharedPairDataset(runtime.data_root/'train/pairs.jsonl',tier=config.get('tier',1),max_pairs=args.limit_train)
    val=SharedPairDataset(runtime.data_root/'val/pairs.jsonl',tier=config.get('tier',1),max_pairs=args.limit_val)
    if not len(train) or not len(val):raise ValueError('Empty dataset')
    if ({x.geo_group for x in train.index}&{x.geo_group for x in val.index}
        or {x.parent_group for x in train.index}&{x.parent_group for x in val.index}):raise ValueError('Train/val group leakage')
    out=args.output.resolve()
    if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    accumulation=config['accumulation'];per_epoch=math.ceil(len(train)/accumulation)
    total_steps=per_epoch*config['epochs'];end=min(total_steps,args.max_steps or total_steps)
    metadata=dict(task='semidense_qrru_v1',config=config,matcher=asdict(mc),
        source_checkpoint=str(args.checkpoint.resolve()),source_sha256=sha256_file(args.checkpoint),
        runtime=asdict(runtime),train_sha256=sha256_file(train.manifest),val_sha256=sha256_file(val.manifest),
        train_pairs=len(train),val_pairs=len(val),total_steps=total_steps,
        dino_sha256=sha256_file(runtime.dino_checkpoint),
        implementation={x.name:sha256_file(x) for x in Path(__file__).parent.glob('*.py')})
    metadata=json.loads(json.dumps(metadata,default=str))
    system=load_first_stage(runtime,args.checkpoint,mc)
    optimizer=torch.optim.AdamW(system.optimizer_groups(config.get('shared_lr',1e-5),config.get('head_lr',1e-4)),weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_steps)
    completed=cursor=epoch=0
    if args.resume:
        state=load_checkpoint(args.resume,model=system,optimizer=optimizer,scheduler=scheduler,
            map_location='cpu',expected_metadata=metadata)
        completed=state['optimizer_step'];cursor=state['data_progress']['cursor'];epoch=state['data_progress']['epoch']
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    system.train()
    while completed<end:
        if cursor==len(train):epoch+=1;cursor=0
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        loader=iter(DataLoader(Subset(train,order[cursor:]),batch_size=1,num_workers=0,
            generator=torch.Generator().manual_seed(seed+epoch)))
        while cursor<len(train) and completed<end:
            started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            count=min(accumulation,len(train)-cursor);records=[]
            for _ in range(count):
                loss,info=batch_loss(system,next(loader),runtime.device)
                (loss/count).backward();records.extend(info);cursor+=1
            frozen_bad=[name for name,param in system.shared.named_parameters() if not param.requires_grad and param.grad is not None]
            if frozen_bad:raise RuntimeError(f'Frozen gradients: {frozen_bad[:3]}')
            group_grads={group['name']:float(torch.stack([p.grad.detach().float().square().sum() for p in group['params'] if p.grad is not None]).sum().sqrt()) for group in optimizer.param_groups}
            grad=torch.nn.utils.clip_grad_norm_([x for x in system.parameters() if x.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();completed+=1
            row=dict(step=completed,epoch=epoch,cursor=cursor,gradient_norm=float(grad),
                seconds=time.perf_counter()-started,records=records,gradient_groups=group_grads,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if completed%config['save_every']==0 or completed==end:
                save_checkpoint(out/'latest.pt',model=system,optimizer=optimizer,scheduler=scheduler,
                    optimizer_step=completed,data_progress=dict(cursor=cursor,epoch=epoch),metadata=metadata)
            if completed%config['validate_every']==0 or completed==end:
                validate(system,val,runtime.device,out,completed)
    return 0


if __name__=='__main__':raise SystemExit(main())
