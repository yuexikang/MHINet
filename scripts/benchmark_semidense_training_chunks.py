"""Fixed-input forward/backward timing and gradient audit; no optimizer updates."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.pretraining.data import SharedPairDataset
from mhinet.downstream.training import SemidenseSystem
from mhinet.downstream.semidense import SemidenseConfig,SemidenseMatcher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.repeats<1:raise ValueError('repeats must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    meta=state['metadata']
    runtime=replace(RuntimePaths.from_json(meta['config']['runtime']),device=args.device)
    if sha256_file(runtime.dino_checkpoint)!=meta['dino_sha256']:raise ValueError('DINO identity mismatch')
    shared,_=build_shared_network(runtime,lora=False)
    config=SemidenseConfig(**meta['matcher'])
    model=SemidenseSystem(shared,SemidenseMatcher(config)).to(args.device)
    model.load_state_dict(state['model'],strict=True);del state
    model.train()
    dataset=SharedPairDataset(runtime.data_root/'train/pairs.jsonl',tier=3)
    indices=[0,100,1000]
    groups=model.optimizer_groups()
    parameters=[p for g in groups for p in g['params']]
    records=[]
    report=dict(scope='Three fixed tier3 training pairs; full shared forward + Lc/Lf/Lq + backward; no optimizer update. Excludes data IO, transfer, gradient clipping, optimizer, validation and visualization; not a convergence experiment.',
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256_file(args.checkpoint),
        manifest=str(dataset.manifest),manifest_sha256=sha256_file(dataset.manifest),
        script_sha256=sha256_file(Path(__file__)),matcher=asdict(config),
        implementation={str(x):sha256_file(x) for x in Path('mhinet/downstream').glob('*.py')},
        gpu=torch.cuda.get_device_name(args.device),torch=torch.__version__,indices=indices,
        chunks=[32,64,128],repeats=args.repeats,deterministic=True,records=records)

    def step(batch,seed):
        model.zero_grad(set_to_none=True);torch.manual_seed(seed)
        torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
        events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        started=time.perf_counter();events[0].record()
        features=model(batch['images']);events[1].record()
        loss,info=model.matcher.training_losses(features,batch['H_gt_norm'],batch['mask_A_overlap'],batch['mask_B_overlap'])
        events[2].record();loss.backward();events[3].record()
        torch.cuda.synchronize(args.device)
        timing=dict(wall_ms=1000*(time.perf_counter()-started),
            shared_forward_ms=events[0].elapsed_time(events[1]),
            loss_forward_ms=events[1].elapsed_time(events[2]),
            backward_ms=events[2].elapsed_time(events[3]),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
        return float(loss.detach()),info[0],timing

    with torch.cuda.device(args.device):
        for position,index in enumerate(indices):
            sample=dataset[index]
            batch={k:v[None].to(args.device) for k,v in sample.items() if isinstance(v,torch.Tensor)}
            baseline=None
            alternatives=[64,128] if position%2==0 else [128,64]
            for chunk in [32]+alternatives:
                model.matcher.config=replace(config,window_chunk=chunk)
                step(batch,index)  # Per-configuration warmup excluded.
                for repeat in range(args.repeats):
                    loss,info,timing=step(batch,index)
                    if chunk==32 and repeat==0:
                        baseline=(loss,info,[None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters])
                    gradient_comparison=None
                    if repeat==0:
                        offset=0;gradient_comparison={}
                        for group in groups:
                            sums=torch.zeros(3,device=args.device,dtype=torch.float64)
                            for param in group['params']:
                                ref=baseline[2][offset];offset+=1
                                if (ref is None)!=(param.grad is None):raise RuntimeError('Gradient presence changed')
                                if ref is None:continue
                                actual=param.grad.detach();reference=ref.to(args.device)
                                delta=actual-reference
                                sums[0]+=delta.double().square().sum()
                                sums[1]+=reference.double().square().sum()
                                sums[2]=torch.maximum(sums[2],delta.abs().max().double())
                            values=sums.cpu().tolist()
                            gradient_comparison[group['name']]=dict(relative_l2=(values[0]/max(values[1],1e-30))**.5,max_abs=values[2])
                    row=dict(index=index,pair_id=sample['pair_id'],chunk=chunk,repeat=repeat,
                        loss=loss,loss_delta=loss-baseline[0],metrics=info,timing=timing,gradient_comparison=gradient_comparison)
                    records.append(row)
                    print(json.dumps(dict(index=index,chunk=chunk,repeat=repeat,**timing,loss_delta=row['loss_delta'])),flush=True)
            del baseline,batch
    report['complete']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
