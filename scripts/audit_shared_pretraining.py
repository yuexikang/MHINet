"""Bounded real-weight alignment, branch-gradient and frozen-base audit (not training)."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from mhinet.config import RuntimePaths
from mhinet.pretraining.model import build_shared_network
from mhinet.pretraining.data import SharedPairDataset
from mhinet.pretraining.loss import descriptor_loss
from mhinet.engine.ghim_losses import ghim_loss


def base_hash(model):
    digest = hashlib.sha256()
    for name, p in model.dino.named_parameters():
        if 'lora_' not in name:
            digest.update(name.replace('.base.','.').encode())
            digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.manual_seed(0)
    runtime=replace(RuntimePaths.from_json('configs/runtime_paths.quadrant.server.json'),device='cuda:0')
    dataset=SharedPairDataset(runtime.data_root/'train/pairs.jsonl',tier=1,max_pairs=1)
    batch=next(iter(DataLoader(dataset,batch_size=1)))
    batch={k:v.cuda() for k,v in batch.items() if isinstance(v,torch.Tensor)}
    model,_=build_shared_network(runtime)
    initial_hash=base_hash(model)
    model.eval()
    with torch.no_grad():
        before=model(batch['images'])
        ref={k:v.cpu() for k,v in before['pyramid'].items()}
        h0=before['H0_norm'].cpu()
    del before
    model.configure(lora=True)
    model.eval()
    with torch.no_grad():
        after=model(batch['images'])
        errors={f'D{k}':float((v.cpu()-ref[k]).abs().max()) for k,v in after['pyramid'].items()}
        errors['H0']=float((after['H0_norm'].cpu()-h0).abs().max())
    assert all(v==0 for v in errors.values()), errors
    del after,ref
    model.train()
    optimizer=torch.optim.AdamW(model.optimizer_groups(),weight_decay=1e-4)
    report={'zero_init_max_abs':errors,'steps':[]}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result=model(batch['images'])
        ghim=ghim_loss(result['stage1'],batch['H_gt_norm'],batch['mask_A_overlap'])['total']
        desc,_=descriptor_loss(result['pyramid'],batch['H_gt_norm'],batch['mask_A_overlap'],batch['mask_B_overlap'],queries=256)
        mvt=next(p for p in model.mvt.parameters() if p.requires_grad)
        adapter=model.dino.blocks[8].attn.qkv.lora_B
        branches={}
        for name, loss in [('GHIM',ghim),('CGMDP',desc)]:
            gradients=torch.autograd.grad(loss,(mvt,adapter),retain_graph=True,allow_unused=True)
            branches[name]=[None if v is None else float(v.float().norm()) for v in gradients]
            assert all(v is not None and torch.isfinite(v).all() and v.abs().sum()>0 for v in gradients), branches
        (ghim+desc).backward()
        norms={name:float(sum(p.grad.float().square().sum() for n,p in model.named_parameters()
            if name in n and p.grad is not None).sqrt()) for name in ('lora_A','lora_B')}
        assert norms['lora_B']>0 and (step==0 or norms['lora_A']>0)
        assert all(p.grad is None and not p.requires_grad for n,p in model.dino.named_parameters() if 'lora_' not in n)
        assert result['call_counts']['dino']==1 and result['call_counts']['mvt']==1
        assert result['call_counts']['dedode_scale1']==0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        report['steps'].append(dict(step=step+1,ghim=float(ghim.detach()),descriptor=float(desc.detach()),
            branch_MVT_LoRA_B_norms=branches,lora_gradient_norms=norms,calls=result['call_counts']))
    report['frozen_base_sha256_before']=initial_hash
    report['frozen_base_sha256_after']=base_hash(model)
    assert initial_hash==report['frozen_base_sha256_after']
    report['passed']=True
    report['peak_memory_gb']=torch.cuda.max_memory_allocated()/1e9
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
