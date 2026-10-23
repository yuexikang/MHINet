"""N1/N2 and matched-compute controls; clean + one local B rectangle per pair."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from mhinet.config import RuntimePaths
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.frozen_locator import initialize_locator
from mhinet.v2.d16_experiment.supervision import correspondence_loss
from mhinet.v2.d16_experiment.train import write,singleton
from mhinet.v2.local_occlusion.occlusion import make_occlusion,augmented_losses
from mhinet.v2.local_occlusion.evaluation import validate


def gradient_norms(loss, model):
    modules=[('mvt',model.mvt),('head',model.stage1_head)]
    params=[p for _,module in modules for p in module.parameters() if p.requires_grad]
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    result={};offset=0
    for name,module in modules:
        count=sum(p.requires_grad for p in module.parameters())
        result[name]=float(sum((g.float().square().sum() for g in grads[offset:offset+count] if g is not None),loss.new_zeros(())).sqrt())
        offset+=count
    return result


def append_csv(path, row):
    exists=path.exists()
    with path.open('a') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def save_metrics(output, result):
    write(output/f'epoch_{result["epoch"]:03d}.json',result)
    for group,metric in [('overall',result['clean']),*result['groups'].items()]:
        append_csv(output/'epochs.csv',dict(epoch=result['epoch'],group=group,**metric))
    for row in result['clean_pairs']:
        append_csv(output/'paired_clean.csv',dict(epoch=result['epoch'],**row))
    with (output/'paired_occlusion.jsonl').open('a') as stream:
        for row in result['occlusion_pairs']:stream.write(json.dumps(dict(epoch=result['epoch'],**row))+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['E1','E2','E_geo','E_cls_mat'],required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--config',default='mhinet/v2/final_round/e_config.json')
    args=parser.parse_args();c=json.loads(Path(args.config).read_text())
    output=Path(c['output'])/(args.variant+('_smoke' if args.smoke else ''));output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    model=build_fresh_shared(replace(RuntimePaths.from_json(c['runtime']),device='cuda:0'))
    init=initialize_locator(SimpleNamespace(shared=model),c)
    model.set_training_groups(mvt=True,vgg=False,dedode=False,ghim_head=True)
    init['fresh_modules']=['VGG / CGMDP (unused)']
    init.update(trainable_modules=['MVT','MatchEmbedding/Flow Head'],frozen_modules=['DINOv3','VGG','decoder'],
        seed=c['seed'],scope='Only original DINO plus pretrained MVT/head; not U/B/C trained weights')
    write(output/'initialization.json',init);write(output/'config.json',dict(c,variant=args.variant))
    selection=json.loads(Path(c['selection']).read_text());train=GeometryPairDataset(c['dataset'],selection['train']);val=GeometryPairDataset(c['dataset'],selection['val'])
    optimizer=torch.optim.AdamW([dict(params=model.mvt.parameters(),lr=c['mvt_lr']),dict(params=model.stage1_head.parameters(),lr=c['head_lr'])],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c['epochs'],eta_min=c['min_lr'])
    params=[p for p in model.parameters() if p.requires_grad];baseline='B'
    active={'E1':{'keep','keep_geo','keep_match'},'E2':{'occ'},'E_geo':{'keep_geo'},'E_cls_mat':{'keep','keep_match'}}[args.variant];best=-1;train_forwards=val_forwards=updates=0
    wall_start=time.time()
    if not args.smoke:
        write(output/'status.json',dict(status='initial_validation',epoch=0))
        result=validate(model,val,c,output,0,full=True);val_forwards+=result['validation_forwards'];save_metrics(output,result)
    for epoch in range(1,c['epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True);started=time.time();skipped=0
        loader=DataLoader(train,batch_size=1,shuffle=True,collate_fn=singleton,num_workers=2,
            generator=torch.Generator().manual_seed(c['seed']+epoch))
        for index,batch in enumerate(loader):
            images=batch['images'].to('cuda:0');gt=batch['geometry'].to('cuda:0');pid=batch['pair_id'][0]
            count=min(c['accumulation'],len(train)-(index//c['accumulation'])*c['accumulation'])
            stage=model(images,pyramid_scales=())['stage1'];train_forwards+=1
            clean,_=correspondence_loss(stage,gt,baseline,c)
            record=dict(epoch=epoch,index=index,pair_id=pid,clean_loss=float(clean.detach()))
            if index==0:record['clean_gradient_norms']=gradient_norms(clean,model)
            (clean/count).backward()
            modified,labels,metadata=make_occlusion(images,gt,pid,epoch,c['seed'])
            aug_stage=model(modified,pyramid_scales=())['stage1'];train_forwards+=1
            losses=augmented_losses(aug_stage,gt,labels,c)
            if index==0:
                record['weighted_augmented_gradient_norms']={k:gradient_norms(v,model) for k,v in losses.items()}
                record['gradient_measurement']='unclipped, before accumulation scaling; once per epoch, same for controls'
            total=sum(v if k in active else v*0 for k,v in losses.items())
            (total/count).backward()
            record.update(metadata);record['augmented_terms']={k:float(v.detach()) for k,v in losses.items()}
            record['active_augmented_terms']=sorted(active);skipped+=metadata['skipped']
            with (output/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if args.smoke:
                assert any(p.grad is not None and bool((p.grad!=0).any()) for p in model.mvt.parameters())
                assert all(p.grad is None for m in (model.dino,model.vgg,model.dedode) for p in m.parameters())
                write(output/'status.json',dict(status='smoke_passed',record=record));print(json.dumps(record),flush=True);return
            if (index+1)%c['accumulation']==0 or index+1==len(train):
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            if index%32==0:
                status=dict(status='training',epoch=epoch,pair=index+1,total=len(train),train_forwards=train_forwards,optimizer_updates=updates)
                write(output/'status.json',status);print(json.dumps(status),flush=True)
        scheduler.step();training_seconds=time.time()-started
        write(output/'status.json',dict(status='validation',epoch=epoch))
        result=validate(model,val,c,output,epoch,full=epoch in (15,30));val_forwards+=result['validation_forwards']
        result.update(training_seconds=training_seconds,epoch_seconds=time.time()-started,skipped_augmentations=skipped,
                      train_forwards=train_forwards,validation_forwards_total=val_forwards,optimizer_updates=updates)
        save_metrics(output,result)
        state=dict(epoch=epoch,mvt=model.mvt.state_dict(),head=model.stage1_head.state_dict())
        torch.save(state,output/'latest.pt')
        if result['clean']['flow_p3']>best:
            best=result['clean']['flow_p3'];torch.save(state,output/'best_p3.pt')
        if epoch==15:torch.save(state,output/'middle.pt')
        print(json.dumps(dict(event='validation',epoch=epoch,clean=result['clean'],occlusion=result['occlusion']['fixed_mixture'])),flush=True)
    write(output/'status.json',dict(status='completed',epoch=epoch,best_clean_p3=best,train_forwards=train_forwards,
        validation_forwards=val_forwards,optimizer_updates=updates,wall_seconds=time.time()-wall_start,
        compute_note='2 full D16 forwards per training pair in all controls/treatments; one extra per pair vs original 1.252 forward/pair. Time includes per-term gradient probes.'))



if __name__=='__main__':main()
