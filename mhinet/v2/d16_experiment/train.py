"""Three matched-budget D16 experiments; no D8/D2/QRRU forward or updates."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from mhinet.config import RuntimePaths
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.frozen_locator import initialize_locator
from .supervision import correspondence_loss, absent_image_loss, evaluate, absent_image_metrics, diagnostic


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def singleton(items):
    return items[0]


def aggregate(rows):
    result={}
    for key in rows[0]:
        if key in ('pair_id','group'):continue
        values=[(r[key],r['points']) for r in rows if r.get(key) is not None]
        if key in ('points','ignore_queries','real_negative_queries','legacy_border_false_negative'):
            result[key]=sum(v for v,n in values)
        elif key=='legacy_epe':
            # Historical reports averaged pairs, so retain that convention here.
            result[key]=sum(v for v,n in values)/len(values) if values else None
        else:
            result[key]=sum(v*n for v,n in values)/sum(n for v,n in values) if sum(n for v,n in values) else None
    return result


def plot_case(output, epoch, batch, stage, gt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records,targets,prediction=diagnostic(stage,gt)
    group=batch['group']; name=f'{group}_epoch_{epoch:03d}'
    write(output/(name+'.json'),dict(pair_id=batch['pair_id'][0],worst_reference_queries=records))
    images=batch['images'][0].permute(0,2,3,1).cpu().numpy()
    target=targets['target'].cpu().numpy();pred=prediction.cpu().numpy()
    source=targets['source'].cpu().numpy();valid=targets['positive'].cpu().numpy()
    error=np.linalg.norm(pred-target,axis=-1)
    chosen=np.arange(1024)[valid & (np.arange(1024)%3==0)]
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    axes[0].imshow(images[0]);axes[0].set_title('A: reference-valid query EPE (px)')
    dots=axes[0].scatter(source[valid,0],source[valid,1],c=error[valid],s=9,cmap='turbo',vmin=0,vmax=32)
    fig.colorbar(dots,ax=axes[0],shrink=.7)
    axes[1].imshow(images[1]);axes[1].set_title('B: reference -> prediction residual, 1:1 px')
    axes[1].quiver(target[chosen,0],target[chosen,1],pred[chosen,0]-target[chosen,0],pred[chosen,1]-target[chosen,1],
                   error[chosen],angles='xy',scale_units='xy',scale=1,cmap='turbo',clim=(0,32),width=.003)
    axes[2].imshow(images[1]);axes[2].set_title('B: worst query Top-10 spatial candidates')
    if records:
        r=records[0];xy=np.array([p['xy'] for p in r['top10']])
        axes[2].scatter(xy[:,0],xy[:,1],c=['lime' if p['in_positive_region'] else 'red' for p in r['top10']],s=40)
        for i,(x,y) in enumerate(xy):axes[2].text(x,y,str(i+1),color='white',fontsize=9)
        axes[2].scatter(*r['reference_xy'],marker='*',s=180,c='yellow',label='H reference')
        axes[2].scatter(*r['predicted_xy'],marker='x',s=90,c='cyan',label='prediction')
        axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xlim(-.5,511.5);ax.set_ylim(511.5,-.5);ax.axis('off')
    fig.suptitle(f'{group} | epoch {epoch} | {batch["pair_id"][0]} | approximate H reference')
    fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=140);plt.close(fig)


@torch.no_grad()
def validate(model,dataset,device,output,epoch,visualize=False):
    model.eval();rows=[];absent=[];seen=set()
    for batch in DataLoader(dataset,batch_size=1,collate_fn=singleton,num_workers=2):
        images=batch['images'].to(device);gt=batch['geometry'].to(device)
        stage=model(images,pyramid_scales=())['stage1']
        rows.append(dict(pair_id=batch['pair_id'][0],group=batch['group'],**evaluate(stage,gt)))
        if batch['group'] not in seen:
            blank=images.clone();blank[:,1]=0
            absent.append(absent_image_metrics(model(blank,pyramid_scales=())['stage1'],gt.mask_a))
            if visualize and batch['group'] in ('G1','G2','G3'):plot_case(output,epoch,batch,stage,gt)
            seen.add(batch['group'])
    return dict(epoch=epoch,overall=aggregate(rows),groups={g:aggregate([r for r in rows if r['group']==g]) for g in sorted(seen)},
                absent_probe={k:sum(r[k] for r in absent)/len(absent) for k in absent[0]},pairs=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['U','B','C'],required=True)
    parser.add_argument('--config',default='mhinet/v2/d16_experiment/config.json');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    output=Path(config['output'])/(args.variant+('_smoke' if args.smoke else ''));output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(config['seed']);random.seed(config['seed']);np.random.seed(config['seed'])
    device='cuda:0';runtime=replace(RuntimePaths.from_json(config['runtime']),device=device)
    model=build_fresh_shared(runtime);initialize_locator(SimpleNamespace(shared=model),config)
    model.set_training_groups(mvt=True,vgg=False,dedode=False,ghim_head=True)
    write(output/'config.json',dict(config,variant=args.variant))
    selection=json.loads(Path(config['selection']).read_text())
    train=GeometryPairDataset(config['dataset'],selection['train']);val=GeometryPairDataset(config['dataset'],selection['val'])
    optimizer=torch.optim.AdamW([{'params':model.mvt.parameters(),'lr':config['mvt_lr']},
        {'params':model.stage1_head.parameters(),'lr':config['head_lr']}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['min_lr'])
    parameters=[p for p in model.parameters() if p.requires_grad]
    write(output/'status.json',dict(status='initial_validation',epoch=0))
    if not args.smoke:
        initial=validate(model,val,device,output,0,True);write(output/'epoch_000.json',initial)
        print(json.dumps(dict(event='baseline_A',**initial['overall'])),flush=True)
    best=float('inf')
    for epoch in range(1,config['epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True);started=time.time()
        loader=DataLoader(train,batch_size=1,shuffle=True,collate_fn=singleton,num_workers=2,
                          generator=torch.Generator().manual_seed(config['seed']+epoch))
        for index,batch in enumerate(loader):
            images=batch['images'].to(device);gt=batch['geometry'].to(device)
            count=min(config['accumulation'],len(train)-(index//config['accumulation'])*config['accumulation'])
            stage=model(images,pyramid_scales=())['stage1'];loss,metrics=correspondence_loss(stage,gt,args.variant,config)
            (loss/count).backward()
            # Same additional inputs and forward/backward budget for U/B/C.
            if index%4==0:
                blank=images.clone();blank[:,1]=0
                absent_stage=model(blank,pyramid_scales=())['stage1']
                negative=absent_image_loss(absent_stage,gt.mask_a,config)
                ((negative if args.variant=='C' else negative*0)/count).backward()
                metrics['controlled_negative_loss']=float(negative.detach())
            if args.smoke:
                gradients={name:sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
                           for name,module in [('mvt',model.mvt),('head',model.stage1_head),('dino',model.dino),('vgg',model.vgg),('decoder',model.dedode)]}
                assert gradients['mvt']>0 and gradients['head']>0
                assert all(gradients[n]==0 for n in ('dino','vgg','decoder'))
                print(json.dumps(dict(loss=float(loss.detach()),gradients=gradients,metrics=metrics)),flush=True)
                write(output/'status.json',dict(status='smoke_passed',gradients=gradients));return
            if (index+1)%config['accumulation']==0 or index+1==len(train):
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
            if index%16==0:
                status=dict(status='training',epoch=epoch,pair=index+1,total=len(train),loss=float(loss.detach()))
                write(output/'status.json',status);print(json.dumps(status),flush=True)
            with (output/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch,pair=index+1,loss=float(loss.detach()),**metrics))+'\n')
        scheduler.step();write(output/'status.json',dict(status='validation',epoch=epoch))
        result=validate(model,val,device,output,epoch,epoch in (1,15,30))
        result['seconds']=time.time()-started;result['learning_rates']=scheduler.get_last_lr()
        write(output/f'epoch_{epoch:03d}.json',result)
        state=dict(epoch=epoch,mvt=model.mvt.state_dict(),head=model.stage1_head.state_dict())
        torch.save(state,output/'latest.pt')
        if result['overall']['epe']<best:
            best=result['overall']['epe'];torch.save(state,output/'best.pt')
        if epoch==15:torch.save(state,output/'middle.pt')
        print(json.dumps(dict(event='validation',**result['overall'],epoch=epoch)),flush=True)
    write(output/'status.json',dict(status='completed',epoch=epoch,best_epe=best))


if __name__=='__main__':main()
