"""Adapt D8/D2 to Hangzhou with the original Flow locator frozen, without QRRU."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.training import supervised_matching_loss
from .run import load, measure, sample, CHECKPOINT
from .matching import infer


def singleton(items):
    return items[0]


@torch.no_grad()
def validate(model, matcher, dataset, device):
    model.eval();matcher.eval();rows=[]
    for batch in dataset:
        gt=batch['geometry'].to(device)
        shared=model(batch['images'].to(device))
        result=infer(shared,matcher,'local')
        fine,_,_=measure(result['points_a'],result['points_b'],gt)
        initial=result['points_a']+sample(shared['flow16_norm'],result['points_a'])*256
        before,_,_=measure(result['points_a'],initial,gt)
        rows.append(dict(pair_id=batch['pair_id'][0],group=batch['group'],d2=fine,flow_same_points=before))
    total=lambda key:sum(r['d2'][key] for r in rows)
    return dict(p3=total('correct3')/max(1,total('n')),p1=total('correct1')/max(1,total('n')),
        epe=total('error_sum')/max(1,total('visible')),matches=total('n')/len(rows),
        correct3=total('correct3')/len(rows),pairs=rows)


def save(path, model, matcher, optimizer, scheduler, epoch, score, config):
    # Only downstream weights; frozen DINO/MVT/Flow remain in their source files.
    torch.save(dict(vgg=model.vgg.state_dict(),dedode=model.dedode.state_dict(),
        temperatures={k:v for k,v in matcher.state_dict().items() if not k.startswith('qrru.')},
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,score=score,
        config=config,matcher=asdict(matcher.config)),path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    out=Path('outputs/flow_d8_d2/training'+('_smoke' if args.smoke else ''));out.mkdir(parents=True,exist_ok=False)
    config=dict(seed=42,epochs=30,learning_rate=8e-5,min_lr=1e-6,patience=6,min_epochs=5,min_delta=.001,
        accumulation=4,selection='outputs/hangzhou_cleaned_restart/selection.json',
        initial_downstream=CHECKPOINT,flow='weight/mhi_v2_u0.pt',variant='local',qrru=False,
        supervision='existing D8 positive focal + D2 uncertainty-region focal; approximate H',
        monitor='validation D2 P@3; initial checkpoint is eligible')
    selection=json.loads(Path(config['selection']).read_text())
    assert not ({r['anchor_block_id'] for r in selection['train']} & {r['anchor_block_id'] for r in selection['val']})
    root='/home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4'
    train=GeometryPairDataset(root,selection['train']);val=GeometryPairDataset(root,selection['val'])
    model,matcher=load(args.device)
    model.set_training_groups(mvt=False,vgg=True,dedode=True,ghim_head=False)
    matcher.requires_grad_(True);matcher.qrru.requires_grad_(False)
    params=[p for module in (model,matcher) for p in module.parameters() if p.requires_grad]
    config['trainable_parameters']=sum(p.numel() for p in params)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    optimizer=torch.optim.AdamW(params,lr=config['learning_rate'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,30,eta_min=config['min_lr'])
    if args.smoke:
        initial=dict(p3=0.)
    else:
        initial=validate(model,matcher,val,args.device)
        (out/'epoch_000.json').write_text(json.dumps(initial,indent=2))
        save(out/'best.pt',model,matcher,optimizer,scheduler,0,initial['p3'],config)
    best=initial['p3'];bad=0
    print(json.dumps(dict(event='initial',**{k:v for k,v in initial.items() if k!='pairs'})),flush=True)
    for epoch in range(1,31):
        model.train();matcher.train();matcher.qrru.eval();start=time.perf_counter();losses=[]
        loader=DataLoader(train,batch_size=1,shuffle=True,collate_fn=singleton,num_workers=2,
            generator=torch.Generator().manual_seed(42+epoch))
        optimizer.zero_grad(set_to_none=True)
        for index,batch in enumerate(loader):
            shared=model(batch['images'].to(args.device))
            loss,metrics=supervised_matching_loss(matcher,shared,batch['geometry'].to(args.device))
            count=min(4,len(train)-(index//4)*4)
            (loss/count).backward();losses.append(float(loss.detach()))
            if (index+1)%4==0 or index+1==len(train):
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                if args.smoke:
                    gradients={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                        for name,module in dict(dino=model.dino,mvt=model.mvt,flow=model.stage1_head,vgg=model.vgg,dedode=model.dedode,qrru=matcher.qrru).items()}
                    assert all(gradients[k]==0 for k in ('dino','mvt','flow','qrru'))
                    assert gradients['vgg']>0 and gradients['dedode']>0
                    print(json.dumps(dict(event='gradient_check',**gradients)),flush=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if args.smoke:
                    print('SMOKE PASSED',flush=True);return
        scheduler.step()
        result=validate(model,matcher,val,args.device)
        (out/f'epoch_{epoch:03d}.json').write_text(json.dumps(result,indent=2))
        improved=result['p3']>best+config['min_delta']
        if improved:
            best=result['p3'];bad=0
            save(out/'best.pt',model,matcher,optimizer,scheduler,epoch,best,config)
        else:bad+=1
        save(out/'latest.pt',model,matcher,optimizer,scheduler,epoch,result['p3'],config)
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)),lr=optimizer.param_groups[0]['lr'],
            seconds=time.perf_counter()-start,best_p3=best,bad_epochs=bad,**{k:v for k,v in result.items() if k!='pairs'})
        with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if epoch>=config['min_epochs'] and bad>=config['patience']:break
    (out/'status.json').write_text(json.dumps(dict(status='complete',last_epoch=epoch,best_p3=best)))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
