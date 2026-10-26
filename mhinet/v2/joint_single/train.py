"""Domain-specific joint pilot: 3 adaptation epochs, then up to 27 joint epochs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from mhinet.v2.qrru_corners.experiment import selections, singleton, load_models
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.flow_d8_d2.run import measure
from mhinet.v2.flow_d8_d2.matching import infer as discrete_infer
from .path import forward, infer, source_grid, sample
from .loss import losses

ROOT=Path('outputs/joint_single')


def configure(model,matcher,joint):
    model.set_training_groups(mvt=joint,vgg=True,dedode=True,ghim_head=joint)
    matcher.requires_grad_(True)
    model.train();matcher.train()


def parameters(model,matcher):
    return [dict(name=name,params=[p for p in module.parameters() if p.requires_grad],lr=lr)
            for name,module,lr in [('mvt',model.mvt,1e-5),('flow',model.stage1_head,1e-5),
                ('vgg',model.vgg,2e-5),('decoder',model.dedode,2e-5),('matcher',matcher,8e-5)]]


@torch.no_grad()
def evaluate(model,matcher,dataset,device,baseline=False):
    model.eval();matcher.eval();rows=[]
    for batch in dataset:
        shared=model(batch['images'].to(device));gt=batch['geometry'].to(device)
        r=infer(shared,matcher)
        a=r['fine_source']
        p0=a+sample(shared['flow16_norm'][0],a)*256
        p8=(r['p8'][:,None]+(a.reshape(-1,16,2)-r['source'][:,None])).reshape(-1,2)
        record=dict(pair_id=batch['pair_id'][0],group=batch['group'])
        stages={}
        for name,b in [('flow',p0),('d8',p8),('d2',r['p2']),('final',r['final'])]:
            stats,error,visible=measure(a,b,gt)
            good=visible & (error<=3) & (b>=0).all(-1) & (b<=511).all(-1)
            stats['correct3']=int(good.sum())
            stages[name]=stats
            if name=='d2':before=good
            if name=='final':
                record['damaged3']=int((before&~good).sum());record['rescued3']=int((~before&good).sum())
        grid16=source_grid(device).reshape(64,64,2)[::2,::2].reshape(-1,2)+4
        # Exact D16 cell centers: 7.5,23.5,...,503.5 pixels.
        dense=(shared['flow16_target_norm'][0].flatten(1).T+1)*256-.5
        stages['dense_flow']=measure(grid16,dense,gt)[0]
        if baseline:
            old=discrete_infer(shared,matcher,'local')
            stages['old_discrete_d2']=measure(old['points_a'],old['points_b'],gt)[0]
        record['stages']=stages;rows.append(record)
    summary={}
    for name in rows[0]['stages']:
        values=[r['stages'][name] for r in rows]
        total=lambda key:sum(v[key] for v in values)
        summary[name]=dict(p3=total('correct3')/max(1,total('n')),p1=total('correct1')/max(1,total('n')),
            epe=total('error_sum')/max(1,total('visible')),matches=total('n')/len(rows),correct3=total('correct3')/len(rows))
    summary['damaged3']=sum(r['damaged3'] for r in rows)/max(1,sum(r['stages']['d2']['correct3'] for r in rows))
    return dict(summary=summary,pairs=rows)


def checkpoint(path,model,matcher,epoch,config):
    state={k:v.detach().cpu() for k,v in model.state_dict().items()
           if not k.startswith('descriptor.encoder.frozen_dinov3.model.')}
    torch.save(dict(shared=state,matcher=matcher.state_dict(),epoch=epoch,config=config),path)


def smoke(model,matcher,dataset,device,out):
    configure(model,matcher,True)
    groups=parameters(model,matcher)
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    reports=[]
    for step in range(2):
        torch.manual_seed(42+step)
        batch=dataset[step];gt=batch['geometry'].to(device)
        shared=model(batch['images'].to(device))
        # More sources in this gradient audit avoid a coverage-empty random example.
        grid=source_grid(device);source=grid[torch.randperm(len(grid),device=device)[:128]]
        r=forward(shared,matcher,source)
        for key in ('p0','p8','p2'):r[key].retain_grad()
        total,metrics,final=losses(shared,matcher,r,gt)
        optimizer.zero_grad(set_to_none=True)
        final.backward()
        coords={key:float(r[key].grad.abs().sum()) if r[key].grad is not None else 0 for key in ('p0','p8','p2')}
        grad={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
              for name,module in [('flow_head',model.stage1_head),('mvt',model.mvt),('vgg',model.vgg),('decoder',model.dedode),('qrru',matcher.qrru)]}
        assert all(x>0 for x in coords.values()),coords
        assert grad['flow_head']>0 and grad['mvt']>0,grad
        assert all(p.grad is None for p in model.dino.parameters())
        if step==1:assert grad['vgg']>0 and grad['decoder']>0,grad
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],1.,error_if_nonfinite=True)
        optimizer.step()
        reports.append(dict(step=step,only_final_loss=float(final.detach()),coordinates=coords,parameters=grad,**metrics))
    (out/'gradients.json').write_text(json.dumps(reports,indent=2))
    # Exercise the complete training loss separately after checking final-only gradients.
    optimizer.zero_grad(set_to_none=True)
    shared=model(batch['images'].to(device));r=forward(shared,matcher,source)
    total,metrics,_=losses(shared,matcher,r,gt);total.backward()
    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],1.,error_if_nonfinite=True)
    print(json.dumps(dict(status='passed',gradients=reports,total=float(total.detach()))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['hangzhou','synthetic'],required=True)
    p.add_argument('--device',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    out=ROOT/(args.domain+('_smoke' if args.smoke else ''));out.mkdir(parents=True,exist_ok=False)
    selected=selections();spec=selected[args.domain]
    train=GeometryPairDataset(spec['root'],spec['train']);val=GeometryPairDataset(spec['root'],spec['val'])
    model,matcher=load_models(args.device)
    matcher.config=replace(matcher.config,iterations=1,inference_iterations=1)
    matcher.qrru.iterations=1
    if args.smoke:
        smoke(model,matcher,train,args.device,out);return
    config=dict(domain=args.domain,seed=42,epochs=30,adaptation_epochs=3,patience=6,min_epochs=8,
        accumulation=4,training_d8_queries=64,output_cap=2048,corner_loss=False,qrru_iterations=1,
        dataset='same selection as qrru_corners; separate-domain pilot, not mixed-domain training',
        learning_rates=dict(mvt=1e-5,flow=1e-5,vgg=2e-5,decoder=2e-5,matcher=8e-5),
        initial_downstream='outputs/flow_d8_d2/training/best.pt',initial_flow='weight/mhi_v2_u0.pt',
        matchability='original exact-H loss on synthetic; no approximate-H Mat labels; detached in geometric prior',
        validation_metric='same-domain final P@3; initial model eligible',
        train_ids=[x['pair_id'] for x in spec['train']],val_ids=[x['pair_id'] for x in spec['val']])
    (out/'config.json').write_text(json.dumps(config,indent=2))
    configure(model,matcher,True)
    groups=parameters(model,matcher);all_params=[p for g in groups for p in g['params']]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,30,eta_min=1e-6)
    initial=evaluate(model,matcher,val,args.device,baseline=True)
    (out/'epoch_000.json').write_text(json.dumps(initial))
    for domain,data in selected.items():
        test=GeometryPairDataset(data.get('test_root',data['root']),data['test'])
        before=evaluate(model,matcher,test,args.device,baseline=True)
        (out/f'initial_test_{domain}.json').write_text(json.dumps(before))
    best=initial['summary']['final']['p3'];best_epoch=0;bad=0
    checkpoint(out/'best.pt',model,matcher,0,config)
    print(json.dumps(dict(event='initial',**initial['summary'])),flush=True)
    for epoch in range(1,31):
        configure(model,matcher,epoch>3)
        torch.manual_seed(42000+epoch)
        loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=2,collate_fn=singleton,
            generator=torch.Generator().manual_seed(42+epoch))
        optimizer.zero_grad(set_to_none=True);values=[];started=time.perf_counter()
        for index,batch in enumerate(loader):
            shared=model(batch['images'].to(args.device));gt=batch['geometry'].to(args.device)
            grid=source_grid(args.device);source=grid[torch.randperm(len(grid),device=args.device)[:64]]
            r=forward(shared,matcher,source)
            loss,metrics,_=losses(shared,matcher,r,gt)
            count=min(4,len(train)-(index//4)*4)
            (loss/count).backward()
            if (index+1)%4==0 or index+1==len(train):
                torch.nn.utils.clip_grad_norm_(all_params,1.,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            values.append(dict(loss=float(loss.detach()),**metrics))
        scheduler.step()
        result=evaluate(model,matcher,val,args.device)
        (out/f'epoch_{epoch:03d}.json').write_text(json.dumps(result))
        score=result['summary']['final']['p3']
        if score>best+.001:
            best=score;best_epoch=epoch;bad=0;checkpoint(out/'best.pt',model,matcher,epoch,config)
        else:bad+=1
        if epoch==3:bad=0
        checkpoint(out/'last.pt',model,matcher,epoch,config)
        means={k:float(np.mean([v[k] for v in values])) for k in ('loss','flow','d8_global','d8_local','d2','final')}
        coverage={k:sum(v[k]['covered'] for v in values)/max(1,sum(v[k]['visible'] for v in values)) for k in ('d8_counts','d2_counts')}
        record=dict(epoch=epoch,stage='joint' if epoch>3 else 'adapt',seconds=time.perf_counter()-started,
            train=means,coverage=coverage,best_epoch=best_epoch,bad_epochs=bad,**result['summary'])
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if epoch>=8 and bad>=6:break
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(state['shared'],strict=False);matcher.load_state_dict(state['matcher'])
    for domain,data in selected.items():
        dataset=GeometryPairDataset(data.get('test_root',data['root']),data['test'])
        result=evaluate(model,matcher,dataset,args.device,baseline=True)
        (out/f'test_{domain}.json').write_text(json.dumps(result))
    (out/'status.json').write_text(json.dumps(dict(status='complete',epochs=epoch,best_epoch=best_epoch,best_p3=best)))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
