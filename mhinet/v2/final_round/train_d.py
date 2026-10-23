"""Frozen-feature 2x2 decoder/data study; same 508 forwards and 64 updates/epoch."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from mhinet.config import RuntimePaths
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.losses import flow_head_loss
from mhinet.v2.d16_experiment.supervision import evaluate
from mhinet.v2.d16_experiment.train import write,aggregate
from mhinet.v2.local_occlusion.train import append_csv
from mhinet.v2.local_occlusion.evaluation import rejection
from mhinet.v2.d16_experiment.supervision import targets
from .decoder import CoordinateModel

ROOT=Path('outputs/d16_final_round')


@torch.no_grad()
def validate(model, datasets):
    model.eval();result={}
    for domain,dataset in datasets.items():
        pairs=[]
        for batch in dataset:
            gt=batch['geometry'].to('cuda:0');stage=model(batch['images'].to('cuda:0'))['stage1']
            pairs.append(dict(pair_id=batch['pair_id'][0],group=batch['group'],**evaluate(stage,gt),**rejection(stage,targets(gt)['positive'])))
        result[domain]=dict(overall=aggregate(pairs),groups={g:aggregate([r for r in pairs if r['group']==g]) for g in sorted({r['group'] for r in pairs})},pairs=pairs)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['D00','D01','D10','D11']);p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    config=json.loads(Path('mhinet/v2/local_occlusion/config.json').read_text());selection=json.loads((ROOT/'selection.json').read_text())
    output=ROOT/'decoder'/(args.variant+f'_seed{args.seed}'+('_smoke' if args.smoke else ''));output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    shared=build_fresh_shared(replace(RuntimePaths.from_json(config['runtime']),device='cuda:0'))
    base=Path('outputs/d16_local_occlusion/B0/latest.pt');state=torch.load(base,map_location='cpu',weights_only=False);assert state['epoch']==30
    shared.mvt.load_state_dict(state['mvt']);shared.stage1_head.load_state_dict(state['head'])
    shared.set_training_groups(mvt=False,vgg=False,dedode=False,ghim_head=False)
    model=CoordinateModel(shared,local=args.variant[1]=='1')
    mixed=args.variant[2]=='1'
    train=GeometryPairDataset(selection['roots']['hz'],selection['hz_train'])
    synthetic=GeometryPairDataset(selection['roots']['rgb_train'],selection['rgb_train'])
    validation={key:GeometryPairDataset(selection['roots']['hz'] if key=='hangzhou' else selection['roots']['rgb_dev'],selection['hz_dev' if key=='hangzhou' else 'rgb_dev']) for key in ['hangzhou','rgb']}
    params=[p for p in model.coordinate.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=1e-6)
    write(output/'config.json',dict(variant=args.variant,seed=args.seed,epochs=30,learning_rate=5e-5,min_lr=1e-6,accumulation=4,weight_decay=.01,
        baseline=str(base),baseline_epoch=30,selection=str(ROOT/'selection.json'),synthetic=mixed,coordinate='local3x3_plus_8px_tanh' if model.local else 'legacy_decoder',
        trainable_parameters=sum(p.numel() for p in params),matchability='fixed B0 original decoder',geo='existing robust Geo, weak H dead-zone retained; mat/cls weights zero'))
    forwards=updates=0;wall_start=time.time()
    if not args.smoke:
        write(output/'status.json',dict(status='initial_validation',epoch=0))
        initial=validate(model,validation);write(output/'epoch_000.json',dict(epoch=0,**initial))
        print(json.dumps(dict(event='initial',metrics={k:r['overall'] for k,r in initial.items()})),flush=True)
    best=-1
    for epoch in range(1,31):
        model.train();optimizer.zero_grad(set_to_none=True);started=time.time()
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        for index,source_index in enumerate(order):
            first=train[source_index]
            second=synthetic[((epoch-1)*254+index)%len(synthetic)] if mixed else train[(source_index+127)%len(train)]
            count=min(4,len(train)-(index//4)*4);losses=[]
            for domain,batch in [('hangzhou',first),('rgb' if mixed else 'hangzhou',second)]:
                gt=batch['geometry'].to('cuda:0');images=batch['images'].to('cuda:0')
                result=model(images);stage=result['stage1'];forwards+=1
                loss,_=flow_head_loss(stage,geometry=gt,mat_weight=0.,cls_weight=0.)
                (loss/(2*count)).backward();losses.append(float(loss.detach()))
                if args.smoke:
                    assert all(p.grad is None for p in model.shared.parameters())
                    assert any(p.grad is not None and bool((p.grad!=0).any()) for p in params)
                    assert torch.equal(stage['coarse_matchability'],shared(images,pyramid_scales=())['stage1']['coarse_matchability'])
                    if not model.local:
                        # Before the first update, cloned legacy coordinates exactly reproduce B0.
                        old=shared(images,pyramid_scales=())['stage1']['coarse_warp']
                        assert torch.allclose(stage['coarse_warp'],old,atol=1e-6,rtol=1e-6)
                    else:
                        info=result['coordinate_diagnostic'];assert info['delta_px'].abs().max()==0
                        assert torch.isfinite(stage['coarse_warp']).all()
                    write(output/'status.json',dict(status='smoke_passed',domain=domain,loss=float(loss.detach())))
            if args.smoke:print('Passed frozen-feature, unchanged-matchability and initial-coordinate checks.',flush=True);return
            if (index+1)%4==0 or index+1==len(train):
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            with (output/'training.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,step=index+1,pair_ids=[first['pair_id'][0],second['pair_id'][0]],losses=losses))+'\n')
            if index%32==0:
                status=dict(status='training',epoch=epoch,step=index+1,train_forwards=forwards,optimizer_updates=updates)
                write(output/'status.json',status);print(json.dumps(status),flush=True)
        scheduler.step();training_seconds=time.time()-started
        write(output/'status.json',dict(status='validation',epoch=epoch))
        result=validate(model,validation);write(output/f'epoch_{epoch:03d}.json',dict(epoch=epoch,training_seconds=training_seconds,**result))
        for domain,r in result.items():
            for group,metrics in [('overall',r['overall']),*r['groups'].items()]:append_csv(output/'epochs.csv',dict(epoch=epoch,domain=domain,group=group,**metrics))
            for row in r['pairs']:append_csv(output/'paired.csv',dict(epoch=epoch,domain=domain,**row))
        state=dict(epoch=epoch,variant=args.variant,seed=args.seed,base_checkpoint=str(base),coordinate=model.coordinate.state_dict())
        torch.save(state,output/'latest.pt')
        if result['hangzhou']['overall']['flow_p3']>best:
            best=result['hangzhou']['overall']['flow_p3'];torch.save(state,output/'best_p3.pt')
        if epoch==15:torch.save(state,output/'middle.pt')
        print(json.dumps(dict(event='validation',epoch=epoch,metrics={k:r['overall'] for k,r in result.items()})),flush=True)
    write(output/'status.json',dict(status='completed',epoch=30,best_development_p3=best,train_forwards=forwards,optimizer_updates=updates,wall_seconds=time.time()-wall_start))


if __name__=='__main__':main()
