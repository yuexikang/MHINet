"""Loss-by-loss gradient audit and matched short counterfactual updates, no long retraining."""
import argparse
import json
from pathlib import Path
import torch
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.qrru_corners.experiment import selections,load_models
from .train import configure,parameters,evaluate
from .path import forward,coarse,fine,source_grid,sample
from .loss import local_loss,position_loss,flow_loss,global_d8


def components(shared,matcher,r,gt):
    l8,t8,v8,_=local_loss(r['source'],r['p8'],r['candidates8'],r['logits8'],r['valid8'],gt)
    l2,t2,v2,_=local_loss(r['fine_source'],r['p2'],r['candidates2'],r['logits2'],r['valid2'],gt)
    g8=position_loss(r['p8'],t8,v8,0 if gt.exact else gt.tolerance_px)
    g2=position_loss(r['p2'],t2,v2,0 if gt.exact else gt.tolerance_px)
    return dict(flow=flow_loss(shared,gt),global8=global_d8(shared,matcher,gt),
        cls8=l8-g8,geo8=g8,cls2=l2-g2,geo2=g2,
        final=position_loss(r['final'],t2,v2,0 if gt.exact else gt.tolerance_px))


def detached_coordinates(shared,matcher,source):
    other={**shared,'flow16_norm':shared['flow16_norm'].detach()}
    c=coarse(other,matcher,source)
    return fine(other,matcher,{**c,'p8':c['p8'].detach()})


def dotcos(a,b):
    return float((a.flatten()@b.flatten())/(a.norm()*b.norm()).clamp_min(1e-30))


@torch.no_grad()
def dense_eval(model,dataset,device):
    model.eval();total=0.;n=0;correct=0
    grid=source_grid(device).reshape(64,64,2)[::2,::2].reshape(-1,2)+4
    for batch in dataset:
        gt=batch['geometry'].to(device);s=model(batch['images'].to(device),pyramid_scales=())
        target,known,visible=gt.query((grid+.5)/256-1)
        pred=s['flow16_target_norm'][0].flatten(1).T
        err=((pred-target)*256).norm(dim=-1);good=known&visible
        total+=float(err[good].sum());n+=int(good.sum());correct+=int((good&(err<=3)).sum())
    return dict(visible_epe=total/max(1,n),visible_p3=correct/max(1,n))


def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',required=True);p.add_argument('--device',required=True);p.add_argument('--routed-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    root=Path('outputs/joint_diagnosis')/args.domain;root.mkdir(parents=True,exist_ok=True)
    selection=selections()[args.domain]
    train=GeometryPairDataset(selection['root'],selection['train'])
    val=GeometryPairDataset(selection['root'],selection['val'][::4])
    model,matcher=load_models(args.device);matcher.qrru.iterations=1
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('descriptor.encoder.frozen_dinov3.model.')}
    initial_matcher={k:v.detach().cpu().clone() for k,v in matcher.state_dict().items()}
    # Coordinate ramp sampled at D8 centers is an exact unit/convention check.
    coords=source_grid(args.device).reshape(64,64,2).permute(2,0,1)
    ramp_error=float((sample(coords,source_grid(args.device))-source_grid(args.device)).abs().max())
    audit=[]
    configure(model,matcher,True)
    for index in range(0 if args.routed_only else 8):
        torch.manual_seed(700+index)
        batch=train[index];gt=batch['geometry'].to(args.device);shared=model(batch['images'].to(args.device))
        grid=source_grid(args.device);src=grid[torch.randperm(len(grid),device=args.device)[:64]]
        r=forward(shared,matcher,src);terms=components(shared,matcher,r,gt)
        target,known,visible=gt.query((grid.reshape(64,64,2)[::2,::2].reshape(-1,2)+4+.5)/256-1)
        warp=shared['flow16_target_norm'];err=((warp[0].flatten(1).T-target)*256).norm(dim=-1)
        reference=err[known&visible].mean()
        inputs=(warp,shared['context'])
        ref=torch.autograd.grad(reference,inputs,retain_graph=True,allow_unused=True)
        stored={};record=dict(pair_id=batch['pair_id'][0],terms={})
        for name,value in terms.items():
            gradients=torch.autograd.grad(value,inputs,retain_graph=True,allow_unused=True)
            stored[name]=[torch.zeros_like(x) if g is None else g.detach() for g,x in zip(gradients,inputs)]
            record['terms'][name]=dict(loss=float(value.detach()),**{
                key:dict(norm=float(g.norm()),cos_to_flow_epe=dotcos(g,rr))
                for key,g,rr in zip(('flow_coordinates','shared_features'),stored[name],ref)})
        combined=[sum(stored[k][j] for k in stored) for j in range(2)]
        record['total']={key:dict(norm=float(g.norm()),cos_to_flow_epe=dotcos(g,rr)) for key,g,rr in zip(('flow_coordinates','shared_features'),combined,ref)}
        audit.append(record)
    if audit:
        (root/'gradient_audit.json').write_text(json.dumps(dict(coordinate_ramp_max_error=ramp_error,pairs=audit),indent=2))
    print('GRADIENT AUDIT COMPLETE',flush=True)
    variants=['routed'] if args.routed_only else ['full','no_coordinate','no_class_coordinate','freeze_mvt','flow_only']
    results=json.loads((root/'counterfactuals.json').read_text()) if args.routed_only else {}
    for variant in variants:
        model.load_state_dict(initial,strict=False);matcher.load_state_dict(initial_matcher)
        configure(model,matcher,True)
        if variant=='freeze_mvt':model.set_training_groups(mvt=False,vgg=True,dedode=True,ghim_head=True)
        if variant=='flow_only':
            model.set_training_groups(mvt=True,vgg=False,dedode=False,ghim_head=True)
            matcher.requires_grad_(False)
        groups=[g for g in parameters(model,matcher) if g['params']]
        optimizer=torch.optim.AdamW(groups,weight_decay=.01)
        params=[p for g in groups for p in g['params']]
        report=dict(before=dense_eval(model,val,args.device),steps=[])
        for step in range(32):
            torch.manual_seed(1000+step);model.train();matcher.train()
            batch=train[step];gt=batch['geometry'].to(args.device)
            shared=model(batch['images'].to(args.device));grid=source_grid(args.device)
            src=grid[torch.randperm(len(grid),device=args.device)[:64]]
            if variant=='flow_only':terms={'flow':flow_loss(shared,gt)}
            else:
                working={**shared,'flow16_norm':shared['flow16_norm'].detach()} if variant=='no_coordinate' else shared
                if variant=='routed':
                    f=shared['flow16_norm'];working={**shared,'flow16_norm':f.detach()+.01*(f-f.detach())}
                r=forward(working,matcher,src);terms=components(shared,matcher,r,gt)
                if variant in ('no_class_coordinate','routed'):
                    stopped=detached_coordinates(shared,matcher,src)
                    separated=components(shared,matcher,stopped,gt)
                    terms['cls8']=separated['cls8'];terms['cls2']=separated['cls2']
            loss=sum(terms.values());optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            if step in (0,7,15,31):
                report['steps'].append(dict(updates=step+1,**dense_eval(model,val,args.device)))
        report['after']=evaluate(model,matcher,val,args.device)['summary']
        results[variant]=report
        (root/'counterfactuals.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(dict(variant=variant,before=report['before'],steps=report['steps'])),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
