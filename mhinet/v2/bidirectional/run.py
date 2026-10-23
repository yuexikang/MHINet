"""A/B frozen inference and paired metrics on development data only."""
import argparse,json,random,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from mhinet.v2.u0 import load_u0
from mhinet.v2.frozen_locator import load_locator_weights
from mhinet.v2.config import SemidenseConfigV2
from mhinet.v2.semidense import SemidenseMatcherV2
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import normalized_d16_grid
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers
from mhinet.ops.geometry import safe_project_points
from .cycle import cycle_consistency,reliability_features,modulate

ROOT=Path('outputs/d16_bidirectional')
VARIANTS=('B0','B1','B2','B3','B4_uniform')


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def prepare():
    path=ROOT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    old=json.loads(Path('outputs/d16_final_round/selection.json').read_text())
    rng=random.Random(20261023);cal=set()
    for i,g in enumerate(sorted({r['G'] for r in old['rgb_dev']})):
        rows=[r for r in old['rgb_dev'] if r['G']==g];rng.shuffle(rows)
        cal.update(r['pair_id'] for r in rows[:43 if i<2 else 42])
    s=dict(roots=old['roots'],hangzhou=old['hz_dev'],rgb=old['rgb_dev'],rgb_calibration_ids=sorted(cal))
    write(path,s);return s


def load_models():
    shared=load_u0()
    state=torch.load('outputs/hangzhou_cleaned_restart/v2_20261022/best.pt',map_location='cpu',weights_only=False,mmap=True)
    params={k[len('shared.'):]:v for k,v in state['model'].items() if k.startswith('shared.')}
    missing,unexpected=shared.load_state_dict(params,strict=False)
    assert not unexpected and all(k.startswith('descriptor.encoder.frozen_dinov3.model.') for k in missing)
    load_locator_weights(shared,'weight/mhi_v2_u0.pt')
    matcher=SemidenseMatcherV2(SemidenseConfigV2(**state['matcher'])).cuda()
    matcher.load_state_dict({k[len('matcher.'):]:v for k,v in state['model'].items() if k.startswith('matcher.')})
    shared.requires_grad_(False);matcher.requires_grad_(False);shared.eval();matcher.eval()
    return shared,matcher


def timed(call):
    torch.cuda.synchronize();start=time.perf_counter();value=call();torch.cuda.synchronize()
    return value,(time.perf_counter()-start)*1000


def query_labels(gt,xy):
    target,known,visible=gt.query(xy)
    _,legal,_=safe_project_points(gt.H,xy)
    pixel=(target+1)*256-.5
    positive=visible & (pixel>=1).all(-1) & (pixel<=510).all(-1)
    negative=known & legal & ((pixel < -1)|(pixel>512)).any(-1) if gt.exact else torch.zeros_like(visible)
    return target,visible,positive,negative


def point_metrics(a,b,gt):
    truth,known,visible=gt.query((a+.5)/256-1)
    error=((truth+1)*256-.5-b).norm(dim=-1)
    n=len(a);denom=max(1,n)
    result=dict(matches=n,reference_visible=int(visible.sum()),reference_invisible_rate=float((~visible).sum())/denom,
        epe_visible=float(error[visible].mean()) if visible.any() else None,
        epe_gt32=float((error>32).sum())/denom)
    for t in (1,3,5):
        count=int(((error<=t)&visible).sum());result[f'ncm{t}']=count;result[f'p{t}']=count/denom
    return result


def parent_metrics(result,gt,d2_target,d2_visible):
    a,b=result['coarse_a'],result['coarse_b'];truth,_,vis=gt.query((a+.5)/256-1)
    correct=vis & (((truth+1)*256-.5-b).abs()<=4).all(-1)
    mapping=torch.full((4096,),-1,device=a.device,dtype=torch.long)
    si=((a+.5)/8).floor().long();ti=((b+.5)/8).floor().long()
    if len(si):mapping[si[:,1]*64+si[:,0]]=ti[:,1]*64+ti[:,0]
    y,x=torch.meshgrid(torch.arange(256,device=a.device),torch.arange(256,device=a.device),indexing='ij')
    source_parent=(y//4*64+x//4).flatten()
    target_parent=((d2_target+1)*32).floor().long()
    covered=d2_visible & (mapping[source_parent]==target_parent[:,1]*64+target_parent[:,0])
    fine=point_metrics(result['fine_all_a'],result['fine_all_b'],gt)
    return dict(d8_parents=len(a),d8_correct=int(correct.sum()),d8_correct_rate=float(correct.sum())/max(1,len(a)),
        d2_reference_queries=int(d2_visible.sum()),d2_covered=int(covered.sum()),d2_coverage=float(covered.sum())/max(1,int(d2_visible.sum())),
        d2_candidates=fine['matches'],d2_ncm3=fine['ncm3'],d2_p3=fine['p3'])


def figure(path,batch,array):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    axes[0,0].imshow(batch['images'][0,0].permute(1,2,0));axes[0,0].set_title('A')
    axes[0,1].imshow(batch['images'][0,1].permute(1,2,0));axes[0,1].set_title('B')
    panels=[(axes[0,2],np.where(array['visible'],array['epe'],np.nan),32,'Reference EPE (px)'),
        (axes[1,0],array['matchability'],1,'U0 Matchability'),
        (axes[1,1],np.where(array['cycle_valid'],array['cycle'],np.nan),64,'Cycle error (px); white invalid'),
        (axes[1,2],array['reliability'],1,'Gaussian cycle reliability (sigma=16px)')]
    for ax,v,top,title in panels:
        im=ax.imshow(v.reshape(32,32),vmin=0,vmax=top,cmap='turbo');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.65)
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(batch['pair_id'][0]);fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',choices=['hangzhou','rgb'],required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);selection=prepare();output=ROOT/(args.domain+('_smoke' if args.smoke else ''));output.mkdir(exist_ok=False)
    shared,matcher=load_models();base=matcher.config
    rows=selection[args.domain];root=selection['roots']['hz' if args.domain=='hangzhou' else 'rgb_dev']
    dataset=GeometryPairDataset(root,rows);records=[];shown=set();cal=set(selection['rgb_calibration_ids'])
    for index,(row,batch) in enumerate(zip(rows,dataset)):
        images=batch['images'].cuda();gt=batch['geometry'].to('cuda:0')
        if index==0:
            warm=shared(images,pyramid_scales=(8,4,2));matcher.infer(warm);shared(images.flip(1),pyramid_scales=());del warm
        forward,fms=timed(lambda:shared(images,pyramid_scales=(8,4,2)))
        backward,rms=timed(lambda:shared(images.flip(1),pyramid_scales=()))
        cycle,cms=timed(lambda:cycle_consistency(forward,backward))
        xy=normalized_d16_grid(1,device=images.device,dtype=torch.float32)[0].permute(1,2,0).reshape(-1,2)
        target,visible,pos,neg=query_labels(gt,xy)
        pred=forward['flow16_target_norm'][0].permute(1,2,0).reshape(-1,2)
        array=dict(epe=(pred-target).norm(dim=-1)*256,visible=visible,positive=pos,negative=neg,
            cycle=cycle['error_px'].flatten(),cycle_valid=cycle['valid'].flatten(),boundary=cycle['boundary_or_outside'].flatten(),
            reliability=cycle['reliability'].flatten(),matchability=forward['flow16_matchability'].flatten(),
            features=reliability_features(forward,backward,cycle)[0].permute(1,2,0).reshape(1024,8))
        array={k:v.cpu().numpy() for k,v in array.items()};np.savez_compressed(output/(row['pair_id']+'.npz'),**array)
        if row['G'] not in shown:
            figure(output/(row['G']+'.jpg'),batch,array);shown.add(row['G'])
        role='calibration' if args.domain=='rgb' and row['pair_id'] in cal else 'evaluation'
        if role=='evaluation' or args.smoke:
            d2xy=normalized_cell_centers(256,256,device=images.device).reshape(-1,2)
            d2target,_,d2visible=gt.query(d2xy)
            for variant in VARIANTS:
                matcher.config=replace(base,flow_guidance_enabled=variant!='B0')
                modified=modulate(forward,cycle,variant)
                result,ms=timed(lambda:matcher.infer(modified)[0])
                record=dict(pair_id=row['pair_id'],group=row['G'],variant=variant,
                    **point_metrics(result['points_a'],result['points_b'],gt),**parent_metrics(result,gt,d2target,d2visible),
                    forward_ms=fms,reverse_ms=rms if variant in ('B2','B3','B4_uniform') else 0.,
                    cycle_ms=cms if variant in ('B2','B3','B4_uniform') else 0.,matching_ms=ms,
                    total_ms=fms+ms+(rms+cms if variant in ('B2','B3','B4_uniform') else 0.))
                records.append(record)
                with (output/'paired.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(dict(domain=args.domain,pair=index+1,total=len(rows),role=role)),flush=True)
        if args.smoke:break
    write(output/'status.json',dict(status='completed',pairs=index+1,variants=VARIANTS,no_training=True))

if __name__=='__main__':main()
