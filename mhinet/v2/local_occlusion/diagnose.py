"""Read-only epoch-30 decoder attribution on identical validation queries."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mhinet.config import RuntimePaths
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.d16_experiment.supervision import targets, grid
from mhinet.v2.d16_experiment.train import write

AUDIT = {'val_G3_12add035b2b2190ad8b131b2', 'val_G3_a4e0a479df8364b40227e2c5',
         'val_G3_d675c8471dc323c373c41661'}


def ideal_decode(head, gt, dtype):
    """Diagnostic only: bilinear H-coordinate oracle -> original PE/decoder."""
    xy = (grid(gt.H.device)+.5)/256-1
    weights = targets(replace(gt, exact=True))['weights']
    visible = targets(gt)['positive']
    probability = F.pad(weights, (0, 1))
    probability[~visible] = 0
    probability[~visible, -1] = 1
    projection = F.linear(xy.to(dtype), (head.position_scale*head.position_frequencies).to(dtype))
    pe = torch.cat((projection.sin(), projection.cos()), -1)
    pe = torch.cat((pe, head.no_match_embedding.to(pe)[None]), 0)
    embedding = (probability @ pe.float()).reshape(1, 32, 32, -1)
    warp, _ = head.match_embedding_decoder(embedding)
    return (warp[0].permute(1, 2, 0).reshape(-1, 2).float()+1)*256-.5


def stats(values, keep):
    e = values['epe'][keep]
    return dict(points=len(e), epe=float(e.mean()) if len(e) else None,
                p3=float((e<=3).mean()) if len(e) else None,
                ideal_epe=float(values['ideal_epe'][keep].mean()) if len(e) else None,
                ideal_p3=float((values['ideal_epe'][keep]<=3).mean()) if len(e) else None)


def conditions(a):
    v = a['valid']; e = a['epe']; d = a['border_distance']
    sets = {'all':v, 'rank1':v & (a['rank']==1), 'rank_above1':v & (a['rank']>1),
            'positive_mass_ge_0.9':v & (a['positive_mass']>=.9),
            'positive_top1_ge_0.9':v & (a['positive_top1_probability']>=.9),
            'high_confidence_epe_gt8':v & (a['positive_mass']>=.9) & (e>8)}
    for lo,hi in [(0,8),(8,16),(16,32),(32,float('inf'))]:
        sets[f'border_{lo}_{hi}'] = v & (d>=lo) & (d<hi)
    for lo,hi in [(0,3),(3,8),(8,32),(32,float('inf'))]:
        sets[f'error_{lo}_{hi}'] = v & (e>lo if lo else e>=0) & (e<=hi)
    return {k:stats(a,m) for k,m in sets.items()}


def draw(output, batch, gt, data, query, audit=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images=batch['images'][0].permute(0,2,3,1).numpy()
    fig,ax=plt.subplots(2,3,figsize=(14,9))
    ax[0,0].imshow(images[0]);ax[0,0].scatter(*data['source'][query],c='yellow',s=30);ax[0,0].set_title('A source query')
    ax[0,1].imshow(images[1]);ax[0,1].scatter(*data['target'][query],c='yellow',marker='*',s=120)
    ax[0,1].scatter(*data['prediction'][query],c='cyan',marker='x',s=80)
    top=data['top_xy'][query];ax[0,1].scatter(top[:,0],top[:,1],s=15,c='red');ax[0,1].set_title('B reference(*), prediction(x), Top10(red)')
    x,y=data['target'][query];ax[0,2].imshow(images[1]);ax[0,2].set_xlim(max(0,x-48),min(512,x+48));ax[0,2].set_ylim(min(512,y+48),max(0,y-48))
    ax[0,2].scatter(x,y,marker='*',s=120,c='yellow');ax[0,2].scatter(*data['prediction'][query],marker='x',s=60,c='cyan');ax[0,2].set_title('Local B reference patch, +/-48 px')
    ax[1,0].imshow(gt.mask_a[0].cpu(),vmin=0,vmax=1);ax[1,0].set_title('Original A validity mask')
    ax[1,1].imshow(gt.mask_b[0].cpu(),vmin=0,vmax=1);ax[1,1].set_title('Original B validity mask')
    if audit:
        q=(torch.arange(512,device=gt.H.device)+.5)/256-1
        yy,xx=torch.meshgrid(q,q,indexing='ij');p=torch.stack((xx,yy),-1)
        target,_,_=gt.query(p)
        warped=F.grid_sample(batch['images'][:,1].to(gt.H.device),target[None],align_corners=False)[0].permute(1,2,0).cpu()
        ax[1,2].imshow(.5*images[0]+.5*warped.numpy());ax[1,2].set_title('A / H-warped B overlay (reference only)')
    else:
        ax[1,2].bar(np.arange(1,11),data['top_prob'][query]);ax[1,2].set_title('Top10 probability; spatial candidates')
    for a in ax.flat:
        if a is not ax[1,2] or audit:a.axis('off')
    fig.suptitle(f'{batch["pair_id"][0]} q={query}; EPE={data["epe"][query]:.2f}; rank={data["rank"][query]}; positive mass={data["positive_mass"][query]:.3f}')
    fig.tight_layout();fig.savefig(output,dpi=110);plt.close(fig)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['U','B','C'],required=True);args=p.parse_args()
    config=json.loads(Path('mhinet/v2/d16_experiment/config.json').read_text())
    output=Path('outputs/d16_local_occlusion/diagnostics')/args.variant;output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    model=build_fresh_shared(replace(RuntimePaths.from_json(config['runtime']),device='cuda:0'))
    path=Path(config['output'])/args.variant/'latest.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['epoch']==30
    model.mvt.load_state_dict(state['mvt']);model.stage1_head.load_state_dict(state['head']);model.eval()
    rows=json.loads(Path(config['selection']).read_text())['val'];dataset=GeometryPairDataset(config['dataset'],rows)
    all_data=[];pair_metrics=[];closest={};saved_high=0
    for index,batch in enumerate(dataset):
        gt=batch['geometry'].to('cuda:0');out=model(batch['images'].to('cuda:0'),pyramid_scales=());s=out['stage1'];t=targets(gt)
        pred=(s['coarse_warp'][0].float().permute(1,2,0).reshape(-1,2)+1)*256-.5
        prob=s['match_probabilities'][0].float();spatial=prob[:,:1024];region=t['region'];pos=spatial.masked_fill(~region,0)
        top_prob,top=spatial.topk(10,-1);best=pos.max(-1).values
        ideal=ideal_decode(model.stage1_head,gt,out['pair_descriptors'].dtype)
        delta=pred-t['target'];boundary=torch.minimum(t['target']+.5,511.5-t['target']).amin(-1)
        tensors=dict(valid=t['positive'],source=t['source'],target=t['target'],prediction=pred,delta=delta,
            epe=delta.norm(dim=-1),ideal_prediction=ideal,ideal_epe=(ideal-t['target']).norm(dim=-1),
            rank=(spatial>best[:,None]).sum(-1)+1,positive_mass=pos.sum(-1),positive_top1_probability=best,
            top_xy=t['source'][top],top_prob=top_prob,entropy=-(prob*prob.clamp_min(1e-30).log()).sum(-1),
            top10_mass=top_prob.sum(-1),border_distance=boundary,dustbin=prob[:,-1],matchability=s['coarse_matchability'][0,0].flatten())
        data={k:v.cpu().numpy() for k,v in tensors.items()};pid=batch['pair_id'][0]
        np.savez_compressed(output/(pid+'.npz'),**data)
        all_data.append(data);pair_metrics.append(dict(pair_id=pid,group=batch['group'],**conditions(data)))
        high=np.where(data['valid'] & (data['positive_mass']>=.9) & (data['epe']>8))[0]
        if len(high):
            q=int(high[np.argmax(data['epe'][high])]);draw(output/(pid+'_high_confidence.jpg'),batch,gt,data,q);saved_high+=1
        if pid in AUDIT:
            q=int(np.where(data['valid'],data['epe'],-1).argmax());draw(output/(pid+'_audit.jpg'),batch,gt,data,q,True)
            write(output/(pid+'_metadata.json'),dict(row=rows[index],metrics=pair_metrics[-1],
                 interpretation='H/mask overlay is visual evidence only; label error versus unobservable correspondence is not established automatically.'))
        if batch['group'] in ('G1','G3'):
            wanted_p,wanted_e=(.902,22.51) if batch['group']=='G1' else (.944,34.42)
            # Search both possible meanings of quoted "correct probability".
            for name in ('positive_mass','positive_top1_probability'):
                distance=np.abs(data[name]-wanted_p)*100+np.abs(data['epe']-wanted_e)
                distance[~data['valid']]=np.inf;q=int(distance.argmin());key=batch['group']+'_'+name
                if key not in closest or float(distance[q])<closest[key]['distance']:
                    closest[key]=dict(pair_id=pid,query=q,distance=float(distance[q]),epe=float(data['epe'][q]),
                        positive_mass=float(data['positive_mass'][q]),positive_top1_probability=float(data['positive_top1_probability'][q]),
                        rank=int(data['rank'][q]),delta=data['delta'][q].tolist(),ideal_epe=float(data['ideal_epe'][q]))
                    draw(output/(key+'_quoted_case_nearest.jpg'),batch,gt,data,q)
        print(json.dumps(dict(variant=args.variant,pair=index+1,pair_id=pid)),flush=True)
    merged={k:np.concatenate([a[k] for a in all_data]) for k in all_data[0]}
    reduced={k:np.concatenate([a[k] for a,row in zip(all_data,rows) if row['pair_id'] not in AUDIT]) for k in all_data[0]}
    report=dict(variant=args.variant,checkpoint=str(path),epoch=30,full=conditions(merged),
        excluding_three_diagnostic_only=conditions(reduced),pairs=pair_metrics,quoted_cases_nearest=closest,
        high_confidence_pair_visualizations=saved_high,
        ideal_distribution='H bilinear oracle; invalid queries dustbin. Diagnostic reference, not true geometry or inference.')
    write(output/'summary.json',report)
    print(json.dumps(report['full']),flush=True)


if __name__=='__main__':main()
