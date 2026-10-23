"""One sealed evaluation per locked model; no training or threshold selection."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from mhinet.config import RuntimePaths
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.d16_experiment.supervision import evaluate,targets
from mhinet.v2.d16_experiment.train import aggregate,write
from mhinet.v2.local_occlusion.evaluation import query_arrays,rejection
from mhinet.v2.local_occlusion.train import append_csv
from .decoder import CoordinateModel
from .calibrate import counts,rates

ROOT=Path('outputs/d16_final_round')


def load_model(variant):
    config=json.loads(Path('mhinet/v2/local_occlusion/config.json').read_text())
    shared=build_fresh_shared(replace(RuntimePaths.from_json(config['runtime']),device='cuda:0'))
    if variant.startswith('D'):base=Path('outputs/d16_local_occlusion/B0/latest.pt')
    elif variant.startswith('E'):base=ROOT/'ablation'/variant/'latest.pt'
    else:base=Path('outputs/d16_local_occlusion')/variant/'latest.pt'
    state=torch.load(base,map_location='cpu',weights_only=False)
    assert state['epoch']==30
    shared.mvt.load_state_dict(state['mvt']);shared.stage1_head.load_state_dict(state['head'])
    shared.requires_grad_(False);shared.eval()
    if variant.startswith('D'):
        model=CoordinateModel(shared,local=variant[1]=='1')
        checkpoint=ROOT/'decoder'/f'{variant}_seed42'/'latest.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['epoch']==30
        model.coordinate.load_state_dict(state['coordinate'])
    else:model=shared;checkpoint=base
    model.requires_grad_(False);model.eval()
    calibration='B0' if variant.startswith('D') else variant
    threshold=json.loads((ROOT/'calibration'/calibration/'threshold.json').read_text())
    return model,checkpoint,threshold


@torch.no_grad()
def benchmark(model,images):
    for _ in range(10):model(images,pyramid_scales=())
    torch.cuda.synchronize();elapsed=[]
    for _ in range(30):
        start=time.perf_counter();model(images,pyramid_scales=());torch.cuda.synchronize()
        elapsed.append(1000*(time.perf_counter()-start))
    return dict(device=torch.cuda.get_device_name(),batch_pairs=1,input_size=512,warmup=10,repeats=30,
        median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),samples_ms=elapsed,
        registered_parameters=sum(p.numel() for p in model.parameters()),
        extra_coordinate_parameters=sum(p.numel() for p in model.coordinate.parameters()) if hasattr(model,'coordinate') else 0,
        note='DINO/MVT/D16 forward only; includes frozen original Matchability head plus independent coordinate head in D variants. Registered parameters include unused stored pyramid modules; not an executed-parameter count.')


def figure(path,batch,a,pred,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=a['positive'];error=np.where(p,a['epe'],np.nan).reshape(32,32)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    axes[0].imshow(batch['images'][0,0].permute(1,2,0));axes[0].set_title('A')
    axes[1].imshow(batch['images'][0,1].permute(1,2,0))
    ids=np.flatnonzero(p)[::6];delta=pred-target
    axes[1].quiver(target[ids,0],target[ids,1],delta[ids,0],delta[ids,1],angles='xy',scale_units='xy',scale=1,color='cyan',width=.003)
    axes[1].set(xlim=(0,511),ylim=(511,0),title='B: reference-to-prediction residual, 1:1 px')
    im=axes[2].imshow(error,vmin=0,vmax=32,cmap='turbo');fig.colorbar(im,ax=axes[2],shrink=.7);axes[2].set_title('EPE at A queries (original px)')
    bad=p & (a['epe']>16) & (a['matchability']>=.9)
    axes[3].imshow(bad.reshape(32,32),vmin=0,vmax=1);axes[3].set_title('EPE>16 and Matchability>=0.9')
    for ax in axes:ax.axis('off')
    fig.suptitle(batch['pair_id'][0]);fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True);args=parser.parse_args()
    decision=json.loads((ROOT/'decision.json').read_text());assert decision['configuration_locked']
    output=ROOT/'sealed_test'/args.variant;output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);model,checkpoint,cal=load_model(args.variant)
    # Saved threshold only, never choose it on these test queries.
    t=cal['threshold'];selection=json.loads((ROOT/'test_selection.json').read_text());result={};timing=None
    for domain,role in [('hangzhou','hz_test'),('rgb','rgb_test')]:
        rows=selection[role];dataset=GeometryPairDataset(selection['roots']['hz' if domain=='hangzhou' else 'rgb_test'],rows)
        pairs=[];shown=set()
        for row,batch in zip(rows,dataset):
            images=batch['images'].to('cuda:0');gt=batch['geometry'].to('cuda:0')
            if timing is None:timing=benchmark(model,images);write(output/'latency.json',timing)
            stage=model(images,pyramid_scales=())['stage1'];labels=targets(gt)
            labels['negative']=torch.zeros_like(labels['positive']);labels['boundary']=labels['negative']
            a=query_arrays(stage,labels);c=counts(a,t)
            pred=(stage['coarse_warp'][0].float().permute(1,2,0).reshape(-1,2)+1)*256-.5
            target=labels['target'];border=((target<7.5)|(target>503.5)).any(-1).cpu().numpy() & a['positive']
            metrics=evaluate(stage,gt)
            record=dict(pair_id=row['pair_id'],group=row['G'],block=row.get('anchor_block_id',row.get('source_relative_path')),**metrics,
                **rejection(stage,labels['positive']),**{f'calibrated_{k}':v for k,v in rates(c).items()},
                border_points=int(border.sum()),border_epe=float(a['epe'][border].mean()) if border.any() else None,
                border_p3=float((a['epe'][border]<=3).mean()) if border.any() else None,
                high_confidence_error_count=int((a['positive'] & (a['epe']>16) & (a['matchability']>=.9)).sum()),counts=c)
            pairs.append(record);append_csv(output/'paired.csv',dict(domain=domain,**{k:v for k,v in record.items() if k!='counts'}))
            np.savez_compressed(output/f'{domain}_{row["pair_id"]}.npz',**a,border=border,prediction=pred.cpu().numpy(),target=target.cpu().numpy())
            if row['G'] not in shown:
                figure(output/f'{domain}_{row["G"]}.jpg',batch,a,pred.cpu().numpy(),target.cpu().numpy());shown.add(row['G'])
        total={k:sum(r['counts'][k] for r in pairs) for k in pairs[0]['counts']}
        # aggregate only the original metrics; supplemental quantities have their own denominators.
        original=[{k:r[k] for k in metrics} for r in pairs]
        result[domain]=dict(overall=aggregate(original),groups={g:aggregate([{k:r[k] for k in metrics} for r in pairs if r['group']==g]) for g in sorted(shown)},
            calibrated=rates(total),pairs=pairs)
        print(json.dumps(dict(variant=args.variant,domain=domain,overall=result[domain]['overall'])),flush=True)
    write(output/'summary.json',dict(variant=args.variant,checkpoint=str(checkpoint),threshold=t,threshold_scope='Hangzhou artificial rectangle calibration; RGB transfer is descriptive only',**result))

if __name__=='__main__':main()
