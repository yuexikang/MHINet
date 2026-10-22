"""Clean and held-out artificial-occlusion metrics, with paired query counts."""
import numpy as np
import torch
from mhinet.v2.d16_experiment.supervision import evaluate, targets
from mhinet.v2.d16_experiment.train import aggregate
from .occlusion import make_occlusion


def rejection(stage, positive):
    probability=stage['match_probabilities'][0].float()
    matchability=stage['coarse_matchability'][0,0].float().flatten()
    dust=probability.argmax(-1)==1024
    low=matchability<.5
    def mean(v):return float(v[positive].float().mean()) if positive.any() else None
    return dict(false_reject=mean(dust|low),dustbin_false_reject=mean(dust),matchability_false_reject=mean(low))


def query_arrays(stage, labels):
    p=stage['match_probabilities'][0].float()
    pred=(stage['coarse_warp'][0].float().permute(1,2,0).reshape(-1,2)+1)*256-.5
    return {k:v.detach().cpu().numpy() for k,v in dict(positive=labels['positive'],negative=labels['negative'],
        boundary=labels['boundary'],epe=(pred-labels['target']).norm(dim=-1),
        dustbin_top1=p.argmax(-1)==1024,dustbin=p[:,-1],
        matchability=stage['coarse_matchability'][0,0].float().flatten()).items()}


def auc(labels, score):
    """Mann-Whitney AUROC, exact average ranks for ties; positive = observable."""
    npos=int(labels.sum());nneg=len(labels)-npos
    if not npos or not nneg:return None
    _,inverse,counts=np.unique(score,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts);ranks=ends-(counts-1)/2
    return float((ranks[inverse][labels].sum()-npos*(npos+1)/2)/(npos*nneg))


def summarize(arrays):
    a={k:np.concatenate([r[k] for r in arrays]) for k in arrays[0]}
    p,n,b=a['positive'],a['negative'],a['boundary'];m=a['matchability'];dust=a['dustbin_top1']
    def avg(v,mask):return float(v[mask].mean()) if mask.any() else None
    return dict(positive_count=int(p.sum()),negative_count=int(n.sum()),boundary_count=int(b.sum()),
        negative_dustbin_top1=avg(dust,n),negative_matchability_mean=avg(m,n),
        negative_matchability_quantiles=np.quantile(m[n],[0,.25,.5,.75,1]).tolist() if n.any() else None,
        positive_false_reject=avg(dust|(m<.5),p),positive_dustbin_false_reject=avg(dust,p),
        positive_matchability_false_reject=avg(m<.5,p),positive_p3=avg(a['epe']<=3,p),positive_epe=avg(a['epe'],p),
        boundary_epe=avg(a['epe'],b),boundary_p3=avg(a['epe']<=3,b),
        matchability_auroc=auc(p[p|n],m[p|n]))


def plot(output, batch, modified, labels, metadata, stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    original=batch['images'][0].permute(0,2,3,1).numpy()
    axes[0,0].imshow(original[0]);axes[0,0].set_title('Original A')
    axes[0,1].imshow(original[1]);axes[0,1].set_title('Original B')
    axes[0,2].imshow(modified[0,1].permute(1,2,0).cpu());axes[0,2].set_title(f'Modified B; area={metadata["actual_area"]:.3f}')
    state=np.full((1024,3),.5);state[labels['positive'].cpu()]=[.1,.8,.2];state[labels['negative'].cpu()]=[.9,.1,.1]
    axes[0,3].imshow(state.reshape(32,32,3));axes[0,3].set_title('A queries: green P / red N / gray Ignore')
    dust=stage['match_probabilities'][0,:,-1].reshape(32,32).float().cpu()
    axes[1,0].imshow(dust,vmin=0,vmax=1,cmap='magma');axes[1,0].set_title('Dustbin probability [0,1]')
    axes[1,1].imshow(stage['coarse_matchability'][0,0].float().cpu(),vmin=0,vmax=1,cmap='viridis');axes[1,1].set_title('Matchability [0,1]')
    pred=(stage['coarse_warp'][0].float().permute(1,2,0).reshape(-1,2)+1)*256-.5
    keep=labels['positive'];ref=labels['target'][keep].cpu().numpy();delta=(pred-labels['target'])[keep].cpu().numpy()
    axes[1,2].imshow(modified[0,1].permute(1,2,0).cpu());axes[1,2].quiver(ref[::3,0],ref[::3,1],delta[::3,0],delta[::3,1],angles='xy',scale_units='xy',scale=1,color='cyan',width=.003)
    axes[1,2].set_title('Positive residual on B, 1:1 px')
    error=(pred-labels['target']).norm(dim=-1).cpu().numpy();error[~keep.cpu().numpy()]=np.nan
    im=axes[1,3].imshow(error.reshape(32,32),vmin=0,vmax=32,cmap='turbo');fig.colorbar(im,ax=axes[1,3],shrink=.6);axes[1,3].set_title('Positive EPE on A queries (px)')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(batch['pair_id'][0]);fig.tight_layout();fig.savefig(output,dpi=110);plt.close(fig)


@torch.no_grad()
def validate(model, dataset, config, output, epoch, full=False):
    model.eval();clean=[];buckets={};pair_rows=[];forwards=0;shown=set()
    for batch in dataset:
        images=batch['images'].to('cuda:0');gt=batch['geometry'].to('cuda:0');pid=batch['pair_id'][0]
        stage=model(images,pyramid_scales=())['stage1'];forwards+=1
        clean.append(dict(pair_id=pid,group=batch['group'],**evaluate(stage,gt),**rejection(stage,targets(gt)['positive'])))
        # Fixed independent validation seed and epoch=0: identical masks across models/epochs.
        cases=[(None,None)]
        if full:cases += [(a,f) for a in (.05,.10,.20) for f in (0.,1.)]+[(.10,.5)]
        for area,fill in cases:
            modified,labels,metadata=make_occlusion(images,gt,pid,0,config['validation_seed'],area=area,fill=fill)
            stage=model(modified,pyramid_scales=())['stage1'];forwards+=1
            key='fixed_mixture' if area is None else f'area_{area:.2f}_rgb_{round(fill*255)}'
            arrays=query_arrays(stage,labels);buckets.setdefault(key,[]).append(arrays)
            pair_rows.append(dict(pair_id=pid,group=batch['group'],condition=key,**metadata,**summarize([arrays])))
            if full and area is None and batch['group'] not in shown:
                plot(output/f'{batch["group"]}_epoch_{epoch:03d}.jpg',batch,modified,labels,metadata,stage)
                shown.add(batch['group'])
    return dict(epoch=epoch,clean=aggregate(clean),groups={g:aggregate([r for r in clean if r['group']==g]) for g in sorted({r['group'] for r in clean})},
        clean_pairs=clean,occlusion={k:summarize(v) for k,v in buckets.items()},occlusion_pairs=pair_rows,validation_forwards=forwards)
