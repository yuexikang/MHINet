"""No optimizer updates: dustbin, calibrated matchability and gradient diagnosis."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from mhinet.config import RuntimePaths
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.frozen_locator import initialize_locator
from mhinet.v2.d16_experiment.supervision import targets,correspondence_loss
from mhinet.v2.local_occlusion.occlusion import make_occlusion,augmented_losses
from mhinet.v2.local_occlusion.evaluation import query_arrays
from mhinet.v2.d16_experiment.train import write
from .prepare import balanced

ROOT=Path('outputs/d16_final_round')


def distribution(values):
    if not len(values):return dict(count=0)
    return dict(count=len(values),mean=float(values.mean()),quantiles=dict(zip(['min','p05','p25','p50','p75','p95','p99','max'],
        np.quantile(values,[0,.05,.25,.5,.75,.95,.99,1]).tolist())))


def threshold(clean,positive):
    def cap(x):
        x=np.sort(x);return float(x[min(int(np.floor(.01*len(x))),len(x)-1)])
    return min(cap(clean),cap(positive))


def counts(a,t):
    p,n=a['positive'],a['negative'];rejected=a['matchability']<t;accepted=p & ~rejected
    return dict(p=int(p.sum()),n=int(n.sum()),fp=int((p & rejected).sum()),tp=int((n & rejected).sum()),
        accepted=int(accepted.sum()),accepted_p3=int((accepted & (a['epe']<=3)).sum()),
        accepted_error=float(a['epe'][accepted].sum()))


def rates(c):
    return dict(negative_recall=c['tp']/c['n'] if c['n'] else None,positive_false_reject=c['fp']/c['p'] if c['p'] else None,
        precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None,
        positive_coverage=c['accepted']/c['p'] if c['p'] else None,
        accepted_p3=c['accepted_p3']/c['accepted'] if c['accepted'] else None,
        accepted_epe=c['accepted_error']/c['accepted'] if c['accepted'] else None)


def sum_counts(rows):return {k:sum(r[k] for r in rows) for k in rows[0]}


def bootstrap(rows):
    blocks=sorted({r['block'] for r in rows});aggregated=[sum_counts([r['counts'] for r in rows if r['block']==b]) for b in blocks]
    rng=np.random.default_rng(20261023);values=[]
    for _ in range(1000):values.append(rates(sum_counts([aggregated[i] for i in rng.integers(0,len(blocks),len(blocks))])))
    return dict(blocks=len(blocks),interval95={k:np.quantile([v[k] for v in values if v[k] is not None],[.025,.975]).tolist()
                for k in values[0] if any(v[k] is not None for v in values)})


def curves(output,arrays,t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=np.concatenate([a['matchability'][a['positive']|a['negative']] for a in arrays])
    negative=np.concatenate([a['negative'][a['positive']|a['negative']] for a in arrays])
    order=np.argsort(m,kind='stable');score=m[order];label=negative[order]
    ends=np.r_[np.where(score[:-1]!=score[1:])[0],len(score)-1]
    tp=np.cumsum(label)[ends];fp=np.cumsum(~label)[ends]
    recall=tp/negative.sum();fpr=fp/(~negative).sum();precision=tp/(tp+fp)
    np.savez_compressed(output/'curves.npz',matchability=score[ends],negative_recall=recall,positive_false_reject=fpr,precision=precision)
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(np.r_[0,fpr],np.r_[0,recall]);ax[0].axvline(.01,color='red',ls='--');ax[0].set(xlabel='Positive false reject',ylabel='Negative recall',title='ROC: absent is positive class')
    ax[1].plot(recall,precision);ax[1].set(xlabel='Negative recall',ylabel='Precision',title='PR on artificial occlusion queries')
    for mask,name in [(negative,'Artificial negatives'),(~negative,'Observable positives')]:ax[2].hist(m[mask],bins=50,density=True,alpha=.5,label=name)
    ax[2].axvline(t,color='red',ls='--');ax[2].legend();ax[2].set(title='Matchability distributions',xlabel='Matchability')
    fig.tight_layout();fig.savefig(output/'operating_curves.jpg',dpi=140);plt.close(fig)


@torch.no_grad()
def collect(model,rows,root,split,output,full):
    collected=[]
    for row,batch in zip(rows,GeometryPairDataset(root,rows)):
        gt=batch['geometry'].to('cuda:0');images=batch['images'].to('cuda:0')
        cases=[('clean',None,None),('mixture',None,None)]
        if full:cases += [(f'area{a}_rgb{round(f*255)}',a,f) for a in (.05,.1,.2) for f in (0.,1.)]
        for name,area,fill in cases:
            if name=='clean':
                labels=targets(gt);labels['negative']=torch.zeros_like(labels['positive']);labels['boundary']=labels['negative']
                modified=images;meta={}
            else:modified,labels,meta=make_occlusion(images,gt,row['pair_id'],0,20261023,area=area,fill=fill)
            stage=model(modified,pyramid_scales=())['stage1'];a=query_arrays(stage,labels)
            p=stage['match_probabilities'][0].float()
            a['dustbin_margin']=(p[:,-1].log()-p[:,:1024].max(-1).values.log()).cpu().numpy()
            a['ignore']=~(a['positive']|a['negative'])
            record=dict(pair_id=row['pair_id'],group=row['G'],block=row['anchor_block_id'],condition=name,**meta)
            path=output/f'{split}_{row["pair_id"]}_{name}.npz';np.savez_compressed(path,**a)
            collected.append((record,a))
        if len(collected)%32==0:print(json.dumps(dict(event=split,pairs=len(collected)//len(cases))),flush=True)
    return collected


def norm(loss,model):
    groups={'mvt':list(model.mvt.parameters()),'coordinate_decoder':list(model.stage1_head.match_embedding_decoder.net.parameters())+list(model.stage1_head.match_embedding_decoder.warp_head.parameters()),
            'no_match_logit':[model.stage1_head.no_match_logit]}
    params=[p for group in groups.values() for p in group];grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    result={};offset=0
    for k,params in groups.items():
        gs=grad[offset:offset+len(params)];offset+=len(params)
        result[k]=float(sum((g.float().square().sum() for g in gs if g is not None),loss.new_zeros(())).sqrt())
        if k=='no_match_logit':result['no_match_logit_signed']=float(gs[0]) if gs[0] is not None else 0.
    return result


def diagnose_gradients(model,rows,root,config,variant):
    result=[]
    for row,batch in zip(rows,GeometryPairDataset(root,rows)):
        gt=batch['geometry'].to('cuda:0');images=batch['images'].to('cuda:0')
        clean,_=correspondence_loss(model(images,pyramid_scales=())['stage1'],gt,'U' if variant in ('U0','N1') else 'B',config)
        values={'clean':norm(clean,model)};del clean
        modified,labels,meta=make_occlusion(images,gt,row['pair_id'],30,config['seed'])
        terms=augmented_losses(model(modified,pyramid_scales=())['stage1'],gt,labels,config)
        for key,loss in terms.items():values[key]=norm(loss,model)
        result.append(dict(pair_id=row['pair_id'],gradient_norms=values,**meta))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True);args=parser.parse_args()
    config=json.loads(Path('mhinet/v2/local_occlusion/config.json').read_text());selection=json.loads((ROOT/'selection.json').read_text())
    output=ROOT/'calibration'/args.variant;output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    model=build_fresh_shared(replace(RuntimePaths.from_json(config['runtime']),device='cuda:0'))
    initialize_locator(SimpleNamespace(shared=model),config);model.set_training_groups(mvt=True,vgg=False,dedode=False,ghim_head=True)
    scalars={'initial':float(model.stage1_head.no_match_logit.detach())}
    source=(ROOT/'ablation'/args.variant) if args.variant.startswith('E') else Path('outputs/d16_local_occlusion')/args.variant
    for name,file in [('middle','middle.pt'),('final','latest.pt')]:
        state=torch.load(source/file,map_location='cpu',weights_only=False)
        scalars[name]=float(state['head']['no_match_logit'])
    assert state['epoch']==30
    model.mvt.load_state_dict(state['mvt']);model.stage1_head.load_state_dict(state['head']);model.eval()
    calibration=collect(model,selection['hz_calibration'],selection['roots']['hz'],'calibration',output,False)
    def select_values(records,condition,label):return np.concatenate([a['matchability'][a[label]] for r,a in records if r['condition']==condition])
    t=threshold(select_values(calibration,'clean','positive'),select_values(calibration,'mixture','positive'))
    # Write threshold before the independent operating set is evaluated.
    write(output/'threshold.json',dict(threshold=t,rule='reject iff matchability < threshold',source='hz_calibration',max_empirical_positive_fpr=.01))
    operating=collect(model,selection['hz_operating'],selection['roots']['hz'],'operating',output,True)
    pairs=[]
    for r,a in operating:pairs.append(dict(**r,counts=counts(a,t)))
    summaries={}
    for condition in sorted({r['condition'] for r in pairs}):
        rows=[r for r in pairs if r['condition']==condition];summaries[condition]=dict(**rates(sum_counts([r['counts'] for r in rows])),counts=sum_counts([r['counts'] for r in rows]),bootstrap=bootstrap(rows),
            groups={g:rates(sum_counts([r['counts'] for r in rows if r['group']==g])) for g in sorted({r['group'] for r in rows})})
    mixture=[a for r,a in operating if r['condition']=='mixture'];curves(output,mixture,t)
    negative_dust=np.concatenate([a['dustbin'][a['negative']] for a in mixture]);negative_margin=np.concatenate([a['dustbin_margin'][a['negative']] for a in mixture])
    positive_margin=np.concatenate([a['dustbin_margin'][a['positive']] for a in mixture])
    gradient_rows=balanced(selection['hz_train'],32,20261023)
    gradients=diagnose_gradients(model,gradient_rows,selection['roots']['hz'],config,args.variant)
    write(output/'gradients.json',dict(mode='replayed at epoch30 eval-mode, weighted terms, no optimizer step; hypothetical disabled terms also shown',pairs=gradients))
    historical=[]
    for line in (source/'training.jsonl').open():
        row=json.loads(line)
        if 'clean_gradient_norms' in row:historical.append(row)
    write(output/'historical_gradients.json',historical)
    passed=summaries['clean']['positive_false_reject']<=.01 and summaries['mixture']['positive_false_reject']<=.01 and summaries['mixture']['negative_recall']>=.5
    write(output/'summary.json',dict(variant=args.variant,checkpoint=str(source/'latest.pt'),threshold=t,no_match_logit=scalars,
        negative_dustbin_probability=distribution(negative_dust),negative_margin=distribution(negative_margin),positive_margin=distribution(positive_margin),
        negative_dustbin_top1=float((negative_margin>0).mean()),operating=summaries,pairs=pairs,passed=passed))
    print(json.dumps(dict(event='completed',variant=args.variant,threshold=t,passed=passed,operating=summaries['mixture'])),flush=True)


if __name__=='__main__':main()
