"""Frozen, paired stage substitutions: separate selection from localization."""
import argparse
import json
from pathlib import Path
import torch
from mhinet.v2.qrru_corners.experiment import selections, load_models
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.flow_d8_d2.run import measure
from mhinet.v2.flow_d8_d2.matching import infer as discrete
import torch.nn.functional as F
from . import path


def expanded_fine(shared, matcher, source, center):
    """Only expand B to 8x8; preserve the moving-grid decoder, without a new prior."""
    axis=(torch.arange(8,device=source.device)-3.5)*2
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    candidates=center[:,None]+torch.stack((x,y),-1).reshape(1,64,2)
    d2=shared['pyramid'][2][0].float()
    a=F.normalize(path.sample(d2[0],source.reshape(-1,16,2)),dim=-1)
    b=F.normalize(path.sample(d2[1],candidates),dim=-1)
    logits=a@b.transpose(-1,-2)/matcher.tau_f
    valid=(candidates>=0).all(-1)&(candidates<=511).all(-1)
    logits=logits.masked_fill(~valid[:,None],-1e4)
    lp=logits.log_softmax(-1)+logits.log_softmax(-2)
    expanded=candidates[:,None].expand(-1,16,-1,-1).reshape(-1,64,2)
    point,_,_,_=path.decode(lp.reshape(-1,64),expanded,8)
    best=lp.argmax(-1);reverse=lp.argmax(-2)
    mutual=(reverse.gather(1,best)==torch.arange(16,device=source.device)[None])
    mutual &= valid.gather(1,best)
    return point,mutual.flatten(),expanded


def summarize(rows):
    output={}
    for name in rows[0]['metrics']:
        values=[row['metrics'][name] for row in rows]
        total=lambda k:sum(v[k] for v in values)
        output[name]=dict(p3=100*total('correct3')/max(1,total('n')),
            points=total('n')/len(values),correct3=total('correct3')/len(values),
            epe=total('error_sum')/max(1,total('visible')))
    return output


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',required=True,choices=['synthetic','hangzhou']);p.add_argument('--device',required=True);p.add_argument('--limit',type=int,default=64)
    args=p.parse_args();torch.set_num_threads(2)
    model,matcher=load_models(args.device);model.eval();matcher.eval()
    spec=selections()[args.domain];ds=GeometryPairDataset(spec['root'],spec['val'][:args.limit])
    out=Path('outputs/joint_diagnosis/attribution');out.mkdir(parents=True,exist_ok=True);rows=[]
    for batch in ds:
        shared=model(batch['images'].to(args.device));gt=batch['geometry'].to(args.device)
        old=discrete(shared,matcher,'local');new=path.infer(shared,matcher)
        metrics={}
        def add(name,a,b,keep=None):
            if keep is not None:a,b=a[keep],b[keep]
            metrics[name]=measure(a,b,gt)[0]
        add('old_final',old['points_a'],old['points_b'])
        add('new_final_pre_qrru',new['fine_source'],new['p2'])
        add('new_final_qrru',new['fine_source'],new['final'])
        # Same D8 parents: use the top 128 original discrete parents, not new selection.
        source=old['coarse_a'][:128];parent=old['coarse_b'][:128]
        r=path.coarse(shared,matcher,source)
        add('same_parents_flow',source,r['p0']);add('same_parents_old_d8',source,parent);add('same_parents_new_d8',source,r['p8'])
        # Native D2 lattice centered on the discrete D8 parent is exactly the legacy grid.
        for parent_kind,center in [('old_parent',parent),('new_parent',r['p8'])]:
            f=path.fine(shared,matcher,{**r,'p8':center})
            a=f['fine_source'];lp=f['logits2'].reshape(-1,16,16)
            best=lp.argmax(-1);reverse=lp.argmax(-2)
            mutual=(reverse.gather(1,best)==torch.arange(16,device=args.device)[None]).flatten()
            valid=f['valid2'].gather(1,best.flatten()[:,None]).flatten()
            hard=f['candidates2'][torch.arange(len(a),device=args.device),best.flatten()]
            add(parent_kind+'_hard_all',a,hard);add(parent_kind+'_soft_all',a,f['p2'])
            add(parent_kind+'_hard_mutual',a,hard,mutual&valid)
            add(parent_kind+'_soft_mutual',a,f['p2'],mutual&valid)
            # Identical source points: can D2 improve on interpolated Flow alone?
            flow=a+path.sample(shared['flow16_norm'][0],a)*256
            add(parent_kind+'_flow_same_sources',a,flow)
            truth,known,visible=gt.query((a+.5)/256-1);truth=(truth+1)*256-.5
            c=f['candidates2'];v=f['valid2']
            oracle=c[torch.arange(len(a),device=args.device),(c-truth[:,None]).norm(dim=-1).masked_fill(~v,torch.inf).argmin(-1)]
            add(parent_kind+'_oracle_candidate',a,oracle)
            expanded,keep,c8=expanded_fine(shared,matcher,a,center)
            add(parent_kind+'_8_soft_all',a,expanded)
            add(parent_kind+'_8_soft_mutual',a,expanded,keep)
            v8=(c8>=0).all(-1)&(c8<=511).all(-1)
            closest=(c8-truth[:,None]).norm(dim=-1).masked_fill(~v8,torch.inf).argmin(-1)
            add(parent_kind+'_8_oracle_candidate',a,c8[torch.arange(len(a),device=args.device),closest])
        rows.append(dict(pair_id=batch['pair_id'][0],metrics=metrics))
        (out/(args.domain+'.json')).write_text(json.dumps(dict(summary=summarize(rows),pairs=rows),indent=2))
        print(args.domain,len(rows),flush=True)
    print(json.dumps(summarize(rows),indent=2),flush=True)

if __name__=='__main__':main()
