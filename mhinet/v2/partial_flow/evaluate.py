"""Calibration-only thresholds and paired held-out evaluation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from mhinet.v2.local_occlusion.evaluation import auc
from .dataset import load_config, dump
from .labels import MATCH, NO_MATCH, UNKNOWN
from .nomatch_head import NoMatchHead


def average_precision_score(labels, score):
    """Step-integral precision/recall with all tied scores grouped together."""
    order=np.argsort(-score,kind="stable");labels=labels[order];score=score[order]
    ends=np.r_[np.flatnonzero(np.diff(score)!=0),len(score)-1]
    tp=np.cumsum(labels)[ends];precision=tp/(ends+1)
    recall=tp/tp[-1]
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def load_records(split):
    out=Path(load_config()['output'])
    rows=[json.loads(s) for s in (out/'pairs.jsonl').read_text().splitlines()]
    records=[]
    for row in rows:
        if row['split'] != split: continue
        for direction in (('ab',) if split=='hz' else ('ab','ba')):
            records.append(torch.load(out/'cache'/f'{row["pair_id"]}_{direction}.pt',weights_only=False,map_location='cpu'))
    return records


def baseline_scores(record):
    d,m=record['p_dustbin'].astype(np.float64),record['matchability'].astype(np.float64)
    return dict(A0=d,A1=1-m,A2=1-m*(1-d))


def calibrate(records,scores,config):
    """Conservative empirical quantiles, guarding each positive data type."""
    thresholds=[]
    for rate in (config['calibration_positive_reject'],config['calibration_positive_no_match']):
        quantiles=[]
        for kind in ('full','partial','texture'):
            values=np.concatenate([s[r['state']==MATCH] for r,s in zip(records,scores) if r['row']['kind']==kind])
            # 'higher' and <= acceptance keep empirical rejected positives <= rate.
            quantiles.append(float(np.quantile(values,1-rate,method='higher')))
        thresholds.append(max(quantiles))
    low,high=thresholds
    # Equality at the upper threshold counts as rejection. Move to the next
    # float so tied calibration scores cannot silently violate the constraint.
    high=float(np.nextafter(max(high,low),np.inf))
    return [low,high]


def states(score,thresholds):
    low,high=thresholds
    out=np.full(score.shape,UNKNOWN,np.int8)
    out[score<=low]=MATCH;out[score>=high]=NO_MATCH
    return out


def fraction(a,mask):
    return float(np.mean(a[mask])) if mask.any() else None


def metrics(records,scores,thresholds,subset='all'):
    truth=np.concatenate([r['state'] for r in records])
    negative_kind=np.concatenate([r['negative_kind'] for r in records])
    epe=np.concatenate([r['epe_px'] for r in records])
    kind=np.concatenate([np.full(1024,r['row']['kind']) for r in records])
    selected=np.ones(len(truth),bool)
    if subset=='geometry': selected=kind=='partial'
    elif subset=='texture': selected=(kind=='texture') & ((truth!=NO_MATCH) | (negative_kind==2))
    elif subset=='full': selected=kind=='full'
    score=np.concatenate(scores).astype(np.float64)
    pred=states(score,thresholds)
    pos=selected & (truth==MATCH);neg=selected & (truth==NO_MATCH);known=pos|neg
    accept=pred==MATCH;correct=pos & (epe<=3)
    result=dict(positive=int(pos.sum()),negative=int(neg.sum()),ignored=int((selected & (truth==UNKNOWN)).sum()),
        false_reject=fraction(~accept,pos),no_match_recall=fraction(pred==NO_MATCH,neg),
        accepted_p3=fraction(epe<=3,pos&accept),correct_retention=fraction(accept,correct),
        r_correct3=fraction(accept & (epe<=3),pos),correct_lost=int((correct & ~accept).sum()),
        raw_epe=float(epe[pos].mean()) if pos.any() else None,
        raw_rmse=float(np.sqrt(np.mean(epe[pos]**2))) if pos.any() else None,
        raw_tail32=fraction(epe>32,pos),
        state_fractions={name:fraction(pred==state,selected) for state,name in enumerate(('MATCH','NO_MATCH','UNCERTAIN'))},
        known_uncertain=fraction(pred==UNKNOWN,known))
    for t in (1,3,5,10):result[f'raw_p{t}']=fraction(epe<=t,pos)
    if pos.any() and neg.any():
        result.update(auroc=float(auc(truth[known]==NO_MATCH,score[known])),
            auprc=float(average_precision_score((truth[known]==NO_MATCH).astype(int),score[known])))
    else: result.update(auroc=None,auprc=None)
    return result


def summarize_signals(records):
    report={}
    for name in ('full','geometry','texture'):
        keep=[r for r in records if r['row']['kind']==dict(full='full',geometry='partial',texture='texture')[name]]
        report[name]={}
        for label,value in (('positive',MATCH),('negative',NO_MATCH)):
            mask=np.concatenate([(r['state']==value) & (((r['state']!=NO_MATCH) | (r['negative_kind']==2)) if name=='texture' else True) for r in keep])
            d={}
            for key in ('p_dustbin','matchability','dustbin_logit_margin','spatial_entropy','top1_top2_margin'):
                v=np.concatenate([r[key] for r in keep])[mask]
                d[key]=dict(mean=float(v.mean()),p05=float(np.quantile(v,.05)),p50=float(np.median(v)),p95=float(np.quantile(v,.95))) if len(v) else None
            d['count']=int(mask.sum())
            d['dustbin_top1']=fraction(np.concatenate([r['dustbin_top1'] for r in keep]),mask)
            report[name][label]=d
    return report


def baseline_diagnosis():
    config=load_config();out=Path(config['output']);records=load_records('cal')
    baselines={name:[baseline_scores(r)[name] for r in records] for name in ('A0','A1','A2')}
    thresholds={name:calibrate(records,values,config) for name,values in baselines.items()}
    results={name:{s:metrics(records,values,thresholds[name],s) for s in ('all','full','geometry','texture')} for name,values in baselines.items()}
    # Phase A gate is calibration-only. No held-out result influences training.
    adequate=any(v['geometry']['no_match_recall']>=.95 and v['texture']['no_match_recall']>=.5 and
        max(v[s]['false_reject'] for s in ('full','geometry','texture'))<=.01 for v in results.values())
    report=dict(thresholds=thresholds,baselines=results,signals=summarize_signals(records),
        train_head=not adequate,gate='calibration geometry recall >=95%, texture >=50%, false rejection <=1% in each type')
    dump(out/'stage_a.json',report)
    print(json.dumps(dict(train_head=report['train_head'],baselines=results),indent=2))


def hz_metrics(records,scores,thresholds):
    report={}
    for group in ('G0','G1','G2','G3'):
        ids=[i for i,r in enumerate(records) if r['row']['kind']==group]
        if not ids:continue
        e=np.concatenate([records[i]['reference_epe_px'] for i in ids])
        mat=np.concatenate([records[i]['matchability'] for i in ids])
        dust=np.concatenate([records[i]['p_dustbin'] for i in ids])
        entropy=np.concatenate([records[i]['spatial_entropy'] for i in ids])
        gap=np.concatenate([records[i]['top1_top2_margin'] for i in ids])
        top=np.concatenate([records[i]['dustbin_top1'] for i in ids])
        state=states(np.concatenate([scores[i] for i in ids]),thresholds)
        accept=state==MATCH;valid=np.isfinite(e);bad=valid & (e>32) & (mat>=.9)
        report[group]=dict(reference_points=int(valid.sum()),raw_reference_p3=fraction(e<=3,valid),
            raw_reference_epe=float(e[valid].mean()) if valid.any() else None,
            accepted_reference_p3=fraction(e<=3,valid&accept),coverage=float(accept.mean()),
            reference_correct_rejected=int(((e<=3)&~accept).sum()),reference_correct_retention=fraction(accept,valid&(e<=3)),
            high_mat_reference_bad32=int(bad.sum()),high_mat_bad_accepted=int((bad&accept).sum()),
            high_mat_bad_rejected=fraction(~accept,bad),high_mat_bad_mean_dustbin=float(dust[bad].mean()) if bad.any() else None,
            contradictions=dict(dustbin_top1_high_mat=int((top&(mat>=.9)).sum()),
                low_dust_low_mat=int(((dust<.01)&(mat<.1)).sum()),
                high_mat_small_top_gap=int(((mat>=.9)&(gap<.1)).sum()),
                high_mat_high_entropy=int(((mat>=.9)&(entropy>5)).sum())),
            state_fractions={n:float((state==v).mean()) for v,n in enumerate(('MATCH','NO_MATCH','UNCERTAIN'))})
    return report


def evaluate():
    config=load_config();out=Path(config['output'])
    stage=json.loads((out/'stage_a.json').read_text())
    cal,ev,hz=load_records('cal'),load_records('eval'),load_records('hz')
    groups=dict(cal=cal,eval=ev,hz=hz)
    scores={s:{name:[baseline_scores(r)[name] for r in records] for name in ('A0','A1','A2')} for s,records in groups.items()}
    thresholds=stage['thresholds']
    if stage['train_head']:
        for name in ('T1','T2'):
            checkpoint=torch.load(out/f'{name}.pt',weights_only=False,map_location='cpu')
            head=NoMatchHead().eval();head.load_state_dict(checkpoint['head'])
            with torch.no_grad():
                for s,records in groups.items():
                    scores[s][name]=[head(r['features'][None]).sigmoid()[0].numpy().astype(np.float64) for r in records]
            thresholds[name]=calibrate(cal,scores['cal'][name],config)
    # Thresholds frozen here, before any held-out score summaries.
    dump(out/'thresholds.json',thresholds)
    results={}
    for name,values in scores['eval'].items():
        result={s:metrics(ev,values,thresholds[name],s) for s in ('all','full','geometry','texture')}
        result['by_overlap']={str(overlap):metrics([r for r in ev if r['row']['requested_overlap']==overlap],
            [v for r,v in zip(ev,values) if r['row']['requested_overlap']==overlap],thresholds[name]) for overlap in (1.,.2,.4,.6,.8)}
        result['hangzhou_reference_only']=hz_metrics(hz,scores['hz'][name],thresholds[name])
        result['calibration']={s:metrics(cal,scores['cal'][name],thresholds[name],s) for s in ('full','geometry','texture')}
        results[name]=result
    baseline_best=max(results[n]['texture']['no_match_recall'] for n in ('A0','A1','A2'))
    gates={}
    for name in ('T1','T2'):
        if name not in results:continue
        r=results[name]
        gates[name]=dict(positive_false_reject=max(r[s]['false_reject'] for s in ('full','geometry','texture'))<=.01,
            texture_recall=r['texture']['no_match_recall']>=.5,
            texture_gain=r['texture']['no_match_recall']>=baseline_best+.15,
            geometry_retained=r['geometry']['no_match_recall']>=results['A1']['geometry']['no_match_recall']-.005,
            original_coordinates_unchanged=all(r[s]['raw_p3']==results['A1'][s]['raw_p3'] for s in ('all','full','geometry','texture')),
            uncertain_bounded=max(r[s]['state_fractions']['UNCERTAIN'] for s in ('full','geometry','texture'))<=config['max_uncertain_fraction'])
    passed=[n for n,g in gates.items() if all(g.values())]
    decision=max(passed,key=lambda n:results[n]['texture']['no_match_recall']) if passed else 'U0'
    report=dict(config=config,thresholds=thresholds,results=results,gates=gates,decision=decision,
        independent_geographic_blocks={'cal':config['split_counts'][1],'eval':config['split_counts'][2]},
        note='Hangzhou reference metrics are NOT real visibility ground truth. Raw coordinates use one identical frozen U0 cache for every method.')
    dump(out/'results.json',report)
    with (out/'paired.csv').open('w') as f:
        writer=None
        for i,r in enumerate(ev):
            for name,values in scores['eval'].items():
                m=metrics([r],[values[i]],thresholds[name]);m.pop('state_fractions')
                row=dict(pair_id=r['row']['pair_id'],direction=r['direction'],kind=r['row']['kind'],mother_id=r['row']['mother_id'],overlap=r['row']['requested_overlap'],method=name,**m)
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(row),lineterminator="\n");writer.writeheader()
                writer.writerow(row)
    # Compact decision arrays support figures without repeating inference.
    torch.save(dict(eval=scores['eval'],hz=scores['hz']),out/'decision_scores.pt')
    print(json.dumps(dict(decision=decision,gates=gates),indent=2))


if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--stage-a',action='store_true');a=p.parse_args()
    baseline_diagnosis() if a.stage_a else evaluate()
