"""Compact A/B decision and report; thresholds use RGB calibration only."""
import csv,json
from pathlib import Path
import numpy as np
from mhinet.v2.local_occlusion.evaluation import auc
from .run import ROOT,VARIANTS,write


def cat(records):return {k:np.concatenate([r[k] for r in records]) for k in records[0]}


def scores(a):return dict(matchability=a['matchability'],cycle=a['reliability'],product=a['matchability']*a['reliability'])


def operating(a,t,score):
    p,n=a['positive'],a['negative'];reject=score<t;accepted=a['visible']&~reject
    return dict(positive_count=int(p.sum()),negative_count=int(n.sum()),negative_recall=float(reject[n].mean()) if n.any() else None,
        positive_false_reject=float(reject[p].mean()) if p.any() else None,
        reference_coverage=float(accepted.sum()/max(1,a['visible'].sum())),accepted_ncm3=int((accepted&(a['epe']<=3)).sum()),
        accepted_p3=float((a['epe'][accepted]<=3).mean()) if accepted.any() else None)


def analysis(a):
    p=a['visible'];finite=p&a['cycle_valid'];high=p&(a['matchability']>=.9)
    hard=high&((a['epe']<=3)|(a['epe']>32));bad=a['epe'][hard]>32
    summary=dict(reference_points=int(p.sum()),reference_epe=float(a['epe'][p].mean()),reference_p3=float((a['epe'][p]<=3).mean()),
        reference_tail32=float((a['epe'][p]>32).mean()),reference_ncm3=int((p&(a['epe']<=3)).sum()),
        cycle_valid_reference_fraction=float(a['cycle_valid'][p].mean()),boundary_reference_fraction=float(a['boundary'][p].mean()),
        high_matchability_points=int(high.sum()),high_matchability_bad32=int((high&(a['epe']>32)).sum()),
        high_matchability_bad_auroc={name:auc(bad,1-value[hard]) for name,value in scores(a).items()})
    bins=[]
    for lo,hi in [(0,1),(1,3),(3,8),(8,16),(16,32),(32,float('inf'))]:
        sel=p&a['cycle_valid']&(a['cycle']>=lo)&(a['cycle']<hi)
        bins.append(dict(bin=f'{lo}-{hi}',points=int(sel.sum()),epe=float(a['epe'][sel].mean()) if sel.any() else None,
            p3=float((a['epe'][sel]<=3).mean()) if sel.any() else None,tail32=float((a['epe'][sel]>32).mean()) if sel.any() else None))
    sel=p&~a['cycle_valid'];bins.append(dict(bin='invalid',points=int(sel.sum()),epe=float(a['epe'][sel].mean()) if sel.any() else None,p3=float((a['epe'][sel]<=3).mean()) if sel.any() else None,tail32=float((a['epe'][sel]>32).mean()) if sel.any() else None))
    summary['cycle_bins']=bins
    summary['matchability_bins']=[]
    for lo,hi in [(0,.1),(.1,.5),(.5,.9),(.9,.99),(.99,1.000001)]:
        sel=p&(a['matchability']>=lo)&(a['matchability']<hi)
        summary['matchability_bins'].append(dict(bin=f'{lo}-{min(hi,1.)}',points=int(sel.sum()),
            epe=float(a['epe'][sel].mean()) if sel.any() else None,p3=float((a['epe'][sel]<=3).mean()) if sel.any() else None,
            tail32=float((a['epe'][sel]>32).mean()) if sel.any() else None))
    return summary


def bootstrap_delta(left,right,blocks):
    ids=sorted(left);groups=sorted({blocks[k] for k in ids});rng=np.random.default_rng(20261023)
    differences=np.array([left[k]['p3']-right[k]['p3'] for k in ids]);group=np.array([groups.index(blocks[k]) for k in ids])
    sums=np.bincount(group,weights=differences);counts=np.bincount(group)
    pick=rng.integers(0,len(groups),(1000,len(groups)));values=sums[pick].sum(1)/counts[pick].sum(1)
    return dict(delta_p3=float(differences.mean()),interval95=np.quantile(values,[.025,.975]).tolist(),blocks=len(groups))


def main():
    s=json.loads((ROOT/'selection.json').read_text());cal=set(s['rgb_calibration_ids']);data={};perpair={}
    for domain in ('hangzhou','rgb'):
        records=[];perpair[domain]={}
        for row in s[domain]:
            path=ROOT/domain/(row['pair_id']+'.npz')
            with np.load(path) as archive:a={k:archive[k] for k in archive.files}
            perpair[domain][row['pair_id']]=a;records.append(a)
        data[domain]=cat(records)
    calibration=cat([a for pid,a in perpair['rgb'].items() if pid in cal]);evaluation=cat([a for pid,a in perpair['rgb'].items() if pid not in cal])
    thresholds={name:float(np.sort(value[calibration['positive']])[int(.01*calibration['positive'].sum())]) for name,value in scores(calibration).items()}
    ops={name:operating(evaluation,thresholds[name],value) for name,value in scores(evaluation).items()}
    aresult=dict(thresholds=thresholds,rgb_operating=ops,hangzhou=analysis(data['hangzhou']),rgb_evaluation=analysis(evaluation),
        groups={g:analysis(cat([perpair['hangzhou'][r['pair_id']] for r in s['hangzhou'] if r['G']==g])) for g in ('G0','G1','G2','G3')})
    a_pass=any(ops[n]['negative_recall']>=.5 and ops[n]['positive_false_reject']<=.01 and ops[n]['negative_recall']>=ops['matchability']['negative_recall']+.1 for n in ('cycle','product'))
    bresult={};b_pass=False;paired=[]
    for domain in ('hangzhou','rgb'):
        rows=[json.loads(line) for line in (ROOT/domain/'paired.jsonl').open()];paired.extend([dict(domain=domain,**r) for r in rows])
        expected=64 if domain=='hangzhou' else 128
        assert len(rows)==expected*len(VARIANTS)
        models={v:{r['pair_id']:r for r in rows if r['variant']==v} for v in VARIANTS}
        means={v:{k:float(np.mean([r[k] for r in values.values() if r[k] is not None])) for k in next(iter(values.values())) if k not in ('pair_id','group','variant')} for v,values in models.items()}
        groups={g:{v:{k:float(np.mean([r[k] for r in values.values() if r['group']==g and r[k] is not None])) for k in ('p3','ncm3','matches','d8_correct_rate','d2_coverage')} for v,values in models.items()} for g in sorted({r['group'] for r in rows})}
        blocks={r['pair_id']:r.get('anchor_block_id',r['pair_id']) for r in s[domain]}
        comparisons={v+'-'+b:bootstrap_delta(models[v],models[b],blocks) for v in ('B2','B3') for b in ('B0','B1','B4_uniform')}
        passed={v:all(means[v]['p3']>=means[b]['p3']+.005 and means[v]['ncm3']>=means[b]['ncm3'] and means[v]['d2_coverage']>=means[b]['d2_coverage'] for b in ('B0','B1','B4_uniform')) for v in ('B2','B3')}
        b_pass|=any(passed.values());bresult[domain]=dict(means=means,groups=groups,paired=comparisons,passed=passed)
    aresult['calibration_positive_invalid_cycle']=float((~calibration['cycle_valid'][calibration['positive']]).mean())
    aresult['evaluation_positive_invalid_cycle']=float((~evaluation['cycle_valid'][evaluation['positive']]).mean())
    write(ROOT/'A.json',aresult);write(ROOT/'B.json',bresult)
    with (ROOT/'reliability_bins.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['domain','signal','bin','points','epe','p3','tail32']);w.writeheader()
        for domain in ('hangzhou','rgb_evaluation'):
            for signal in ('cycle','matchability'):
                w.writerows(dict(domain=domain,signal=signal,**r) for r in aresult[domain][signal+'_bins'])
    decision=dict(A_gate=a_pass,B_gate=b_pass,start_C=a_pass or b_pass,sigma_px=16.,threshold_source='rgb_calibration128 only',
        reason='Small frozen-feature reliability head is justified under preregistered gate.' if a_pass or b_pass else 'No clear gain under fixed gates; retain U0 and original downstream, no C training or further experiments.')
    write(ROOT/'decision.json',decision)
    with (ROOT/'paired.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    lines=['# 双向Flow0与D8可靠性实验','',f'配置：U0 epoch30；冻结下游为清理后杭州v2 best epoch7；Cycle高斯sigma=16px。A/B不训练，未使用封存测试。',
        '','## A：对应有效性与高置信错误','',
        '|域|参考EPE(px)|参考P@3|>32px|可计算Cycle的参考有效点|高Mat错误数/高Mat点数|Cycle/Mat错误AUROC|','|---|---:|---:|---:|---:|---|---|']
    for domain in ('hangzhou','rgb_evaluation'):
        r=aresult[domain];au=r['high_matchability_bad_auroc'];lines.append(f'|{domain}|{r["reference_epe"]:.3f}|{100*r["reference_p3"]:.2f}%|{100*r["reference_tail32"]:.2f}%|{100*r["cycle_valid_reference_fraction"]:.2f}%|{r["high_matchability_bad32"]}/{r["high_matchability_points"]}|{au["cycle"]} / {au["matchability"]}|')
    lines+=['','原始定位指标包含全部参考有效查询，不按Cycle筛点。高Mat定义≥0.9；错误AUROC仅区分其中EPE>32与≤3的点，中间点仍计原始定位指标。边界插值无效处记为未知Cycle、可靠性0，另报计数，不能解释为已经证明无对应。','',
        '|分数|校准阈值|评价几何无对应召回|评价有效点误拒|全参考点接受覆盖率|接受后P@3|接受正确点数|','|---|---:|---:|---:|---:|---:|---:|']
    for n,r in ops.items():lines.append(f'|{n}|{thresholds[n]:.7f}|{100*r["negative_recall"]:.2f}%|{100*r["positive_false_reject"]:.2f}%|{100*r["reference_coverage"]:.2f}%|{100*r["accepted_p3"]:.2f}%|{r["accepted_ncm3"]}|')
    lines+=['','几何无对应标签仅来自精确RGB H投影图外>1px，正点同时满足裁剪掩膜和目标内侧1px；其他位置忽略。不是杭州自然遮挡/时相变化标注。','', '## B：完整候选下的几何奖励调制','']
    for domain,r in bresult.items():
        lines += [f'### {domain}','', '|组|D8父匹配正确率|D2覆盖率|最终P@1/3/5|NCM@3|输出数|>32px|总耗时ms|','|---|---:|---:|---|---:|---:|---:|---:|']
        for v,m in r['means'].items():lines.append(f'|{v}|{100*m["d8_correct_rate"]:.2f}%|{100*m["d2_coverage"]:.2f}%|{100*m["p1"]:.2f}/{100*m["p3"]:.2f}/{100*m["p5"]:.2f}%|{m["ncm3"]:.1f}|{m["matches"]:.1f}|{100*m["epe_gt32"]:.2f}%|{m["total_ms"]:.1f}|')
    lines+=['','B0外观；B1原Matchability；B2 Mat×Cycle；B3仅Cycle；B4_uniform按D16有效点平均奖励权重均匀缩小Mat。全部4096²候选，QRRU4轮、输出上限2048，无按可靠性硬删候选。各项为逐对均值；NCM和输出同时报告，P的分母是所有最终输出。D2覆盖以全图参考有效D2查询作分母，未选中父窗口也计入。',
        '','耗时含完整共享金字塔前向，B2/B3/B4额外计反向DINO/MVT/Flow和Cycle；不是只计D8。相同域所有组在同卡顺序评价，各输入先共享固定特征。各组对照配对95%区间见B.json；杭州按空间块bootstrap，RGB按源图对bootstrap。杭州仅近似H参考一致性。',
        '','## C启动裁决','',json.dumps(decision,ensure_ascii=False),
        '','本轮未触发C。互相一致的错误仍可能存在，不能以Cycle低误差证明对应真实正确。']
    if not decision['start_C']:
        lines+=['','## 最终回答与保留项','',
            '1. **能识别一部分高Matchability的参考错误，但不能当作真值。** 杭州高Mat点中，Cycle对>32px与≤3px参考误差的AUROC=0.951，优于Mat的0.774；这只是近似H参考错误的排序能力。RGB评价没有高Mat且>32px的样本，该项AUROC不可定义，不能填成完美分类。Matchability/Cycle分箱与EPE、P@3、长尾的关系见reliability_bins.csv；所有G组原始计数见A.json。',
            '2. **没有证明改善D8/D2及最终匹配。** B2−B1杭州P@3为−0.070pp（配对空间块95%区间−0.199～+0.052pp），RGB为+0.025pp（−0.002～+0.057pp）。杭州D2覆盖还从41.42%降至41.22%。与等平均奖励量B4相比也没有达到有实际意义的预定收益，额外反向使约66/56ms增至87/75ms。维持原B1。',
            '3. **不启动C。** 几何无对应与高Mat错误不是同一任务。原Mat在该精确合成域已达到99.53%召回、0.95%有效点误拒；双向信号没有提供额外无对应检出或稳定下游收益。未新增有效性头、未训练任何模块、未追加超参数搜索。',
            '',
            'Cycle阈值为何为0：校准的几何有效正点中3.57%无法在反向D16中心支持范围内插值（评价3.64%）。这些查询的R=0，若阈值>0便同时拒绝整批正点，违反1%误拒约束；严格小于0则不拒绝任何点。0%召回体现本轮保守边界处理与低误拒工作点的限制，不等于Cycle对所有错误都无判别力。不能把这些点按GT删掉再宣称工作点改善。',
            '',
            '最终保留U0 epoch30、原MatchEmbeddingDecoder及原D8/D2/QRRU，正式默认不启用Cycle调制。Cycle可作为只读诊断工具。RGB的几何裁剪无对应结论不等于自然遮挡/时相变化识别；杭州不存在新增可见性真值，未将其用于有效性训练。',
            '',
            '计时限制：B0也复用共同前向中已计算的U0头，纯外观专用优化可能更快；总时间未含图像读取。B4按D16有效预测的平均权重对齐，空间高斯奖励在完整D8矩阵中的总和不必逐元素一致。源图独立划分不等于地理邻近独立；这是开发集结论，未把既往封存测试重新用于调参。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(decision,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
