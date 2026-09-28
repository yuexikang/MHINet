"""Summarize paired chunk accuracy, never treating an incomplete cohort as final."""
from pathlib import Path
import argparse,json,sys,time,html
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/semidense_chunk_accuracy_val3848'
CHUNKS=(32,128,256,512,1024)


def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(data);t.replace(p)


def mean(xs):return float(np.mean(xs)) if xs else None


def summarize():
    records=[];registrations=[];done=True
    for shard in range(2):
        folder=OUT/f'shard_{shard}';p=folder/'pairs.jsonl'
        done &= (folder/'completed.json').exists()
        if (folder/'registration.json').exists():registrations.append(json.loads((folder/'registration.json').read_text()))
        if p.exists():
            for line in p.read_text().splitlines():
                try:records.append(json.loads(line))
                except json.JSONDecodeError:pass
    indices=[r['index'] for r in records]
    if len(indices)!=len(set(indices)):raise ValueError('Duplicate validation indices')
    if any(set(r['variants'])!=set(map(str,CHUNKS)) for r in records):raise ValueError('Incomplete per-pair comparison')
    complete=bool(done and len(records)==3848)
    if complete:
        if set(indices)!=set(range(3848)) or len({r['pair_id'] for r in records})!=3848:raise ValueError('Invalid full cohort')
        for key in ['checkpoint_sha256','manifest_sha256','script_sha256']:
            if len({r[key] for r in registrations})!=1:raise ValueError(f'Shard {key} differs')
    groups={}
    for chunk in CHUNKS:
        variants=[r['variants'][str(chunk)] for r in records];comparisons=[v['comparison'] for v in variants]
        epe_delta=[r['variants'][str(chunk)]['metrics']['final']['overlap_finite_EPE']-r['variants']['32']['metrics']['final']['overlap_finite_EPE'] for r in records if r['variants'][str(chunk)]['metrics']['final']['overlap_finite_EPE'] is not None and r['variants']['32']['metrics']['final']['overlap_finite_EPE'] is not None]
        precision_delta=[r['variants'][str(chunk)]['metrics']['final']['thresholds']['1.0']['precision']-r['variants']['32']['metrics']['final']['thresholds']['1.0']['precision'] for r in records]
        stages={}
        for stage in ['coarse','fine','final']:
            rows=[v['metrics'][stage] for v in variants if stage in v['metrics']];epe=[r['overlap_finite_EPE'] for r in rows if r['overlap_finite_EPE'] is not None]
            stages[stage]=dict(reported_pairs=len(rows),epe_pairs=len(epe),epe=mean(epe),matches=mean([r['matches'] for r in rows]),precision={t:mean([r['thresholds'][t]['precision'] for r in rows]) for t in ['1.0','3.0','5.0','10.0']},success_at5px=mean([r['success_20_correct_at_5px'] for r in rows]))
        targets=sum(c['compared_targets'] for c in comparisons)
        times=[v['timing']['wall_total_ms'] for v in variants]
        groups[str(chunk)]=dict(stages=stages,failure_rate=mean([v['failure_reason']!='none' for v in variants]),
            source_set_changed_pairs=sum(not c['source_sets_equal'] for c in comparisons),source_order_changed_pairs=sum(not c['source_order_equal'] for c in comparisons),
            added_sources=sum(c['added_sources'] for c in comparisons),dropped_sources=sum(c['dropped_sources'] for c in comparisons),ambiguous_duplicate_pairs=sum(c['duplicate_sources'] for c in comparisons),
            compared_targets=targets,common_target_mean_delta_px=sum((c['common_target_mean_delta_px'] or 0)*c['compared_targets'] for c in comparisons)/targets if targets else None,
            common_target_max_delta_px=max((c['common_target_max_delta_px'] for c in comparisons if c['common_target_max_delta_px'] is not None),default=None),
            paired_epe=dict(pairs=len(epe_delta),mean_delta=mean(epe_delta),max_abs_delta=max(map(abs,epe_delta),default=None)),
            paired_precision1=dict(mean_delta=mean(precision_delta),improved_pairs=sum(x>1e-12 for x in precision_delta),worse_pairs=sum(x< -1e-12 for x in precision_delta)),
            wall_mean_ms=mean(times),wall_p95_ms=float(np.percentile(times,95)) if times else None,peak_allocated_bytes=max((v['peak_allocated_bytes'] for v in variants),default=0))
    report=dict(complete=complete,pairs=len(records),expected_pairs=3848,scope='paired full tier3 validation; C checkpoint; only inference window_chunk differs',registrations=registrations,groups=groups)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n';write(OUT/'comparison.json',text);write(ROOT/'artifacts/semidense_chunk_accuracy_results.json',text)
    def fmt(v):return '—' if v is None else f'{v:.6f}'
    rows=[]
    for chunk,g in groups.items():
        vals=[chunk,fmt(g['stages']['final']['precision']['1.0']),fmt(g['stages']['final']['epe']),fmt(g['failure_rate']),str(g['source_set_changed_pairs']),fmt(g['common_target_max_delta_px']),fmt(g['wall_mean_ms'])]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>推理分块精度对照</title><style>body{{font-family:sans-serif;margin:30px}}td,th{{padding:9px 16px;border-bottom:1px solid #ddd}}</style><h1>推理分块精度对照</h1><p>{'完整验证已完成' if complete else '运行中，部分样本不作最终结论'}：{len(records)}/3848 对</p><p>同一第三档C权重，同一第三档验证集；只改推理分块。训练分块、12000点上限、4轮QRRU和H0残差先验保持不变。</p><table><tr><th>分块</th><th>最终precision@1px</th><th>重叠区EPE(px)</th><th>失败率</th><th>点集合改变对数</th><th>公共点最大位移(px)</th><th>平均总耗时(ms)</th></tr>{''.join(rows)}</table><p>按源点对齐，排序变化与点集合变化分别统计；公共点位移不覆盖新增/丢失点，须同时看点集合变化。EPE仅统计有效重叠匹配；精度/失败率为0～1比例。两张同型GPU分片，不作跨硬件速度结论。</p><p><a href="comparison.json">完整指标、点数及逐对差异汇总</a> · <a href="../semidense_lr_comparison.html">训练看板</a></p></html>'''
    write(OUT/'index.html',body)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        r=summarize();print(json.dumps(dict(pairs=r['pairs'],complete=r['complete'])),flush=True)
        if r['complete'] or not args.watch:break
        time.sleep(60)


if __name__=='__main__':main()
