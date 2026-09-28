"""Refresh a read-only C/D held-out comparison until both full evaluations finish."""
import argparse
import html
import json
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/semidense_tier3_cd_test_comparison'
EXPECTED = 2000


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def mean(values):
    return sum(values)/len(values) if values else None


def timing_distribution(values):
    if not values:return None
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),
        p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),min=float(a.min()),max=float(a.max()))


def collect(name):
    root = ROOT/f'outputs/semidense_tier3_lr_{name}_test2000'
    records = []
    if (root/'pairs.jsonl').exists():
        for line in (root/'pairs.jsonl').read_text().splitlines():
            try: records.append(json.loads(line))
            except json.JSONDecodeError: pass
    stages = {}
    for stage in ('coarse', 'fine', 'final'):
        rows = [r[stage] for r in records if stage in r]
        epe = [r['overlap_finite_EPE'] for r in rows if r['overlap_finite_EPE'] is not None]
        stages[stage] = dict(reported_pairs=len(rows),epe_pairs=len(epe),overlap_EPE=mean(epe),
            matches=mean([r['matches'] for r in rows]),
            precision={t:mean([r['thresholds'][t]['precision'] for r in rows]) for t in ('1.0','3.0','5.0','10.0')},
            success20_at5px=mean([int(r['success_20_correct_at_5px']) for r in rows]))
    complete=(root/'report.json').exists() and len(records)==EXPECTED
    registration=json.loads((root/'registration.json').read_text()) if (root/'registration.json').exists() else None
    timing_rows=[r for r in records if 'timing' in r]
    timing=dict(wall_total_ms=timing_distribution([r['milliseconds'] for r in records]),
        module_ms={k:timing_distribution([r['timing']['module_ms'].get(k,0) for r in timing_rows])
            for k in sorted({k for r in timing_rows for k in r['timing']['module_ms']})},profiled_pairs=len(timing_rows))
    return dict(timing=timing,pairs=len(records),complete=complete,registration=registration,stages=stages,
        failure_rate=mean([int(r['failure_reason']!='none') for r in records]),
        mean_ms=mean([r['milliseconds'] for r in records])), [r['pair_id'] for r in records]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        c,ids_c=collect('c');d,ids_d=collect('d');complete=c['complete'] and d['complete']
        if complete:
            if len(set(ids_c))!=EXPECTED or ids_c!=ids_d:raise ValueError('C/D test cohorts differ')
            if c['registration']['manifest_sha256']!=d['registration']['manifest_sha256']:raise ValueError('Test manifests differ')
        timing_warning=None
        if complete and c['mean_ms'] and d['mean_ms'] and max(c['mean_ms'],d['mean_ms'])/min(c['mean_ms'],d['mean_ms'])>1.5:
            timing_warning='本轮 C/D 耗时差异较大；未记录测试期间 GPU 负载，不能将差异归因于模型权重或据此比较速度。'
        report=dict(timing_warning=timing_warning,scope='independent synthetic tier3 test, macro mean per pair',complete=complete,
            expected_pairs=EXPECTED,C=c,D=d,
            note='EPE only averages pairs with finite overlap matches; precision uses all predicted matches. Stage coverage and success/failure must be read together. No fitted-H accuracy claim.')
        write(OUT/'comparison.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        fmt=lambda v:'—' if v is None else f'{v:.5f}'
        rows=[]
        for stage,label in [('coarse','粗匹配'),('fine','细匹配'),('final','QRRU 最终')]:
            for key,title in [('overlap_EPE','重叠区 EPE'),('matches','平均点数'),('success20_at5px','至少20个正确点的成功率@5px')]:
                rows.append(f'<tr><td>{label} {title}</td><td>{fmt(c["stages"][stage][key])}</td><td>{fmt(d["stages"][stage][key])}</td></tr>')
            for t in ['1.0','3.0','5.0']:
                rows.append(f'<tr><td>{label} precision@{t}px</td><td>{fmt(c["stages"][stage]["precision"][t])}</td><td>{fmt(d["stages"][stage]["precision"][t])}</td></tr>')
        for key,label in [('failure_rate','推理失败率'),('mean_ms','平均推理耗时(ms)')]:rows.append(f'<tr><td>{label}</td><td>{fmt(c[key])}</td><td>{fmt(d[key])}</td></tr>')
        labels=(c['registration'] or d['registration'] or {}).get('module_labels',{})
        timing_html='<h2>模块耗时分布 (ms/影像对)</h2><p>非重叠 CUDA event 区间，含 CPU 提交间隙；QRRU 包含输出整理。单次最终同步，仍有计时开销。以下分布均按每对影像统计。</p><table><tr><th>模块</th><th>C 均值 / P50 / P95 / P99</th><th>D 均值 / P50 / P95 / P99</th></tr>'
        keys=['wall_total_ms']+list(labels)
        for key in keys:
            cells=[]
            for group in (c,d):
                v=group['timing']['wall_total_ms'] if key=='wall_total_ms' else group['timing']['module_ms'].get(key)
                cells.append(' / '.join(fmt(v[k]) for k in ('mean','p50','p95','p99')) if v else '—')
            label='总推理墙钟时间' if key=='wall_total_ms' else labels[key]
            timing_html+=f'<tr><td>{html.escape(label)}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>'
        timing_html+='</table><p>各模块均值可以相加；各模块分位数不可直接相加。P90、最小和最大值见完整汇总。</p>'
        body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>C/D 第三档独立测试</title>
<style>body{{font-family:sans-serif;margin:30px;color:#243047}}td,th{{padding:8px 18px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style>
<h1>C/D 第三档独立测试</h1><p>{'完整测试已完成' if complete else '测试进行中；当前均值为部分样本，不作最终排名'}：C {c['pairs']}/{EXPECTED}，D {d['pairs']}/{EXPECTED}</p>
<p>同一独立合成测试集，有精确同母图真值。不是原始跨时相配准精度。EPE 为原图目标像素；精度/成功率为0～1比例。</p>
<p>EPE 只统计存在有限重叠对应点的样本，须结合成功率及失败率。粗/细阶段可能因提前失败而缺记录，各阶段实际分母见原始汇总。耗时来自不同 GPU 的并行任务，仅供参考。</p>
<p>{html.escape(timing_warning or '')}</p>
<table><tr><th>指标</th><th>C</th><th>D</th></tr>{''.join(rows)}</table>{timing_html}
<p><a href="comparison.json">完整汇总及各阶段样本数</a> · <a href="../semidense_lr_comparison.html">返回训练看板</a></p></html>'''
        write(OUT/'index.html',body)
        print(json.dumps(dict(C=c['pairs'],D=d['pairs'],complete=complete)),flush=True)
        if complete or not args.watch:break
        time.sleep(30)


if __name__=='__main__':main()
