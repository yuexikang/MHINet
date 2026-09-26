"""Refresh a read-only C/D held-out comparison until both full evaluations finish."""
import argparse
import html
import json
import time
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
    return dict(pairs=len(records),complete=complete,registration=registration,stages=stages,
        failure_rate=mean([int(r['failure_reason']!='none') for r in records]),
        mean_ms=mean([r['milliseconds'] for r in records])), [r['pair_id'] for r in records]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        c,ids_c=collect('c');d,ids_d=collect('d');complete=c['complete'] and d['complete']
        if complete:
            if len(set(ids_c))!=EXPECTED or ids_c!=ids_d:raise ValueError('C/D test cohorts differ')
            if c['registration']['manifest_sha256']!=d['registration']['manifest_sha256']:raise ValueError('Test manifests differ')
        report=dict(scope='independent synthetic tier3 test, macro mean per pair',complete=complete,
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
        body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>C/D 第三档独立测试</title>
<style>body{{font-family:sans-serif;margin:30px;color:#243047}}td,th{{padding:8px 18px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style>
<h1>C/D 第三档独立测试</h1><p>{'完整测试已完成' if complete else '测试进行中；当前均值为部分样本，不作最终排名'}：C {c['pairs']}/{EXPECTED}，D {d['pairs']}/{EXPECTED}</p>
<p>同一独立合成测试集，有精确同母图真值。不是原始跨时相配准精度。EPE 为原图目标像素；精度/成功率为0～1比例。</p>
<p>EPE 只统计存在有限重叠对应点的样本，须结合成功率及失败率。粗/细阶段可能因提前失败而缺记录，各阶段实际分母见原始汇总。耗时来自不同 GPU 的并行任务，仅供参考。</p>
<table><tr><th>指标</th><th>C</th><th>D</th></tr>{''.join(rows)}</table>
<p><a href="comparison.json">完整汇总及各阶段样本数</a> · <a href="../semidense_lr_comparison.html">返回训练看板</a></p></html>'''
        write(OUT/'index.html',body)
        print(json.dumps(dict(C=c['pairs'],D=d['pairs'],complete=complete)),flush=True)
        if complete or not args.watch:break
        time.sleep(30)


if __name__=='__main__':main()
