"""Live C/D/E accuracy and module latency comparison on one held-out cohort."""
from pathlib import Path
import argparse,html,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_semidense_cd_test import collect,write
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/semidense_tier3_cde_test_comparison'


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        groups={};ids={}
        for k in 'cde':groups[k.upper()],ids[k]=collect(k)
        complete=all(g['complete'] for g in groups.values())
        if complete:
            if not ids['c']==ids['d']==ids['e'] or len(set(ids['c']))!=2000:raise ValueError('Test pair IDs differ')
            if len({g['registration']['manifest_sha256'] for g in groups.values()})!=1:raise ValueError('Test manifests differ')
            if any(g['timing']['profiled_pairs']!=2000 for g in groups.values()):raise ValueError('Incomplete module timing')
        data=dict(complete=complete,expected_pairs=2000,groups=groups,note='Independent synthetic tier3; macro per-pair metrics. Different execution periods/GPUs: latency is not a controlled speed ranking. D and E GPU telemetry stored separately.')
        write(OUT/'comparison.json',json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        write(ROOT/'artifacts/semidense_cde_timed_test_results.json',json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        rows=[]
        def fmt(x):return '—' if x is None else f'{x:.5f}'
        def row(label,vals):rows.append('<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+fmt(v)+'</td>' for v in vals)+'</tr>')
        for stage,label in [('coarse','粗匹配'),('fine','细匹配'),('final','QRRU最终')]:
            for key,title in [('overlap_EPE','EPE'),('matches','点数'),('success20_at5px','成功率@5px（至少20点）')]:row(label+' '+title,[g['stages'][stage][key] for g in groups.values()])
            for t in ['1.0','3.0','5.0']:row(label+f' precision@{t}px',[g['stages'][stage]['precision'][t] for g in groups.values()])
        row('推理失败率',[g['failure_rate'] for g in groups.values()])
        for stat in ['mean','p50','p95','p99']:
            row('总推理耗时(ms) '+stat,[(g['timing']['wall_total_ms'] or {}).get(stat) for g in groups.values()])
        labels=(groups['C']['registration'] or {}).get('module_labels',{})
        for key,label in labels.items():row(label+' 均值(ms)',[(g['timing']['module_ms'].get(key) or {}).get('mean') for g in groups.values()])
        progress='；'.join(f'{k} {g["pairs"]}/2000' for k,g in groups.items())
        body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>C/D/E 测试对照</title><style>body{{font-family:sans-serif;margin:30px}}td,th{{padding:8px 16px;border-bottom:1px solid #ddd}}</style><h1>C/D/E 第三档测试对照</h1><p>{'完整测试已完成' if complete else '测试进行中，部分样本不作最终排名'}：{progress}</p><p>同一2000对独立合成测试，非真实跨时相精度；EPE为目标原图像素，精度为0～1。EPE只统计有效重叠匹配，须结合失败率与各阶段分母（见JSON）。</p><p>不同执行时段/物理GPU，耗时不能直接作为受控速度排名。D已覆盖旧异常记录重测，D/E每30秒记录GPU状态。各模块为非重叠CUDA event区间（含CPU提交间隙）。</p><table><tr><th>指标</th><th>C</th><th>D 重测</th><th>E</th></tr>{''.join(rows)}</table><p><a href="comparison.json">完整结果和模块分位数</a> · <a href="../semidense_lr_comparison.html">训练看板</a></p></html>'''
        write(OUT/'index.html',body);print(progress,flush=True)
        if complete or not args.watch:break
        time.sleep(30)


if __name__=='__main__':main()
