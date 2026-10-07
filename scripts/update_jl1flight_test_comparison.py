"""Expose NCM/NM from complete four-run reports without rerunning inference."""
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1] / 'outputs/jl1flight_pre_post_tests_512_20261004'
PROTOCOL_URL = 'https://app.notion.com/p/3eabcd67ec6b81e88327cadb0fd4826d'
JOBS = [(p, d) for p in ('before', 'after') for d in ('GoogleEarth', 'JL1Flight')]
COUNTS = {'GoogleEarth': 500, 'JL1Flight': 1755}


def augment(metrics):
    return dict(metrics, NCM=metrics['mean_ncm'], NM=metrics['mean_matches'])


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    reports, groups, stage_reports, stages = {}, {}, {}, []
    rows = []
    for phase, dataset in JOBS:
        name = f'{phase}_{dataset}'
        folder = ROOT/name
        report = json.loads((folder/'report.json').read_text())
        pairs = [json.loads(x) for x in (folder/'pairs.jsonl').read_text().splitlines()]
        if not report['complete'] or len(pairs) != COUNTS[dataset]:
            raise ValueError(f'Incomplete results: {name}')
        m = augment(report['overall'])
        if m['pairs'] != len(pairs): raise ValueError(f'Pair count mismatch: {name}')
        # The denominator includes every pair, including zero-match failures.
        nm = statistics.mean(r['metrics']['matches'] for r in pairs)
        if abs(nm-m['NM']) > 1e-8: raise ValueError('NM aggregation mismatch')
        for t in ('1','3','5'):
            ncm = statistics.mean(r['metrics']['ncm'][t] for r in pairs)
            if abs(ncm-m['NCM'][t]) > 1e-8: raise ValueError('NCM aggregation mismatch')
            if any(r['metrics']['ncm'][t] > r['metrics']['matches'] for r in pairs):
                raise ValueError('NCM cannot exceed NM')
        reports[name] = m
        groups[name] = {g:augment(v) for g,v in report['by_group'].items()}
        row = dict(dataset=dataset, phase=phase, pairs=m['pairs'],
            **{f'P@{t}px':m['precision'][t] for t in ('1','3','5')},
            **{f'NCM@{t}px':m['NCM'][t] for t in ('1','3','5')}, NM=m['NM'],
            **{f'SR@{t}px':m['sr'][t] for t in ('1','3','5')},
            RMSE=m['rmse_correct5_or_failure10'], inference_ms=m['inference_ms_mean'])
        rows.append(row)
        for stage in ('coarse', 'fine', 'final'):
            records = [r['metrics'] if stage=='final' else r['stages'][stage] for r in pairs]
            successful = sum(r['sr']['3'] for r in records)
            rmse = [r['registration_rmse'] for r in records if r['registration_rmse'] is not None]
            stage_row = dict(dataset=dataset,phase=phase,stage=stage,
                precision={t:statistics.mean(r['precision'][t] for r in records) for t in ('1','3','5')},
                sr3=successful/len(records),sr3_pairs=successful,
                median_registration_grid_rmse=statistics.median(rmse))
            stages.append(stage_row)
        stage_reports[name] = {r['stage']:r for r in stages if r['dataset']==dataset and r['phase']==phase}
    deltas = {}
    for dataset in COUNTS:
        a,b = reports[f'before_{dataset}'],reports[f'after_{dataset}']
        deltas[dataset] = dict(
            precision_change_pp={t:100*(b['precision'][t]-a['precision'][t]) for t in ('1','3','5')},
            sr3_change_pp=100*(b['sr']['3']-a['sr']['3']),
            NCM_change={t:b['NCM'][t]-a['NCM'][t] for t in ('1','3','5')},NM_change=b['NM']-a['NM'])
    result = dict(reports=reports,by_group=groups,stages=stage_reports,after_minus_before=deltas,
        metric_definitions=dict(protocol_url=PROTOCOL_URL,coordinate_unit='stored_512_target_pixels',
            NCM='Number of GT-correct matches at <=1/3/5px in GT-visible geometric support and image bounds; mean over ALL pairs, empty output=0.',
            NM='All final model-returned match pairs; mean over ALL image pairs, empty output=0.',
            support='Headline metrics retain the existing public score_matches geometric-support convention. Raster mask statistics remain separately recorded in each pair mask_metrics.',
            precision='Pair-macro mean, not mean NCM divided by mean NM.',
            RMSE='SR@3 success: GT-correct <=5px match RMSE. Failure=10. Undefined successful/no-correct pairs separately counted.',
            timing='Recorded timings retained; GPU runs and system load differ, so this is not a controlled speed comparison.'),
        summary_updated=datetime.now(timezone.utc).isoformat(),
        inference_rerun=False)
    write_json(ROOT/'comparison.json',result)
    with (ROOT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labels=['数据集','权重','影像对','P@1px','P@3px','P@5px','NCM@1px','NCM@3px','NCM@5px','NM','SR@3px','RMSE(px)','推理ms/对','详情']
    tables=[];markdown=['| '+' | '.join(labels[:-2])+' |','| '+' | '.join(['---']*len(labels[:-2]))+' |']
    for row in rows:
        phase,dataset=row['phase'],row['dataset']
        cells=[dataset,'微调前' if phase=='before' else '微调后',str(row['pairs']),
            *[f'{100*row[f"P@{t}px"]:.2f}%' for t in ('1','3','5')],
            *[f'{row[f"NCM@{t}px"]:.2f}' for t in ('1','3','5')],f'{row["NM"]:.2f}',
            f'{100*row["SR@3px"]:.2f}%',f'{row["RMSE"]:.3f}']
        markdown.append('| '+' | '.join(cells)+' |')
        name=f'{phase}_{dataset}'
        tables.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+
            f'<td>{row["inference_ms"]:.2f}</td><td><a href="{name}/report.json">报告</a></td></tr>')
    stage_html=[]
    for r in stages:
        cells=[r['dataset'],'微调前' if r['phase']=='before' else '微调后',r['stage'],
            *[f'{100*r["precision"][t]:.2f}%' for t in ('1','3','5')],
            f'{100*r["sr3"]:.2f}%',f'{r["median_registration_grid_rmse"]:.3f}']
        stage_html.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
    content='<!doctype html><meta charset="utf-8"><title>512 微调前后：NCM/NM 对比</title>'
    content+='<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;white-space:nowrap}.scroll{overflow-x:auto}p{max-width:1000px;line-height:1.6}</style>'
    content+='<h1>512 微调前后：GoogleEarth / JL1Flight</h1>'
    content+=f'<p>已核对 <a href="{PROTOCOL_URL}">Notion 指标定义</a>。NCM 是正确匹配点对数量；NM 是总输出点对数量。均按全部影像对平均，无匹配计0。P 为逐对宏平均，不能直接用平均 NCM ÷ 平均 NM 替代。</p>'
    content+='<p>阈值采用存储影像的512目标像素。RMSE：SR@3成功对统计≤5px正确匹配，失败记10。GoogleEarth 标签是名义原始配准上的合成几何。耗时保留原测试记录，运行卡及系统负载不同。</p>'
    content+='<div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in labels)+'</tr>'+''.join(tables)+'</table></div>'
    content+='<h2>各阶段指标</h2><div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in ['数据集','权重','阶段','P@1px','P@3px','P@5px','SR@3px','网格RMSE中位数'])+'</tr>'+''.join(stage_html)+'</table></div>'
    content+='<p><a href="comparison.json">完整结果及 JL t0–t4 分组</a> · <a href="comparison.csv">CSV</a></p>'
    (ROOT/'comparison.html').write_text(content)
    (ROOT/'comparison.md').write_text('\n'.join(markdown)+'\n\nNCM/NM 均为全部影像对平均；GT阈值使用512坐标。推理结果未重跑。\n')
    print(json.dumps(dict(reports={k:dict(NCM=v['NCM'],NM=v['NM']) for k,v in reports.items()},output=str(ROOT/'comparison.html')),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
