"""Expose NCM/NM from complete four-run reports without rerunning inference."""
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1] / 'outputs/jl1flight_pre_post_tests_512_20261004'
PROTOCOL_URL = 'https://app.notion.com/p/3eabcd67ec6b81e88327cadb0fd4826d'
JOBS = [(p, d) for p in ('before', 'after') for d in ('GoogleEarth', 'JL1Flight')]
DATASETS = tuple(dict.fromkeys(dataset for _, dataset in JOBS))


def augment(metrics):
    return dict(metrics, NCM=metrics['mean_ncm'], NM=metrics['mean_matches'])


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def validate_dataset_records(name, phase, dataset, registration, report, pairs):
    """Reject stale or mixed predictions even when their row counts agree."""
    if registration['phase'] != phase or registration['dataset'] != dataset:
        raise ValueError(f'Registration identity mismatch: {name}')
    if report['registration']['phase'] != phase or report['registration']['dataset'] != dataset:
        raise ValueError(f'Report registration identity mismatch: {name}')
    manifest = registration['manifest']
    if report['registration']['manifest'] != manifest:
        raise ValueError(f'Report/registration manifest mismatch: {name}')
    manifest_path = Path(manifest['path'])
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest['sha256']:
        raise ValueError(f'Current dataset manifest SHA256 mismatch: {name}')
    expected = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    expected_ids = [row['pair_id'] for row in expected]
    actual_ids = [row['pair_id'] for row in pairs]
    if len(set(expected_ids)) != len(expected_ids) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError(f'Duplicate pair_id: {name}')
    if not report['complete'] or not pairs or len(pairs) != manifest['pairs']:
        raise ValueError(f'Incomplete results: {name}')
    if len(expected) != manifest['pairs'] or report['overall']['pairs'] != len(pairs):
        raise ValueError(f'Pair count mismatch: {name}')
    if actual_ids != expected_ids:
        raise ValueError(f'Prediction pair_id/order differs from current dataset manifest: {name}')
    return manifest


def main():
    reports, groups, stage_reports, stages = {}, {}, {}, []
    dataset_manifests = {}
    curation_path = ROOT/'googleearth_curation.json'
    curation = json.loads(curation_path.read_text()) if curation_path.exists() else None
    rows = []
    for phase, dataset in JOBS:
        name = f'{phase}_{dataset}'
        folder = ROOT/name
        report = json.loads((folder/'report.json').read_text())
        registration = json.loads((folder/'registration.json').read_text())
        pairs = [json.loads(x) for x in (folder/'pairs.jsonl').read_text().splitlines() if x.strip()]
        manifest = validate_dataset_records(name, phase, dataset, registration, report, pairs)
        if dataset in dataset_manifests and manifest != dataset_manifests[dataset]:
            raise ValueError(f'Before/after dataset manifest mismatch: {dataset}')
        dataset_manifests[dataset] = manifest
        if dataset == 'GoogleEarth' and curation is not None:
            if curation['status'] != 'complete' or curation['current_manifest'] != manifest:
                raise ValueError(f'Curation/current manifest mismatch: {name}')
            if set(curation['excluded_pair_ids']) & {r['pair_id'] for r in pairs}:
                raise ValueError(f'Excluded pair still present: {name}')
        m = augment(report['overall'])
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
    for dataset in DATASETS:
        a,b = reports[f'before_{dataset}'],reports[f'after_{dataset}']
        deltas[dataset] = dict(
            precision_change_pp={t:100*(b['precision'][t]-a['precision'][t]) for t in ('1','3','5')},
            sr3_change_pp=100*(b['sr']['3']-a['sr']['3']),
            NCM_change={t:b['NCM'][t]-a['NCM'][t] for t in ('1','3','5')},NM_change=b['NM']-a['NM'])
    result = dict(reports=reports,by_group=groups,stages=stage_reports,after_minus_before=deltas,
        dataset_manifests=dataset_manifests,dataset_curation=curation,
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
    if curation is not None:
        note = f"GoogleEarth 经人工确认删除 {', '.join(curation['excluded_pair_ids'])}：预测正确，原始影像配对与 GT 错误。当前有效 {dataset_manifests['GoogleEarth']['pairs']} 对；JL 数据保持不变。其余预测保留，仅重新汇总指标。"
        content += '<p>'+html.escape(note)+' <a href="googleearth_curation.json">人工排除记录</a></p>'
        markdown.extend(['', note])
    content+='<p>阈值采用存储影像的512目标像素。RMSE：SR@3成功对统计≤5px正确匹配，失败记10。GoogleEarth 标签是名义原始配准上的合成几何。耗时保留原测试记录，运行卡及系统负载不同。</p>'
    content+='<div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in labels)+'</tr>'+''.join(tables)+'</table></div>'
    content+='<h2>各阶段指标</h2><div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in ['数据集','权重','阶段','P@1px','P@3px','P@5px','SR@3px','网格RMSE中位数'])+'</tr>'+''.join(stage_html)+'</table></div>'
    content+='<p><a href="comparison.json">完整结果及 JL t0–t4 分组</a> · <a href="comparison.csv">CSV</a></p>'
    (ROOT/'comparison.html').write_text(content)
    (ROOT/'comparison.md').write_text('\n'.join(markdown)+'\n\nNCM/NM 均为全部影像对平均；GT阈值使用512坐标。推理结果未重跑。\n')
    print(json.dumps(dict(reports={k:dict(NCM=v['NCM'],NM=v['NM']) for k,v in reports.items()},output=str(ROOT/'comparison.html')),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
