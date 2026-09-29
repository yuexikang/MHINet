"""Re-score stored predictions; preserve old all-match and registration RMSE diagnostics."""
import argparse,json,hashlib,html,shutil
from pathlib import Path
import numpy as np
from mrsi_gt_metrics import conditional_rmse


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--success-rule',choices=('ncm20','registration3'),required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads((args.source/'report.json').read_text())
    if not report['complete']:raise ValueError('Source evaluation incomplete')
    rows=[json.loads(x) for x in (args.source/'pairs.jsonl').read_text().splitlines()]
    if len(rows)!=600 or len({r['id'] for r in rows})!=600:raise ValueError('Source pair count mismatch')
    for row in rows:
        path=args.source/'matches'/(row['id'].replace('/','_')+'.npz')
        with np.load(path,allow_pickle=False) as match:
            extra=conditional_rmse(match['points_a'],match['points_b'],*row['sizes'],row['H_gt'],
                success_override=row['metrics']['sr']['3'] if args.success_rule=='registration3' else None)
        if extra['rmse_ncm']!=row['metrics']['ncm']['5']:raise ValueError('Correct match selection differs')
        row['metrics'].update(extra);row['prediction_sha256']=sha(path)
    def enrich(group,selected):
        group.update(rmse_correct5_or_failure10=float(np.mean([r['metrics']['rmse_correct5_or_failure10'] for r in selected])),
            rmse_success_pairs=sum(r['metrics']['rmse_protocol_success'] for r in selected),
            rmse_failure_pairs=sum(not r['metrics']['rmse_protocol_success'] for r in selected),
            rmse_protocol_sr=float(np.mean([r['metrics']['rmse_protocol_success'] for r in selected])))
    enrich(report['overall'],rows)
    for key,group in report['by_modality'].items():enrich(group,[r for r in rows if r['modality']==key])
    report['rmse_revision']=dict(source_report_sha256=sha(args.source/'report.json'),source_pairs_sha256=sha(args.source/'pairs.jsonl'),
        script_sha256=sha(Path(__file__)),metrics_sha256=sha(Path(__file__).with_name('mrsi_gt_metrics.py')),
        scope='Final-stage RMSE re-scored from unchanged prediction NPZs; inference, precision, NCM, registration SR, old RMSE diagnostics, coarse/fine reports unchanged.',
        correct_matches='GT reprojection error <=5 native target pixels with valid GT-visible support',
        source_transform='H_GT, not the model-fitted homography',success_rule=args.success_rule,
        success_definition='NCM@5px>=20' if args.success_rule=='ncm20' else 'Original registration grid RMSE<=3px',
        failure_rmse=10,aggregation='Mean of per-pair RMSE over ALL pairs, including failures; not pooled by NCM',
        note='The supplied excerpt does not specify a correct-match threshold or success criterion; these are explicit implementation choices, not claimed as verified paper definitions.')
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.output/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    labels={'1 optical-optical':'光学–光学','2 optical-infrared':'光学–红外','3 optical-sar':'光学–SAR','4 optical-depth':'光学–深度','5 optical-map':'光学–地图','6 day-night':'昼–夜'}
    lines=[]
    for k,g in [('总体',report['overall']),*report['by_modality'].items()]:
        vals=[str(g['pairs']),*[f'{100*g["precision"][t]:.2f}%' for t in ('1','3','5')],f'{g["rmse_correct5_or_failure10"]:.4f}',f'{100*g["rmse_protocol_sr"]:.2f}%',f'{g["mean_ncm"]["5"]:.1f}',f'{100*g["sr"]["3"]:.2f}%']
        lines.append('<tr><td>'+labels.get(k,k)+'</td>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
        print(labels.get(k,k),g['rmse_correct5_or_failure10'],g['rmse_success_pairs'],g['rmse_failure_pairs'])
    page=f'<!doctype html><meta charset="utf-8"><title>MRSI正确匹配RMSE</title><style>body{{font-family:sans-serif;margin:30px}}td,th{{padding:9px;border-bottom:1px solid #ddd}}</style><h1>E组 expanded MRSI：修订RMSE</h1><p>正确匹配误差≤5px，使用GT变换源点；成功条件：{html.escape(report["rmse_revision"]["success_definition"])}。成功对只统计正确匹配的RMSE；失败对记10。全部影像对宏平均。</p><p>预测未改变，P/NCM和原配准SR保持不变。RMSE成功率与配准SR分别列出，避免混用成功定义。</p><table><tr><th>模态</th><th>对数</th><th>P@1px</th><th>P@3px</th><th>P@5px</th><th>新RMSE(px)</th><th>RMSE成功率</th><th>NCM@5px</th><th>原配准SR@3px</th></tr>{"".join(lines)}</table><p><a href="report.json">完整结果与指标定义</a> · <a href="../{args.source.name}/index.html">原始诊断与匹配图</a></p>'
    (args.output/'index.html').write_text(page)

if __name__=='__main__':main()
