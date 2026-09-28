"""Audited live/final report for the registered short chunk training comparison."""
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/semidense_chunk32_128_short200_20260928'


def read_jsonl(path):
    if not path.exists():return []
    # Writers flush complete records; discard only an unfinished final line.
    text=path.read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def mean(values):
    values=[x for x in values if x is not None]
    return statistics.mean(values) if values else None


def atomic(path,text):
    temporary=path.with_suffix('.tmp');temporary.write_text(text);temporary.replace(path)


def collect(chunk):
    folder=OUT/f'chunk{chunk}';rows=read_jsonl(folder/'train.jsonl')
    metadata=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else None
    validations={}
    for path in sorted((folder/'validation').glob('step_*.json')):
        value=json.loads(path.read_text());validations[str(value['summary']['step'])]=value['summary']
    evaluated=read_jsonl(folder/'cascaded_val128/pairs.jsonl')
    stages={}
    for stage in ('coarse','fine','final'):
        metrics=[x[stage] for x in evaluated if stage in x]
        stages[stage]=dict(pairs=len(metrics),mean_matches=mean([m['matches'] for m in metrics]),
            epe=mean([m.get('overlap_finite_EPE') for m in metrics]),
            epe_pairs=sum(m.get('overlap_finite_EPE') is not None for m in metrics),
            precision={t:mean([m['thresholds'][t]['precision'] for m in metrics]) for t in ('1.0','3.0','5.0')})
    curves=[]
    for row in rows:
        item={k:row[k] for k in ('step','seconds','gradient_norm','lr_shared','lr_head')}
        for key in ('lc','lf','lq'):item[key]=mean([r[key] for r in row['records']])
        item['loss']=sum(item[k] for k in ('lc','lf','lq'));curves.append(item)
    complete=(folder/'completed.json').exists()
    if complete and (len(rows)!=200 or len(evaluated)!=128):raise ValueError('Completed experiment has incomplete records')
    pair_ids=[r['pair_id'] for row in rows for r in row['records']]
    steady=[x for x in curves if x['step']>10]
    return dict(chunk=chunk,complete=complete,steps=len(rows),training_pairs=len(pair_ids),
        source_sha256=metadata['source_sha256'] if metadata else None,
        initial_matcher_sha256=metadata['initial_matcher_sha256'] if metadata else None,
        schedule_steps=metadata['total_steps'] if metadata else None,
        training_order_sha256=hashlib.sha256(json.dumps(pair_ids).encode()).hexdigest(),
        steady_step_seconds=mean([x['seconds'] for x in steady]),
        last50_train={k:mean([x[k] for x in curves[-50:]]) for k in ('loss','lc','lf','lq','gradient_norm')},
        peak_allocated_bytes=max((x['peak_allocated_bytes'] for x in rows),default=0),
        validation=validations,cascaded_pairs=len(evaluated),stages=stages,
        failure_rate=mean([float(x['failure_reason']!='none') for x in evaluated]),curves=curves),rows,evaluated


def chart(groups,key,label):
    series=[];all_values=[]
    for chunk,group in groups.items():
        rows=group['curves'];values=[mean([r[key] for r in rows[max(0,i-19):i+1]]) for i in range(len(rows))]
        series.append((chunk,rows,values));all_values.extend(values)
    if not all_values:return ''
    low,high=min(all_values),max(all_values);span=max(high-low,1e-8)
    lines=[]
    for chunk,rows,values in series:
        points=' '.join(f'{45+500*r["step"]/200:.2f},{160-130*(v-low)/span:.2f}' for r,v in zip(rows,values))
        lines.append(f'<polyline fill="none" stroke="{"#2563eb" if chunk=="32" else "#e56b20"}" stroke-width="2" points="{points}"/>')
    return f'<figure><figcaption>{html.escape(label)}（20步滑动均值）</figcaption><svg viewBox="0 0 570 190"><path d="M45 20V160H545" fill="none" stroke="#999"/><text x="0" y="30">{high:.3g}</text><text x="0" y="160">{low:.3g}</text><text x="45" y="183">0</text><text x="490" y="183">200步</text>{"".join(lines)}</svg></figure>'


def main():
    groups={};training={};evaluation={}
    for chunk in (32,128):groups[str(chunk)],training[chunk],evaluation[chunk]=collect(chunk)
    n=min(len(training[32]),len(training[128]))
    for a,b in zip(training[32][:n],training[128][:n]):
        if any(a[k]!=b[k] for k in ('step','cursor','lr_shared','lr_head')):raise ValueError('Step or schedule mismatch')
        if [r['pair_id'] for r in a['records']]!=[r['pair_id'] for r in b['records']]:raise ValueError('Training data order mismatch')
    for key in ('source_sha256','initial_matcher_sha256','schedule_steps'):
        a,b=[groups[c][key] for c in ('32','128')]
        if a is not None and b is not None and a!=b:raise ValueError(f'Metadata mismatch: {key}')
    count=min(len(evaluation[32]),len(evaluation[128]))
    if [r['pair_id'] for r in evaluation[32][:count]]!=[r['pair_id'] for r in evaluation[128][:count]]:
        raise ValueError('Evaluation cohort mismatch')
    complete=all(g['complete'] for g in groups.values())
    report=dict(complete=complete,matched_training_steps=n,matched_evaluation_pairs=count,groups=groups,
        note='Single seed, 200 updates, fixed first128 tier3 val pairs. Partial results are not a ranking; full-training equivalence unproven. Step timing excludes periodic validation/saving; different GPUs.')
    atomic(OUT/'comparison.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    atomic(ROOT/'artifacts/semidense_chunk32_128_short200_results.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    table=[]
    def fmt(value):return '—' if value is None else f'{value:.6g}'
    def row(label,values):table.append('<tr><td>'+label+'</td>'+''.join('<td>'+fmt(v)+'</td>' for v in values)+'</tr>')
    row('已完成训练步',[g['steps'] for g in groups.values()])
    row('平均训练步耗时(s，排除前10步)',[g['steady_step_seconds'] for g in groups.values()])
    for step in ('0','100','200'):
        for key in ('lc','lf','lq'):
            row(f'第{step}步验证 {key}',[g['validation'].get(step,{}).get('metrics',{}).get(key) for g in groups.values()])
    for stage,label in (('coarse','粗匹配'),('fine','细匹配'),('final','QRRU最终')):
        row(label+' EPE(px)',[g['stages'][stage]['epe'] for g in groups.values()])
        row(label+' precision@1px',[g['stages'][stage]['precision']['1.0'] for g in groups.values()])
    row('推理失败率',[g['failure_rate'] for g in groups.values()])
    charts=''.join(chart(groups,k,label) for k,label in [('loss','总损失'),('lc','Lc'),('lf','Lf'),('lq','Lq'),('gradient_norm','梯度范数'),('seconds','训练步耗时(s)')])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>32/128训练分块对照</title><style>body{{font-family:sans-serif;margin:28px;max-width:1200px}}td,th{{padding:8px 18px;border-bottom:1px solid #ddd}}.charts{{display:grid;grid-template-columns:repeat(2,1fr)}}figure{{margin:12px}}svg{{width:100%}}</style><h1>32/128训练分块：200步对照</h1><p>{'已完成' if complete else '进行中，部分结果不作最终排名'}；第一档C → 第三档，E学习率，seed0，有效batch4。蓝色：32；橙色：128。</p><p>实际相同训练顺序已核对{n}步，级联验证已核对{count}对。固定val前128对，不能代表完整验证集；推理统一1024。</p><table><tr><th>指标</th><th>32</th><th>128</th></tr>{''.join(table)}</table><div class="charts">{charts}</div><p><a href="comparison.json">完整结果</a> · 单seed短程结果不能证明完整训练等价。不同物理GPU，训练耗时不包含定期保存、验证。</p></html>'''
    atomic(OUT/'index.html',page)
    print(json.dumps(dict(complete=complete,steps={c:g['steps'] for c,g in groups.items()},matched_evaluation_pairs=count)))


if __name__=='__main__':main()
