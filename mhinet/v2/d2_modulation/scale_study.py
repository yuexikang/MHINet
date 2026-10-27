"""Data expansion and final evaluation for the predeclared D2 scale study."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from mhinet.v2.qrru_corners.experiment import selections, subset
from .model import load_adapter
from .experiment import evaluate, aggregate, write_json

SMALL=Path('outputs/d2_modulation')
LARGE=Path('outputs/d2_modulation_expanded')
REPORT=Path('reports/d2_modulation_expanded')


def runs():
    result=[('baseline',SMALL/'none')]
    result += [(f'small_{mode}_s{seed}',SMALL/(mode if seed==42 else f'{mode}_seed{seed}'))
               for seed in (42,43,44) for mode in ('self','pair')]
    result += [(f'{budget}_{mode}_s42',LARGE/f'{budget}_{mode}')
               for budget in ('matched','full') for mode in ('self','pair')]
    return result


def plan():
    s=json.loads(Path('outputs/d16_final_round/selection.json').read_text());old=selections()
    rgb,hz=old['synthetic'],old['hangzhou']
    all_test=[json.loads(x) for x in (Path(rgb['test_root'])/'test/pairs.jsonl').read_text().splitlines()]
    previous={r['pair_id'] for r in s['rgb_test']}
    extra=subset([r for r in all_test if r['pair_id'] not in previous],192,'source_dataset')
    selection=dict(train=dict(root=rgb['root'],rows=s['rgb_train']),val=dict(root=rgb['root'],rows=rgb['val']),
        test=dict(root=rgb['test_root'],rows=rgb['test']),extra_test=dict(root=str(Path(rgb['test_root'])/'test'),rows=extra),
        hangzhou_test=dict(root=hz['root'],rows=hz['test']))
    for key in ('pair_id','source_path'):
        groups=[{r[key] for r in selection[k]['rows']} for k in ('train','val','test','extra_test')]
        if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):
            raise ValueError(f'Cross-split overlap in {key}')
    LARGE.mkdir(parents=True,exist_ok=True)
    write_json(LARGE/'selection.json',selection)
    print({k:len(v['rows']) for k,v in selection.items()})


@torch.no_grad()
def extra_evaluation(device):
    torch.set_num_threads(2)
    files=sorted((LARGE/'cache/extra_test').glob('*.pt'))
    expected=len(json.loads((LARGE/'data.json').read_text())['splits']['extra_test']['pairs'])
    if len(files)!=expected:raise RuntimeError('Extra test cache incomplete')
    for name,folder in runs():
        if not (folder/'done.json').exists():raise RuntimeError(f'Run unfinished: {name}')
        if (folder/'best_extra_test.json').exists():continue
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        adapter=load_adapter(folder/'best.pt',device)
        write_json(folder/'best_extra_test.json',evaluate(adapter,files,state['temperature'],device))
        print('extra test '+name,flush=True)


def grouping(row):
    overlap=min(row['overlap_fractions'])
    scale=float(row['equivalent_side_ratio']);scale=max(scale,1/scale)
    return dict(source=row['source_dataset'],
        overlap='<0.4' if overlap<.4 else '0.4–0.7' if overlap<.7 else '>=0.7',
        scale='<1.25' if scale<1.25 else '1.25–1.75' if scale<1.75 else '>=1.75')


def report():
    REPORT.mkdir(parents=True,exist_ok=True)
    selection=json.loads((LARGE/'selection.json').read_text())
    metadata={r['pair_id']:r for split in ('test','extra_test') for r in selection[split]['rows']}
    all_results={};details=[];strata=[]
    for name,folder in runs():
        done=json.loads((folder/'done.json').read_text());config=json.loads((folder/'config.json').read_text())
        history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
        values={split:json.loads((folder/f'best_{split}.json').read_text()) for split in ('val','test','extra_test','hangzhou_test')}
        total_updates=sum(row['updates'] for row in history)
        selected_updates=sum(row['updates'] for row in history if row['epoch']<=done['best_epoch'])
        all_results[name]=dict(done=done,config=config,updates=total_updates,best_updates=selected_updates,
            metrics={split:v['summary'] for split,v in values.items()})
        for split,v in values.items():
            details.extend(dict(run=name,split=split,**row) for row in v['pairs'])
        for split in ('test','extra_test'):
            for axis in ('source','overlap','scale'):
                labels=sorted({grouping(metadata[row['pair_id']])[axis] for row in values[split]['pairs']})
                for label in labels:
                    rows=[row for row in values[split]['pairs'] if grouping(metadata[row['pair_id']])[axis]==label]
                    strata.append(dict(run=name,split=split,axis=axis,group=label,pairs=len(rows),**aggregate(rows)))
    write_json(REPORT/'summary.json',all_results)
    write_json(REPORT/'strata.json',strata)
    (REPORT/'pairs.jsonl').write_text('\n'.join(json.dumps(row) for row in details)+'\n')
    write_json(REPORT/'test_selection.json',{k:dict(root=v['root'],rows=[{f:r[f] for f in ('pair_id','source_dataset','source_path','overlap_fractions','equivalent_side_ratio')} for r in v['rows']]) for k,v in selection.items() if k in ('test','extra_test')})
    text='''# D2 调制：随机种子复现与 RGB_512 扩量

只训练相同82,496参数MLP；Flow、D8、原D2、温度均冻结，固定8×8上下文和4×4→8×8匹配。三种训练设置共用64对RGB验证选模型、64对原测试、192对新增测试和64对杭州测试。新增测试按来源分层，从已登记d16测试256对之外选择；未用于本轮调参或选模型，不宣称原预训练绝对未见过同地区。

小样本254对用seed42/43/44；1024对预算对照每验证周期抽254对，完整扩量每周期遍历1024对；最多30周期、相同余弦计划和早停规则。预算对照的最大更新数相同，早停导致实际更新数和best更新数可能不同，下面均显式报告。扩量只跑seed42，不能将多种子稳定性结论推广到扩量。

## 各组结果

|组|实际更新/最佳更新|最佳周期|RGB验证P@3|原64测试P@3|新增192测试P@3|杭州测试P@3|
|---|---:|---:|---:|---:|---:|---:|
'''
    for name,r in all_results.items():
        text+=f"|{name}|{r['updates']}/{r['best_updates']}|{r['done']['best_epoch']}|"+'|'.join(f"{100*r['metrics'][s]['p3']:.2f}%" for s in ('val','test','extra_test','hangzhou_test'))+'|\n'
    base=all_results['baseline']['metrics']
    text+='\n同点直接Flow：'+ '；'.join(f"{s} {100*base[s]['flow_p3']:.2f}%" for s in ('test','extra_test','hangzhou_test'))+'。所有P@3分母为全部输出点；杭州为近似H一致性。\n'
    text+='\n## 小样本三seed复现\n\n|测试集|self 均值±样本标准差|pair 均值±样本标准差|成对差均值±标准差|pair胜出seed数|\n|---|---:|---:|---:|---:|\n'
    for split in ('test','extra_test','hangzhou_test'):
        a=[all_results[f'small_self_s{s}']['metrics'][split]['p3']*100 for s in (42,43,44)]
        b=[all_results[f'small_pair_s{s}']['metrics'][split]['p3']*100 for s in (42,43,44)]
        d=[y-x for x,y in zip(a,b)]
        text+=f'|{split}|'+ '|'.join(f'{statistics.mean(x):.3f}±{statistics.stdev(x):.3f}' for x in (a,b,d))+f'|{sum(x>0 for x in d)}/3|\n'
    text+='\n## 数据量与更新预算\n\n|模式|新增测试：小样本seed42|1024相同最大预算|1024完整遍历|\n|---|---:|---:|---:|\n'
    for mode in ('self','pair'):
        text+=f'|{mode}|'+ '|'.join(f"{100*all_results[k]['metrics']['extra_test']['p3']:.2f}%" for k in (f'small_{mode}_s42',f'matched_{mode}_s42',f'full_{mode}_s42'))+'|\n'
    text+='\n### 相同更新次数位置的验证对照\n\n两组每周期均254次更新且学习率日程相同；下表比较相同优化步数，避免把早停选择的不同预算混在一起。验证结果不是独立测试结论。\n\n|模式|更新次数|254对验证P@3|1024对验证P@3|\n|---|---:|---:|---:|\n'
    for mode in ('self','pair'):
        small={r['epoch']:r for r in json.loads((SMALL/mode/'history.json').read_text())}
        large={r['epoch']:r for r in json.loads((LARGE/f'matched_{mode}'/'history.json').read_text())}
        for epoch in (5,10,15,20):
            if epoch in small and epoch in large:
                text+=f"|{mode}|{254*epoch}|{100*small[epoch]['p3']:.2f}%|{100*large[epoch]['p3']:.2f}%|\n"
    text+='\n## 新增测试分组结果\n\n分组为描述性统计，不据此重新选模型；重叠率取A/B两侧最小值，尺度取max(s,1/s)。分组存在样本量差异，不能单靠单组变化宣称普适收益。\n\n|分组轴/组|对数|不调制|小样本self42|小样本pair42|扩量self42|扩量pair42|\n|---|---:|---:|---:|---:|---:|---:|\n'
    chosen=['baseline','small_self_s42','small_pair_s42','full_self_s42','full_pair_s42']
    groups=sorted({(r['axis'],r['group']) for r in strata if r['split']=='extra_test'})
    for axis,label in groups:
        lookup={r['run']:r for r in strata if r['split']=='extra_test' and r['axis']==axis and r['group']==label}
        text+=f"|{axis}/{label}|{lookup['baseline']['pairs']}|"+'|'.join(f"{100*lookup[k]['p3']:.2f}%" for k in chosen)+'|\n'
    text+='\n逐对输出、改坏/救回Flow和候选排名见pairs.jsonl，完整分组统计见strata.json。未解冻原网络、未增加D8模块、未更改原主线。具体结论应结合多seed成对差和预算对照，不把更多更新带来的提升全部归于数据量。\n'
    full=all_results['full_pair_s42']['metrics']['extra_test']
    self_full=all_results['full_self_s42']['metrics']['extra_test']
    small=all_results['small_pair_s42']['metrics']['extra_test']
    matched=all_results['matched_pair_s42']['metrics']['extra_test']
    text+=f"\n## 本轮结论\n\n新增测试上，完整扩量pair相对未调制提高 {(full['p3']-base['extra_test']['p3'])*100:.2f} 个百分点，相对同点Flow提高 {(full['p3']-full['flow_p3'])*100:.2f} 个百分点。小样本pair→1024相同最大预算→1024完整训练为 {small['p3']*100:.2f}%→{matched['p3']*100:.2f}%→{full['p3']*100:.2f}%；数据覆盖与更多更新都有贡献，不给出未经严格因素分离的因果占比。\n\n"
    text+=f"完整扩量pair比self高 {(full['p3']-self_full['p3'])*100:.2f} 个百分点，但二者最佳权重经历的更新数不同；同最大预算组的新测试差距接近零。结合小样本三seed，保留pair为当前候选、self为必要对照，尚不能宣称双图条件具有普适优势。\n\n"
    text+='新增测试只覆盖当前抽到的重叠/尺度区间，没有重叠率≥0.7或尺度比<1.25的样本；不外推到这些区间。完整扩量只做seed42，未重复扩量多seed，未合并上一主线或解冻上游。\n'
    (REPORT/'results.md').write_text(text)
    print(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','evaluate','report']);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.action=='plan':plan()
    elif args.action=='evaluate':extra_evaluation(args.device)
    else:report()

if __name__=='__main__':main()
