"""Produce the three-way held-out report only after all runs have completed."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/d2_modulation'))
    p.add_argument('--output',type=Path,default=Path('reports/d2_modulation'));args=p.parse_args()
    results={};histories={}
    for mode in ('none','self','pair'):
        folder=args.root/mode
        done=json.loads((folder/'done.json').read_text())
        results[mode]=dict(**done,config=json.loads((folder/'config.json').read_text()),
            evaluations={split:json.loads((folder/f'best_{split}.json').read_text()) for split in ('val','test','hangzhou_test')})
        histories[mode]=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps({mode:{**{k:v for k,v in r.items() if k!='evaluations'},
        'evaluations':{s:v['summary'] for s,v in r['evaluations'].items()}} for mode,r in results.items()},indent=2))
    (args.output/'pairs.jsonl').write_text('\n'.join(json.dumps(dict(mode=mode,split=split,**row))
        for mode,r in results.items() for split,value in r['evaluations'].items() for row in value['pairs'])+'\n')
    (args.output/'history.json').write_text(json.dumps(histories,indent=2))
    (args.output/'data.json').write_text((args.root/'data.json').read_text())
    text='''# D2 局部双图调制：RGB_512 小规模试验

模型与参数见 docs/mhi_v2/d2_local_modulation.md。训练254对、RGB验证64对选模型，RGB测试64对及杭州测试64对仅在训练结束后评估。Flow、D8及全部原特征网络冻结；仅新MLP学习。A/B均使用8×8上下文，A中心4×4查询与B8×8候选匹配，不执行QRRU或删点。三组共享FP32缓存，初始调制为恒等。

训练损失只有精确H四邻域软标签位置分类。验证最优允许选初始状态；结果好坏不影响实现验收。杭州只有近似H参考一致性。

新缓存入口与前轮基线在131072个验证查询上相差1个3px正确点（0.000763个百分点）；三组均以当前none为对照。恒等初始化与当前none的相似度/坐标完全一致。

|组|完成轮数|最佳轮次|RGB验证P@3|RGB测试P@3|杭州测试P@3|
|---|---:|---:|---:|---:|---:|
'''
    for mode,r in results.items():
        ev=r['evaluations'];text+=f"|{mode}|{r['epochs_run']}|{r['best_epoch']}|"+'|'.join(f"{100*ev[s]['summary']['p3']:.2f}%" for s in ('val','test','hangzhou_test'))+'|\n'
    for split,label in [('test','RGB_512测试'),('hangzhou_test','杭州测试')]:
        text+=f'\n## {label}同点分析\n\n|组|P@1|P@3|P@5|P@10|EPE(px)|正确区域Rank@1/5/10|改坏Flow|救回Flow|输出点数|\n|---|---:|---:|---:|---:|---:|---|---:|---:|---:|\n'
        for mode,r in results.items():
            v=r['evaluations'][split]['summary'];n=max(1,v['n']);cover=max(1,v['covered3'])
            text+=f'|{mode}|'+''.join(f"{100*v[f'correct{k}']/n:.2f}%|" for k in (1,3,5,10))
            text+=f"{v['epe']:.3f}|"+'/'.join(f"{100*v[f'rank{k}']/cover:.2f}%" for k in (1,5,10))
            text+=f"|{v['damaged']}|{v['rescued']}|{v['n']}|\n"
        base=results['none']['evaluations'][split]['summary']
        text+=f"\n同点直接Flow：P@3={100*base['flow_p3']:.2f}%，EPE={base['flow_epe']:.3f}px。Rank分母为窗口内有可见3px正确候选的查询；P@k分母为全部输出点，EPE仅参考可见点。\n"
    rgb={k:r['evaluations']['test']['summary'] for k,r in results.items()}
    delta=(rgb['pair']['p3']-rgb['none']['p3'])*100
    versus_self=(rgb['pair']['p3']-rgb['self']['p3'])*100
    versus_flow=(rgb['pair']['p3']-rgb['pair']['flow_p3'])*100
    text+=f'''\n## 结论边界\n\n双图调制相对基线的RGB测试变化为 {delta:+.2f} 个百分点，相对自调制为 {versus_self:+.2f} 个百分点，相对同点Flow为 {versus_flow:+.2f} 个百分点。

小样本单seed不能证明普适效果。跨图条件价值应同时看pair相对self以及测试变化；只有相对Flow为正且净救回为正才通过“细化优于直接Flow”的目标。模型与训练实现已经独立运行，未将实验权重合入主线。

![训练曲线](curves.png)

复现入口：mhinet/v2/d2_modulation。逐对计数见pairs.jsonl，配置/划分见summary.json与data.json，历史见history.json。best.pt/last.pt留在outputs/d2_modulation对应组目录，未加入Git。\n'''
    (args.output/'results.md').write_text(text)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for mode in ('self','pair'):
        history=histories[mode]
        axes[0].plot([0]+[v['epoch'] for v in history],
            [100*results['none']['evaluations']['val']['summary']['p3']]+[100*v['p3'] for v in history],label=mode)
        axes[1].plot([v['epoch'] for v in history],[v['loss'] for v in history],label=mode)
    base=results['none']['evaluations']['val']['summary']
    axes[0].axhline(base['flow_p3']*100,color='k',linestyle='--',label='Frozen Flow')
    axes[0].set_ylabel('RGB validation P@3 (%)');axes[1].set_ylabel('Training classification loss')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'curves.png',dpi=160);plt.close(fig)
    print(text)

if __name__=='__main__':main()
