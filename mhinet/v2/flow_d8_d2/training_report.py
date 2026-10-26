"""Summarize the completed D8/D2-only adaptation run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def main():
    root=Path('outputs/flow_d8_d2/training');out=Path('reports/flow_d8_d2/training');out.mkdir(parents=True,exist_ok=True)
    history=[json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
    initial=json.loads((root/'epoch_000.json').read_text())
    checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=False,mmap=True)
    epoch=checkpoint['epoch'];best=json.loads((root/f'epoch_{epoch:03d}.json').read_text())
    status=json.loads((root/'status.json').read_text())
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'before.json').write_text(json.dumps(initial,indent=2));(out/'best.json').write_text(json.dumps(best,indent=2))
    lines=['# 杭州D8/D2训练结果（关闭QRRU）','',
        f'已完成{status["last_epoch"]}轮，最佳第{epoch}轮。254训练/64验证对，沿用清理后空间划分。',
        '冻结DINO、MVT、原Flow及Matchability；更新VGG、D8/D2解码器和温度。继承旧下游第7轮权重，非随机重新训练。QRRU不调用、不参与损失或优化器。',
        '最多30轮，AdamW 8e-5，余弦退火，累积4对；验证P@3连续6轮未提升至少0.1个百分点即早停。初始模型也参与最佳选择。',
        '实际推理采用Flow局部D8 ±4格 → D2，最多2048对，不额外截断Matchability。训练标签沿用近似H及已有不确定区域损失。',
        '**以下为杭州近似H参考一致性，不等同于人工标注的真实定位精度；该验证集用于早停，不是独立测试集。**','',
        '|指标|训练前|最佳轮|','|---|---:|---:|']
    for label,key,scale in [('P@1 (%)','p1',100),('P@3 (%)','p3',100),('可见EPE (原图px)','epe',1),('输出点/对','matches',1),('正确点@3/对','correct3',1)]:
        lines.append(f'|{label}|{initial[key]*scale:.2f}|{best[key]*scale:.2f}|')
    lines += ['','|分组|训练前P@3|最佳轮P@3|最佳轮同点Flow P@3|','|---|---:|---:|---:|']
    for group in sorted({r['group'] for r in best['pairs']}):
        def p3(records,key):
            rows=[r[key] for r in records['pairs'] if r['group']==group]
            return 100*sum(r['correct3'] for r in rows)/max(1,sum(r['n'] for r in rows))
        lines.append(f'|{group}|{p3(initial,"d2"):.2f}%|{p3(best,"d2"):.2f}%|{p3(best,"flow_same_points"):.2f}%|')
    from .report import stats
    lines += ['', '## 最佳权重复评', '', '|数据|D8 P@3 前→后|D2 P@3 前→后|D2 EPE 前→后|训练后同点Flow P@3→D2 P@3|', '|---|---:|---:|---:|---:|']
    for domain in ('hangzhou','synthetic'):
        baseline=json.loads((Path('outputs/flow_d8_d2')/domain/'results.json').read_text())
        trained=json.loads((Path('outputs/flow_d8_d2/trained')/domain/'results.json').read_text())
        assert len(baseline)==len(trained)
        (out/f'{domain}_reevaluation.json').write_text(json.dumps(trained,indent=2))
        a,b=stats(baseline,'local','d2'),stats(trained,'local','d2')
        ca,cb=stats(baseline,'local','d8'),stats(trained,'local','d8')
        flow,fine=stats(trained,'local','paired_flow'),stats(trained,'local','paired_d2')
        lines.append(f'|{domain}|{ca[3]:.2f}% → {cb[3]:.2f}%|{a[3]:.2f}% → {b[3]:.2f}%|{a[1]:.2f} → {b[1]:.2f}|{flow[3]:.2f}% → {fine[3]:.2f}%|')
    lines += ['', '## 结论', '', '本轮训练带来有限收益：杭州D2 P@3提高2.87个百分点，正确点约增加65对/图对；P@1基本持平，EPE仅小幅下降。合成精细定位没有退化。', '', '杭州最终同点比较仍为Flow 68.11%、D2 67.03%，G2差距最明显（Flow 67.89%、D2 59.59%）；G0则相对训练前退化。不能据整体P@3上升认定细化已全面优于Flow。保持QRRU关闭，后续优先检查G2的D8父窗口选择、D2候选及近似H误差，再决定是否修改细化结构或监督，不继续盲目延长训练。']
    lines += ['', '![训练曲线](curves.png)', '', '完整最佳权重保存在 `outputs/flow_d8_d2/training/best.pt`，未将大权重上传Git。逐轮指标和前后逐对统计保存在本报告目录。']
    fig,axes=plt.subplots(1,3,figsize=(12,3.5));xs=[0]+[r['epoch'] for r in history]
    for ax,key,label,scale in zip(axes,('p3','epe','correct3'),('Validation P@3 (%)','Visible EPE (px)','Correct points @3 / pair'),(100,1,1)):
        ax.plot(xs,[initial[key]*scale]+[r[key]*scale for r in history],marker='.',markersize=4)
        ax.axvline(epoch,color='green',linestyle='--',label='Selected checkpoint');ax.set_xlabel('Epoch');ax.set_title(label);ax.grid(alpha=.25)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);plt.close(fig)
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
