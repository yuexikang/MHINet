"""Common geographic-reference metrics for the v1/v2 Hangzhou comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_report(output, v2_run):
    runs = {'v1': output, 'v2': v2_run}
    histories = {name: [json.loads(p.read_text()) for p in sorted(root.glob('epoch_*.json'))]
                 for name, root in runs.items()}
    selected = {}
    for name, rows in histories.items():
        if not rows:
            continue
        trained = [r for r in rows if r['epoch'] > 0]
        selected[name] = dict(before=rows[0], best=min(trained or rows, key=lambda r:r['mean']['loss']), after=rows[-1])
    summary = {name:{stage:dict(epoch=row['epoch'], metrics=row['mean']) for stage,row in stages.items()}
               for name,stages in selected.items()}
    (output/'comparison.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=140)
    for name, rows in histories.items():
        epochs = [r['epoch'] for r in rows]
        for ax, key, scale, label in zip(axes, ('pre3','flow_epe_px','matches'), (100,1,1),
                                        ('Reference Pre@3 (%)','Reference Flow EPE (px)','Mean output matches')):
            ax.plot(epochs, [r['mean'].get(key, 0)*scale for r in rows], marker='.', label=name)
            ax.set_xlabel('Epoch'); ax.set_ylabel(label); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output/'comparison.png'); plt.close(fig)
    config = json.loads((output/'config.json').read_text())
    initialization = ('DINO、MVT 和 MatchEmbedding 加载既有权重并冻结，其余活跃模块从头训练'
                      if config.get('frozen_locator_checkpoint') else 'DINO-only 初始化，其他活跃模块从头训练')
    text = ['# 杭州 v1 / v2 同数据、同预算对照', '',
        '本实验没有真实人工对应点或真实非刚性 Flow，指标均相对于近似地理 H。', '',
        f'使用完全相同的 256 对训练、64 对验证样本，{initialization}。AdamW 学习率 2e-4、有效批量 4、最多 30 轮、余弦退火、连续 6 轮无改善早停。', '',
        'v1 保留原生 H0 拟合、匹配与控制点监督；v2 使用 Flow 定位和新增弱监督损失。因此这是完整方案对照，不能单独隔离网络结构因素。两者损失定义不同，不能直接比较总 loss 数值。best 按各自验证 loss 选择，初始状态不参与选模。', '',
        '| 模型 | 状态 | 轮数 | 参考 Pre@1 | 参考 Pre@3 | Flow EPE（像素） | 平均输出点数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name, stages in selected.items():
        for stage, row in stages.items():
            m=row['mean']; title={'before':'训练前','best':'最佳验证 loss','after':'最新记录'}[stage]
            text.append(f"| {name} | {title} | {row['epoch']} | {m.get('pre1',0)*100:.3f}% | {m.get('pre3',0)*100:.3f}% | {m.get('flow_epe_px',0):.3f} | {m.get('matches',0):.1f} |")
    text += ['', '## 各模型内部的训练与验证损失', '',
        '只在同一模型内部比较；两个模型的总损失定义不同。', '',
        '| 模型 | 最佳轮数 | 当轮平均训练损失 | 验证损失 |',
        '|---|---:|---:|---:|']
    for name, stages in selected.items():
        row=stages['best']
        text.append(f"| {name} | {row['epoch']} | {row.get('train_mean',{}).get('loss',float('nan')):.4f} | {row['mean']['loss']:.4f} |")
    text += ['', '## 最佳检查点按几何组比较', '',
        '| 模型 | 几何组 | 对数 | 参考 Pre@3 | 平均输出点数 |',
        '|---|---|---:|---:|---:|']
    for name, stages in selected.items():
        for group in ('G0','G1','G2','G3'):
            pairs = [r for r in stages['best']['pairs'] if r['group'] == group]
            if pairs:
                text.append(f"| {name} | {group} | {len(pairs)} | {sum(r['pre3'] for r in pairs)/len(pairs)*100:.3f}% | {sum(r['matches'] for r in pairs)/len(pairs):.1f} |")
    text += ['', '![共同指标曲线](comparison.png)', '',
        '若两个方案都表现较差，只能说明当前数据、监督和小样本从头训练条件的组合存在困难，不能据此排除初始化、训练预算及优化问题。需要结合原始图像与独立对应点才能判断真实几何误差。']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('output', type=Path); p.add_argument('v2_run', type=Path)
    a=p.parse_args(); write_report(a.output,a.v2_run)
