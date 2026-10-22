"""Generate a human-readable diagnostic / paired experiment report."""
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())


def diagnostic_report(root):
    output=root/'diagnostics';reports={v:read(output/v/'summary.json') for v in ('U','B','C')}
    lines=['# 阶段一：第30轮D16解码诊断','',
        '仅诊断旧U/B/C，不重新训练。所有查询坐标相同；指标相对杭州近似H，不代表独立真实对应。',
        '','## 正区域排序与连续坐标','',
        '|模型|Rank=1查询数|Rank=1 EPE|Rank=1 P@3|Rank>1 EPE|理想概率整体EPE|理想概率整体P@3|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v,d in reports.items():
        f=d['full'];r=f['rank1'];a=f['all']
        lines.append(f'|{v}|{r["points"]}|{r["epe"]:.3f}|{r["p3"]*100:.2f}%|{f["rank_above1"]["epe"]:.3f}|{a["ideal_epe"]:.3f}|{a["ideal_p3"]*100:.2f}%|')
    # Use intersection to avoid comparing different rank=1 subsets.
    paired=[]
    for path in sorted((output/'U').glob('val_*.npz')):
        arrays={v:dict(np.load(output/v/path.name)) for v in reports}
        common=np.logical_and.reduce([a['valid'] & (a['rank']==1) for a in arrays.values()])
        paired.append({v:a['epe'][common] for v,a in arrays.items()})
    lines+=['','三组各自Rank=1子集不同；以下进一步固定三组共同Rank=1的相同查询：','']
    for v in reports:
        e=np.concatenate([p[v] for p in paired]);lines.append(f'- {v}：{len(e)}点，EPE={e.mean():.3f}px，P@3={(e<=3).mean()*100:.2f}%。')
    lines+=['',
        '旧MatchEmbeddingDecoder由三个3×3卷积构成（含输出头），对源D16网格有7×7感受野，并使用零填充。单个查询的最高候选正确，不保证连续坐标正确；邻居概率分布和边界也参与输出。',
        '理想概率诊断使用H坐标四邻域双线性分布，无效查询置dustbin，再进入原PE聚合及Decoder。它改变了整个概率场的分布及邻域输入，不能把结果全部归因于单点解码器，也不能据此将近似H当作精确真值。',
        '','## 方案引用的两个案例（B，第30轮）','']
    for group in ['G1','G3']:
        r=reports['B']['quoted_cases_nearest'][group+'_positive_mass']
        lines.append(f'- {group}：`{r["pair_id"]}`，查询{r["query"]}，正区域概率{r["positive_mass"]:.6f}，排名{r["rank"]}，EPE={r["epe"]:.3f}px，偏差(dx,dy)={r["delta"]}，H理想概率输入EPE={r["ideal_epe"]:.3f}px。')
        lines.append(f'![{group}](B/{group}_positive_mass_quoted_case_nearest.jpg)')
    lines+=['','G1案例在理想概率场下明显改善；G3案例仍偏差很大。两者不能用同一种简单解释概括。暂不修改Decoder。',
        '','## 三个灾难性G3样本','',
        '三对均保留在正式统计中。排除它们仅用于定位长尾来源，不用于重新筛选训练集。',
        '|模型|全量EPE|仅诊断排除3对EPE|全量P@3|排除后P@3|','|---|---:|---:|---:|---:|']
    for v,d in reports.items():
        a=d['full']['all'];b=d['excluding_three_diagnostic_only']['all']
        lines.append(f'|{v}|{a["epe"]:.3f}|{b["epe"]:.3f}|{a["p3"]*100:.2f}%|{b["p3"]*100:.2f}%|')
    observations={
        'val_G3_12add035b2b2190ad8b131b2':'B最差查询的正区域排名775，候选集中在错误位置；H叠图可见局部地貌结构，但该查询局部植被重复。支持“参考候选检索失败”的判断，不能独立证明H绝对准确。',
        'val_G3_a4e0a479df8364b40227e2c5':'B最差查询排名275；B图明显低对比、林地纹理重复，局部对应的可观察性较弱。不能仅凭大EPE认定标签错误或解码器错误。',
        'val_G3_d675c8471dc323c373c41661':'B最差查询排名590；图中住宅区可辨认，但该查询落在重复植被区，Top10集中于另一区域。既有旋转/尺度变化，也有候选检索混淆。'}
    for pid,note in observations.items():
        lines+=['',f'### {pid}',note,f'![审计](B/{pid}_audit.jpg)',f'[H、掩膜路径及尺度旋转原始元数据](B/{pid}_metadata.json)']
    lines+=['','所加载valid_A/B在这些图上为全有效；它们不等于真实重叠/遮挡标注。H投影的图内区域另由geometry.query计算。现有材料不能可靠将每个错误点划分成“真值错误”或“实际不可观察”，报告保留这种不确定性。',
        '','## 逐查询材料','',
        '各模型目录的NPZ保存全部1024查询：参考/预测/理想坐标、偏差向量、有效掩膜、排名、正区域概率质量、正候选最高概率、Top10坐标/概率、完整分布熵、边界距离、dustbin和matchability。summary.json含条件分箱及逐对统计；高置信错误的全部坐标保留在NPZ，每个有此类点的影像对保存最差点局部图；export_patches.py另输出所有高置信>8px查询的A/参考B/预测B局部图拼页，黑色补边表示裁剪超出图像。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def comparison_report(root, config):
    variants=['N1','N2','U0','B0']
    if not all((root/v/'status.json').exists() and read(root/v/'status.json').get('status')=='completed' for v in variants):return
    results={v:read(root/v/'epoch_030.json') for v in variants}
    lines=['# 局部遮挡第30轮配对验收','',
        '统一比较第30轮，不使用各自最优epoch混比。U0/B0具有相同增强输入和前后向预算，但新增项乘零。',
        '','|组|清洁P@3|EPE|>32px|遮挡负例dustbin Top1|未遮挡正例P@3|正例误拒绝|AUROC|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,r in results.items():
        a=r['clean'];o=r['occlusion']['fixed_mixture']
        lines.append(f'|{v}|{a["flow_p3"]*100:.2f}%|{a["epe"]:.3f}|{a["epe_gt32"]*100:.2f}%|{o["negative_dustbin_top1"]*100:.2f}%|{o["positive_p3"]*100:.2f}%|{o["positive_false_reject"]*100:.2f}%|{o["matchability_auroc"]:.4f}|')
    verdicts={}
    for v,baseline in [('N1','U0'),('N2','B0')]:
        r,b=results[v],results[baseline];a,z=r['clean'],b['clean'];o,q=r['occlusion']['fixed_mixture'],b['occlusion']['fixed_mixture']
        checks=dict(clean_p3=a['flow_p3']>=z['flow_p3']-config['p3_tolerance'],
            negative_rejection=o['negative_dustbin_top1']>=q['negative_dustbin_top1']+config['negative_rejection_min_gain'],
            positive_rejection=o['positive_false_reject']<=q['positive_false_reject']+config['positive_false_reject_tolerance'],
            clean_rejection=a['false_reject']<=z['false_reject']+config['positive_false_reject_tolerance'],
            tail=a['epe_gt32']<=z['epe_gt32']+config['tail32_tolerance'],
            groups=all(r['groups'][g]['flow_p3']>=b['groups'][g]['flow_p3']-config['group_p3_tolerance'] for g in ['G1','G3']))
        verdicts[v]=dict(baseline=baseline,checks=checks,passed=all(checks.values()))
        lines+=['',f'## {v} 对 {baseline}',json.dumps(verdicts[v],ensure_ascii=False),
            '通过仅表示当前验证集诊断门槛通过；未完成独立测试集及额外种子复核，不能宣布最终方法。' if all(checks.values()) else '未通过全部预登记门槛；不并入主线，不自动启动N3。']
        # Paired per-image effects for external analysis.
        by_id={p['pair_id']:p for p in b['clean_pairs']}
        pairs=[dict(pair_id=p['pair_id'],group=p['group'],delta_p3=p['flow_p3']-by_id[p['pair_id']]['flow_p3'],
                    delta_epe=p['epe']-by_id[p['pair_id']]['epe']) for p in r['clean_pairs'] if p['flow_p3'] is not None]
        (root/f'{v}_paired_deltas.json').write_text(json.dumps(pairs,indent=2)+'\n')
    lines+=['','## 计算预算','']
    for v in variants:
        s=read(root/v/'status.json');lines.append(f'- {v}：训练前向{s["train_forwards"]}，优化更新{s["optimizer_updates"]}，验证前向{s["validation_forwards"]}，实际墙钟{s["wall_seconds"]:.1f}秒。')
    lines+=['','每组每轮508次训练前向；旧实验每轮318次，因此相对旧U/B/C增加约59.7%前向预算。新增梯度探针各组相同；未用理论FLOPs代替实测，时间及次数分别报告。',
        '','黑/白5%、10%、20%及未见灰色10%结果见各组epoch_030.json的occlusion字段；同一条件同一64对。矩形重采样失败的影像对仍保留，并记录skip原因；不将它们从清洁验证中删除。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');(root/'acceptance.json').write_text(json.dumps(verdicts,indent=2)+'\n')


if __name__=='__main__':
    config=read(Path('mhinet/v2/local_occlusion/config.json'));root=Path(config['output'])
    diagnostic_report(root);comparison_report(root,config)
