"""Export paired evidence and Chinese conclusions without selecting new settings."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mhinet.v2.d16_experiment.train import write

ROOT=Path('outputs/d16_final_round')
OLD=Path('outputs/d16_local_occlusion')


def read(p):return json.loads(p.read_text())


def paired(label,left,right,blocks,output):
    a={r['pair_id']:r for r in left};b={r['pair_id']:r for r in right};assert a.keys()==b.keys()
    rows=[]
    for key in a:
        assert a[key]['points']==b[key]['points']
        rows.append(dict(comparison=label,pair_id=key,block=blocks.get(key,key),points=a[key]['points'],
            **{m:a[key][m]-b[key][m] for m in ('epe','flow_p3','epe_gt32')}))
    path=output/'paired_differences.csv';exists=path.exists()
    with path.open('a') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));
        if not exists:writer.writeheader()
        writer.writerows(rows)
    names=sorted({r['block'] for r in rows});rng=np.random.default_rng(20261023);result={}
    for m in ('epe','flow_p3','epe_gt32'):
        weights=np.array([sum(r['points'] for r in rows if r['block']==g) for g in names])
        numerator=np.array([sum(r[m]*r['points'] for r in rows if r['block']==g) for g in names])
        picks=rng.integers(0,len(names),(1000,len(names)));boot=numerator[picks].sum(1)/weights[picks].sum(1)
        result[m]=dict(difference=float(numerator.sum()/weights.sum()),interval95=np.quantile(boot,[.025,.975]).tolist())
    return dict(blocks=len(names),metrics=result)


def main():
    output=ROOT/'report';output.mkdir(exist_ok=True)
    p=output/'paired_differences.csv'
    if p.exists():p.unlink()
    sel=read(ROOT/'selection.json');blocks={r['pair_id']:r['anchor_block_id'] for r in sel['hz_dev']}
    e={v:read((ROOT/'ablation' if v.startswith('E') else OLD)/v/'epoch_030.json') for v in ('U0','B0','N1','N2','E1','E2')}
    d={v:read(ROOT/'decoder'/f'{v}_seed42'/'epoch_030.json') for v in ('D00','D01','D10','D11')}
    pairs={}
    for v,b in [('E1','B0'),('E2','B0'),('N2','B0'),('N2','E1')]:pairs[v+'-'+b]=paired(v+'-'+b,e[v]['clean_pairs'],e[b]['clean_pairs'],blocks,output)
    for v,b in [('D01','D00'),('D11','D10'),('D10','D00'),('D11','D01')]:
        for domain in ('hangzhou','rgb'):pairs[v+'-'+b+'_'+domain]=paired(v+'-'+b+'_'+domain,d[v][domain]['pairs'],d[b][domain]['pairs'],blocks,output)
    # Difference-in-differences per pair, using a zero reference for the second pairing.
    for domain,models in [('keep_occ',e),('decoder_data',d)]:
        names=('N2','E1','E2','B0') if domain=='keep_occ' else ('D11','D10','D01','D00')
        domains=['hangzhou'] if domain=='keep_occ' else ['hangzhou','rgb']
        for data in domains:
            maps=[{r['pair_id']:r for r in (models[n]['clean_pairs'] if domain=='keep_occ' else models[n][data]['pairs'])} for n in names]
            left=[];right=[]
            for key,r in maps[0].items():
                left.append(dict(pair_id=key,points=r['points'],**{m:maps[0][key][m]-maps[1][key][m]-maps[2][key][m]+maps[3][key][m] for m in ('epe','flow_p3','epe_gt32')}))
                right.append(dict(pair_id=key,points=r['points'],epe=0.,flow_p3=0.,epe_gt32=0.))
            label=domain+'_interaction_'+data;pairs[label]=paired(label,left,right,blocks,output)
    write(output/'paired_intervals.json',pairs)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for v in d:
        folder=ROOT/'decoder'/f'{v}_seed42';epochs=[read(folder/f'epoch_{i:03d}.json') for i in range(31)]
        axes[0].plot(range(31),[100*x['hangzhou']['overall']['flow_p3'] for x in epochs],label=v)
        axes[1].plot(range(31),[100*x['rgb']['overall']['flow_p3'] for x in epochs],label=v)
        axes[2].plot(range(31),[x['rgb']['overall']['epe'] for x in epochs],label=v)
    for ax,title in zip(axes,['Hangzhou reference P@3 (%)','RGB exact P@3 (%)','RGB exact EPE (px)']):ax.set(title=title,xlabel='Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'decoder_curves.jpg',dpi=150);plt.close(fig)
    lines=['# D16 Flow0 最后一轮实验结论','',
        '**开发集裁决：保留U0第30轮的旧MatchEmbeddingDecoder方案；停止本轮局部残差与黑白遮挡扩展。** 独立测试只用于验证边界，不用于重新选模。',
        '','## 正负监督消融','',
        '|模型|杭州P@3|EPE(px)|>32px|人工负例召回|清洁/增强正点误拒绝|','|---|---:|---:|---:|---:|---|']
    for v,r in e.items():
        c=r['clean'];s=read(ROOT/'calibration'/v/'summary.json');a=s['operating']['mixture'];clean=s['operating']['clean']
        lines.append(f'|{v}|{100*c["flow_p3"]:.2f}%|{c["epe"]:.3f}|{100*c["epe_gt32"]:.2f}%|{100*a["negative_recall"]:.2f}%|{100*clean["positive_false_reject"]:.2f}% / {100*a["positive_false_reject"]:.2f}%|')
    lines+=['',
        'E1仅Keep即可达到N2的101.0% EPE改善，P@3还高0.94pp，说明平均EPE下降无需Occ也能实现。但预登记追加条件要求二者P@3绝对差≤0.5pp，实际不满足，因此不追加E_geo/E_cls_mat。不能把Keep内Geo/分类/Mat的贡献继续分开归因。',
        'E2单独Occ使人工负例召回达到81.77%，清洁/增强误拒绝均<1%；Dustbin负例Top1达到69.06%。这否定“共享标量必然不能拒配”的断言。但E2清洁P@3比B0下降2.60pp，超过允许0.5pp，不满足最终保留条件。不得把E2头直接拼接到U0并沿用阈值。',
        '旧N1/N2虽有AUROC改善，却没有满足低误拒绝率工作点；增强Keep与Occ同时训练的结果不等于分别训练的收益相加。交互及配对区间见paired_intervals.json，仅单seed诊断性证据。',
        '','## 固定B0特征的坐标与数据消融','',
        '|组|杭州P@3|杭州EPE|RGB P@3|RGB EPE|','|---|---:|---:|---:|---:|']
    for v,r in d.items():
        h=r['hangzhou']['overall'];s=r['rgb']['overall'];lines.append(f'|{v}|{100*h["flow_p3"]:.2f}%|{h["epe"]:.3f}|{100*s["flow_p3"]:.2f}%|{s["epe"]:.3f}|')
    lines+=['',
        'D01相比D00在RGB上P@3增加22.49pp、EPE由2.547降至1.106px，证明在本预算/旧头配置下，精确合成监督能显著改善合成域连续定位。杭州P@3反降0.43pp，因此没有证实向真实杭州参考一致性的收益。',
        'D10/D11在杭州P@3约47.2%，明显低于旧头约68.4–68.8%。本轮固定3×3＋±8px残差方案不保留。这不能推出所有局部解码器都无效，也不能单凭结果断言D16信息不足；旧头有预训练、新头零残差初始化，起点与曲线已公开。',
        '所有D组特征、候选概率及Matchability完全冻结。排名不变属于实验设计，不能把坐标改善称为跨图候选判别改善。D01/D00测的是等预算合成对替换额外杭州对，不是增加训练量。',
        '','## 条件分支与技术选择','',
        '六组发现性训练均30轮，每组15240次训练前向、1920次更新。未满足条件追加两组；没有开发候选通过门槛，因此按计划不进行seed43/44的四组确认训练，不报告虚构的三seed均值。',
        '最终接口保留512输入、32×32查询、normalized xy及align_corners=False、原flow/target/matchability字段。误差统一原图像素。U0 Matchability保留为分数，但本轮未证明其具备可靠人工遮挡拒配能力，不能把校准阈值标成可部署的遮挡过滤器。',
        'RGB为精确合成H定位；杭州为近似H参考一致性。人工黑白矩形并非自然遮挡真值。不存在额外人工对应点，也不将H称为真实稠密Flow。',
        '','## 独立测试与来源','',
        '原拟选85张GoogleEarth源图的文件名全部出现在早期共享描述子训练/验证清单中，MVT/定位头当时可训练。因此原selection.json保留，最终test_selection.json改为128 FLAIR＋128 WHU，来自原RGB test且与原train/val源身份不交叉。此修改在任何测试推理之前，仅基于来源，不基于测试表现。测试结论不覆盖GE域；外部基础预训练与地理邻近独立性未认证。',
        '杭州测试256对，按历史使用、空间块/base_pair与footprint隔离；校准/工作点各128对因val容量不足使用未参与历史训练的保留原train区域。详细偏差见README、data_audit与test_provenance。']
    tests={v:read(ROOT/'sealed_test'/v/'summary.json') for v in [*e,*d] if (ROOT/'sealed_test'/v/'summary.json').exists()}
    if len(tests)==10:
        test_selection=read(ROOT/'test_selection.json')
        test_blocks={r['pair_id']:r.get('anchor_block_id',r.get('source_relative_path')) for role in ('hz_test','rgb_test') for r in test_selection[role]}
        for v,b in [('B0','U0'),('E1','B0'),('E2','B0'),('N2','B0'),('D01','D00'),('D11','D10'),('D10','D00'),('D11','D01')]:
            for domain in ('hangzhou','rgb'):
                label='test_'+v+'-'+b+'_'+domain
                pairs[label]=paired(label,tests[v][domain]['pairs'],tests[b][domain]['pairs'],test_blocks,output)
        write(output/'paired_intervals.json',pairs)
    lines+=['','|模型|测试杭州P@3|测试杭州EPE|测试RGB P@3|测试RGB EPE|','|---|---:|---:|---:|---:|']
    for v,r in tests.items():
        h=r['hangzhou']['overall'];s=r['rgb']['overall'];lines.append(f'|{v}|{100*h["flow_p3"]:.2f}%|{h["epe"]:.3f}|{100*s["flow_p3"]:.2f}%|{s["epe"]:.3f}|')
    if len(tests)==10:
        lines+=['','独立测试复核：D01−D00在RGB P@3上提升23.51pp（配对源图bootstrap 95%区间21.91～25.16pp），杭州仅+0.12pp（52空间块bootstrap −0.36～+0.58pp）；没有证明杭州稳定收益。D11−D01杭州P@3下降15.49pp（−18.89～−11.80pp）。因此封存数据支持保留旧头并拒绝本轮新局部残差方案，不能把合成域提升推广为真实域提升。']
    lines+=['',f'已完成封存测试：{len(tests)}/10。'+('所有配置保持锁定，未依据测试回头调参。' if len(tests)==10 else '未全部完成前本报告为阶段稿。'),
        '','完整G组、边界、高置信错误、阈值前后精度/覆盖率及逐查询材料在sealed_test各模型目录。推理时延采用同一卡、batch=1、10次预热、30次测量；参数统计包含模型注册但D16路径未执行的金字塔参数，额外坐标头另计，不能混作执行参数量。']
    lines+=['','## 推理开销与训练成本','','|模型|同卡中位数(ms)|P95(ms)|注册参数(M)|额外坐标头(M)|','|---|---:|---:|---:|---:|']
    for v in tests:
        cost=read(ROOT/'sealed_test'/v/'latency.json')
        lines.append(f'|{v}|{cost["median_ms"]:.2f}|{cost["p95_ms"]:.2f}|{cost["registered_parameters"]/1e6:.3f}|{cost["extra_coordinate_parameters"]/1e6:.3f}|')
    lines+=['','B0/N2/E1/E2逐步样本顺序与增强元数据7620步全部一致。每组150次额外autograd.grad梯度探针计入实际时间，未单独计时，不能把全部时间差归因于损失计算。详见budget_comparison.json。',
        '','|新训练组|实际总时间(s)|训练前向|更新|','|---|---:|---:|---:|']
    for v in ('E1','E2','D00','D01','D10','D11'):
        folder=ROOT/'ablation'/v if v.startswith('E') else ROOT/'decoder'/f'{v}_seed42'
        status=read(folder/'status.json')
        lines.append(f'|{v}|{status["wall_seconds"]:.1f}|{status["train_forwards"]}|{status["optimizer_updates"]}|')
    (output/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    if len(tests)==10:
        u=tests['U0'];h=u['hangzhou']['overall'];rgb=u['rgb']['overall']
        one=['# D16 Flow0：最终技术选择（一页）','',
            '**固定U0第30轮，继续使用原MatchEmbeddingDecoder；本轮不采用局部3×3＋残差头，不加入黑白遮挡Occ训练。进入后续模块任务。**','',
            '1. **定位方案**：六组新训练、固定30轮的预登记消融没有产生合格新候选。新残差头在杭州P@3约47%，旧头约68%；不因局部EPE或合成域收益而替换工程基线。',
            '2. **精确监督**：旧头混入RGB512后，开发RGB P@3从72.64%升至95.13%，EPE从2.55降至1.11px；杭州P@3从68.83%降至68.40%。证明本配置的合成域收益，未证明杭州迁移收益。',
            '3. **增强正负归因**：E1只Keep即可取得N2的101%平均EPE改善，不能把该收益归给Occ。E1与N2 P@3相差0.94pp，不满足预登记追加分解条件，Geo与分类/Mat各自贡献未证实。',
            '4. **拒配能力**：E2单独Occ在独立工作点召回81.77%，清洁/增强正点误拒绝0.54%/0.72%；说明共享Dustbin并非必然无效。但清洁P@3比B0下降2.60pp，拒配与定位没有同时达标，因此不部署、不拼头。U0的Matchability只保留分数，无合格遮挡阈值。',
            f'5. **封存测试**：U0杭州P@3={100*h["flow_p3"]:.2f}%，EPE={h["epe"]:.2f}px；独立FLAIR/WHU合成P@3={100*rgb["flow_p3"]:.2f}%，EPE={rgb["epe"]:.2f}px。测试未参与选择。十组测试全量、G组及配对区间均公开；不据测试重新选模。',
            '6. **结论边界**：只有AB间H。杭州是近似H参考一致性，RGB是精确合成定位，黑白矩形是人工负例。不能声称自然遮挡检测、独立人工点像素精度或普适最优。无合格开发候选，按计划跳过多seed确认；单seed结果不冒充三seed结论。','',
            '**后续接口**：输入[B,2,3,512,512]，输出32×32的flow16_target_norm、flow16_norm、flow16_matchability、flow16_valid；align_corners=False，像素坐标=(norm+1)×256−0.5。valid是数值/边界检查，不是遮挡判定。D8/D2/QRRU保持原状。','',
            '**权重**：`outputs/d16_local_occlusion/U0/latest.pt`（epoch30）＋配置指定的冻结DINO；加载入口与坐标协议见final_model.json。运行成本见完整报告。','',
            '**停止条件已触发**：不增加Dustbin网络、遮挡形状或更细特征来挽救本轮负结果。完整对照与来源修正见FINAL_REPORT.md。']
        (output/'ONE_PAGE.md').write_text('\n'.join(one)+'\n')
    write(output/'final_model.json',dict(variant='U0',epoch=30,checkpoint=str(OLD/'U0/latest.pt'),runtime=read(Path('mhinet/v2/local_occlusion/config.json'))['runtime'],
        loader='mhinet.v2.final_round.final_test.load_model("U0")',input_shape=[1,2,3,512,512],query_resolution=[32,32],align_corners=False,
        output_fields={'flow16_target_norm':'B x 2 x 32 x 32 normalized target xy','flow16_norm':'normalized target minus source grid','flow16_matchability':'B x 1 x 32 x 32 score','flow16_valid':'finite in-bounds target and finite probability in [0,1]; NOT a learned occlusion rejection'},
        coordinate_units='normalized xy, pixel=(normalized+1)*256-0.5',error_units='original 512 image pixels',
        validity='Uncalibrated reliability score only; artificial-occlusion deployment gate FAILED',
        diagnostic_threshold=read(ROOT/'calibration/U0/threshold.json')['threshold'],deployment_rejection_threshold=None,
        scope='DINO -> MVT -> D16 Flow0; D8/D2/QRRU unchanged',decision=str(ROOT/'decision.json')))

if __name__=='__main__':main()
