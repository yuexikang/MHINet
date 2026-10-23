"""Summarize the fixed, independent calibration/operating-set diagnosis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/d16_final_round')


def main():
    lines=['# A阶段：独立工作点与Dustbin诊断','',
        '阈值只在校准128对上确定，工作点评价使用另外128对。每组一个阈值，对清洁与增强未遮挡正查询分别施加≤1%校准误拒绝约束。校准约束不保证独立评价误拒绝仍≤1%。',
        '','|模型|阈值|评价清洁正点误拒绝|评价增强正点误拒绝|人工负例召回|召回95%块bootstrap区间|通过|',
        '|---|---:|---:|---:|---:|---|---|']
    reports={}
    for variant in ('U0','B0','N1','N2','E1','E2','E_geo','E_cls_mat'):
        folder=ROOT/'calibration'/variant
        if not (folder/'summary.json').exists():continue
        r=json.loads((folder/'summary.json').read_text());reports[variant]=r
        clean=r['operating']['clean'];occ=r['operating']['mixture'];lo,hi=occ['bootstrap']['interval95']['negative_recall']
        lines.append(f'|{variant}|{r["threshold"]:.6f}|{100*clean["positive_false_reject"]:.2f}%|{100*occ["positive_false_reject"]:.2f}%|{100*occ["negative_recall"]:.2f}%|{100*lo:.2f}%–{100*hi:.2f}%|{r["passed"]}|')
    lines+=['','门槛为评价负例召回≥50%，两类正点误拒绝各≤1%。旧四组均未通过，因此AUROC提升不能替代有效工作点。所有统计仍以H确定的可见性和人工遮挡为参考，不等于真实遮挡检测。',
        '','## Dustbin为何不赢','',
        '|模型|初始z|15轮z|30轮z|负例平均P_dust|负例P95|平均dustbin−最大空间logit|最大margin|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,r in reports.items():
        z=r['no_match_logit'];p=r['negative_dustbin_probability'];m=r['negative_margin']
        lines.append(f'|{v}|{z["initial"]:.5f}|{z["middle"]:.5f}|{z["final"]:.5f}|{p["mean"]:.7f}|{p["quantiles"]["p95"]:.7f}|{m["mean"]:.3f}|{m["quantiles"]["max"]:.3f}|')
    lines+=['','N1/N2的标量确实更新，负例dustbin概率也提升，但0.05量级的z变化远小于数个logit单位的空间峰值优势。空间候选仍然更高。这支持“学到了变化但未达到拒配工作点”，而非“参数完全冻结/没有梯度”。不能只凭系数推断训练强弱。',
        '','## 固定第30轮权重、32对训练诊断样本的加权梯度','',
        '下表是反向重放，不调用optimizer.step；未激活的项也计算为反事实诊断。U0/B0实际只有clean；N1/N2实际包含全部项；E1仅clean与Keep三项，E2仅clean与Occ。历史训练时的首批梯度记录另存historical_gradients.json，不能与重放值混称。',
        '|模型|项|MVT L2均值|坐标解码器L2均值|z梯度均值（带符号）|','|---|---|---:|---:|---:|']
    for v in reports:
        folder=ROOT/'calibration'/v
        g=json.loads((folder/'gradients.json').read_text())['pairs']
        for term in ('clean','occ','keep','keep_geo','keep_match'):
            a=[r['gradient_norms'][term] for r in g]
            lines.append(f'|{v}|{term}|{np.mean([r["mvt"] for r in a]):.6f}|{np.mean([r["coordinate_decoder"] for r in a]):.6f}|{np.mean([r["no_match_logit_signed"] for r in a]):.6f}|')
        arrays=[]
        for path in sorted(folder.glob('operating_*_mixture.npz')):
            with np.load(path) as a:arrays.append({k:a[k] for k in ('positive','negative','dustbin_margin','dustbin')})
        neg_margin=np.concatenate([a['dustbin_margin'][a['negative']] for a in arrays]);pos_margin=np.concatenate([a['dustbin_margin'][a['positive']] for a in arrays]);neg_probability=np.concatenate([a['dustbin'][a['negative']] for a in arrays])
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        ax[0].hist(neg_margin,bins=50,alpha=.6,density=True,label='Negative');ax[0].hist(pos_margin,bins=50,alpha=.5,density=True,label='Positive');ax[0].axvline(0,color='black');ax[0].legend();ax[0].set(xlabel='Dustbin logit - max spatial logit',title=v)
        ax[1].hist(np.log10(neg_probability.clip(1e-30)),bins=50);ax[1].set(xlabel='log10 negative dustbin probability',ylabel='Queries')
        fig.tight_layout();fig.savefig(folder/'dustbin_distribution.jpg',dpi=120);plt.close(fig)
    lines+=['','## 曲线与完整材料','']
    for v in reports:lines += [f'- {v}：[工作点曲线](calibration/{v}/operating_curves.jpg)，[Dustbin分布](calibration/{v}/dustbin_distribution.jpg)，[完整逐对与分层指标](calibration/{v}/summary.json)。']
    lines+=['','## 数据隔离说明','',
        '历史415个杭州样本占据168个空间块，原val清除已用块及footprint交叉后仅75对。新增校准128对取原train113＋val15，工作点128对取原train121＋val7；均未参与历史训练/选模，两个新集合各29块且互不重叠。此处原train是原始数据划分名，新保留的这些区域不进入本轮训练；训练仍是固定254对。',
        '','最终测试与本阶段工作点评价独立；其完成状态和结果见sealed_test及最终报告。完整抽样规则和H约定检查见data_audit.json、geometry_checks.json，RGB测试来源修正见test_provenance.json。']
    (ROOT/'A_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
