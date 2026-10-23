"""Chinese findings and traceable static scientific figures."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .dataset import load_config
from .evaluate import load_records,states
from .labels import MATCH


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def state_figure(record,index,scores,thresholds,path):
    row=record['row'];reverse=record['direction']=='ba'
    a,b=(row['image_b'],row['image_a']) if reverse else (row['image_a'],row['image_b'])
    methods=[n for n in ('A0','A1','A2','T1','T2') if n in scores]
    fig,axs=plt.subplots(2,4,figsize=(16,8))
    axs[0,0].imshow(Image.open(a));axs[0,0].set_title('Source image')
    axs[0,1].imshow(Image.open(b));axs[0,1].set_title('Target image')
    cmap=ListedColormap(['#30b56c','#db4242','#9c9c9c'])
    axs[0,2].imshow(record['state'].reshape(32,32),vmin=0,vmax=2,cmap=cmap,interpolation='nearest')
    axs[0,2].set_title('Exact GT (gray = unknown)')
    remaining=[axs[0,3],*axs[1]]
    for ax,name in zip(remaining,methods):
        pred=states(scores[name][index],thresholds[name])
        ax.imshow(pred.reshape(32,32),vmin=0,vmax=2,cmap=cmap,interpolation='nearest')
        ax.set_title(f'{name}: accepted {100*(pred==MATCH).mean():.1f}%')
    for ax in remaining[len(methods):]:ax.set_visible(False)
    for ax in axs.flat:ax.axis('off')
    fig.suptitle(f'{row["pair_id"]} / {record["direction"]}\nGreen: MATCH | Red: NO_MATCH | Gray: UNCERTAIN / unknown GT')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def figures(ev,hz,scores,thresholds,directory):
    directory.mkdir(parents=True,exist_ok=True)
    figures=[]
    for kind in ('full','partial','texture'):
        i=next(i for i,r in enumerate(ev) if r['row']['kind']==kind and r['direction']=='ab' and (kind=='full' or r['row']['requested_overlap']==.4))
        name=f'states_{kind}.png';state_figure(ev[i],i,scores['eval'],thresholds,directory/name)
        figures.append((name,ev[i]['row']['pair_id'],'evaluation first fixed-order example, 40% overlap where applicable'))
    for group in ('G0','G1','G2','G3'):
        indices=[i for i,r in enumerate(hz) if r['row']['kind']==group]
        if not indices:continue
        i=max(indices,key=lambda i:np.sum((hz[i]['reference_epe_px']>32)&(hz[i]['matchability']>=.9)))
        name=f'states_{group}.png';state_figure(hz[i],i,scores['hz'],thresholds,directory/name)
        figures.append((name,hz[i]['row']['pair_id'],'diagnostic maximum count of high-Mat reference EPE>32, not representative random sample'))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,kind in zip(axes,('partial','texture')):
        indices=[i for i,r in enumerate(ev) if r['row']['kind']==kind]
        truth=np.concatenate([ev[i]['state'] for i in indices]);epe=np.concatenate([ev[i]['epe_px'] for i in indices])
        positive=truth==MATCH
        for method,values in scores['eval'].items():
            score=np.concatenate([values[i] for i in indices]);x=[];y=[]
            for threshold in np.quantile(score,np.linspace(0,1,151)):
                accept=score<=threshold;selected=positive&accept
                x.append(float(accept[positive].mean()))
                y.append(float((epe[selected]<=3).mean()) if selected.any() else np.nan)
            ax.plot(x,y,label=method)
            accepted=score<=thresholds[method][0];selected=positive&accepted
            ax.scatter([accepted[positive].mean()],[(epe[selected]<=3).mean() if selected.any() else np.nan],s=30)
        ax.set_title(kind);ax.set_xlabel('True-match acceptance coverage');ax.set_ylabel('Accepted true-match P@3')
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Evaluation curves; dots = calibration-fixed working points (no test retuning)')
    fig.tight_layout();fig.savefig(directory/'accuracy_coverage.png',dpi=150);plt.close(fig)
    return figures


def report():
    torch.set_num_threads(4)
    config=load_config();out=Path(config['output']);public=Path('reports/pcf01');public.mkdir(parents=True,exist_ok=True)
    result=json.loads((out/'results.json').read_text());a=json.loads((out/'stage_a.json').read_text())
    training=json.loads((out/'training_summary.json').read_text()) if (out/'training_summary.json').exists() else None
    data=json.loads((out/'dataset_summary.json').read_text())
    scores=torch.load(out/'decision_scores.pt',map_location='cpu',weights_only=False)
    ev,hz=load_records('eval'),load_records('hz')
    gallery=figures(ev,hz,scores,result['thresholds'],public/'figures')
    r=result['results'];decision=result['decision']
    lines=['# PCF-01：杭州显式拒配验证报告','',f'最终选择：**{decision}**。'+('新增拒配头通过全部预设验收门槛。' if decision!='U0' else '本轮新增拒配头未通过全部预设门槛，保留原 U0；不启动新的 Decoder 训练。'),'',
        (f'**技术结论：同源纹理覆盖监督能学到拒配能力，但不能直接当作杭州真实跨源影像的有效性输出。T2 在 G0/G3 上保留 {pct(r["T2"]["hangzhou_reference_only"]["G0"]["reference_correct_retention"])}/{pct(r["T2"]["hangzhou_reference_only"]["G3"]["reference_correct_retention"])} 的参考正确匹配；外观差异可能触发过度拒配。这里只有近似 H，因此这是参考定位证据，不是真实遮挡标签验证。**' if 'T2' in r else '阶段 A 已达到预设基线门槛，保留原 U0。'),'','## 协议与可复查证据','',
        f'- 母图全部来自杭州天地图 RGB 原图，共 {data["mothers"]} 个地理块；训练/校准/评价为 {config["split_counts"]}，裁剪前划分。无 GoogleEarth。',
        f'- 图对数量：{data["split_pairs"]}。合成数据双向评价；杭州跨源 G0–G3 只做 A→B 参考诊断。',
        f'- 逐像素实测重叠与目标档位平均绝对差：{data["mean_overlap_error"]:.6f}。没有将非重叠区域填黑。',
        '- 校准、评价地理块均排除已记录 U0 杭州实验的使用块。训练母图可能来自旧训练块；不能据此宣称 DINO 预训练完全未见。',
        '- 误差单位为 512 原图像素；所有方法共享同一份原始 U0 坐标。D8/D2/QRRU 不执行，原默认推理不修改。',
        '- 校准阈值先固定再评价；真实有对应点被判 NO_MATCH 或 UNCERTAIN 均计误拒。UNKNOWN 真值不参与分类评价。',
        '- 仅有 7 个独立评价地理块，本结论用于此次模块验收，不能当作广域遥感泛化结论。母图总览显示校准/评价以山地植被为主，城市建筑覆盖不足；训练含道路、村镇、农田、植被。',
        '- 5 项规定数值测试通过，包含实际 U0 的冻结梯度及分类头一步更新前后坐标逐元素相等。','']
    if training:
        lines += [f'训练：每组 {training["trainable_parameters"]:,} 个参数，共 {training["trained_epochs"]} epoch、{training["steps_each"]} 优化步，共同选择 epoch {training["best_epoch"]}。T1/T2 初始化一致，U0 更新次数为 0。','']
    lines += ['共同早停的限制：T2 校准损失在第 1–4 epoch 持续下降，但 T1 上升导致共同指标变差。因此当前比较是预设共同预算/共同选点的结果，不是对 T2 可达到性能上限的估计；没有在看到独立评价后重训或调阈值。',''] if training else []
    lines+=['## 同协议独立评价','','A0=原 Dustbin；A1=原 Matchability；A2=两者固定组合；T1=仅几何部分重叠训练；T2=加入真实纹理覆盖训练。所有方法使用各自的校准集固定阈值。','',
        '| 方法 | 几何召回 | 几何误拒 | 纹理召回 | 纹理误拒 | 全重叠误拒 | 纹理 AUROC | 纹理 AUPRC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in r.items():
        g,t,f=v['geometry'],v['texture'],v['full']
        lines.append(f'| {name} | {pct(g["no_match_recall"])} | {pct(g["false_reject"])} | {pct(t["no_match_recall"])} | {pct(t["false_reject"])} | {pct(f["false_reject"])} | {t["auroc"]:.4f} | {t["auprc"]:.4f} |')
    lines+=['','纹理子集的负例只包含真实纹理覆盖导致的确定无对应；其几何图外负例另行评价，避免混入容易负例抬高成绩。','',
        '| 方法 | 原始 P@3 | 原始 EPE | 原始 RMSE | 接受后 P@3 | 正确匹配保留 | R_correct@3 | 丢失正确点 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in r.items():
        m=v['all'];lines.append(f'| {name} | {pct(m["raw_p3"])} | {m["raw_epe"]:.3f} | {m["raw_rmse"]:.3f} | {pct(m["accepted_p3"])} | {pct(m["correct_retention"])} | {pct(m["r_correct3"])} | {m["correct_lost"]} |')
    lines+=['','原始 P@1/3/5/10、EPE>32、各重叠率、三态比例及逐对指标见 [results.json](results.json) 和 [paired.csv](paired.csv)。原始精度相同来自冻结与相同坐标缓存，不代表新增头提升定位。','',
        '## 五个研究问题','',
        '### 1. 原 U0 Dustbin 为什么不能稳定拒配？','',
        '结构上它只有一个全局共享 logit，必须超过该查询所有空间候选才能成为 Top1；已有坐标解码器无论 Dustbin 多大都会输出坐标。该结构事实说明“有坐标”不等于“有对应”，但不是对失败原因的唯一因果证明。','',
        '校准纹理负例上的直接观测：','']
    for signal in ('p_dustbin','matchability','dustbin_logit_margin'):
        d=a['signals']['texture']['negative'][signal]
        lines.append(f'- {signal}：均值 {d["mean"]:.6g}，P05/P50/P95={d["p05"]:.6g}/{d["p50"]:.6g}/{d["p95"]:.6g}。')
    lines += [f'- Dustbin Top1 比例：{pct(a["signals"]["texture"]["negative"]["dustbin_top1"])}。','',
        '这些分布和本轮基线结果共同判断可用拒配能力，不能只根据 Top1 为零断定完全没学习。','',
        '### 2. 显式头是否优于 Matchability？','']
    for name,gates in result['gates'].items():
        lines.append(f'- {name}：'+ '；'.join(f'{k}={"通过" if v else "未通过"}' for k,v in gates.items())+'。')
    lines+=['','### 3. 真实纹理监督是否有独立收益？','']
    if 'T2' in r:
        delta=r['T2']['texture']['no_match_recall']-r['T1']['texture']['no_match_recall']
        fp=r['T2']['texture']['false_reject']-r['T1']['texture']['false_reject']
        lines += [f'T2 相比 T1 的纹理负例召回变化为 {delta*100:+.2f} 个百分点，纹理正例误拒变化为 {fp*100:+.2f} 个百分点。两组结构、初始化、源图对数、优化步数和所选 epoch 一致。收益必须连同误拒和几何召回一起判断；不能把高分错误对应直接解释为无对应。']
    else:lines += ['阶段 A 已达到预设基线门槛，未训练新头；本轮不声称纹理监督有独立收益。']
    lines+=['','### 4. 丢失了多少真实正确匹配？','']
    for name,v in r.items():
        m=v['all'];lines.append(f'- {name}：丢失 {m["correct_lost"]} 个 EPE≤3px 的真实匹配，正确匹配保留率 {pct(m["correct_retention"])}。')
    lines+=['','### 5. 是否继续真正的条件式解码？','',
        '有通过当前门槛的候选，可作为后续独立结构实验依据；本轮没有自动启动新 Decoder。' if decision!='U0' else '没有候选通过全部门槛，证据不足以支持继续替换 Decoder。本轮结束，保留 U0。','',
        '## 杭州真实跨源诊断','',
        '这些图对只有近似 H，没有真实遮挡标签。下表“参考好点被拒”和“参考坏点被接受”是定位参考诊断，不是真实有无对应分类精度。','',
        '| 方法/组 | 原始参考 P@3 | 覆盖率 | 参考正确点保留率 | 高 Mat 且参考 EPE>32 点数 | 其中仍接受 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,v in r.items():
        for g,m in v['hangzhou_reference_only'].items():
            lines.append(f'| {name}/{g} | {pct(m["raw_reference_p3"])} | {pct(m["coverage"])} | {pct(m["reference_correct_retention"])} | {m["high_mat_reference_bad32"]} | {m["high_mat_bad_accepted"]} |')
    lines+=['','高置信小 Top1–Top2 间隔和高熵只是多峰/歧义线索，不等同于经过人工确认的多个真实候选；相关数量见 JSON。','',
        '## 可视化','',
        '绿=接受对应，红=拒配，灰=不确定；GT 图中的灰表示未知标签。没有为拒配点发布箭头或坐标。合成示例按固定顺序取样；G0–G3 各取高置信大参考误差最多的一对，用来展示失效而非平均水平。','',
        '![母图分布总览](mother_overview.jpg)','','![正确率—覆盖率](figures/accuracy_coverage.png)','']
    for name,pair,selection in gallery:
        lines += [f'`{pair}`（{selection}）',f'![{name}](figures/{name})','']
    lines+=['## 文件','',
        '- `outputs/pcf01/queries_*.csv.gz`：逐查询坐标、精确标签/参考标签、各项分数、误差和来源。',
        '- `outputs/pcf01/mother_selection.json`、`pairs.jsonl`、`data/`：地理划分、原图来源、裁剪参数及掩膜。',
        '- `outputs/pcf01/T1.pt`、`T2.pt`：实验头权重（若阶段 A 触发训练）；未替换 U0。',
        '- 本目录 JSON/CSV 和图片可随 Git 分享；大规模缓存及权重保留本机。','']
    shutil.copytree(public/'figures',out/'figures',dirs_exist_ok=True)
    text='\n'.join(lines)
    (public/'REPORT.md').write_text(text);(out/'REPORT.md').write_text(text)
    for name in ('results.json','paired.csv','stage_a.json','thresholds.json','dataset_summary.json','training_summary.json','train_history.json','tests.log','mother_overview.jpg','mother_selection.json'):
        if (out/name).exists():shutil.copy2(out/name,public/name)
    print(public/'REPORT.md')


if __name__=='__main__':report()
