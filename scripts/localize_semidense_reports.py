"""Independent Chinese report renderer; never edits training code or model state.

Use --watch while an existing English-exporting trainer is running. Localized
pages live under visualizations/zh and refer to the original diagnostic images.
"""
import argparse
import html
import json
import os
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
CSS='body{font:16px "Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:24px;color:#243047;background:#f6f7f9}img{max-width:100%;background:white}a{color:#1465a8;padding:6px;display:inline-block}summary{font-weight:600;padding:10px;cursor:pointer}section,details{background:white;padding:12px;border-radius:10px;margin-bottom:14px}.note{color:#5b6472;font-size:14px}'


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text,encoding='utf-8');os.replace(temp,path)


def document(title,body,refresh=False):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'+('<meta http-equiv="refresh" content="120">' if refresh else '')+f'<title>{html.escape(title)}</title><style>{CSS}</style><h1>{html.escape(title)}</h1>'+body+'</html>'


def read_json(path):return json.loads(path.read_text())


def figure_title(stem):
    if stem.startswith('features_'):
        fields=stem.split('_');scale=fields[1]
        if fields[-1]=='summary':return f'{scale} 特征概览：固定基 PCA、原始范数、通道标准差'
        return f'{scale} · 图像 {fields[2]} 的特征通道，使用固定色阶'
    if stem.startswith('fine_'):return f'D2 fine 局部窗口 {int(stem.split("_")[-1])+1}：相似度、选点与 H0 残差先验'
    if stem.startswith('qrru_'):return f'QRRU 查询 {int(stem.split("_")[-1])+1}：逐轮坐标、控制流与更新门控'
    return {'stages_matches':'各阶段对应点：D8 coarse → D2 fine → QRRU',
            'h0_support':'H0 预测的双侧重叠支持区域','coarse_cosine':'D8 coarse：固定查询点的 cosine 相似度',
            'coarse_dual_probability':'D8 coarse：dual-softmax 匹配概率（对数色阶）'}.get(stem,stem)


def caption(stem):
    if stem.startswith('features_'):
        return 'PCA 使用第0步的共同投影基；raw norm 表示原始特征范数，channel std 表示通道标准差。channels 为通道编号，fixed scale 为固定色阶。'
    if stem.startswith('fine_'):
        return 'cosine 为相似度；dual probability 为 dual-softmax 概率。绿色＋为 GT，青色×为预测选择。灰色为仅使用 H0 的投影，橙色为 H0＋coarse 残差。右下图把同一组概率显示在实际图像坐标中。'
    if stem.startswith('qrru_'):
        return '上排：第0～4轮的位置与 EPE，单位为 D2 像素。下排：更新门控（0～1）与四象限控制位移。初始窗口为规则网格，不输入 H0；门控不是匹配置信度。'
    if stem.startswith('coarse_'):
        return '绿色＋为 GT，青色×为预测。selected＝已选中；excluded_support＝支持区外；not_mutual＝未通过双向互选；below_threshold＝低于阈值；capped＝超出数量上限。log 表示对数色阶。'
    if stem=='stages_matches':return '显示所有点的位置，并按索引均匀抽取最多100条连线。coarse 为粗匹配，fine 为细匹配，final 为 QRRU 细化后的结果。'
    return '绿色覆盖区域表示 H0 预测的有效支持范围。'


def curves(run,zh):
    path=run/'train.jsonl';rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    if rows:
        steps=[r['step'] for r in rows];fig,axes=plt.subplots(2,3,figsize=(16,9))
        def plot(ax,values,label):
            ax.plot(steps,values,alpha=.22,lw=.6)
            w=min(20,len(values));ax.plot(steps[w-1:],np.convolve(values,np.ones(w)/w,'valid'),label=label,lw=1.2)
        config=read_json(run/'run.json').get('matcher',{})
        plot(axes[0,0],[np.mean([sum(config.get('lambda_'+k,1)*x['l'+k] for k in ('c','f','q')) for x in r['records']]) for r in rows],'总 loss')
        for k in ('lc','lf','lq'):plot(axes[0,0],[np.mean([x[k] for x in r['records']]) for r in rows],k)
        for k,label in [('q_control','控制流 loss'),('q_center','中心坐标 loss'),('fine_coverage','fine 监督覆盖率')]:
            plot(axes[0,1],[np.mean([x[k] for x in r['records']]) for r in rows],label)
        for k,label in [('cgmdp','CGMDP'),('semidense_qrru','匹配温度＋QRRU')]:plot(axes[0,2],[r['gradient_groups'][k] for r in rows],label)
        for k,label in [('lr_shared','CGMDP 学习率'),('lr_head','下游学习率')]:plot(axes[1,0],[r[k] for r in rows],label)
        for k,label in [('tau_c','coarse 温度'),('tau_f','fine 温度')]:plot(axes[1,1],[r[k] for r in rows],label)
        plot(axes[1,2],[r['seconds'] for r in rows],'每步耗时（秒）')
        for ax,title in zip(axes.flat,['GT 解耦监督的训练 loss','QRRU 分项 loss 与有效监督','裁剪前的梯度范数','学习率','可学习温度','训练耗时']):
            ax.set_title(title);ax.set_xlabel('优化步数');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(zh/'training_curves.png',dpi=110);plt.close(fig)
    reports=[read_json(x) for x in sorted((run/'visualizations').glob('step_*/summary.json'))]
    if reports:
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        for stage,label in [('coarse','coarse 粗匹配'),('fine','fine 细匹配'),('final','QRRU 最终结果')]:
            for ax,k in zip(axes,['precision_1px','epe','matches']):
                ax.plot([r['step'] for r in reports],[r['stages'].get(stage,{}).get(k,np.nan) for r in reports],marker='.',label=label)
        for ax,title in zip(axes,['固定验证样本：precision@1px','重叠区域 EPE（原图目标像素）','输出对应点数']):
            ax.set_title(title);ax.set_xlabel('优化步数');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(zh/'cascade_curves.png',dpi=110);plt.close(fig)
    return rows[-1]['step'] if rows else 0


def localize_run(run):
    visual=run/'visualizations';zh=visual/'zh';zh.mkdir(exist_ok=True)
    history=[];page_count=0
    for step in sorted(visual.glob('step_*'),reverse=True):
        links=[]
        for pair in sorted(step.glob('pair_*')):
            if not (pair/'trace.json').exists():continue
            dest=zh/step.name/pair.name/'index.html'
            assets=sorted(list(pair.glob('*.png'))+list(pair.glob('*.gif')))
            freshness=max([p.stat().st_mtime for p in assets]+[(pair/'trace.json').stat().st_mtime])
            if not dest.exists() or dest.stat().st_mtime<freshness:
                trace=read_json(pair/'trace.json');sections=[]
                for image in assets:
                    label=figure_title(image.stem)+('（动画）' if image.suffix=='.gif' else '')
                    url=f'../../../{step.name}/{pair.name}/{image.name}'
                    sections.append(f'<details open><summary>{html.escape(label)}</summary><p class="note">{html.escape(caption(image.stem))}</p><a href="{url}"><img loading="lazy" src="{url}"></a></details>')
                body=f'<p><a href="../../index.html">返回训练总览</a> · <a href="../../../{step.name}/{pair.name}/trace.json">原始记录与坐标单位</a></p>'
                body+='<p class="note">历史 PNG 保留原始图内标注；每张图上方提供中文标题与图注。GT、EPE、PCA、D8、D2、QRRU 等关键词保留。</p>'
                body+=''.join(sections)
                write(dest,document(f'验证样本 {int(pair.name.split("_")[-1])+1} · {trace["pair_id"]}',body))
            links.append(f'<a href="{step.name}/{pair.name}/index.html">样本 {int(pair.name.split("_")[-1])+1}</a>');page_count+=1
        history.append(f'<details open><summary>第 {int(step.name.split("_")[-1])} 步</summary>'+''.join(links)+'</details>')
    step=curves(run,zh)
    body=f'<p>当前已记录第 <strong>{step}</strong> 步。固定验证样本用于纵向比较；GT 仅用于监督评估和图中对照。页面每120秒刷新。</p>'
    body+='<p class="note">训练 loss 来自 GT 解耦监督；下方级联指标来自固定验证子集，不能代替完整验证集结果。淡线为原始值，实线为20步滑动平均。</p>'
    body+=''.join(f'<section><h2>{label}</h2><img src="{name}?v={time.time_ns()}"></section>' for name,label in [('training_curves.png','训练曲线'),('cascade_curves.png','纯预测级联的匹配结果')] if (zh/name).exists())
    write(zh/'index.html',document('半密集匹配训练总览',body+''.join(history),True))
    return dict(run=run.name,step=step,pages=page_count)


def dashboard():
    cards=[]
    for name,gpu in [('a',1),('b',2),('c',1)]:
        run=f'semidense_stable_v2_tier1_lr_{name}_seed0'
        config=read_json(ROOT/f'configs/semidense_tier1_lr_{name}.json')
        report=f'{run}/visualizations/zh/index.html'
        cards.append(f'<section><h2>{name.upper()} 组 · GPU {gpu}</h2><p>CGMDP 学习率：{config["shared_lr"]:g}<br>下游学习率：{config["head_lr"]:g}</p><a href="{report}">打开详细可视化</a> · <a href="{run}.console.log">查看训练日志</a><iframe title="{name.upper()}组训练报告" src="{report}"></iframe></section>')
    body='<p>第三档预训练权重 → 第一档半密集下游训练。各组使用相同初始化、随机种子、数据顺序和验证样本；冻结 DINO、MVT 与 H0 预测分支，每组训练12995步。</p>'
    body+='<style>main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}iframe{width:100%;height:850px;border:0}</style><main>'+''.join(cards)+'</main>'
    write(ROOT/'outputs/semidense_lr_comparison.html',document('半密集下游学习率对照',body,True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=120)
    args=parser.parse_args()
    font=next((p for p in (Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path.home()/'.local/share/fonts/NotoSansCJKsc-Regular.otf') if p.exists()),Path('/missing-chinese-font'))
    if not font.exists():raise FileNotFoundError('Install fonts-noto-cjk for Chinese scientific plots')
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    cache={}
    while True:
        for name in 'abc':
            run=ROOT/f'outputs/semidense_stable_v2_tier1_lr_{name}_seed0'
            if not (run/'run.json').exists():continue
            watched=[run/'train.jsonl']+list((run/'visualizations').glob('step_*/pairs.json'))+list((run/'visualizations').glob('step_*/summary.json'))
            signature=tuple((str(p),p.stat().st_mtime_ns) for p in watched if p.exists())
            if cache.get(name)!=signature:
                try:print(json.dumps(localize_run(run),ensure_ascii=False),flush=True);cache[name]=signature
                except (json.JSONDecodeError,FileNotFoundError):
                    if not args.watch:raise
        dashboard()
        if not args.watch:break
        complete=True
        for name in 'abc':
            run=ROOT/f'outputs/semidense_stable_v2_tier1_lr_{name}_seed0'
            if not (run/'run.json').exists():complete=False;continue
            total=read_json(run/'run.json')['total_steps']
            if not (run/'visualizations'/f'step_{total:07d}'/'summary.json').exists():complete=False
        if complete:break
        time.sleep(max(10,args.interval))


if __name__=='__main__':main()
