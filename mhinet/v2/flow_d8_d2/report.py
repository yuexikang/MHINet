"""Aggregate frozen-stage evaluation and draw representative development pairs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('reports/flow_d8_d2')

def stats(rows,variant,stage):
    values=[r['flow'] if stage=='flow' else r['variants'][variant][stage] for r in rows]
    total=lambda k:sum(v[k] for v in values)
    return [total('n')/len(rows), total('error_sum')/max(1,total('visible'))]+[100*total(f'correct{t}')/max(1,total('n')) for t in (1,3,5,10)]+[total('correct3')/len(rows)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lines=['# Flow → D8 → D2 冻结能力测试','',
        '新分支：`codex/flow-d8-d2`。原 Flow 第30轮 + 旧杭州 D8/D2 第7轮；不训练、不运行 QRRU。这个组合未联合重训。',
        '所有误差均为512原图像素。P@阈值的分母是全部输出，参考不可见位置计为不正确；EPE仅统计参考可见位置。按点汇总，正确点数为每对均值。',
        '杭州64对使用近似H，仅衡量参考一致性；同源合成35对使用精确H。两者都是已使用的开发数据。',
        'local 为 Flow ±4 D8格搜索，local_mat01 另过滤 Matchability<0.1。阈值未经部署校准。','']
    summaries={}
    for domain in ('hangzhou','synthetic'):
        base=Path('outputs/flow_d8_d2')/domain
        rows=json.loads((base/'results.json').read_text());summaries[domain]={}
        lines += [f'## {domain}：{len(rows)} 对','', '|路径/阶段|点数/对|可见EPE|P@1%|P@3%|P@5%|P@10%|正确点@3/对|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for variant,stage in [('global','flow')]+[(v,s) for v in ('global','local','local_mat01') for s in ('d8','d2')]:
            values=stats(rows,variant,stage);name='Flow0' if stage=='flow' else variant+'/'+stage
            summaries[domain][name]=values
            lines.append('|'+name+'|'+'|'.join(f'{x:.2f}' for x in values)+'|')
        lines += ['','同一批D2源点上的比较（仅保留Flow插值有效的位置；相同采样点与参考可见掩码）：','',
            '|路径|Flow EPE → D2 EPE|Flow P@3 → D2 P@3|配对点/全部D2点|搜索覆盖/全部可见D8查询|D8父窗口正确率|匹配耗时ms/对|','|---|---:|---:|---:|---:|---:|---:|']
        for v in ('global','local','local_mat01'):
            a,b=stats(rows,v,'paired_flow'),stats(rows,v,'paired_d2')
            r=[x['variants'][v] for x in rows]
            coverage=sum(x['search_covered'] for x in r)/max(1,sum(x['visible_queries'] for x in r))
            parent=sum(x['correct_d8_parent'] for x in r)/max(1,sum(x['d8']['n'] for x in r))
            fraction=sum(x['paired_d2']['n'] for x in r)/max(1,sum(x['d2']['n'] for x in r))
            lines.append(f'|{v}|{a[1]:.2f} → {b[1]:.2f}|{a[3]:.2f}% → {b[3]:.2f}%|{fraction:.1%}|{coverage:.1%}|{parent:.1%}|{np.mean([x["match_ms"] for x in r]):.1f}|')
        lines += ['',f'共享特征前向平均耗时：{np.mean([r["feature_ms"] for r in rows]):.1f} ms/对（不含文件读取）。','',
            '|分组|Flow P@3|global D2 P@3|local D2 P@3|local_mat01 D2 P@3|','|---|---:|---:|---:|---:|']
        for group in sorted({r['group'] for r in rows}):
            subset=[r for r in rows if r['group']==group]
            p=[stats(subset,'global','flow')[3]]+[stats(subset,v,'d2')[3] for v in ('global','local','local_mat01')]
            lines.append('|'+group+'|'+'|'.join(f'{x:.2f}%' for x in p)+'|')
        # Fixed first example per group; no selection by success.
        for group in sorted({r['group'] for r in rows}):
            i=next(i for i,r in enumerate(rows) if r['group']==group)
            data=np.load(base/f'{i:03d}.npz');fig,axes=plt.subplots(1,4,figsize=(16,4))
            axes[0].imshow(data['image_a'].transpose(1,2,0));axes[0].set_title('Image A')
            for ax,v in zip(axes[1:],('global','local','local_mat01')):
                ax.imshow(data['image_b'].transpose(1,2,0))
                error=np.where(data[v+'_visible'],data[v+'_error'],32)
                points=data[v+'_points_b'];sc=ax.scatter(points[:,0],points[:,1],c=error,s=3,vmin=0,vmax=16,cmap='turbo')
                ax.set_title(v+' D2 endpoint error')
            for ax in axes:ax.set_xlim(-.5,511.5);ax.set_ylim(511.5,-.5);ax.axis('off')
            fig.colorbar(sc,ax=list(axes[1:]),label='Reference error (px)',fraction=.015)
            fig.suptitle(rows[i]['pair_id']);name=f'{domain}_{group}.png';fig.savefig(OUT/name,dpi=130,bbox_inches='tight');plt.close(fig)
            lines += ['',f'首个{group}样本（按数据顺序，未筛选成功案例；不可见点按32px着色）：',f'![{domain} {group}]({name})']
        lines += ['']
    lines += ['', '## 结论', '', '精确合成对验证了局部链路的细定位能力：同点比较，Flow EPE 2.14 → D2 0.85 px，P@3 81.35% → 98.90%。杭州跨源同点 P@3 则为69.25% → 64.16%，尚不能认定现有D8/D2带来定位收益；近似H也无法区分真实地形偏差与预测错误。', '', '局部与全局输出接近。局部候选计算减少，但当前实测耗时没有明显收益，不声称加速。Matchability阈值0.1仅小幅提升最终精度，却将可见查询的搜索覆盖从杭州95.1%降至92.1%、合成99.4%降至94.1%，本轮不建议默认开启。', '', '下一步应固定原Flow，针对真实跨源数据适配D8/D2，再比较同点指标；暂不接QRRU。当前结果不能用于断言整个粗到细架构失效。', '', '验证：全局路径与现有refine=False输出一致；稀疏局部dual-softmax与掩码稠密实现数值一致；QRRU调用保护在全部99对上未触发。逐对统计保存在同目录的两个JSON文件中。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
