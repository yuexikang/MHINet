"""Publish compact review evidence; keep checkpoints/query arrays in experiment outputs."""
import gzip,json,shutil
from pathlib import Path
from .report_final import ROOT,OLD,read


def main():
    names=['U0','B0','N1','N2','E1','E2','D00','D01','D10','D11']
    if not all((ROOT/'sealed_test'/v/'summary.json').exists() for v in names):
        raise RuntimeError('Final tests are not all complete.')
    dest=Path('reports/d16_final_round');dest.mkdir(exist_ok=True)
    for name in ('FINAL_REPORT.md','ONE_PAGE.md','final_model.json','paired_intervals.json','decoder_curves.jpg'):
        shutil.copy2(ROOT/'report'/name,dest/name)
    for name in ('decision.json','keep_gate.json','data_audit.json','geometry_checks.json','test_provenance.json','budget_comparison.json','A_DIAGNOSIS.md'):
        shutil.copy2(ROOT/name,dest/name)
    for path in [ROOT/'selection.json',ROOT/'test_selection.json',ROOT/'report/paired_differences.csv']:
        with path.open('rb') as src,gzip.open(dest/(path.name+'.gz'),'wb') as out:shutil.copyfileobj(src,out)
    for v in names:
        folder=dest/v;folder.mkdir(exist_ok=True)
        source=ROOT/'decoder'/f'{v}_seed42' if v.startswith('D') else (ROOT/'ablation'/v if v.startswith('E') else OLD/v)
        for name in ('config.json','status.json','initialization.json','epoch_000.json','epoch_015.json','epoch_030.json','epochs.csv'):
            if (source/name).exists():shutil.copy2(source/name,folder/('development_'+name))
        test=ROOT/'sealed_test'/v
        for name in ('summary.json','latency.json','paired.csv'):
            shutil.copy2(test/name,folder/('test_'+name))
        for path in test.glob('*.jpg'):shutil.copy2(path,folder/('test_'+path.name))
        cal=ROOT/'calibration'/v
        if cal.exists():
            for name in ('summary.json','threshold.json','gradients.json','historical_gradients.json','operating_curves.jpg','dustbin_distribution.jpg'):
                shutil.copy2(cal/name,folder/('calibration_'+name))
    diagnosis=(dest/'A_DIAGNOSIS.md').read_text()
    for v in names:
        for filename in ('operating_curves.jpg','dustbin_distribution.jpg','summary.json'):
            diagnosis=diagnosis.replace(f'calibration/{v}/{filename}',f'{v}/calibration_{filename}')
    (dest/'A_DIAGNOSIS.md').write_text(diagnosis)
    (dest/'README.md').write_text('''# D16最终实验交付

先读 [一页结论](ONE_PAGE.md) 和 [完整报告](FINAL_REPORT.md)。代码位于 `mhinet/v2/final_round/`，原始计划位于 `mhinet/v2/local_occlusion/FINAL_ROUND_PLAN.md`。

此目录包含所有十组开发/测试结果、逐对CSV、空间块配对区间、梯度、工作点曲线、G组示例及冻结的数据清单。压缩仅用于减少Git体积，无额外校验框架。

完整逐查询NPZ、所有训练日志与权重保留在服务器 `/home/disk1/MHINet/outputs/d16_final_round/`；旧基线权重在 `/home/disk1/MHINet/outputs/d16_local_occlusion/`。最终权重路径/坐标协议及诊断阈值见final_model.json。Git不包含模型权重和数据影像。

图中青色箭头是B图参考位置到预测位置的**残差**，不是A→B完整位移；长度1:1原图像素。高置信错误图使用Matchability≥0.9且EPE>16px；这只是描述性切片，不是部署阈值。每组示例按清单首个样本固定，不挑成功图。
''')

if __name__=='__main__':main()
