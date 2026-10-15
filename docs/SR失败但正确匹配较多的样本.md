# SR失败但正确匹配较多的样本

| 样本 | NCM@3 | NM | P@3 | 全域网格RMSE/px | 网格误差≤3px比例 | 局部1几何RMSE/px | 局部2几何RMSE/px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jl1flight__5760_5040__t2 | 7817 | 12000 | 65.14% | 3.038 | 73.37% | 1.202 | 5.426 |
| jl1flight__0_11520__t0 | 8329 | 12000 | 69.41% | 6.464 | 45.73% | 0.485 | 5.138 |
| googleearth_eval__043 | 6386 | 11995 | 53.24% | 3.964 | 60.87% | 0.653 | 6.379 |
| jl1flight__4320_8640__t2 | 7415 | 11999 | 61.80% | 5.197 | 63.76% | 0.574 | 11.637 |

## 图像说明

第一行依次为原始A、原始B、预测H配准后的A，以及预测配准红青叠加。第二、三行分别放大蓝框与橙框对应的64×64局部：原始B、配准A、预测叠加、GT参考叠加。所有原图均为测试时的512×512输入，没有另外缩放后重新评价。

叠加图中红通道来自B，青通道来自配准后的A。为了在JL亮度差异较大的影像中看清结构，叠加图使用原始A/B各自2%–98%灰度窗口作显示归一化；预测与GT使用完全相同的窗口。原始图与配准RGB不作亮度处理。灰区表示预测变换下没有A像素覆盖。图像亮度、时相变化也可能影响叠加颜色，GT列只作参考。

局部数值是目标框内逐像素经GT反投影到A，再比较预测H与GT投影所得的几何RMSE；它是局部诊断，不替代全域SR使用的20×20源网格评价。

## 样本解读

- JL1117有7817个≤3px正确匹配，且73.37%的可见评价网格点误差≤3px，但网格误差平方平均后开根号为3.038px，仍略超阈值。蓝框局部几何RMSE为1.202px，橙框为5.426px。
- JL80有8329个正确匹配。蓝框处局部几何RMSE仅0.485px，橙框建筑处为5.138px；全域网格RMSE为6.464px。局部对齐较好与整幅通过SR并不等价。
- GE43有6386个正确匹配。蓝框为0.653px，橙框为6.379px，全域为3.964px。GoogleEarth的GT定义为相对名义原始配准叠加的合成几何变换，此图不用于判定原始跨时相GT是否准确。
- JL872作为补充例子，蓝框为0.574px，橙框为11.637px，全域为5.197px。

这些是从SR失败样本中挑出的定性示例，未删除、修改或重算原测试结果。

## 原图、配准图与局部放大

### jl1flight__5760_5040__t2

![原始影像、配准结果与局部放大](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_01117/comparison.png)

[原始A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_01117/original_A.png) · [原始B](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_01117/original_B.png) · [配准A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_01117/registered_A.png) · [详细坐标与H](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_01117/case.json)

### jl1flight__0_11520__t0

![原始影像、配准结果与局部放大](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00080/comparison.png)

[原始A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00080/original_A.png) · [原始B](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00080/original_B.png) · [配准A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00080/registered_A.png) · [详细坐标与H](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00080/case.json)

### googleearth_eval__043

![原始影像、配准结果与局部放大](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/GoogleEarth_00043/comparison.png)

[原始A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/GoogleEarth_00043/original_A.png) · [原始B](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/GoogleEarth_00043/original_B.png) · [配准A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/GoogleEarth_00043/registered_A.png) · [详细坐标与H](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/GoogleEarth_00043/case.json)

### jl1flight__4320_8640__t2

![原始影像、配准结果与局部放大](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00872/comparison.png)

[原始A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00872/original_A.png) · [原始B](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00872/original_B.png) · [配准A](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00872/registered_A.png) · [详细坐标与H](/home/disk1/MHINet/outputs/sr_failure_high_ncm_cases_512_20261015/JL1Flight_00872/case.json)

## 可复现配置

来源：`outputs/h0_position_ablation_512_20261013/A_GoogleEarth` 和 `A_JL1Flight`，即Top-K实验中Full所用的原始A预测。使用已保存的最终匹配与RANSAC预测H，不重新训练、推理或拟合H。

脚本独立核对选中样本的影像SHA、NCM@3，以及全部GT可见20×20网格的配准RMSE。原SR中预测落到画面外的GT可见网格仍参与计算，局部框只用于选取能清晰观察的显示区域，不改变任何指标。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /root/miniconda3/envs/loma-repro/bin/python scripts/visualize_sr_failures.py   --font outputs/sr_failure_high_ncm_cases_512_20261015/fonts/NotoSansSC-Medium.ttf
```

原始可视化记录：`outputs/sr_failure_high_ncm_cases_512_20261015/cases.json`，包含输入路径与SHA、预测H、GT、局部框坐标、全域及局部误差。中文字体为官方Google Fonts Noto Sans SC，来源、哈希及许可保存在输出目录的fonts中。
