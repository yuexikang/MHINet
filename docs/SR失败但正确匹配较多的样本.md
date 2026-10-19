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

## JL平面场景的进一步诊断（2026-10-19）

| 样本 | D2匹配拟合完整H | QRRU后拟合完整H（原结果） | 同一最终匹配拟合相似变换 |
| --- | --- | --- | --- |
| jl1flight__0_11520__t0 | 1.468px | 6.464px | 2.895px |
| jl1flight__5760_5040__t2 | 2.484px | 3.038px | 1.555px |
| jl1flight__4320_8640__t2 | 2.611px | 5.197px | 1.359px |

表中均为相同GT可见20×20源网格上的配准RMSE，以≤3px为成功。

这三个例子在D2阶段都成功，经过QRRU及最终边界筛点、重新拟合后失败。QRRU按局部窗口独立更新点位，现有代码没有强制这些更新共享同一个全局H。该结果说明这些选中样本的细化路径确实使全局几何估计变差，并不能用视差或物理上不存在像素对应来解释。

在固定最终预测坐标、不使用GT选点或拟合参数的条件下，只将估计器改为相似变换RANSAC，三例又都成功。相似变换由旋转、等比例缩放、平移构成，共4个自由度；原始完整H有8个自由度。当前JL测试清单的1755个GT矩阵均符合相似变换形式，透视项为0；这三例原始预测H则包含非零透视项。

估计器诊断参数为cv2.estimateAffinePartial2D，RANSAC阈值3px、maxIters=10000、confidence=0.999、refineIters=10，每对重置seed=0，OpenCV线程2。GT只用于随后计算指标，未用于选择匹配或估计变换参数，没有重新执行神经网络。更换估计器同时改变模型自由度、采样和优化，不能把改善唯一归因于透视参数。

“正确匹配”是误差≤3px，并非零误差或严格独立的几何约束。例如JL80的8329个正确@3匹配中，仅2602个达到≤1px；预测点仍可能存在局部偏差。JL80的正确@3匹配有77.96%位于源图上半，而均匀GT可见评价网格仅39.59%位于上半；JL1117没有相同的上半集中现象，不能把三例都归因于覆盖不足。

范围限制：这里只诊断挑出的三例，不替代全量测试结果。全量JL的D2阶段SR@3为42.85%，最终为43.65%，因此也不能据这三例断言QRRU在整个数据集上总是有害。物理上存在全局对应，与模型能精确定位并稳健估计该变换，是需要分别验证的两件事。

本机完整诊断记录：`outputs/jl_planar_sr_diagnosis_20261019/estimator_comparison.json`，含预测与GT矩阵、源匹配SHA、估计器配置及误差分布。原测试结果未覆盖。
