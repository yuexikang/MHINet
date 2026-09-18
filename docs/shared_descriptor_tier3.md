# 第三档训练与第一档回测

2026-09-18。用户要求回测第一档完整val、继续第三档训练并分析极端外推。两个GPU任务独立运行，保留旧模型/结果；回测完成前不宣称“已确认无遗忘”。

## 第一档完整回测

卡0，第二档最终checkpoint在第一档5772对val上运行完整GHIM+CGMDP：H0四角/有效重叠误差、三个尺度双向full-gallery描述子检索、loss及可视化。没有使用test，没有取前若干对替代完整val。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/miniconda3/envs/loma-repro/bin/python -u -m scripts.evaluate_shared_full --checkpoint /home/disk1/MHINet/outputs/shared_descriptor_frozen_tier2_seed0/latest.pt --tier 1 --output /home/disk1/MHINet/outputs/shared_tier2_backtest_tier1
```

结束后用`python -m scripts.compare_shared_backtest`生成`outputs/shared_tier2_backtest_tier1/comparison.json`：和第一档训练结束时同一val、同一查询种子、同一GT/mask规则比较。H0重叠参考来自已完成的`shared_descriptor_frozen_seed0/h0_overlap_v1/tier1_summary.json`，描述子参考来自该模型`validation/step_012995/summary.json`。报告原始差值及逐对改善比例，不用单种子结果冒充跨种子等效检验。

## 第三档

- 卡1，配置`configs/shared_descriptor_tier3.json`，脚本`scripts/train_shared_descriptor_tier3.sh`。
- 初始化第二档最终checkpoint：`outputs/shared_descriptor_frozen_tier2_seed0/latest.pt`，SHA256 `4af7846f8111a64efae182005ba7fa9d1e90fecf9553815d5b88395d80280d7d`。
- 第三档为几何+辐射+遮挡，掩膜按双侧可见支持监督；只使用第三档train/val，不混入test。
- DINO冻结、无LoRA，MVT/GHIM head/VGG/CGMDP联合训练，VGG BN统计固定，不执行MHIR/D1。
- 延续第二档峰值lr：MVT/head5e-7、VGG2.5e-6、CGMDP5e-6；新optimizer/scheduler，warmup5%+cosine至10%，AdamW wd1e-4、clip1。
- BS1×累积4，一轮；每1000步保存，每5000步及最后完整val。独立目录`outputs/shared_descriptor_frozen_tier3_seed0`。
- 真实8对train/2对val两步冒烟通过，loss分别0.172034/0.211256，峰值9.96GB。两个不同batch的loss不能当作收敛曲线；这是接口/反传/验证检查。

日志：

```bash
tail -n 30 -f /home/disk1/MHINet/outputs/shared_descriptor_frozen_tier3_seed0.console.log
tail -n 10 -f /home/disk1/MHINet/outputs/shared_tier2_backtest_tier1.log
```

极端外推的实测分组、原因及下一版生成约束建议见`docs/homography_extrapolation_analysis.md`。当前数据/标签/损失未因该分析而自动改变。

## 第一档完整回测结果（已完成）

同一5772对第一档val、相同GT支持像素2533713851、相同描述子查询种子。未使用test。

| 指标 | 第一档训练后 | 第二档训练后回测第一档 |
|---|---:|---:|
| H0重叠每对均值，px | 2.113301 | 1.686334 |
| H0重叠每对误差中位数，px | 1.178909 | 1.028497 |
| H0重叠每对误差P90，px | 2.240741 | 1.944974 |
| H0四角MACE，px | 74.711518 | 69.928910 |
| D8双向平均匹配误差，px | 3.254289 | 3.234129 |
| D4双向平均匹配误差，px | 1.601194 | 1.595642 |
| D2双向平均匹配误差，px | 0.829732 | 0.824359 |
| D2 Recall@1px | 70.1314% | 70.6444% |
| 验证描述子损失 | 0.159341 | 0.145992 |
| 非法投影/拟合失败 | 0/0 | 0/0 |

74.7228%的影像对H0重叠误差改善；该验证集上未观察到总体几何能力退化，反而有一致收益。这是单种子同域验证，不等于所有场景或下游任务都不会遗忘。完整对照和SHA见`artifacts/shared_tier2_backtest_tier1.json`，逐对记录在`outputs/shared_tier2_backtest_tier1/validation/step_008663/pairs.jsonl`。

## 第三档训练前H0基线（已完成）

第二档权重在完整第三档3848对val上：重叠每对均值2.305215px，中位数1.593579px，P90为3.689485px；像素加权均值2.170733px，点Recall@1/3/5为35.2183%/84.4293%/93.8839%，无无效投影/拟合失败。报告`outputs/shared_descriptor_frozen_tier2_seed0/h0_overlap_tier3_baseline/tier3_summary.json`。这是第三档训练前基线，不是第三档训练完成结果。
