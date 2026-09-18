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
