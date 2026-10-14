# 固定权重的 H0 推理位置消融

分支：`codex/h0-inference-position-ablation`，建立于2026-10-13。使用已经完成的H0联合RGB512权重，比较H0在D8与D2的使用位置，全程不训练。

全量实验已于2026-10-13 18:01（UTC+8）完成，GoogleEarth 498对、JL1Flight 1755对，四组共9012份匹配结果。2253对共享特征在四组间完全一致，运行前后全部参数及buffers哈希一致；A的2253对输出与旧结果逐点完全相同。CPU独立重新核验全部新NPZ与2253份历史A结果通过。

完整指标、粗细阶段和结论见[全量结果](h0_inference_position_ablation_results_20261013.md)，机器可读指标见[CSV](h0_inference_position_ablation_results_20261013.csv)，一致性证据见[审计JSON](h0_inference_position_ablation_audit_20261013.json)。当前没有一组在两套数据的匹配精度和SR上全面胜出，默认配置仍为A。

| 实验 | D8 H0 Mask | D2 H0几何引导 |
|---|---|---|
| A 原始模型 | 使用 | 使用 |
| B 仅D8几何约束 | 使用 | 不使用 |
| C 仅D2几何引导 | 不使用 | 使用 |
| D 无H0几何引导 | 不使用 | 不使用 |

原实现的D2除了局部warp，还使用H0预测的overlap mask。本实验把这两处共同归入D2几何引导：关闭时使用目标粗匹配D8单元的规则4×4子格，仅检查影像边界。A保留原行为；D不读取H0矩阵或H0有效性门控，避免保留隐性几何依赖。共享网络仍计算同一组特征及H0预测，D忽略该预测。

## 固定条件

- 权重：`weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt`。
- SHA256：`43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4`。
- DINOv3、MVT、CGMDP、VGG、GHIM、匹配温度与QRRU参数以及所有buffers固定，eval/no_grad，无优化器。
- 每对只执行一次共享特征提取，四组复用同一个D8/D2/H0字典；运行前后记录完整模型SHA和逐对特征SHA。
- 打分函数、温度、阈值、互选规则、置信度组合、max_coarse/max_matches不变。实际分数数值会随候选集合及采样位置改变，不能解释为四组每条分数均相同。
- 输入512、推理分块1024；GE498对、JL1755对，同一清理后的cohort，GE431/432保持排除。
- 每对A与2026-10-12保存的points_a/points_b/confidence严格逐点一致；每数据集首对再对旧`be04d1f`推理代码验证。

## 指标与证据

P@1/3/5为逐对宏平均；NCM@1/3/5和NM按全部影像对平均，无匹配计0。H仅从预测匹配RANSAC拟合（3px、10000迭代、confidence0.999、seed0），GT仅用于随后评估。SR@3为GT可见20×20网格配准RMSE≤3px。论文RMSE使用SR@3成功对的≤5px正确匹配，失败记10；特殊undefined情况明确计数。记录粗、细、最终三阶段及H0有效/无效子集。

每组保存逐对JSONL、完整匹配NPZ、影像对索引、清单/输入/实现/权重哈希；完整模型哈希核对后才把8组报告置为complete。统计图及中文HTML/CSV/JSON由报告器生成。

GPU2真实预检已完成GE与JL各1对，A/B/C/D共8个结果：A与旧缓存及旧infer完全相同，所有模型参数/buffers和两对共享特征SHA不变，最大allocated约1.30GiB、reserved约1.50GiB。此有限预检不是全量结论。CPU测试包括位置开关23项、runner7项、计时恢复3项，均通过。

## 执行

全量输出：`outputs/h0_position_ablation_512_20261013`。物理GPU2有其他任务，显存预算6GiB且启动需额外4GiB空余，不停止其他任务；耗时只作记录，不作为受控速度结论。

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/loma-repro/bin/python -u scripts/evaluate_h0_position_ablation.py \
  --device cuda:0 --memory-budget-gib 6 \
  --output outputs/h0_position_ablation_512_20261013
```

不传`--device`时在CPU排队真正空闲GPU；`--gpu`可限制物理卡候选。`--prepare-only`只做CPU全量数据/基线/权重检查。`--limit`必须另用输出目录，避免有限验证与正式全量记录混用。
