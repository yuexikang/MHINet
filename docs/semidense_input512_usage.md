# 半密集模型的 512 输入选项

状态核对：2026-10-14。当前 [D8→D2 密集匹配下游主路线](mainline_d8_d2.md)使用 RGB512，配置及已完成的 H0 联合权重明确记录 `input_size=512`。代码中的 `SemidenseConfig` 通用默认值仍是 784；缺少该字段的旧权重也按 784 解释，两者不能混为主路线的输入设置。

512 支持已从 `codex/semidense-input512-study` 集成到 `main`。
本次集成引入正式半密集版本 `7cddb2a` 的匹配、训练、计时及可视化实现，并在其上适配分辨率；推理分块保持 1024，训练分块保持 32。

## 配置与运行

- 配置字段：`matcher.input_size`，允许 `784` 或 `512`；旧权重没有此字段时按 784 解释。
- 模板：`configs/semidense_qrru.json` 和 `configs/semidense_qrru_512.json`。
- 当前主路线训练配置：`configs/semidense_rgb512_h0_joint_5epoch.json`；已完成权重：`weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt`。该轮训练已完成，不需要为切换主路线重新训练。
- 训练、半密集评估、计时评估和 MRSI 评估入口支持 `--input-size 512`。
- MRSI 评估默认使用当前仓库实现，无须指定临时 implementation-root。

以下为旧 E 权重的 MRSI 512 评估示例，不是当前 H0 联合权重的双数据集结果入口；本文不将该示例列为已完成的全量结果：

```bash
python scripts/evaluate_e_mrsi_gt.py \
  --checkpoint outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt \
  --manifest artifacts/mrsi_expanded_e_manifest.jsonl \
  --input-size 512 --output outputs/mrsi_expanded_e_gt_512
```

该评估脚本输出原始匹配和诊断指标。论文口径 RMSE 继续由 `scripts/rescore_mrsi_correct_match_rmse.py` 后处理：成功沿用配准 SR@3px，正确匹配采用 5px，失败 RMSE=10。本次没有更改指标定义或覆盖既有结果。

从已完成的半密集权重开启 512 训练使用 `scripts/train_semidense_next_tier.py`，传入 `--config`、`--checkpoint`、`--output` 及可选的 `--input-size 512`。加载完整模型权重，重新建立优化器和调度器。配置模板使用 quadrant runtime、tier 1，正式启动前应指定所需数据配置与学习率。不要通过 `--resume` 将旧训练中途改成另一分辨率。

## 特征与坐标

| 项目 | 784 | 512 |
|---|---:|---:|
| DINO token 网格 | 49×49 | 32×32 |
| D8 粗匹配 | 98×98 | 64×64 |
| D4 | 196×196 | 128×128 |
| D2 精匹配 | 392×392 | 256×256 |

匹配点经动态网格归一化后映射回各自原图尺寸。默认 A 配置中，H0 为 D8 提供预测重叠支持区 mask，为 D2 提供重叠支持区 mask 和 H 残差 warp；它不进入 QRRU。D8 并非限制每个点只能在 H0 邻域搜索。QRRU 保持 4 次迭代；可加载 E 权重运行 512，但旧 E 的结果与完成 RGB512 训练后的结果属于不同权重阶段。

## 当前全量证据

H0 联合 RGB512 训练已完成 5 epoch、25,535/25,535 步；GoogleEarth 498 对、JL1Flight 1755 对全量测试已完成，见[训练及双数据集结果](semidense_rgb512_h0_joint_test512_results.md)和[训练记录](semidense_rgb512_h0_joint_5epoch.txt)。固定同权重的 [H0 推理位置消融](h0_inference_position_ablation_results_20261013.md)已完成四组测试并核对 A 与历史输出逐点一致，默认仍为 A。

这些证据覆盖所列权重及数据集，不意味着 E 权重的 MRSI 512 示例也已完成。

## 512 集成时的历史验证

历史验证记录保留在 `codex/semidense-input512-study` 分支的 `artifacts/semidense_input512_checks.json`；main 上的可复现入口为 `scripts/check_semidense_input_size.py`。

- E 权重严格加载；三对固定 MRSI 影像分别运行 784 与 512，全部得到匹配。
- 三对 784 的坐标与置信度和原保存结果逐位一致；两种尺寸下计时包装与普通推理输出一致。
- 512 特征图、匹配和 QRRU 轨迹可视化成功生成。
- 512 实际数据的 Lc+Lf+Lq 反向传播成功，共享可训练模块和匹配头均有梯度，冻结模块没有梯度；没有执行优化器更新。

上述集成检查只证明当时的代码路径可用，当时没有启动训练；后续完整训练及测试进度以上节为准。LoMa 参考核只扩展 D8 尺寸检查，原始校验值及局部修改说明保存在 `artifacts/loma_downstream_source.json`，仍对修改后的文件执行完整性校验。
