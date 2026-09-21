# 稳定数据三档完整一轮续训

2026-09-21 登记。目的：检验完成完整数据遍历是否改善 2,000 步短程适应后的 H0 退化；不是已证实的修复。已知最差五例主要是 matchability 筛选放入错误对应，故保留原损失、拟合器及阈值，结束后同时比较描述子与 H0 指标，不能仅依据总 loss 判断成功。

训练共享 GHIM + CGMDP（D8/D4/D2），不训练 MHIR、不计算 D1。冻结 DINOv3，不启用 LoRA；MVT、GHIM head、VGG 和累计解码器参与训练，VGG BN 运行统计固定。三档使用同一超参：物理 BS1 × 累积4，seed0，AdamW、weight decay 1e-4、梯度裁剪1、每方向每尺度1024查询；峰值 LR 为 MVT/head 1e-7、VGG 5e-7、CGMDP decoder 1e-6。每阶段重建优化器，5% warmup + cosine 降到峰值10%。阶段内恢复则恢复优化器、调度器及数据游标。

| 阶段 | 数据 | train / val 对数 | 优化步 | 全量 val 步点 |
|---|---|---:|---:|---|
| 1 | 几何 | 51978 / 5772 | 12995 | 5000、10000、12995 |
| 2 | 几何+辐射 | 34652 / 3848 | 8663 | 5000、8663 |
| 3 | 几何+辐射+干扰 | 34652 / 3848 | 8663 | 5000、8663 |

实际数据：`/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2`；运行配置 `configs/runtime_paths.stable_v2.server.json`。仅 train 训练、val 验证，不使用 test 选择参数。每500步及结束保存 latest.pt。

第一档初始化 `/home/disk1/MHINet/outputs/shared_descriptor_frozen_tier3_seed0/latest.pt`，SHA256 `5a9ed14cc32a1a4ff3a843b737410da13d795b79c42a9d33d82a068226e30386`，不是退化后的2000步权重，也不是随机初始化。

新输出固定为 `/home/disk1/MHINet/outputs/shared_stable_v2_full_tier{1,2,3}_seed0`。第二档读取第一档 latest.pt，第三档读取第二档 latest.pt。后两份权重是未来产物，尚未生成；启动器检查上一档配置、完整训练步数和最终完整 val 已落盘。同目录再次执行自动恢复，不覆盖旧实验。这组固定串接入口不接受输出目录或诊断步数覆盖；需要另一组实验时另外登记配置。

依次运行（上一档完成后再运行下一档）：

```bash
cd /home/disk1/MHINet
GPU_ID=1 bash scripts/train_shared_stable_v2_full_tier1.sh
GPU_ID=1 bash scripts/train_shared_stable_v2_full_tier2.sh
GPU_ID=1 bash scripts/train_shared_stable_v2_full_tier3.sh
```

入口使用 `/root/miniconda3/envs/loma-repro/bin/python`，不需要重新激活环境。含训练/验证进度条。可加 `--check-only` 只检查资源和前序完成状态而不训练。

结束后与 stable_v2 三档训练前完整 val 基线比较 H0 重叠投影均值/P90/超阈值尾部、四角误差、失败率及 D8/D4/D2 retrieval。最终模型另回测第一、二档完整 val 检查遗忘；该回测不在上述训练入口自动执行。总 loss 下降不等于 H0 精度改善。
