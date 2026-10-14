# MHINet

当前主路线（2026-10-14）：**共享 GHIM/CGMDP → D8 粗匹配 → D2 细匹配 → 4 轮 QRRU**，以 RGB512 H0 联合训练版本作为可复用基线。

路线决策、默认设置和证据边界见[主路线说明](docs/mainline_d8_d2.md)；所有文档的用途与过时项见[文档入口](docs/README.md)和[筛查清单](docs/document_status_20261014.md)。

## 模型与命名

GHIM 通过冻结的 DINOv3 和可训练 MVT 提供 H0 及跨图像上下文；CGMDP 融合上下文与 VGG 特征，累计解码至 D2。下游消费 D8 和 D2，先生成粗父对应，再展开局部细对应并由 QRRU 精修匹配点。

本路线统一称“D8→D2 密集匹配下游”。现有实现与检查点仍命名为 `SemidenseSystem` / `SemidenseMatcher` / `semidense_qrru_v1`：返回的是经互选、置信筛选和数量截断后的匹配集合，并非每个像素都有对应。D4 是共享金字塔的中间层；D1 不启用。结构与梯度说明见[模型架构](docs/model_architecture.md)。

DINOv3 冻结，LoRA 暂不启用。当前 H0 联合配置训练 MVT、GHIM/H0、VGG、CGMDP 与匹配/QRRU，固定 VGG BN running statistics；损失为 `Lc + Lf + Lq + Lh0`。匹配先验仍 detach，GHIM 头通过独立 Lh0 获得监督。

## 当前基线与结果

- 权重：`weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt`，已完成 5 epoch、25,535 步，尚未做 JL 微调。
- [联合训练及两数据集测试](docs/semidense_rgb512_h0_joint_test512_results.md)：GoogleEarth 498 对、JL1Flight 1755 对，SR@3 分别为 47.59% / 43.65%。这不是已验证精度突破。
- [H0 推理位置消融结果](docs/H0推理位置消融实验结果.md)：默认保留 A，即 D8 H0 mask 与 D2 H0 几何引导都开启；其余组为固定权重对照，不按测试集结果改默认。
- 后续实施顺序见[实验计划](docs/experiment_plan.md)，累计诊断见[研究阅读指南](docs/research/mhinet_breakthrough/reading_guide.md)。

## 当前入口

在 `/home/disk1/MHINet` 使用已有 `loma-repro` 环境。主线配置为 `configs/semidense_rgb512_h0_joint_5epoch.json`，runtime 为 `configs/runtime_paths.mhinet_rgb512.server.json`。

```bash
# 查看可训练匹配路线的入口参数
conda run --no-capture-output -n loma-repro python -m mhinet.cli train-semidense --help
conda run --no-capture-output -n loma-repro python -m mhinet.cli evaluate-semidense --help

# 在新输出目录复现已登记 H0 联合训练；按实际空闲设备选择 GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/train_semidense_rgb512_h0.sh outputs/your_new_h0joint_run
```

训练 launcher 在目标目录已有 `latest.pt` 时自动恢复；新目录从 E 组权重初始化并建立新优化器。已完成的基线目录保留原样，新实验使用独立输出目录。`evaluate-semidense` 是按 tier 的原生坐标匹配评价入口，不等同于最新 Test_Data 的拟合 H / SR 协议；正式对照以已冻结的清单、实现和指标为准，详见[主路线说明](docs/mainline_d8_d2.md)。

`evaluate-dense` 对应早期 P3/P6 零样本消费者；`train` / `resume` / `evaluate` 对应旧 MHIR 路线。源码与旧入口保留供历史复现，日常主线使用上述 `*-semidense` 入口。

## 代码与资源

| 路径 | 用途 |
| --- | --- |
| `mhinet/pretraining/`、`mhinet/models/` | 共享 GHIM/CGMDP 及历史 MHIR 实现 |
| `mhinet/downstream/semidense.py`、`qrru.py` | 当前 D8→D2 匹配与 QRRU |
| `mhinet/downstream/training.py`、`train.py` | 独立 H0 监督、参数组与训练/恢复 |
| `scripts/train_semidense_rgb512_h0.sh` | 当前 H0 联合训练 launcher |
| `scripts/evaluate_h0_position_ablation.py` | 固定权重 H0 位置消融 |
| `weight/README.txt`、`weight/manifest.json` | 常用权重与来源登记 |

LoMa/LoRetta 源码、官方初始化权重和数据路径由 runtime 配置登记。Test_Data 只用于冻结协议评价，不用于选择超参、checkpoint、H0 开关或点数。过时方案和说明已删除，清单见[文档状态](docs/document_status_20261014.md)；保留的 MHIR tiny gate、H0→H6 实验结果和评估注册表按其原始范围阅读。
