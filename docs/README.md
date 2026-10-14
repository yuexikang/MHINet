# MHINet 文档入口

2026-10-14起，当前主路线为 **共享GHIM/CGMDP → D8粗匹配 → D2细匹配 → 4轮QRRU**。现有 `semidense` 代码与检查点名称继续使用。完整逐文件状态见[文档筛查清单](document_status_20261014.md)。

## 当前规范与操作

| 文档 | 用途 |
| --- | --- |
| [主路线](mainline_d8_d2.md) | 路线决策、基线、默认设置、当前入口和评价契约 |
| [模型架构](model_architecture.md) | 共享特征、粗细匹配、QRRU及训练梯度边界 |
| [实验计划](experiment_plan.md) | 后续最小机制对照、跨域验证与验收顺序 |
| [512输入说明](semidense_input512_usage.md) | 当前512配置与784兼容、历史集成验证 |

## 当前可复用证据

| 文档 | 证据范围 |
| --- | --- |
| [H0联合训练记录](semidense_rgb512_h0_joint_5epoch.txt) | 已完成5epoch/25535步、监督、配置与SHA |
| [双数据集结果](semidense_rgb512_h0_joint_test512_results.md) | 当前H0joint基线、498 GE与1755 JL完整测试 |
| [H0位置消融实验定义](h0_inference_position_ablation.md) | 固定权重2×2设计、开关范围与审计 |
| [H0位置消融中文结果](H0推理位置消融实验结果.md) | 四组最终/阶段结果、协议与结论 |
| [H0位置消融全量结果](h0_inference_position_ablation_results_20261013.md) | 结果及CSV/JSON旁证入口 |
| [JL微调迁移分析](jl1flight_finetune_transfer_analysis.md) | 历史域适配收益与旧域退化；供当前研究保留反证 |

## 累计研究与历史资料

研究阅读顺序见[阅读指南](research/mhinet_breakthrough/reading_guide.md)：累计记忆 → 最新有效交付 → 原始轮次证据。路线与任务要求保存在稳定README/brief；latest和累计记忆由研究脚本更新。历史round/runs/evidence保留当时判断和证据边界。

9份旧方案、4份部分过时说明及3份旧首页/架构/计划快照已删除，具体文件见[删除清单](document_status_20261014.md)。保留的历史实验结果在顶部标明范围，原数字、源SHA与结果附件继续用于追溯；当前数据、来源和恢复合同统一见[主路线](mainline_d8_d2.md)。

`evaluation_summary.md`及其registry按旧H0/MHIR几何评估范围阅读，重新扫描可能重写该文件。`benchmark_semidense_v1.md`是2026-09-29历史基准；当前baseline与指标以本页链接的H0joint记录及主路线为准。日期较早的结果仍可引用其登记范围，顶部“历史”提示限制当前执行用途。
