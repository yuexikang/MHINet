# MHINet研究成果阅读指南

建议按“累计判断 → 最新进展 → 相关证据 → 最小实验”阅读。第一次先看前两份，不必按时间顺序通读全部轮次。

## 1. 先了解目前总体判断

读[累计研究记忆](/home/disk1/MHINet/docs/research/mhinet_breakthrough/research_memory.md)，重点看“历史判断的修正与关联”和“已完成诊断、未解项和优先级”。这里汇总稳定事实、支持/反证、旧观点如何收窄，以及下一步为什么值得做。开头标明证据截止时间；它是压缩总结，细节可沿所引报告回查。

目前成果主要是问题定位、已有预测的CPU诊断、论文机制与可证伪实验方案，尚未通过新训练和受控验证证明精度突破。当前重要边界是：密集输出和QRRU在合成验证可以有效，真实跨域低精度还不能统一归因于点数或QRRU。

## 2. 再看最近一次交付新增了什么

读[最新报告](/home/disk1/MHINet/docs/research/mhinet_breakthrough/latest.md)，按三个问题阅读：本轮补了什么证据？它保留、修正或否定了哪个旧判断？下一项最小验证是什么？新的定位材料或数据资格审计是实验前置工作，应看它通向哪个精度问题，不将资料查齐等同模型提升。

`latest.md`始终指向最近一次成功或limited交付，failed轮不会替换它。查看[运行状态](/home/disk1/MHINet/docs/research/mhinet_breakthrough/state.json)中的`last_attempt_at`、`last_status`、`last_success_at`、`last_report`和`next_focus`，能区分“最近一次尝试”与“最近一份成果”。

## 3. 按关心的问题回查专题

| 想回答的问题 | 建议阅读 |
| --- | --- |
| 点数是否过多，删点是否真的提高精度 | [10月12日18时点数专题](/home/disk1/MHINet/docs/research/mhinet_breakthrough/runs/20261012T180001/report.md)，结合累计记忆中的后续修正 |
| 数据分布、GT和失败样本有什么问题 | [数据与逐对结果](/home/disk1/MHINet/docs/research/mhinet_breakthrough/round_0001_data_results.md) |
| H0、fine、QRRU、损失和训练方式限制在哪里 | [模型与训练审计](/home/disk1/MHINet/docs/research/mhinet_breakthrough/round_0001_model_training.md) |
| 新论文的哪些机制能迁移、成本多大 | [论文研究](/home/disk1/MHINet/docs/research/mhinet_breakthrough/round_0001_literature.md)，版本/发表状态以之后的核验为准 |
| 最初为什么优先考虑这些方向 | [首轮综合报告](/home/disk1/MHINet/docs/research/mhinet_breakthrough/round_0001_synthesis.md)，作为历史起点阅读 |

每轮`runs/<时间戳>/report.md`保留当时判断。累计记忆和最新报告中的有证据修正应优先于较早的假设；遇到矛盾时沿路径核查原始范围与指标，不直接将“新报告”当成必然正确。新增记忆机制之后，每轮另存`research_memory.md`快照，便于追踪判断的演变。

## 4. 分清结论的证据等级

- **实现事实/已有观测：**有源码位置、数据清单、保存预测或可复算统计支持；只适用于所列数据和协议。
- **机制假设：**解释低精度的候选原因，需要控制其他变量，不能直接当作因果结论。
- **最小实验方案：**未来怎么验证、支持与否定标准是什么；列出方案不代表已经执行。
- **性能突破：**需要新受控实验和隔离验证证据。`completed`仅表示该研究轮次完成；`limited`表示交付存在核验缺口，局部已核实证据仍可用；`failed`没有完整交付。状态都不自动表示模型变好。

## 5. 点数研究必须一起看的指标

| 指标 | 阅读重点 |
| --- | --- |
| P@1/3/5 | 输出匹配中达到误差阈值的比例；删掉困难点可能提高P |
| NCM与NM | 正确匹配数与总输出数；P提高但NCM减少可能只是取舍 |
| 空间覆盖 | 正确匹配是否分布于足够区域，而不是集中在少数容易位置 |
| SR与全对网格RMSE | 预测匹配拟合H后的全域配准质量；同时看成功率和大误差尾部 |
| 条件RMSE、RANSAC内点率 | 前者可能只反映选定子集，后者只描述预测自洽，不能代替全域GT精度 |

例如报告中的“局部截断提高P、却减少NCM”说明精度与数量的取舍，不等于整体配准改善。比较删点方法时，要看同一候选池、相同K、相同拟合预算的置信筛选/随机/空间均匀对照，不能仅依据P选择点数。

## 6. 准备决定下一项实验时

优先看累计记忆的未解项，再看对应报告的最小方案：是否单因素、是否有固定基线和来源隔离开发集、是否能区分替代解释、结果怎样才支持或否定假设、需要补什么缓存以及预算。当前全量最终坐标/真实域同点轨迹等缺口会限制可执行对照；没有这些数据时，研究应明确缺件并推进独立可做的分支。

日常只需查看累计记忆和最新报告；源码、NPZ、机器JSON及私有运行日志留给需要复算或排查时使用。定时运行方式见[README](/home/disk1/MHINet/docs/research/mhinet_breakthrough/README.md)。
