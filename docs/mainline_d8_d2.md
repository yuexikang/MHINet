# 主路线：D8→D2 密集匹配下游

决策日期：2026-10-14（北京时间）。按用户要求，后续架构说明、训练、开发验证和研究任务统一以 **共享 GHIM/CGMDP → D8 粗匹配 → D2 细匹配 → 4 轮 QRRU** 为当前主路线。

本页记录路线与基线；性能结论以对应实验登记和结果为准。[模型架构](model_architecture.md)定义接口与梯度，[实验计划](experiment_plan.md)定义后续顺序，[文档筛查清单](document_status_20261014.md)解释旧文档的用途。

## 路线边界

- 共享 GHIM 生成 H0 与 MVT 上下文，CGMDP 融合 VGG 特征并累计解码至 D2；匹配下游消费 D8、D2。D4 保留为累计解码中间层，D1 不启用。
- 当前工作分辨率为 **512×512**，D8 为64×64、D2为256×256；784 保留为兼容/历史对照输入。
- 默认 H0 设置保留 **A**：D8 预测 overlap mask 开启，D2 H0 warp 与预测 overlap mask 开启。QRRU 不接收 H0。固定权重 B/C/D 消融不会自动改变默认路线。
- “密集匹配下游”是本路线的统一名称。实现仍为 `SemidenseSystem` / `SemidenseMatcher`，检查点协议仍为 `semidense_qrru_v1`；互选、合法性过滤和输出上限意味着返回匹配集合，不保证每个像素有对应。
- 原六轮 MHIR、早期 P3/P6 零样本消费者及 LoRA/D1 方案作为历史证据或独立对照保留。现行入口、监督和验收按本页执行。

## 可复用基线

| 项目 | 当前登记 |
| --- | --- |
| 权重 | `weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt` |
| 权重登记 SHA256 | `43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4` |
| 完成进度 | 5 epoch，25,535/25,535 optimizer steps；未做 JL 微调 |
| 训练配置 | [semidense_rgb512_h0_joint_5epoch.json](../configs/semidense_rgb512_h0_joint_5epoch.json) |
| runtime | [runtime_paths.mhinet_rgb512.server.json](../configs/runtime_paths.mhinet_rgb512.server.json) |
| 初始化 | E 组完整匹配权重，新优化器/新余弦调度 |
| 训练/验证 | RGB512 train 40,849 对、val 5,106 对；JL 不进入该轮训练 |
| 冻结 | DINOv3；LoRA 不启用；VGG BN running statistics 固定 |
| 可训练 | MVT、GHIM/H0、VGG、CGMDP、匹配温度与 QRRU |
| 损失 | `Lc+Lf+Lq+Lh0`；`Lh0=geo+0.01mat+0.0001cls+0.05H` |
| 训练超参 | seed42、单卡累积8、共享/头部初始LR均8e-5、5epoch余弦衰减 |
| 训练采样 | D8粗监督1024查询、D2细监督64窗口、QRRU128查询 |
| 推理 | D8 mutual、D2局部4×4展开、QRRU4轮、推理窗口分块1024 |
| 输出限制 | `max_coarse=6000`、`max_matches=12000`、置信阈值0 |

训练采样预算不限制推理查询数。匹配细窗口中的 H0 仍 detach，GHIM 头由独立 Lh0 获得直接监督；训练损失验证使用 GT 解耦标签，级联推理只使用图像和预测先验。详见[完成训练记录](semidense_rgb512_h0_joint_5epoch.txt)。

当前训练代码会为 `freeze_h0=false` 且未显式指定监督权重的配置默认启用 Lh0；旧 RGB512/JL 配置不构成严格复现历史 Lc/Lf/Lq 的现行推荐命令。新任务使用上述显式 H0 联合配置并登记来源。

## 数据、来源与加载合同

RGB512 train来源为FLAIR RGB 29,360对、GoogleEarth Current 7,701对、WHU 3,788对。加载器以 `(source_dataset, source_path)` 检查来源分组隔离；数据没有完整经纬度，来源隔离不等于已证明地理隔离。同母图合成变换提供几何GT，跨母图原始图像不能自动赋单位变换标签；没有可靠GT的图像仅作定性检查。

共享网络每对只计算一次。`mhinet.pretraining.model.load_shared`加载共享bundle，bundle保存MVT/GHIM/VGG/累计解码器，不包含冻结DINO大权重或匹配头；加载时校验DINO checkpoint SHA。当前完整匹配权重通过`SemidenseSystem`严格加载，不用共享bundle替代完整匹配检查点。

历史LoMa参考核来源提交为`899fdb99a4076e312196bbbf99a3c329739b5d7a`，文件SHA和512形状适配见[来源登记](../artifacts/loma_downstream_source.json)，原迁移parity见[审计记录](../artifacts/loma_downstream_parity.json)。来源登记中的零样本/no-trainable-head字段仅描述参考核迁移阶段，当前QRRU及训练路径由模型架构定义。

如复查旧共享H0，应按[GT可见域评价实现](../mhinet/pretraining/overlap_metrics.py)确定支持集，非法预测/拟合失败保留在Recall分母；可用`scripts/evaluate_shared_overlap.py`独立补评估。该历史H0_overlap口径不混同当前匹配P/拟合H的SR。

## 恢复合同与已有审计

新检查点保存模型、优化器、调度器、数据进度和逐rank RNG/buffers；DDP捕获本rank CUDA状态，分布式验证临时共用rank0 buffers，验证后恢复各rank原值。配置、数据、模型/实现及执行策略按记录核对，旧检查点缺失的其他rank状态无法事后精确恢复。

验证按固定manifest索引分片并逐样本设种子，策略记录为`per_manifest_index_v1`；`--validation-pairs 0`为全量，恢复省略参数时沿用检查点策略。回退恢复在任务锁下整理日志，超出断点的记录归档至`resume_history`。实现见[恢复与验证管理](../mhinet/engine/semidense_runtime.py)。

2026-09-30已有恢复审计记录：28项CPU检查通过；独立双rank dropout/BN/AdamW中断接续的下一步权重与buffers和连续路径逐值一致；分片验证与串行逐样本一致；第5750步真实权重完成恢复。这些是原版本的恢复证据，不是跨版本任意恢复承诺，也不证明当前匹配精度提升。原说明由Git历史追溯。

## 已完成结果与默认设置

| 数据集 | 对数 | P@1 | P@3 | P@5 | 严格 SR@3 | 论文 RMSE | 平均 NCM@5 | 平均 NM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GoogleEarth | 498 | 8.25% | 33.24% | 53.42% | 47.59% | 6.520 | 6340.16 | 11772.68 |
| JL1Flight | 1755 | 7.10% | 37.92% | 59.51% | 43.65% | 6.791 | 7112.03 | 11774.41 |

来源：[H0 联合训练与两数据集测试](semidense_rgb512_h0_joint_test512_results.md)，原始结果保留于 `outputs/semidense_rgb512_h0joint_test512_20261012/`。联合监督已修复 H0 头缺少直接梯度，但没有证明明显跨域精度提升。

[H0 推理位置消融](H0推理位置消融实验结果.md)复用同权重、同特征及同测试清单，A 与历史同权重输出逐点一致。B/C/D 没有跨域一致最优结论，默认 A 保留；D2 开关同时改变 warp 与支持区，不能拆成单独 warp 因果结果。旧 JL 域适配已完成且有域内收益/旧域退化证据，见[迁移分析](jl1flight_finetune_transfer_analysis.md)，其权重阶段单独标识。

## 运行入口

| 用途 | 当前入口 | 范围 |
| --- | --- | --- |
| 主路线训练 | `scripts/train_semidense_rgb512_h0.sh` / `python -m mhinet.cli train-semidense` | H0独立监督与D8→D2/QRRU训练 |
| 匹配评价 | `python -m mhinet.cli evaluate-semidense` | 按manifest tier评价原生坐标匹配；不包含最新拟合H/SR完整契约 |
| 固定H0位置消融 | `scripts/evaluate_h0_position_ablation.py` | 当前权重、冻结Test_Data清单和A/B/C/D协议 |
| 历史零样本 | `python -m mhinet.cli evaluate-dense` | P3/P6消费者，无当前可训练QRRU链 |
| 历史MHIR | `python -m mhinet.cli train/resume/evaluate` | H0→H6单应精修与旧tiny gate |

在仓库目录使用已有 `loma-repro` 环境，例如在独立新目录复现登记训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_semidense_rgb512_h0.sh outputs/your_new_h0joint_run
```

设备按实际空闲情况选择。目标目录存在 `latest.pt` 时 launcher 自动恢复；新目录从 E 初始化。配置或数据变化另注册新任务，保持已完成基线目录、检查点和原始报告可追溯。512/784 兼容与历史验证见[输入选项](semidense_input512_usage.md)。

最新双数据集结果的 cohort、实现哈希与评价器由原结果目录的 `plan.json` / `registration.json` 记录；不能把旧 `benchmark_semidense_v1.sh` 的 stable_v2 清单直接当作 RGB512/Test_Data 当前基准。

## 评价契约

- 点级 P@1/3/5 以全部返回匹配为分母，按影像对宏平均；NCM/NM 按全部影像对平均，无匹配计0。当前 Test_Data 阈值使用存储影像512目标像素。
- 配准 H 仅由预测匹配拟合，GT随后参与评分。已登记 RANSAC 阈值3px、最大10000迭代、confidence0.999、seed0；固定登记不构成独立多随机种子统计。
- 严格 SR@1/3/5 根据预测 H 在 GT 可见20×20源网格上的配准RMSE判定，全部影像对参与统计。至少20个5px正确点的旧成功率另列。
- “论文RMSE”为SR@3成功对的GT误差≤5px正确匹配RMSE、失败填10；它不替代未封顶全域网格误差及失败尾部。成功但无正确匹配时单列无定义。
- 阶段分别报告D8、D2及最终QRRU输出；精修前后集合可能受截断/合法性过滤影响，同点因果对照要保留对应关系和mask。
- 开发同时观察空间覆盖、失败率、延迟与显存；不同系统负载下的历史耗时不作为受控速度结论。训练loss、tiny可学习性和研究交付状态均不自动代表精度提升。

Test_Data 不用于选择超参、checkpoint、开关或点数。后续机制在独立开发集锁定后再做冻结评价，执行顺序见[当前实验计划](experiment_plan.md)。
