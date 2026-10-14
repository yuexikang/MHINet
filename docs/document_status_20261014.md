# 文档筛查与删除清单（2026-10-14）

当前主路线为 **共享GHIM/CGMDP → D8粗匹配 → D2细匹配 → 4轮QRRU**，详见[主路线](mainline_d8_d2.md)。

原有39份顶层说明文档经筛查后，按用户“删除过时说明，然后推送”的要求，删除9份历史方案和4份部分过时说明，保留26份当前规范、操作、实验结果及有效参考。另移除上一轮建立的3份旧首页/架构/计划快照，共删除16份说明文件。

有效数据、来源和恢复合同已集中到[主路线](mainline_d8_d2.md)。当前入口为[仓库README](../README.md)、[docs入口](README.md)、[模型架构](model_architecture.md)和[实验计划](experiment_plan.md)。实验数字、原始SHA、机器附件与历史研究结果继续保留。

## 已删除说明

| 文件（原docs路径） | 删除前分类 | 删除依据 |
| --- | --- | --- |
| `batch_size_and_visualization.md` | 历史方案 | E01仅训练adapter/MHIR、784输入六轮四角L1/七张图、旧2500 val及test_e00入口不属当前主路线；旧吞吐/BF16漂移记录保留，不能直接升级当前默认batch。 |
| `mhinet_rgb512_jl1_training_plan.md` | 历史方案 | 尚未启动、合成2epoch、JL约20%留出及manifest未实现已被实际5epoch训练替代；旧launcher不能严格复现当前监督代码下的历史训练。 |
| `quadrant_three_tiers.md` | 历史方案 | v1三档数据生成协议仍可复现但无稳定几何安全界，被stable_v2替代；当前runtime仍single_parent/尚待切换属于旧登记。 |
| `real_image_evaluation.md` | 部分过时 | MHIR evaluate/report/MACE工具口径可留作历史分支说明；train36000/val2500/test1000、E00可定量test和未发现正式checkpoint陈述已过时；当前single_parent原图test无精确GT，半密集用独立匹配指标。 |
| `semidense_resume_audit.md` | 部分过时 | rank RNG/buffers、日志/验证恢复机制可参考；旧6265步与仅Lc/Lf/Lq/H0无梯度属于9月30日状态，后续已完成并加入Lh0。 |
| `shared_descriptor_pretraining.md` | 历史方案 | 共享描述子冻结/LoRA对照是早期预训练协议，正式两组后来已完成；784、第一档一轮不是当前训练设置。共享bundle加载器及DINO/MVT每对一次、无MHIR/D1边界仍适用对应入口，LoRA当前默认不启用。 |
| `shared_descriptor_tier2.md` | 历史方案 | tier2续训设置及2步冒烟记录保留；当前训练不再限定v1 tier2一轮。H0_overlap GT支持定义/非法预测分母工具仍有用途；后续完整结果与tier3回测接续。 |
| `shared_downstream_integration.md` | 部分过时 | 登记时双下游并列、零样本no_grad、固定784、未迁入RRU/fine可训练头、基线GPU运行中、未拟合H评价均属迁移初期；现已D8→D2→4轮QRRU训练和正式H拟合评价。原LoMa核SHA/parity和D4不直接消费但累计解码经过D4仍有效。 |
| `shared_stable_v2_full_curriculum.md` | 历史方案 | 三档共享预训练协议有效历史，后两份权重未来尚未生成已过时；磁盘三阶段最终val均存在（12995/8663/8663）。此阶段无MHIR/QRRU不是当前匹配训练结论。 |
| `single_parent_v2.md` | 部分过时 | 停用跨母图单位标签原则仍适用；single_parent_v2已被三档和stable_v2/RGB512接续。scripts/test.sh NO-GT原图工具仍存在，但H0+六轮图属历史MHIR，不能描述当前D8→D2匹配输出。 |
| `temporal_dataset_v1.md` | 历史方案 | 跨past/current母图以单位基础变换生成H标签的近似假设已被single_parent_v2明确停用；全量尚未生成属于旧状态，不能作为当前数据协议。 |
| `training_frozen_dino.md` | 历史方案 | 旧 temporal4 单位母图变换标签、六轮 MHIR 四角监督和20k预算；默认 runtime/输出已改为 single_parent，与当前 D8→D2→QRRU 主路线不同。 |
| `training_frozen_dino_mvt.md` | 历史方案 | 旧 temporal4/旧 train35996-val2500 路径和10k MHIR预算；正文默认输出自身矛盾；实际脚本已用 runtime_paths.single_parent.server.json。冻结MVT属旧实验，当前RGB512只冻结DINO。 |
| `history/mhir_readme_legacy.md` | 旧入口快照 | 首页已改为D8→D2；移除工作树内重复的六轮MHIR入口。 |
| `history/model_architecture_mhir.md` | 旧架构快照 | 旧六轮H更新合同由Git历史追溯。 |
| `history/experiment_plan_mhir.md` | 旧计划快照 | 旧P0–P6/E01/TINY-6/D1计划由Git历史追溯。 |

原13份说明及改写前的首页/架构/计划可从删除前提交 `4ec2ea0cd4d966a3b921b5688d5666f6a55e5f0f` 追溯。删除记录使用文本文件名，避免把已移除路径当作可打开入口。历史实现日志/研究报告中的原路径引用按其记录时间阅读。

## 保留的原有顶层文档

| 文档 | 状态 | 当前用途与范围 |
| --- | --- | --- |
| [H0推理位置消融实验结果.md](H0推理位置消融实验结果.md) | 当前基线证据 | 中文完整结果及实验配置；默认A，单次固定权重结论，无统一跨域最优组。 |
| [benchmark_semidense_v1.md](benchmark_semidense_v1.md) | 历史基准 | 2026-09-29 stable_v2第三档C/D/E基准与回归登记保留；当前冻结范围、权重与指标由H0joint记录接续。 |
| [cleanup_candidates_20260920.md](cleanup_candidates_20260920.md) | 历史结果 | 过去已授权删除47权重/11缓存的精确记录仍有效；第二批暂缓和当时无运行进程不是新授权或实时状态。与路线替换无直接冲突，不应据此继续删除。 |
| [evaluation_summary.md](evaluation_summary.md) | 历史结果 | 自动生成的MHIR/H0指标登记总表非整个项目当前总榜；全表只覆盖旧E00及工程权重，保留其split/口径约束。重新scan会重写，因此标签宜通过索引解释并考虑生成器同步。 |
| [experiment_plan.md](experiment_plan.md) | 当前规范 | 当前D8→D2基线、独立开发、机制对照与跨域验收；旧MHIR计划已移除。 |
| [h0_inference_position_ablation.md](h0_inference_position_ablation.md) | 当前基线证据 | 已完成固定权重2×2设计与审计说明；按10月13日协议复查，非待启动训练。 |
| [h0_inference_position_ablation_results_20261013.md](h0_inference_position_ablation_results_20261013.md) | 当前基线证据 | 已完成四组全量结果，A与历史同权重输出一致；不能据测试结果改默认。 |
| [homography_extrapolation_analysis.md](homography_extrapolation_analysis.md) | 历史结果 | v1极端合法投影外推的13468 val/训练审计及反事实噪声试验仍有效，不能据此否定旧重叠区精度；建议下一版/尚待回测已接续实现。 |
| [jl1flight_512_finetune_5epoch.txt](jl1flight_512_finetune_5epoch.txt) | 历史结果 | 旧RGB512→JL域适配训练记录，已完成4215/4215步；该轮无独立H0监督，不作为当前H0joint启动配置。 |
| [jl1flight_finetune_transfer_analysis.md](jl1flight_finetune_transfer_analysis.md) | 历史结果 | 旧JL域适配前后498GE/1755JL完整结果及BN/几何偏差分析仍有效；非新H0joint权重结果或模块因果消融。 |
| [model_architecture.md](model_architecture.md) | 当前规范 | 当前GHIM/CGMDP→D8→D2→QRRU接口和训练梯度；旧六轮MHIR说明已移除。 |
| [mrsi_expanded_e_evaluation.md](mrsi_expanded_e_evaluation.md) | 历史结果 | E组784输入的600对MRSI跨模态结果保留；旧main缺QRRU/必须detached worktree已过时，当前实现已集成。 |
| [pretraining_mcnet_d2_summary.md](pretraining_mcnet_d2_summary.md) | 历史结果 | 32个受控H0条件、D8/D4/D2各两轮及TINY-6 gate是旧MHIR局部可学习性证据，不能当D8→D2匹配主路线验收；未执行E00/E01是当时状态。 |
| [results_mcnet_d2.md](results_mcnet_d2.md) | 历史结果 | 把D8/D4/D2各两轮MCNet四角H更新称当前主线，需限制为历史MHIR；tiny gate、失败轨迹、非逐位恢复检查均保留有效。 |
| [results_real_image_val16.md](results_real_image_val16.md) | 历史结果 | 仅两步工程权重、旧16对val、H0→H6 MACE/网格评价；不是当前密集匹配精度或全量排名，数字不因时间早失效。 |
| [semidense_input512_usage.md](semidense_input512_usage.md) | 当前操作 | 当前RGB512主线输入为512；库默认及缺字段旧权重仍为784。新增完整H0joint证据，旧E/MRSI示例单列。 |
| [semidense_rgb512_h0_joint_5epoch.txt](semidense_rgb512_h0_joint_5epoch.txt) | 当前基线证据 | 已完成5epoch/25535步、独立Lh0、冻结范围与源SHA记录；数值保留。 |
| [semidense_rgb512_h0_joint_test512_results.md](semidense_rgb512_h0_joint_test512_results.md) | 当前基线证据 | 最新H0joint双数据集完整测试，GE498/JL1755；当前默认A。 |
| [shared_descriptor_results.md](shared_descriptor_results.md) | 历史结果 | 工程诊断与严确定性/恢复/bundle审计有效；正式两组尚未启动仅为2026-09-16状态，磁盘两组均已到validation/step_012995。2步小val不作训练效果结论。 |
| [shared_descriptor_tier3.md](shared_descriptor_tier3.md) | 历史结果 | 含5772对第一档完整回测和训练前tier3 H0基线，有效但非tier3最终分数；第三档启动中的状态已过时，现有validation/step_008663。 |
| [shared_stable_v2_adaptation.md](shared_stable_v2_adaptation.md) | 历史结果 | 2000步适应登记+已完成三档基线有效；下一步/正在运行语气已过时；未接可训练fine/QRRU、未拟合最终H仅指早期P3/P6推理消费者。 |
| [stable_geometry_v2.md](stable_geometry_v2.md) | 专项参考 | stable_v2生成/验收的安全界、单母图和split定义仍适用该数据；目的为后续MCNet四角更新是历史动机，生成后未切runtime/未训练是当时状态，不是当前主路线。 |
| [stable_geometry_v2_extrapolation.md](stable_geometry_v2_extrapolation.md) | 历史结果 | 134750对全量几何安全审计与6对敏感性控制结果仍有效，不证明现模型匹配成功；下一步尚需模型基线已由后续基线完成。 |
| [stable_v2_adaptation_regressions.md](stable_v2_adaptation_regressions.md) | 历史结果 | 固定5对H0尾部回归反事实说明matchability/拟合外点污染，仍可作为机制证据，非全val分布也非最新H0联合/推理位置消融因果结论。 |
| [temporal4_batch_probe.md](temporal4_batch_probe.md) | 历史结果 | 旧frozen_dino_mvt/MHIR BS1-BS2吞吐和BF16输出漂移证据仍有效，但不证明现模型的batch精度/吞吐，也不能用当时数据标签作当前精度依据。 |
| [temporal4_correlation_recompute_probe.md](temporal4_correlation_recompute_probe.md) | 历史结果 | 旧MHIR相关性activation checkpoint开关的同权重梯度/资源结论，非当前半密集QRRU通用速度结论。 |

## 研究目录与附件

| 文件/目录 | 状态与阅读规则 |
| --- | --- |
| [研究README](research/mhinet_breakthrough/README.md)、[brief](research/mhinet_breakthrough/brief.md)、[阅读指南](research/mhinet_breakthrough/reading_guide.md) | 当前稳定入口，已置顶D8→D2主路线并保留原精度上限/论文/点数研究要求 |
| [累计记忆](research/mhinet_breakthrough/research_memory.md)、[latest](research/mhinet_breakthrough/latest.md) | 动态成果，由脚本更新；根据其证据截止时间回查原报告。本次不手改动态成果 |
| `research/mhinet_breakthrough/round_0001_*.md` | 首轮历史研究；“下一小时/每小时”、源码行号及早期缺件按当时快照阅读，修正沿累计记忆回查 |
| `research/mhinet_breakthrough/runs/`、`evidence/` | 历史研究与可复算证据；保留时间、原判断、支持/反证及局部核验缺口，不改写成当前结论 |
| [accuracy_ceiling_request](research/mhinet_breakthrough/accuracy_ceiling_request_20261013.md)、[paper_reference_clarification](research/mhinet_breakthrough/paper_reference_clarification_20261013.md) | 用户研究要求与指标参考记录，未解机制继续保留；论文数字不混同当前返回点P/严格SR |
| `research/mhinet_breakthrough/state.json`、`schedule.json` | 既有运行/调度状态；本次路线文档整理不修改调度 |
| `research/mhinet_breakthrough/manifest.json`、`response.schema.json`、`smoke_status.json`、`progress_check_20261012.json` | 来源、输出schema与运行链路历史证据；不作模型精度证明 |
| [H0消融CSV](h0_inference_position_ablation_results_20261013.csv)、[H0消融审计JSON](h0_inference_position_ablation_audit_20261013.json)及结果报告链接的图表 | 对应10月13日固定权重实验的机器附件与图表，完整保留 |

14时研究快照可能记录本页/主路线当时尚未写入；本次整理已补齐这两份稳定文件。该快照保留原运行时事实，不作为持续缺件状态。

`evaluation_summary.md`可能被旧MHIR扫描器重写，其历史范围由本清单持续说明；scan不把它扩展为当前匹配总榜。原有未提交研究内容保留，研究轮次可能继续生成；本次文档清理没有改变代码、权重、数据或启动训练/评估；定时研究的动态文件保留既有状态。

链接核对还发现`evaluation_summary.md`的7份工程summary原路径已不存在；registry内对应summary快照与原SHA仍在，原数字继续保留。当前规范和新增链接均可解析，历史失效输出链接在原表顶部明确说明。
