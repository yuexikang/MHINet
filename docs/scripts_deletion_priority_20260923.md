# scripts 删除优先级审查（2026-09-23）

本次只做静态检查，不删除脚本、不启动GPU诊断。未来是否复用无法绝对判定；优先级按当前主线、现存输入、调用关系和未提交工作给出。

- P1：优先删除候选，6个。旧固定任务已结束或其硬编码输入不存在；删除后失去对应诊断的快捷重跑入口。
- P2：建议先归档，6个。一次性启动器、旧数据生成或旧实验汇总，不能称为完全无用。
- P3：低删除优先级。旧路线仍被测试/其他入口引用，或属于权重复现链。
- 保留：当前主线、复用审计、数据工具、未提交工作。

`scripts/__pycache__/` 可直接清理，但不属于脚本。删除任何源码前还需同步其文档/调用方；静态搜索未发现引用不等于没有外部调用。

| 文件 | 优先级 | 依据 |
|---|---|---|
| `analyze_homography_extrapolation.py` | 保留 | audit_stable_extrapolation 和测试直接导入。 |
| `analyze_tiny_conditions.py` | P2 | 旧 MHIR tiny 条件分析；源结果仍存在，属于归档候选而非失效代码。 |
| `audit_shared_pretraining.py` | 保留 | 可复用特征对齐和梯度审计。 |
| `audit_stable_extrapolation.py` | 保留 | 稳定数据几何外推审计，测试直接导入。 |
| `audit_stable_geometry.py` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `check_batched_forward.py` | 保留 | 支持传入runtime，可复用的batch一致性校验。 |
| `check_correlation_recompute.py` | P1 | 旧 MHIR 相关性重计算诊断，写死已不存在的 temporal4 数据。 |
| `check_dense_migration.py` | P3 | 密集下游迁移一致性校验；仅放弃该对照路线后考虑归档。 |
| `check_frozen_dino.py` | P1 | 一次性两步冒烟，写死 temporal4 runtime；该数据目录不存在。 |
| `check_frozen_dino_mvt.py` | P1 | 一次性两步冒烟，写死旧 scale_pairs runtime；该数据目录不存在。 |
| `check_random_initialization.py` | 保留 | 未跟踪的独立消融工作，不纳入此次删除。 |
| `check_semidense_resume.py` | 保留 | 可复用恢复一致性验证；删除旧输出不使它失效。 |
| `check_shared_bundle.py` | 保留 | 共享导出/训练权重一致性验证。 |
| `check_shared_resume.py` | 保留 | 参数化共享训练恢复比较。 |
| `compare_shared_backtest.py` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `diagnose_adaptation_regressions.py` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `diagnose_batch_precision.py` | P1 | 旧 batch 数值漂移定位，写死已不存在的 scale_pairs 数据。 |
| `evaluate_shared_full.py` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `evaluate_shared_overlap.py` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `generate_stable_three_tiers.sh` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `generate_temporal_dataset.sh` | P2 | single_parent_v2 生成入口，旧路线未用于当前主线；它本就用于创建数据，目录不存在不构成脚本失效。 |
| `generate_test_tier3.sh` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `generate_three_tiers.sh` | P2 | 旧 quadrant_tiers_v1 生成入口，主线换用 stable_v2；旧数据仍存在，不能说已损坏。 |
| `launch_semidense_lr_compare.py` | P2 | 固定 A/B 两组的一次性启动器，产物/登记已存在时主动拒绝重跑；保留作实验复现模板也有价值。 |
| `localize_semidense_reports.py` | 保留 | 中文看板进程正在使用。 |
| `preview_dataset_tiers.py` | 保留 | 未跟踪且可切换root复用的数据预览。 |
| `probe_temporal_batch.py` | P1 | 旧 temporal4 batch 性能探测，写死已不存在的数据和旧配置。 |
| `refit_regression_diagnostics.py` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `run_shared_stable_full.py` | 保留 | 三个 stable_v2 shell 入口的共同实现。 |
| `summarize_batch_visualization.py` | P1 | 固定读取已删除的 engineering_visualization_seven_smoke/run.json，当前直接运行会缺文件。 |
| `summarize_pretraining.py` | P2 | 旧 MHIR tiny 汇总器，仍可重建报告；不是当前共享预训练主线的通用汇总器。 |
| `test.sh` | P3 | 旧模型原图可视化入口；入口测试仍引用。 |
| `test_e00.sh` | P3 | H0 独立基线；入口测试仍引用，不等同于冒烟脚本。 |
| `train_e01.sh` | P3 | 旧 MHIR/E01 训练；tests/test_entrypoints.py 引用，且存在未提交修改。 |
| `train_frozen_dino.sh` | P3 | 旧 GHIM/MHIR 路线入口，可归档；依赖 train_frozen_dino_mvt.sh。 |
| `train_frozen_dino_mvt.sh` | P3 | 被 train_frozen_dino、tier1_one_epoch_a/b 及 test_temporal_ready 引用，不能单独删。 |
| `train_semidense_next_tier.py` | 保留 | A 第二档正在使用；阶段接续入口。 |
| `train_shared_descriptor.sh` | P3 | 早期共享预训练入口；重要权重初始化链的复现代码。 |
| `train_shared_descriptor_lora.sh` | P3 | LoRA 对照入口，尚有对照价值。 |
| `train_shared_descriptor_tier2.sh` | P3 | 早期共享第二档入口；保留历史课程复现。 |
| `train_shared_descriptor_tier3.sh` | P3 | 早期共享第三档入口；保留历史课程复现。 |
| `train_shared_stable_v2_full_tier1.sh` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `train_shared_stable_v2_full_tier2.sh` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `train_shared_stable_v2_full_tier3.sh` | 保留 | 仍用于稳定数据生成、验收、共享主线复现或回归/遗忘评估；暂无删除依据。 |
| `train_shared_stable_v2_tier1_adapt.sh` | P2 | 2000步筛查实验已结束，现主线为完整三档；仅复现该筛查时需要。 |
| `train_tier1_one_epoch_a.sh` | 保留 | 未跟踪的旧独立实验；不是当前半密集 Run A，先保留。 |
| `train_tier1_one_epoch_b.sh` | 保留 | 未跟踪的旧独立实验；不是当前半密集 Run B，先保留。 |
| `unit_tests.sh` | 保留 | 单元测试统一入口。 |
| `validate_dense_loma.sh` | P3 | 仍可做密集/半密集对照，非当前训练入口。 |
| `verify_stable_dataset.py` | 保留 | 稳定数据完整性审计，测试直接导入。 |

## 核实的关键事实

- `runtime_paths.temporal4.server.json` 的 GoogleEarth_temporal4_v1 不存在。
- `runtime_paths.server.json` 的 GoogleEarth_scale_pairs 不存在。
- `runtime_paths.single_parent.server.json` 的 GoogleEarth_single_parent_v2 不存在。
- quadrant_tiers_v1 与 quadrant_tiers_stable_v2 数据目录存在。
- train_frozen_dino_mvt.sh 是多个包装脚本的公共入口，不能独立删。
- 当前清单未修改或删除原有文件；包括已有未提交改动和用户已删除的文档。
