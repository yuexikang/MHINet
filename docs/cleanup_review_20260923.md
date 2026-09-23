# 项目清理审查清单（2026-09-23）

盘点时间：2026-09-23T10:07:10.144255+00:00。只整理审查资料，未删除、移动或改写现有代码、权重、日志及验证结果。

## 建议先审这三类

| 类别 | 数量/逻辑占用 | 建议 |
|---|---:|---|
| 可重建缓存 | 14 个目录，0.89 MiB | 可清理；运行程序会再生成 |
| 工程验证权重 | 8 个，7.89 GiB | 建议仅删除权重，保留配置、测试、日志和审计结论；会失去原权重复验能力 |
| 历史正式训练权重 | 80 个，90.11 GiB | 需要逐个审查，不能把此总量当作可释放量；最终/最佳/已评估/被引用者优先保留 |

outputs 目录实际占用约 129 GiB（du 盘点），逻辑文件大小与实际占用有差异。没有发现 .tmp/.bak/.orig/.rej 或 ~ 结尾的遗留临时文件。磁盘空间充足，优先减少混淆而非追求删除量。

## 正在运行与保护链

- `3861460    05:09:24 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.downstream.train --config /home/disk1/MHINet/configs/semidense_tier1_lr_c.json --checkpoint /home/disk1/MHINet/outputs/shared_stable_v2_full_tier3_seed0/shared_descriptor.pt --output /home/disk1/MHINet/outputs/semidense_stable_v2_tier1_lr_c_seed0`
- `3867440       18:37 /root/miniconda3/envs/loma-repro/bin/python -u scripts/train_semidense_next_tier.py --config configs/semidense_tier2_lr_a.json --checkpoint outputs/semidense_stable_v2_tier1_lr_a_seed0/latest.pt --output /home/disk1/MHINet/outputs/semidense_stable_v2_tier2_lr_a_seed0`
- `3867507       17:56 /root/miniconda3/envs/loma-repro/bin/python scripts/localize_semidense_reports.py --watch`

保护：stable_v2 三档共享训练及其前序 shared_descriptor 冻结三档、LoRA 对照；半密集第一档 A/B/C 和 A 第二档；训练所用 DINO/LoMa 外部权重。A 第二档仍依赖 A 第一档 final checkpoint，A 第一档依赖 shared_stable_v2_full_tier3 的导出。latest.pt 与 shared_descriptor.pt 分别用于恢复训练/模型导出，不是重复文件。

当前训练模块的实现哈希参与恢复校验。不要在清理中移动或重写 mhinet/downstream 下文件。项目外数据和权重不在本清单范围。

## 第一批：工程权重精确候选

这些多为成功的冒烟/恢复测试，不是失败模型。清理理由是可重跑、不是正式模型；引用信息见 JSON。保留所在目录其他文件。

| 文件 | GiB | 显式引用数 |
|---|---:|---:|
| `outputs/diagnostics/stable_v2_tier1_adapt_smoke/latest.pt` | 1.64 | 0 |
| `outputs/diagnostics/stable_v2_tier1_adapt_smoke/shared_descriptor.pt` | 0.41 | 0 |
| `outputs/semidense_qrru_engineering_v1/latest.pt` | 0.97 | 1 |
| `outputs/semidense_qrru_replay_v2/full/latest.pt` | 0.97 | 1 |
| `outputs/semidense_qrru_replay_v2/split/latest.pt` | 0.97 | 1 |
| `outputs/semidense_qrru_resume_v1/latest.pt` | 0.97 | 0 |
| `outputs/semidense_visual_smoke_v1/latest.pt` | 0.97 | 0 |
| `outputs/semidense_visual_smoke_v2/latest.pt` | 0.97 | 0 |

其中 replay_v2/full 与 split 是逐位恢复一致性验证的两份证据，不能仅凭用途相近认定内容重复。engineering_v1 也被工程级联评估引用，删除后无法用原权重复算，只剩已归档结论。

## 第二批：历史正式权重逐个审查

四组旧训练均保留原始日志、逐对结果、验证摘要和最终权重。不是因历史数据局限就删除全部结果。下表列出每个权重的验证/引用证据；“无显式引用”不证明无动态依赖；尚未统一复算各组最佳指标，暂不指定最佳权重。

| 文件 | GiB | 验证摘要数 | 显式引用数 | 建议 |
|---|---:|---:|---:|---|
| `outputs/E01_heads/checkpoints/step_0000500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0001000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0001500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0002000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0002500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0003000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0003500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0004000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0004500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0005000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0005500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0006000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0006500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0007000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0007500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0008000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0008500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0009000.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0009500.pt` | 0.84 | 1 | 2 | 历史正式权重待审 |
| `outputs/E01_heads/checkpoints/step_0010000.pt` | 0.84 | 1 | 3 | 保留历史最终权重 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0001000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0002000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0003000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0004000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0005000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0006000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0007000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0008000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0009000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0010000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0011000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0012000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0013000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0014000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0015000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0016000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0017000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0018000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0019000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0/checkpoints/step_0020000.pt` | 1.00 | 1 | 3 | 保留历史最终权重 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0000500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0001000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0001500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0002000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0002500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0003000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0003500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0004000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0004500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0005000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0005500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0006000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0006500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0007000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0007500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0008000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0008500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0009000.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0009500.pt` | 1.00 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0/checkpoints/step_0010000.pt` | 1.00 | 1 | 3 | 保留历史最终权重 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0001000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0002000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0003000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0004000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0005000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0006000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0007000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0008000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0009000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0010000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0011000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0012000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0013000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0014000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0015000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0016000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0017000.pt` | 1.66 | 1 | 3 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0018000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0019000.pt` | 1.66 | 1 | 2 | 历史正式权重待审 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0020000.pt` | 1.66 | 1 | 3 | 保留历史最终权重 |

## 失败、中断与退化的区别

- `outputs/batch_precision_diagnosis.log:26`：Float/BFloat16 类型错误。后续 fp32_position 日志已有数值诊断；旧失败日志体积很小，建议作为修复证据保留。
- `outputs/engineering_batch_visualization_smoke_start.log:23`：CUDA OOM；bs1 目录是后续运行。旧启动目录可归档，但留错误日志。
- `outputs/tier1_a_console.log`、`tier1_b_console.log`：末尾是 KeyboardInterrupt，属中断，不足以认定实现失败；也不是当前 semidense A/B。相关未提交代码需要保留。
- `outputs/tiny_overfit/mcnet_d4_full.log`、`mcnet_d8_full.log`：早段有 Traceback，尾部已有输出摘要。不能按字符串命中删除整份实验。
- `shared_stable_v2_tier1_adapt_seed0`：2000步是已登记的短程预算，虽调度总长12995步，也不应标成意外未完成；H0退化案例及回归诊断保留。
- 空启动/小目录 E01_heads_bs_2、E01_heads_seed0、GHIM_joint_frozen_dino_mvt_temporal4_seed0：只有启动状态不足以证明最终失败，列为可归档而非直接删除。

## 代码、配置、文档的整理建议

| 范围 | 发现/依据 | 建议 |
|---|---|---|
| mhinet/downstream/loma_reference | semidense.py、supervision.py、visualize.py 直接导入 | 必须保留，不能按旧密集实现删除 |
| dense.py、evaluate.py、密集匹配 tests | CLI 仍有 evaluate-dense；check_dense_migration 和回归测试引用 | 保留为密集对照；可在目录索引标为可选路线 |
| mhinet/engine、models/iterator 等 MHIR 路线 | 旧训练入口和测试仍使用 | 可标记历史/可选路线；暂无足够证据认定死代码 |
| diagnostics、check_*、tests | 记录 batch/precision/resume 等回归 | 保留源代码，优先清产物权重 |
| temporal4 / quadrant旧数据 configs 和 generate脚本 | 历史实验复现入口 | 可归档分组，不能直接移走导致路径失效 |
| MHINet_server_handoff_v1.2 | 原始设计与history，占用约454 KiB | 保留文档档案，节省空间价值很低 |
| docs/shared_stable_v2_full_curriculum.md | 仍写后两档“尚未生成”，实际已完成 | 建议补完成状态并保留原登记时间；不是删掉 |
| artifacts/semidense_qrru_engineering.json | formal_training_started=false 是当时工程快照 | 保留快照，加索引说明已有正式实验 |
| docs/semidense_lr_comparison_results.md | 仍有“后续再启动第二档”的当时结论 | 补链接至新训练登记；保持评估结论历史可追溯 |
| Git 已有脏文件与未跟踪代码 | initialization、tier1脚本、registry等 | 全部保护，不纳入自动清理 |

本次没有确认可直接删除的生产源文件。删除旧路线需要单独做依赖迁移和回归验证，不能把当前未参与训练等同于无用。

## 所有输出目录一览

| 目录 | 逻辑 GiB | 最近训练步/配置总步 | 状态 | 建议 |
|---|---:|---|---|---|
| `outputs/E00_h0_seed0` | 0.00 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/E01_heads` | 17.18 | 10000/— | completed | 历史正式训练；逐个审查中间权重 |
| `outputs/E01_heads_bs_2` | 0.00 | 29/— | running | 保留结果/证据 |
| `outputs/E01_heads_seed0` | 0.00 | 104/— | running | 保留结果/证据 |
| `outputs/E01_heads_test_step10000` | 0.01 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0` | 20.61 | 20000/— | completed | 历史正式训练；逐个审查中间权重 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0_test_step20000` | 0.01 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0` | 20.60 | 10000/— | completed | 历史正式训练；逐个审查中间权重 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0_test_step10000` | 0.01 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/GHIM_joint_frozen_dino_mvt_temporal4_seed0` | 0.00 | 13/— | running | 保留结果/证据 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0` | 33.64 | 20000/— | completed | 历史正式训练；逐个审查中间权重 |
| `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0_test_step17000` | 0.02 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/diagnostics` | 2.49 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/e01_gate_refusal_smoke` | 0.00 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/engineering_batch_visualization_smoke` | 0.00 | —/— | running | 保留结果/证据 |
| `outputs/engineering_batch_visualization_smoke_bs1` | 0.01 | 2/— | completed | 保留结果/证据 |
| `outputs/engineering_visualization_seven_smoke` | 0.01 | 2/— | completed | 保留结果/证据 |
| `outputs/mhinet_minimal_smoke` | 0.00 | 2/— | completed | 保留结果/证据 |
| `outputs/mhinet_resume_smoke` | 0.00 | 2/— | completed | 保留结果/证据 |
| `outputs/mhinet_resume_smoke_mcnet_d2_v2` | 0.00 | 2/— | completed | 保留结果/证据 |
| `outputs/mhinet_uninterrupted_smoke_mcnet_d2_v2` | 0.00 | 2/— | completed | 保留结果/证据 |
| `outputs/previews` | 0.00 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/real_image_accuracy_val16` | 0.01 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/real_image_accuracy_val16_verified` | 0.01 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/semidense_qrru_engineering_v1` | 0.98 | 2/2 | 日志达到配置步数 | 仅权重列为可清理候选；保留日志和报告 |
| `outputs/semidense_qrru_replay_v2` | 1.94 | —/— | 无统一运行状态 | 仅权重列为可清理候选；保留日志和报告 |
| `outputs/semidense_qrru_resume_v1` | 0.97 | 2/2 | 日志达到配置步数 | 仅权重列为可清理候选；保留日志和报告 |
| `outputs/semidense_stable_v2_tier1_lr_a_seed0` | 3.62 | 12995/12995 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/semidense_stable_v2_tier1_lr_b_seed0` | 3.64 | 12995/12995 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/semidense_stable_v2_tier1_lr_c_seed0` | 2.07 | 5940/12995 | 训练中，进程已核对 | 保留主线/对照/初始化链 |
| `outputs/semidense_stable_v2_tier2_lr_a_seed0` | 1.30 | 325/8663 | 训练中，进程已核对 | 保留主线/对照/初始化链 |
| `outputs/semidense_visual_smoke_v1` | 1.03 | 1/1 | 日志达到配置步数 | 仅权重列为可清理候选；保留日志和报告 |
| `outputs/semidense_visual_smoke_v2` | 1.03 | 1/1 | 日志达到配置步数 | 仅权重列为可清理候选；保留日志和报告 |
| `outputs/shared_descriptor_frozen_seed0` | 2.24 | 12995/12995 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/shared_descriptor_frozen_tier2_seed0` | 2.16 | 8663/8663 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/shared_descriptor_frozen_tier3_seed0` | 2.16 | 8663/8663 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/shared_descriptor_lora_seed0` | 2.24 | 12995/12995 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/shared_stable_v2_full_tier1_seed0` | 2.25 | 12995/12995 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/shared_stable_v2_full_tier2_seed0` | 2.16 | 8663/8663 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/shared_stable_v2_full_tier3_seed0` | 2.16 | 8663/8663 | 日志达到配置步数 | 保留主线/对照/初始化链 |
| `outputs/shared_stable_v2_tier1_adapt_seed0` | 2.11 | 2000/12995 | 按登记2000步完成短程实验；非全轮失败 | 保留主线/对照/初始化链 |
| `outputs/shared_tier2_backtest_tier1` | 0.05 | —/— | 无统一运行状态 | 保留主线/对照/初始化链 |
| `outputs/shared_tier3_stable_v2_baseline_tier1` | 0.05 | —/— | 无统一运行状态 | 保留主线/对照/初始化链 |
| `outputs/shared_tier3_stable_v2_baseline_tier2` | 0.04 | —/— | 无统一运行状态 | 保留主线/对照/初始化链 |
| `outputs/shared_tier3_stable_v2_baseline_tier3` | 0.04 | —/— | 无统一运行状态 | 保留主线/对照/初始化链 |
| `outputs/stable_geometry_smoke_v2` | 0.08 | —/— | 无统一运行状态 | 保留结果/证据 |
| `outputs/tier1_a_seed0` | 0.00 | 434/— | running | 保留结果/证据 |
| `outputs/tier1_b_seed0` | 0.00 | 434/— | running | 保留结果/证据 |
| `outputs/tiny_overfit` | 0.51 | —/— | 无统一运行状态 | 保留结果/证据 |

“日志达到配置步数”只说明训练循环进度，不自动证明全量验证已完成或模型有效。预览、基线、回测、工程审计即使没有 run.json 也不属于垃圾。所有正式 validation/test 的摘要、逐对指标与可视化建议保留，不同数据版本/split/GT监督与纯预测级联结果不能混合比较。

## 审查方式

建议分别决定：①仅清缓存；②清上列8个工程权重；③旧正式权重保留哪些步点；④仅给历史代码/文档加分类索引。当前不执行任何删除。

逐文件路径、字节数、mtime、显式引用位置和错误行号见 `artifacts/cleanup_review_20260923.json`；目录表可用 `artifacts/cleanup_review_20260923_outputs.csv` 筛选。执行任何清理前需重新核对运行进程、文件时间/大小和实际依赖；此次静态审查不是动态依赖完备性证明。

2026-09-20 已删除的47个工程checkpoint不再计入本轮候选；原清理审计保留。
