# 项目清理候选清单（2026-09-20）

状态：2026-09-20 用户批准后，第一批47个checkpoint和11个缓存目录已按精确清单删除；下文保留原筛选说明。删除前核对权重大小、真实路径、缓存内容及运行进程，删除后确认47个权重均不存在。未进入回收站，无法直接恢复，工程测试可重跑；实际outputs占用由约152G降至102G。后续运行Python会重新生成缓存。第二批和明确保留项未删除。

## 第一批：建议确认后删除

- Python 缓存：11 个 __pycache__ 目录，合计约 0.88 MiB，可重新生成。
- 工程测试 checkpoint：47 个 .pt，逻辑大小合计 50.36 GiB。仅删除下列清单中的权重，不删除所在目录的日志、JSON、图片、报告或脚本。
- 删除这些权重后，不能直接恢复当时的冒烟训练或检查其参数；可以重新运行工程测试。正式模型不在此清单中。
- 不依据名字推断内容完全相同；本次没有做大文件逐字节去重，空间为逻辑大小，实际释放量受文件系统快照/共享块影响。

| 工程输出目录 | 权重数 | GiB |
|---|---:|---:|
| outputs/diagnostics/shared_frozen_deterministic_continuous | 2 | 2.05 |
| outputs/diagnostics/shared_frozen_deterministic_split | 2 | 2.05 |
| outputs/diagnostics/shared_frozen_final_continuous | 2 | 2.05 |
| outputs/diagnostics/shared_frozen_final_split | 2 | 2.05 |
| outputs/diagnostics/shared_frozen_smoke | 2 | 2.05 |
| outputs/diagnostics/shared_frozen_strict_continuous | 2 | 2.05 |
| outputs/diagnostics/shared_frozen_strict_split | 2 | 2.05 |
| outputs/diagnostics/shared_frozen_v1 | 2 | 2.05 |
| outputs/diagnostics/shared_lora_final_continuous | 2 | 2.06 |
| outputs/diagnostics/shared_lora_final_split | 2 | 2.06 |
| outputs/diagnostics/shared_lora_smoke | 2 | 2.06 |
| outputs/diagnostics/shared_lora_strict_continuous | 2 | 2.06 |
| outputs/diagnostics/shared_lora_strict_split | 2 | 2.06 |
| outputs/diagnostics/shared_lora_v1 | 2 | 2.06 |
| outputs/diagnostics/shared_tier2_overlap_smoke | 2 | 2.05 |
| outputs/diagnostics/shared_tier3_smoke | 2 | 2.05 |
| outputs/diagnostics/tier1_afss_continuous | 1 | 1.66 |
| outputs/diagnostics/tier1_afss_resume | 1 | 1.66 |
| outputs/diagnostics/tier1_afss_resume_v2 | 2 | 3.31 |
| outputs/diagnostics/tier1_random_init_smoke | 1 | 1.66 |
| outputs/diagnostics/tier1_uniform_smoke | 1 | 1.66 |
| outputs/engineering_visualization_seven_smoke | 1 | 0.84 |
| outputs/engineering_batch_visualization_smoke_bs1 | 2 | 1.68 |
| outputs/mhinet_minimal_smoke | 1 | 0.84 |
| outputs/mhinet_resume_smoke | 2 | 1.69 |
| outputs/mhinet_resume_smoke_mcnet_d2_v2 | 2 | 1.67 |
| outputs/mhinet_uninterrupted_smoke_mcnet_d2_v2 | 1 | 0.84 |

## 第二批：暂缓，需要逐个核对正式 checkpoint 引用

E01_heads（约18G）、三个 GHIM_joint_*temporal4* 正式训练目录（约21G、21G、34G）有大量中间checkpoint。历史跨时相标签存在问题，但这些结果仍是实验溯源证据，不能直接认定整个目录无用。后续可保留最终、最佳、已评估及被其他配置引用的checkpoint，再单列其余文件；本清单不授权删除任何正式checkpoint。

## 明确保留

- 四个 shared_descriptor 正式输出：frozen_seed0、lora_seed0、frozen_tier2_seed0、frozen_tier3_seed0；latest.pt 与 shared_descriptor.pt 用途不同，不能作为重复文件合并。
- outputs/shared_tier2_backtest_tier1，所有正式评估、逐对指标、可视化和评价总表。
- outputs/tiny_overfit、loss_spikes_8293_8480 诊断图片和脚本、稳定几何v2试生成及预览。新版全量尚未在本轮验收，保留试生成证据。
- docs、artifacts、implementation_log.md、原始设计包及其history。history不作为实施指令，不等于应删除。
- 所有测试代码、batch/precision/resume审计脚本。历史问题回归测试仍有用途。
- Git脏工作区中的已有改动和未跟踪脚本，包括initialization.py、preview_dataset_tiers.py、两组tier1配置/训练脚本等。未跟踪不等于没用。
- 本项目之外的数据、LoMa、预训练权重不在本次范围。

## 引用与运行检查

检查了当前 configs/scripts/mhinet 中显式工程输出路径，scripts/summarize_batch_visualization.py 仍读取工程可视化目录，因此不删整个目录；仅候选.pt。文档和实验报告中的历史引用保留。未做动态路径依赖完备性证明，删除前需再次检查运行进程与引用。盘点时没有发现训练/生成/评估进程，只有用户 tail 日志进程；这不代表未来执行删除时也无进程。

## 精确文件清单

机器可读清单：artifacts/cleanup_candidates_20260920.json。第一批仅限其中 checkpoint_candidates 与 cache_candidates，执行前复核文件未变化。建议按清单移动至明确的临时回收目录，确认无误再永久删除；移动本身不释放空间。

- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_deterministic_continuous/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_deterministic_continuous/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_deterministic_split/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_deterministic_split/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_final_continuous/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_final_continuous/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_final_split/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_final_split/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_smoke/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_smoke/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_strict_continuous/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_strict_continuous/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_strict_split/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_strict_split/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_v1/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_frozen_v1/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_final_continuous/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_final_continuous/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_final_split/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_final_split/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_smoke/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_smoke/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_strict_continuous/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_strict_continuous/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_strict_split/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_strict_split/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_v1/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_lora_v1/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_tier2_overlap_smoke/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_tier2_overlap_smoke/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_tier3_smoke/latest.pt`
- `/home/disk1/MHINet/outputs/diagnostics/shared_tier3_smoke/shared_descriptor.pt`
- `/home/disk1/MHINet/outputs/diagnostics/tier1_afss_continuous/checkpoints/step_0000007.pt`
- `/home/disk1/MHINet/outputs/diagnostics/tier1_afss_resume/checkpoints/step_0000005.pt`
- `/home/disk1/MHINet/outputs/diagnostics/tier1_afss_resume_v2/checkpoints/step_0000005.pt`
- `/home/disk1/MHINet/outputs/diagnostics/tier1_afss_resume_v2/checkpoints/step_0000007.pt`
- `/home/disk1/MHINet/outputs/diagnostics/tier1_random_init_smoke/checkpoints/step_0000002.pt`
- `/home/disk1/MHINet/outputs/diagnostics/tier1_uniform_smoke/checkpoints/step_0000004.pt`
- `/home/disk1/MHINet/outputs/engineering_visualization_seven_smoke/checkpoints/step_0000002.pt`
- `/home/disk1/MHINet/outputs/engineering_batch_visualization_smoke_bs1/checkpoints/step_0000001.pt`
- `/home/disk1/MHINet/outputs/engineering_batch_visualization_smoke_bs1/checkpoints/step_0000002.pt`
- `/home/disk1/MHINet/outputs/mhinet_minimal_smoke/checkpoints/step_0000002.pt`
- `/home/disk1/MHINet/outputs/mhinet_resume_smoke/checkpoints/step_0000001.pt`
- `/home/disk1/MHINet/outputs/mhinet_resume_smoke/checkpoints/step_0000002.pt`
- `/home/disk1/MHINet/outputs/mhinet_resume_smoke_mcnet_d2_v2/checkpoints/step_0000001.pt`
- `/home/disk1/MHINet/outputs/mhinet_resume_smoke_mcnet_d2_v2/checkpoints/step_0000002.pt`
- `/home/disk1/MHINet/outputs/mhinet_uninterrupted_smoke_mcnet_d2_v2/checkpoints/step_0000002.pt`
