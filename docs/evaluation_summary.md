# 评估总表

> 文档状态（2026-10-14）：**历史结果**。本文是自动登记的旧 H0/MHIR 几何评估表，覆盖 E00 与工程权重，不是当前密集匹配实验总榜。不同 split、子集与指标版本的限制仍有效；重新扫描生成时可能覆盖本页状态提示，应同时阅读文档状态索引。
> 当前路线：[D8→D2 主路线](mainline_d8_d2.md)；分类依据：[文档状态索引](document_status_20261014.md)。

2026-10-14路径核对：下表7份工程权重summary的原输出路径已不存在；`artifacts/evaluation_registry.json`仍保留这7条记录的summary快照与原内容SHA。历史数字保留，原路径链接不表示对应运行目录仍可访问。

自动登记独立评估及训练中验证；每次以原始summary路径+内容SHA去重，同一路径的新结果保留为新记录。
只汇总已找到的真实影像评估summary；tiny-overfit、性能探测及单元测试不冒充精度评估。
不同split、影像子集、架构及指标版本不可直接排名。误差为有效输出上的统计，必须同时看失败率。
旧记录没有AUC或独立推理计时时显示“—”，不补造；旧中位数口径与新版本不同。

| 记录/原始结果 | 类型 | split/对数 | H0均值px | 最终均值px | 中位/P90 px | 成功@1/3/5 | AUC@1/3/5 | 失败率 | 显存GB | 推理/墙钟ms | 指标版本 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [E00_h0_seed0 · 678f86](/home/disk1/MHINet/outputs/E00_h0_seed0/metrics/summary.json) | 独立评估 / 预训练H0 | test/1000 | 15.413 | 15.413 | 9.491/25.342 | 0.000/0.029/0.137 | 0.000/0.004/0.031 | 0.000 | 1.138 | 47.849/108.631 | real_image_v2_ecdf_auc_quantile_median |
| [engineering_batch_visualization_smoke_bs1 · ea4902](/home/disk1/MHINet/outputs/engineering_batch_visualization_smoke_bs1/validation/step_0000001/summary.json) | 训练中验证 / 工程权重 | val/2 | 6.612 | 6.605 | 5.917/7.155 | 0.000/0.000/0.000 | —/—/— | 0.000 | 2.079 | —/1602.145 | legacy_lower_median |
| [engineering_batch_visualization_smoke_bs1 · 69a49b](/home/disk1/MHINet/outputs/engineering_batch_visualization_smoke_bs1/validation/step_0000002/summary.json) | 训练中验证 / 工程权重 | val/2 | 6.612 | 6.613 | 5.832/7.237 | 0.000/0.000/0.000 | —/—/— | 0.000 | 2.079 | —/1550.231 | legacy_lower_median |
| [engineering_visualization_seven_smoke · 3204e5](/home/disk1/MHINet/outputs/engineering_visualization_seven_smoke/validation/step_0000002/summary.json) | 训练中验证 / 工程权重 | val/2 | 6.612 | 6.613 | 5.832/7.238 | 0.000/0.000/0.000 | —/—/— | 0.000 | 2.079 | —/1714.687 | legacy_lower_median |
| [mhinet_minimal_smoke · 7bc01f](/home/disk1/MHINet/outputs/mhinet_minimal_smoke/validation/step_0000002/summary.json) | 训练中验证 / 工程权重 | val/2 | — | — | —/— | 0.000/0.000/0.000 | —/—/— | 0.000 | 1.648 | —/255.175 | legacy_lower_median |
| [mhinet_resume_smoke · d4c037](/home/disk1/MHINet/outputs/mhinet_resume_smoke/validation/step_0000002/summary.json) | 训练中验证 / 工程权重 | val/2 | — | — | —/— | 0.000/0.000/0.000 | —/—/— | 0.000 | 1.647 | —/393.355 | legacy_lower_median |
| [mhinet_resume_smoke_mcnet_d2_v2 · c509a6](/home/disk1/MHINet/outputs/mhinet_resume_smoke_mcnet_d2_v2/validation/step_0000002/summary.json) | 训练中验证 / 工程权重 | val/2 | 6.612 | 6.598 | 5.919/7.142 | 0.000/0.000/0.000 | —/—/— | 0.000 | 1.638 | —/138.223 | legacy_lower_median |
| [mhinet_uninterrupted_smoke_mcnet_d2_v2 · 350c19](/home/disk1/MHINet/outputs/mhinet_uninterrupted_smoke_mcnet_d2_v2/validation/step_0000002/summary.json) | 训练中验证 / 工程权重 | val/2 | 6.612 | 6.598 | 5.919/7.142 | 0.000/0.000/0.000 | —/—/— | 0.000 | 1.638 | —/121.653 | legacy_lower_median |
| [real_image_accuracy_val16 · 0abd3f](/home/disk1/MHINet/outputs/real_image_accuracy_val16/metrics/summary.json) | 独立评估 / 工程权重 | val/16 | 4.157 | 4.018 | 3.778/6.292 | 0.000/0.375/0.750 | 0.000/0.083/0.269 | 0.000 | 2.061 | 405.054/644.720 | legacy_lower_median |
| [real_image_accuracy_val16_verified · 01f854](/home/disk1/MHINet/outputs/real_image_accuracy_val16_verified/metrics/summary.json) | 独立评估 / 工程权重 | val/16 | 4.157 | 4.018 | 3.913/6.292 | 0.000/0.375/0.750 | 0.000/0.083/0.269 | 0.000 | 2.061 | 367.961/591.850 | real_image_v2_ecdf_auc_quantile_median |

成功率/AUC/失败率以[0,1]比例表示。显存列为peak allocated，reserved和H0～各轮完整轨迹见机器可读记录。
权重路径/hash、数据清单hash、原始summary快照保存在 `artifacts/evaluation_registry.json`；缺失的运行时hash保持缺失，不把当前磁盘文件hash冒充运行时证据。
登记时间是入表时间，source_mtime是原文件修改时间，不冒充未知的实验开始时间。
