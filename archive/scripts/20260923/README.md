# 2026-09-23 脚本归档

用户批准将审查清单中的 P2 六个脚本归档。源码原样保留，SHA256见 manifest.json。

这些是历史实验入口，不是当前运行入口。部分脚本通过自身位置定位项目根目录，因此**不要直接在归档位置运行**。需要复现时，先将指定文件复制回项目的 `scripts/` 原路径，确认没有同名文件，再核对数据、配置、输出目录和 GPU。固定 A/B 启动器遇到已有输出或登记会拒绝运行。

| 文件 | 归档原因 |
|---|---|
| launch_semidense_lr_compare.py | 第一档 A/B 一次性启动已完成 |
| train_shared_stable_v2_tier1_adapt.sh | 旧2000步适应筛查 |
| generate_three_tiers.sh | 旧 quadrant_tiers_v1 生成入口 |
| generate_temporal_dataset.sh | 旧 single_parent_v2 生成入口 |
| analyze_tiny_conditions.py | 旧 MHIR tiny 条件分析 |
| summarize_pretraining.py | 旧 MHIR tiny 证据汇总 |

原文档里的 `scripts/文件名` 是历史执行路径，可在此找到归档版本。P1 六个脚本已删除，需要恢复时可查 Git 历史。
