# MHI v1 冻结档案

代码固定为 `mhi-v1` → `522d9bd5565e88267e72866bd829fb05274762e4`。独立源码工作区：`/home/disk1/MHINet-v1`。此目录保存基线资料，MHI v2 的计划和新增证据位于 [../v2/](../v2/README.md)。

| 文件 | 用途 |
| --- | --- |
| [freeze_manifest.json](freeze_manifest.json) | 原冻结清单，字节与原 SHA256 保持不变 |
| [archive_index.json](archive_index.json) | 每份新增归档的来源、提交和 SHA256 |
| [configs/training_snapshot.json](configs/training_snapshot.json) | 正式权重内嵌的完整训练元数据 |
| [configs/runtime_snapshot.json](configs/runtime_snapshot.json) | 原 runtime 对象快照，保留历史绝对路径 |
| [reports/H0_position_ablation.md](reports/H0_position_ablation.md) | v1 精度基线和指标定义 |
| `reports/h0_position_ablation/` | 历史计划、摘要、审计及两个数据集 A 组报告的小文件副本 |
| `historical/configs/`、`historical/scripts/` | 从冻结清单登记的历史提交取回的原始配置与脚本，按原始哈希校验 |
| [provenance/loma_downstream_source.json](provenance/loma_downstream_source.json) | v1 来源核登记副本，旧入口依赖的原位置仍保留 |

正式权重为 `weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt`，SHA256 为 `43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4`，进度 25535/25535。大资源仍使用冻结清单中的绝对路径。

独立源码工作区只隔离源码。旧 evaluator 仍从 checkpoint 读取历史绝对 runtime 路径，该路径在原 main 中缺失；此处的归档副本不会自动改变旧 evaluator 的读取行为。进行 v1 回放前必须显式加载归档 runtime、映射源码/输出目录，并记录实际 `mhinet.__file__`，避免误用 v2 的 editable install。v1 回放输出独立放入 `outputs/mhi_v1_replay/`，不能覆盖历史预测。准备预检通过不代表 GPU 回放已经通过。

冻结清单中的旧相对档案引用可查 [../relocations.json](../relocations.json)。不要为修复这些历史引用而重写冻结清单或原权重。
