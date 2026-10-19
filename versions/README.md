# MHI 版本档案

MHI v1 和 MHI v2 分开存放，先从对应版本入口进入。

| 内容 | MHI v1 | MHI v2 |
| --- | --- | --- |
| 版本入口 | [v1/README.md](v1/README.md) | [v2/README.md](v2/README.md) |
| 源码工作目录 | `/home/disk1/MHINet-v1`，detached `mhi-v1` | `/home/disk1/MHINet`，`codex/mhi-v2` |
| 档案 | `versions/v1/` | `versions/v2/` |
| 算法代码 | 标签 `mhi-v1` 中的原代码 | 新增专属实现进入 `mhinet/v2/` |
| 配置 | 冻结快照 `versions/v1/configs/`，历史原件 `versions/v1/historical/configs/` | `configs/mhi_v2/` |
| 专属脚本与测试 | v1 工作目录中的原入口 | `scripts/mhi_v2/`、`tests/v2/` |
| 权重和原始运行结果 | 使用冻结清单中的原路径 | `weight/mhi_v2/`、`outputs/mhi_v2/` |
| 验收证据 | 历史报告在 `versions/v1/reports/` | `versions/v2/evidence/` |

v1 的 Git 标签保持 `522d9bd5565e88267e72866bd829fb05274762e4`，不为整理文件移动标签。v1 冻结清单保持字节不变，[路径迁移表](relocations.json) 记录旧档案位置到新位置的对应关系。v1 中的小型配置/报告来自带哈希的历史原件；大权重、数据及历史预测不复制、不搬迁。

当前仓库原 `mhinet/models/`、`mhinet/downstream/`、默认 CLI、`configs/` 历史文件和 `artifacts/loma_downstream_source.json` 保留为 v1 兼容底座，不能因分档破坏旧入口。v2 复用它们的稳定功能时使用明确 import，新增算法和入口只写入 v2 目录。历史名称 `stable_v2`、`single_parent_v2` 和协议 `v1.2` 仍属于旧研究命名，不代表本次 MHI v2。

当前唯一实现的 v2 专属脚本是准备预检。P1–P5 的算法、训练与精度验收按 [v2 清单](v2/ACCEPTANCE.md) 逐条进行。
