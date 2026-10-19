# MHI v2 实验工作区

分支：`codex/mhi-v2`。v1 基线单独保存在 [../v1/](../v1/README.md) 与源码工作区 `/home/disk1/MHINet-v1`。

| 入口 | 用途 |
| --- | --- |
| [PLAN.md](PLAN.md) | 研究边界、五阶段方案及 A–F 消融矩阵 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 27项逐条执行清单、前置、证据及通过标准 |
| [acceptance_status.json](acceptance_status.json) | 每项实际状态、证据引用和下一项 |
| [experiment_plan.json](experiment_plan.json) | 机器可读研究登记，不能作为训练配置执行 |
| [proposal.md](proposal.md) | 用户原始方案档案 |
| [evidence/preparation.json](evidence/preparation.json) | 本轮实际资源预检报告，仅验收准备阶段 |

从当前仓库根目录执行准备预检：

```bash
python scripts/mhi_v2/preflight.py \
  --baseline-worktree /home/disk1/MHINet-v1 \
  --verify-large-hashes \
  --output versions/v2/evidence/preparation.json
```

报告已存在时默认拒绝覆盖；复验使用新的输出文件名以保留证据。此命令只读检查版本身份、分档、资源哈希和数据清单，不加载模型或使用 GPU。

准备阶段通过后，下一项为 **V2-P1-01：接口与数值验收协议**。模型代码、训练与精度均按清单分别签收；没有实际证据的项目保持未开始。算法专属实现进入 `mhinet/v2/`，配置、入口、测试分别进入 `configs/mhi_v2/`、`scripts/mhi_v2/`、`tests/v2/`；大输出和新权重分别进入 `outputs/mhi_v2/`、`weight/mhi_v2/`。
