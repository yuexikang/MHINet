# MHI v2 实验工作区

分支：`codex/mhi-v2`。v1 基线单独保存在 [../v1/](../v1/README.md) 与源码工作区 `/home/disk1/MHINet-v1`。

2026-10-20（北京时间）状态：`mhinet/v2/` 的 Flow0、D8 soft prior、Direct-D8、Discrepancy/DGFO、严格迁移与训练/评价入口已实现；57 项 CPU 测试通过，[真实 GPU 工程验收](evidence/real_feature_acceptance_20261020_final.json) 为 `passed`。Flow Head 的 2 步实数据训练烟测及断点恢复也已通过。正式长程已于北京时间2026-10-20 03:24:37在 GPU0 启动 Flow Head，C/D/E 排队；实际步数和完成状态以调度输出目录中的 `campaign.json` 及各阶段 `status.json` 为准。上述短程证据不代表完整训练或精度收益。

当前 v2 约定已实现：输入只支持 **512×512**；最终输出上限 **`max_output_matches=2048` 对/影像对**，可显式设置正整数。最终按置信度稳定选取，不足时不补点。D16/D8/D4/D2 分别为32/64/128/256；QRRU 前候选预算 **`max_matches=12000`** 与最终输出上限分开登记。

| 入口 | 用途 |
| --- | --- |
| [PLAN.md](PLAN.md) | 研究边界、五阶段方案及 A–F 消融矩阵 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 27项逐条执行清单、前置、证据及通过标准 |
| [acceptance_status.json](acceptance_status.json) | 每项实际状态、证据引用和下一项 |
| [experiment_plan.json](experiment_plan.json) | 机器可读研究登记，不能作为训练配置执行 |
| [proposal.md](proposal.md) | 用户原始方案档案 |
| [evidence/preparation.json](evidence/preparation.json) | 本轮实际资源预检报告，仅验收准备阶段 |
| [evidence/real_feature_acceptance_20261020_final.json](evidence/real_feature_acceptance_20261020_final.json) | 两对真实 val 与三步 QRRU 梯度诊断的 GPU 工程验收 |
| [evidence/resource_acceptance_20261020.json](evidence/resource_acceptance_20261020.json) | 已通过的资源与耗时/显存验收 |
| [campaign.json](../../outputs/mhi_v2/campaign_rgb512_seed42_20261020/campaign.json) | 已启动 Flow Head 与排队 C/D/E 的实时状态 |
| [训练配置](../../configs/mhi_v2/README.md) | Flow Head → 同预算 C/D/E 的独立运行配置 |
| [脚本入口](../../scripts/mhi_v2/README.md) | preflight、真实验收与串行训练调度 |

P0 准备阶段的历史命令（该脚本校验当时的 planned 状态；当前模型工程复验使用 `validate_real.py` 并指定新的证据路径）：

```bash
python scripts/mhi_v2/preflight.py \
  --baseline-worktree /home/disk1/MHINet-v1 \
  --verify-large-hashes \
  --output versions/v2/evidence/preparation.json
```

报告已存在时默认拒绝覆盖；复验使用新的输出文件名以保留证据。此命令只读检查版本身份、分档、资源哈希和数据清单，不加载模型或使用 GPU。

P0 准备阶段4项、62项资源预检与8项分档核验的历史证据保持原样。后续逐项签收以 [acceptance_status.json](acceptance_status.json) 为准，工程、训练与精度分别登记。算法专属实现进入 `mhinet/v2/`，配置、入口、测试分别进入 `configs/mhi_v2/`、`scripts/mhi_v2/`、`tests/v2/`；大输出和新权重分别进入 `outputs/mhi_v2/`、`weight/mhi_v2/`。

V2-P2-04 的完整 D8/D2 retention/coverage 诊断仍待完成，清单保持未签收；P5 精度验收尚未开始。工程报告通过不能替代这些证据。

正式协议登记为：Flow Head 先跑 seed42、5 epoch、25535 次 optimizer 更新，再从同一 Flow Head 最佳权重独立初始化 C/D/E，各跑相同 seed42、5 epoch、25535 次更新。QRRU 每批目标为 50% 合格预测 D2 + 50% GT 有界噪声；预测修正范围为 6 D2 像素，合格预测不足时以噪声样本补齐，并记录实际比例。固定原 RGB_512 的 train=40849/val=5106，只用 train/val，不读取封存评价集。源图身份交叉为 0，尚不能证明地理邻近或真正空间隔离；本轮只有单 seed42，不能声称多 seed 稳定性。
