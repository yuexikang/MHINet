# MHINet 服务器实施入口 · 交接 v1.2

2026-09-07。结构 v1.0 / 损失 v1.1 / 训练协议 v1.2。当前交付是设计和实施计划，尚无新模型训练结果。用户将在服务器另开会话实施。

## 当前唯一主线

保留现有 Stage1、共享 DINOv3/MVT 与 VGG/DeDoDe。DINOv3 始终冻结；MVT、VGG、DeDoDe 和新增层允许联合训练。Stage1 head 参数默认冻结，但联合训练时保留 head/fitter 对 MVT 输入的梯度。D8/D4/D2/D1 各两轮、共八轮 H-guided local correlation → CNN 四角残差 → 可微四角 H 重建。默认八轮等权 proposal 四角 L1。当前不实现 Planar Reliability。

## 阅读和实施顺序

1. 读本文件、[模型](docs/01_full_model.md)、[张量契约](docs/03_implementation_contract.md)。
2. 读[训练协议](docs/02_training_and_validation.md)、[可微性审计](docs/05_autograd_and_joint_training.md)。
3. 按[实施路线与验收](docs/06_implementation_roadmap.md)依次完成 P0–P6。
4. 按[实验表](docs/07_experiment_plan.md)执行 E00、E01，再进入 E02–E05；不要把所有可选实验列为首批任务。
5. 模型配置为 `configs/mhinet_v1.json`；可训练参数组为 `configs/training_profiles.json`。运行路径用 `configs/runtime_paths.example.json` 的实际填充值，源码审计见[复用边界](docs/08_source_reuse_and_provenance.md)。

配置是实施目标，当前尚无消费这些 JSON 的训练入口；配置变更须同步文档并记录原因。history/ 仅是历史，不属于实施要求。既有架构图主要表达结构，冻结策略以训练 v1.2 为准。

## 上传与开始新会话

将整个 `MHINet_server_handoff_v1.2` 文件夹或同名 ZIP 上传并解压。在服务器使用标准 Python 执行下面的包完整性检查，参数改成真实解压路径：

```bash
python /path/to/MHINet_server_handoff_v1.2/verify_package.py /path/to/MHINet_server_handoff_v1.2
```

这是包校验命令，不是训练命令。随后把 [HANDOFF_PROMPT.md](HANDOFF_PROMPT.md) 的正文发给服务器新会话，并给出真实设计目录与 2-step 仓库位置。新会话先核对资源、实现接口、验证反传，再开始训练。

权重、数据和 LoRetta 第三方源码没有放进本包；本包不包含 2-step 或 MCNet 仓库副本。实际权重与数据须在服务器定位。不要把未知资源路径、GPU 型号或运行时间写成已确定事实。

## 完成与未完成

已完成：源码/PDF审视、模型层定义、几何与损失公式的 NumPy 数值核验、梯度风险清单、训练与实验计划。待完成：PyTorch 实现、真实参数统计、真实 checkpoint 兼容性、autograd/AMP/显存测试、训练和测试集评估。所有实验状态均为待执行。
