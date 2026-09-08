# MHINet · 多尺度 Homography 迭代模型

**当前入口：[START_HERE.md](START_HERE.md)。** 结构 v1.0、损失 v1.1、训练/交接 v1.2。2026-09-07。

完整设计已整理成服务器实施包；[新会话提示词](HANDOFF_PROMPT.md)可直接交接。

- [完整模型](docs/01_full_model.md)：四尺度、八次 H 迭代，新增模块 2,544,160 参数（设计计数）。
- [训练协议](docs/02_training_and_validation.md)与[可微性说明](docs/05_autograd_and_joint_training.md)：冻结 DINO，逐步开放 DeDoDe/VGG/MVT 联合训练。
- [实现契约](docs/03_implementation_contract.md)与[分阶段路线](docs/06_implementation_roadmap.md)：代码边界、依赖、验收条件。
- [损失审视](docs/04_loss_review.md)：主线序列四角 L1，FGO 和辅助项是独立候选。
- [实验表](docs/07_experiment_plan.md)：必做、机制消融和条件实验；机器可读表在 experiments/。
- [架构图](figures/architecture.png) / [SVG](figures/architecture.svg)，[层尺寸](artifacts/layer_shapes.csv)。

本版“多尺寸”指同一对 784×784 输入的多尺度特征迭代，首版 B=1。暂不加入 Planar。方案与实验预算是待验证设计，尚未实现或训练网络。history/ 中旧冻结策略和旧混合损失仅供追溯。
