# MHI v2 专属算法目录

此处已实现独立 MHI v2。顺序与逐项签收见 [ACCEPTANCE.md](../../versions/v2/ACCEPTANCE.md) 和 [验收状态](../../versions/v2/acceptance_status.json)。

Flow0 接口、D8 soft prior、Direct-D8 fine matching、Discrepancy/DGFO、严格版本化迁移、RGB_512 数据适配、独立损失及训练评价代码均放在此目录。可复用旧模块的稳定功能；使用 wrapper、子类或显式适配保留旧 checkpoint 参数键名，不往原 `models/`、`downstream/` 注入 v2 专属条件分支，不改写 v1 来源核或默认 CLI。

显式入口为 `python -m mhinet.v2.train` 与 `python -m mhinet.v2.evaluate`，训练配置见 [配置目录](../../configs/mhi_v2/README.md)。只支持512×512输入；最终稳定置信度上限2048与 QRRU 前12000候选分别控制。Flow 模式跳过 H Fitter；GT 仅用于监督/预测后的评价。

57 项 CPU 测试和 [真实 GPU 验收](../../versions/v2/evidence/real_feature_acceptance_20261020_final.json) 已通过；后者只覆盖两对 val 和三步 QRRU 梯度诊断，未提供完整精度结论。Flow Head 两步实数据训练及断点恢复烟测已通过。正式 Flow Head 已在 GPU0 启动，C/D/E 排队；实际状态从 `campaign.json` 和阶段 `status.json` 读取。本轮固定原 train/val、单 seed42；源图身份隔离不能解释为真正空间隔离。
