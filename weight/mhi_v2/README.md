# MHI v2 权重目录

本目录保存正式验收后导出的 v2 权重，使用独立文件名并在 `versions/v2/` 的对应验收证据中登记 SHA256、来源 checkpoint、task/schema、配置与实现提交。训练中的 `latest.pt`、`best.pt` 与各 epoch checkpoint 保存在 `outputs/mhi_v2/<实验ID>/<阶段>/<运行ID>/`，不视为已发布权重。

v1 权重继续使用冻结清单中的路径；不移动、覆盖或改名。v2 训练仅迁移明确允许的模型权重，新建优化器，不恢复 v1 optimizer 状态。

当前 v2 实现、57项CPU测试、真实GPU工程验收及 Flow Head 两步训练/断点恢复烟测已通过；正式 Flow Head 已在 GPU0 启动，C/D/E 排队，实际状态以 `campaign.json` 与阶段 `status.json` 为准，尚未发布正式 v2 权重。首轮配置见 [配置目录](../../configs/mhi_v2/README.md)：seed42、各5 epoch/25535更新，统一512输入、最终2048与QRRU前12000候选。短程烟测 checkpoint 只作工程证据，不能作为正式精度结果。
