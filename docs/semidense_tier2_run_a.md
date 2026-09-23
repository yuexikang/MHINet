# Run A 第二档接续训练

使用 Run A 第一档最终 `latest.pt`，完整继承 shared、温度和 QRRU 参数，严格匹配 state_dict。以 stable_v2 tier=2 训练一个 epoch；CGMDP/VGG 初始学习率 1e-5，下游初始学习率 1e-4。新建 AdamW 和余弦调度，避免继承第一档已衰减到零的学习率。冻结 DINO、MVT 和 GHIM/H0 分支。

独立入口 `scripts/train_semidense_next_tier.py` 不改动其他运行任务的训练实现。配置、启动命令和 PID 见 `artifacts/semidense_tier2_lr_a_registration.json`。每100步保存，保留既有可视化与验证频率；中文看板新增第二档 A 组。
