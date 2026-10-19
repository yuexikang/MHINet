# MHI v2 配置目录

独立 v2 运行配置已实现，使用 `mhi.v2-run.v1` schema 和 `mhi_semidense_v2` task，由 `python -m mhinet.v2.train` 读取。研究登记 [experiment_plan.json](../../versions/v2/experiment_plan.json) 仍是非可执行的计划，不能作为运行配置。

默认 `input_size=512`，只允许512×512模型输入；`max_output_matches=2048` 允许显式正整数覆盖。该字段只限制 QRRU 与有效性过滤后的最终输出；`max_matches=12000` 作为 QRRU 前候选预算单独保留，不从旧 checkpoint 把12000误继承成最终输出默认值。

| 配置 | 训练内容 | seed / 预算 | 最佳权重选择 |
| --- | --- | --- | --- |
| [flow_head_rgb512_seed42.json](flow_head_rgb512_seed42.json) | 仅 Flow Head，其余冻结 | 42 / 5 epoch、25535 更新 | 每 epoch 完整 val 的 Flow EPE 最低 |
| [qrru_c_rgb512_seed42.json](qrru_c_rgb512_seed42.json) | 原 QRRU | 同上 | 每 epoch 完整 val 的级联 Pre@3 最高 |
| [qrru_d_rgb512_seed42.json](qrru_d_rgb512_seed42.json) | QRRU + Discrepancy | 同上 | 同上 |
| [qrru_e_rgb512_seed42.json](qrru_e_rgb512_seed42.json) | QRRU + Discrepancy + DGFO | 同上 | 同上 |
| [runtime.rgb512.server.json](runtime.rgb512.server.json) | 当前机器的资源路径 | 不属于训练预算 | 不适用 |

每步累积 8 对，每 epoch 为 `ceil(40849/8)=5107` 次更新。中途 64 对 val 是诊断；选模使用每 epoch 完整 5106 对 val。C/D/E 使用同一冻结 Flow Head 最佳来源、共同 D2 缓存、数据顺序、seed、预算与选择规则，各自独立初始化优化器。QRRU 每批目标 50% 合格预测 D2 + 50% GT 噪声；预测点只在 GT 可见且修正范围≤6 D2 像素时入选，数量不足用噪声补齐并记录实际比例。

保留原 RGB_512 train/val 划分（40849/5106），源图身份交叉为 0；这不等于地理邻近隔离。只使用 train/val，封存 GoogleEarth/JL1Flight 不参与本轮训练与选型。第一轮只有 seed42。正式 Flow Head 已于北京时间2026-10-20 03:24:37在 GPU0 启动，C/D/E 排队；实际状态以调度目录 `campaign.json` 与各阶段 `status.json` 为准。旧协议缺省值、正式 v1 配置快照保留原样。
