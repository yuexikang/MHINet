# CGMDP A0–A4研究产出

五组均正常完成；A0/A1为冻结参照，A2–A4各10轮。RGB训练1024对、验证64对、测试256对，杭州test_controlled64对。不是完整RGB数据集训练，也不是杭州完整test_real。

- analysis.md：原因分析与建议。
- results.md / results.json：固定窗口及重新匹配测试指标、改坏/救回、覆盖、排名及调制统计。
- training_history.json：逐轮loss、验证、学习率、显存、用时及参数统计。
- calibration.json：训练集32对估计的各尺度通道RMS。
- config.json / data.json：运行配置与样本数量。
- legacy_archive.json / smoke.json：旧三尺度研究归档和小规模验证。

逐对测试记录及父点名单、权重、完整日志保留在本地outputs/cgmdp_residual_research；Git上传的是训练历史与测试聚合数据，不含影像和权重。

完整RGB 40849对两阶段训练位于codex/flow-d8-d2分支reports/full_rgb_d2_10epoch_20261028。
