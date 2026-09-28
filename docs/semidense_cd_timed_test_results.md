# C/D 第三档计时测试结果

状态：D重测进行中。D按用户要求在原目录重测覆盖，C保持原结果。GPU1（与C此前使用的物理卡相同），权重、数据和计时代码不变。

同一独立合成测试集，共2000对，有同母图精确真值；不是原始跨时相配准测试。CUDA event区间包含CPU提交间隙，QRRU包含投影与输出整理；读图、预热、GT评估、绘图和写盘不计入。

进度：C 2000/2000；D 0/2000。未完成时不作最终比较。

GPU状态和同卡进程每30秒采样至 outputs/semidense_tier3_lr_d_test2000.gpu.jsonl，不能据此保证两次采样间完全没有其他负载。


逐对记录在原测试目录；完整汇总为 artifacts/semidense_cd_timed_test_results.json。启动、覆盖范围与完成校验见 artifacts/semidense_cd_timed_test_registration.json。
