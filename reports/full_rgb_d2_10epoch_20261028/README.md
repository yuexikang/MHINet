# 完整RGB两阶段训练与测试（2026-10-28）

状态：Flow、冻结Flow→D2均完成总计10轮，早停关闭，全部最终测试完成。训练40849对、验证5106对、RGB测试5106对、杭州test_real235对。

Flow的best为第4轮；D2的best为第1轮，均按RGB验证P@3选择。latest是第10轮。D2始终冻结原Flow第4轮best，未换用第10轮Flow。QRRU关闭。

两阶段从原早停运行继续到10轮；原历史包含旧调度配置，恢复时余弦周期压缩至10轮。D2第10轮曾中断，随后从第9轮完整检查点重跑并完成。

|阶段|权重|测试集|推理路径|P@1|P@3|P@5|P@10|EPE(px)|
|---|---|---|---|---:|---:|---:|---:|---:|
|d2|best|hangzhou_test_real|global_no_flow|25.993%|62.141%|78.701%|89.579%|10.9722|
|d2|best|hangzhou_test_real|global_flow|25.924%|62.206%|78.809%|89.659%|10.9556|
|d2|best|hangzhou_test_real|flow|11.205%|46.428%|68.344%|87.104%|14.8629|
|d2|best|hangzhou_test_real|local_flow|22.301%|62.682%|78.652%|89.572%|11.6957|
|d2|best|test|global_no_flow|77.853%|97.197%|97.868%|97.990%|0.8773|
|d2|best|test|global_flow|78.201%|97.628%|98.307%|98.429%|0.9001|
|d2|best|test|flow|77.796%|98.902%|99.541%|99.715%|0.9955|
|d2|best|test|local_flow|90.202%|98.338%|98.799%|98.887%|0.6980|
|d2|latest|hangzhou_test_real|global_no_flow|23.677%|62.461%|77.614%|88.903%|11.1695|
|d2|latest|hangzhou_test_real|global_flow|24.275%|62.778%|77.855%|89.158%|11.2130|
|d2|latest|hangzhou_test_real|flow|11.205%|46.428%|68.344%|87.104%|14.8629|
|d2|latest|hangzhou_test_real|local_flow|22.503%|62.910%|77.971%|89.148%|11.6219|
|d2|latest|test|global_no_flow|87.479%|95.577%|96.120%|96.223%|0.6912|
|d2|latest|test|global_flow|88.054%|96.217%|96.770%|96.874%|0.6917|
|d2|latest|test|flow|77.796%|98.902%|99.541%|99.715%|0.9955|
|d2|latest|test|local_flow|92.504%|97.345%|97.773%|97.854%|0.5632|
|flow|best|hangzhou_test_real|flow|11.205%|46.428%|68.344%|87.104%|14.8629|
|flow|best|test|flow|77.796%|98.902%|99.541%|99.715%|0.9955|
|flow|latest|hangzhou_test_real|flow|13.074%|48.287%|69.037%|86.467%|15.0631|
|flow|latest|test|flow|89.818%|98.863%|99.571%|99.719%|0.8009|

**分母注意**：表内Flow P@k分母为有效可见D16点；D2各路径P@k分母为全部输出点，不能跨阶段直接比较百分数。EPE只统计已知可见真值，阈值为512原图像素。JSON保留n/visible/正确点数及原始p3和p3_visible，便于重新计算。杭州仅为近似H参考一致性。

文件：config/data/done、flow_history.json、d2_history.json、results.json。history包含各轮loss、验证、学习率、显存及用时。表和results.json保留所有8套测试文件的完整聚合计数及全部推理路径。逐对原始测试记录、完整运行日志和权重保留于本地outputs/full_rgb_d2_10epoch_20261028，本报告不上传图像或模型权重。

这是完整RGB实验，区别于CGMDP A0–A4的1024训练/256测试/杭州64对研究。
