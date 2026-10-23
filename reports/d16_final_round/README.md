# D16最终实验交付

先读 [一页结论](ONE_PAGE.md) 和 [完整报告](FINAL_REPORT.md)。代码位于 `mhinet/v2/final_round/`，原始计划位于 `mhinet/v2/local_occlusion/FINAL_ROUND_PLAN.md`。

此目录包含所有十组开发/测试结果、逐对CSV、空间块配对区间、梯度、工作点曲线、G组示例及冻结的数据清单。压缩仅用于减少Git体积，无额外校验框架。

完整逐查询NPZ、所有训练日志与权重保留在服务器 `/home/disk1/MHINet/outputs/d16_final_round/`；旧基线权重在 `/home/disk1/MHINet/outputs/d16_local_occlusion/`。最终权重路径/坐标协议及诊断阈值见final_model.json。Git不包含模型权重和数据影像。

图中青色箭头是B图参考位置到预测位置的**残差**，不是A→B完整位移；长度1:1原图像素。高置信错误图使用Matchability≥0.9且EPE>16px；这只是描述性切片，不是部署阈值。每组示例按清单首个样本固定，不挑成功图。
