# MHI v2：QRRU 优先

当前分支 `codex/mhi-v2`，输入 **512×512**，最终最多 **2048** 对。

**2026-10-21：旧 RGB_512 长程实验已暂停。** C/D 完成但收益极小，新 Flow 损失未胜过同预算旧损失；E 停在 5107 步。转向 QRRU 推理优化，以及杭州数据的 64 对、128 步小试。

当前只需看 [结论、代码入口与执行计划](QRRU_RESTART.md)。配置在 [hangzhou_pilot.json](../../configs/mhi_v2/hangzhou_pilot.json)，两个入口：

```bash
python -m mhinet.v2.benchmark --config configs/mhi_v2/hangzhou_pilot.json --output NEW_OUTPUT
python -m mhinet.v2.pilot --config configs/mhi_v2/hangzhou_pilot.json
```

新入口使用普通配置、日志和权重；不依赖旧实验的哈希登记链。`pilot` 输出的是小型 QRRU 权重文件，加载基础 checkpoint 后再加载其中的 `qrru` 字段即可，不冒充完整系统 checkpoint。

历史材料保留供查阅：[RGB_512 实验记录](RGB512_HISTORY.md)、[旧结果快照](evidence/progress_20261021_1734.json)、[v1 固定基线](../v1/README.md)。旧 train/evaluate/campaign 工具属于上一轮流程，当前不自动运行。
