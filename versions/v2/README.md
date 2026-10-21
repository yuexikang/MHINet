# MHI v2：杭州联合训练

## 当前主线：杭州从头联合训练

只保留原始 DINOv3 预训练权重并冻结，其余活跃模块重新初始化。新增 homography / dense_flow / sparse_points / unlabeled 几何接口；杭州近似 H 使用弱监督。256 对训练、64 对验证，最多 30 epoch，余弦退火和早停，自动记录 QRRU 前/中/后及每次迭代。

见 [实际调用路径、接口和训练细节](GEOMETRY_TRAINING.md)。入口：`python -m mhinet.v2.train_hangzhou --config configs/mhi_v2/hangzhou_fresh_joint.json`。运行结果位于 `outputs/mhi_v2/hangzhou_fresh_joint_20261021`；下文为上一轮记录。

当前分支 `codex/mhi-v2`，输入 **512×512**，最终最多 **2048** 对。

**2026-10-21：旧 RGB_512 长程实验已暂停。** C/D 完成但收益极小，新 Flow 损失未胜过同预算旧损失；E 停在 5107 步。转向 QRRU 推理优化，以及杭州数据的 64 对、128 步小试。

当前只需看 [结论、代码入口与执行计划](QRRU_RESTART.md)。配置在 [hangzhou_pilot.json](../../configs/mhi_v2/hangzhou_pilot.json)，两个入口：

```bash
python -m mhinet.v2.benchmark --config configs/mhi_v2/hangzhou_pilot.json --output NEW_OUTPUT
python -m mhinet.v2.pilot --config configs/mhi_v2/hangzhou_pilot.json
```

新入口使用普通配置、日志和权重；不依赖旧实验的哈希登记链。`pilot` 输出的是小型 QRRU 权重文件，加载基础 checkpoint 后再加载其中的 `qrru` 字段即可，不冒充完整系统 checkpoint。

历史材料保留供查阅：[RGB_512 实验记录](RGB512_HISTORY.md)、[旧结果快照](evidence/progress_20261021_1734.json)、[v1 固定基线](../v1/README.md)。旧 train/evaluate/campaign 工具属于上一轮流程，当前不自动运行。
