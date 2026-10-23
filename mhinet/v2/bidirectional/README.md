# 双向 Flow0 诊断

本轮已完成A/B，未训练任何模型。Cycle能排序杭州高置信参考错误，但在几何无对应低误拒工作点和最终匹配上没有明确增益，因此不启动C，不替换U0或下游默认路径。

- `cycle.py`：交换输入的反向预测由调用者提供，采样其绝对坐标映射，输出像素Cycle误差、有效插值掩膜与高斯可靠性。只调制额外D8奖励，不删除候选。
- `run.py`：复用U0、冻结定位头加载、原semidense/D8/D2/QRRU推理，输出A查询记录及B0/B1/B2/B3/均匀降权对照。
- `report.py`：仅使用RGB开发校准子集选择阈值，输出配对指标与C条件裁决。
- `test_cycle.py`：恒等/逆平移、无效映射/边界两项CPU数值测试。

运行环境为loma-repro，实验输出在`outputs/d16_bidirectional`，已有结果目录不会覆盖。执行前固定的范围、阈值规则和C启动条件见PLAN.md；最终报告见`reports/d16_bidirectional/REPORT.md`。

```bash
python -m pytest mhinet/v2/bidirectional/test_cycle.py -q
CUDA_VISIBLE_DEVICES=0 python -m mhinet.v2.bidirectional.run --domain hangzhou
CUDA_VISIBLE_DEVICES=1 python -m mhinet.v2.bidirectional.run --domain rgb
python -m mhinet.v2.bidirectional.report
```

下游固定为`outputs/hangzhou_cleaned_restart/v2_20261022/best.pt`（epoch7），MVT/Flow头固定加载`weight/mhi_v2_u0.pt`（U0 epoch30）。这不是重新挑选或训练一个下游模型。主分支的U0默认接口未加阈值或Cycle过滤。
