# MHI v2 算法

当前优先优化基础 QRRU 推理，并用杭州数据进行小规模训练。见 [当前结论与计划](../../versions/v2/QRRU_RESTART.md)。

- `qrru.py`：递归细化；训练保留轨迹，普通推理只返回末轮。
- `coarse_matcher.py`：D8 全局匹配，可选整矩阵或分块。
- `semidense.py`：D8 → D2 → QRRU；按置信度分批细化，满足最终有效输出预算后可提前停止。
- `pilot.py`：杭州数据抽样、基础权重加载、近似地理参考弱监督、小试前后评价。
- `benchmark.py`：512 实际推理速度和精度对照。

运行配置：`configs/mhi_v2/hangzhou_pilot.json`。输入仅 512，最终输出默认最多 2048。推理的 `inference_iterations` 与训练 `iterations` 独立；`max_matches` 是 QRRU 前候选上限。`qrru_early_stop` 在凑满有效输出后停止，诊断模式仍处理完整候选。

`train.py`、`evaluate.py`、`flow_loss/` 保留上一轮 RGB_512 实验的恢复与追溯用途。当前小试不依赖这些入口的逐文件哈希和注册流程。v1 文件及基线工作区保持独立。
