# D2 局部调制实验

详细设计、数据、参数及验收：[实验计划](../../../docs/mhi_v2/d2_local_modulation.md)。这是当前分支独立入口，未修改主线默认推理。

A/B 各取最终 D2 特征的 8×8 上下文，masked mean 生成双向共享的残差通道调制。A 中心4×4查询匹配B全部8×8候选。MLP 768→64→512，82,496参数；末层零初始化。输出后L2归一化、dual-softmax、峰值邻域软坐标。训练只更新调制器。

```bash
python -m mhinet.v2.d2_modulation.check
python -m mhinet.v2.d2_modulation.experiment prepare --device cuda:0
python -m mhinet.v2.d2_modulation.experiment train --mode none --device cuda:0
# 两个独立训练任务，可在两个终端并行运行
python -m mhinet.v2.d2_modulation.experiment train --mode self --device cuda:0
python -m mhinet.v2.d2_modulation.experiment train --mode pair --device cuda:1
# 三组全部完成后
python -m mhinet.v2.d2_modulation.report
```

无标签在线推理，使用与训练相同的冻结上游加载方式：

```python
import torch
from mhinet.v2.qrru_corners.experiment import load_models
from mhinet.v2.d2_modulation import load_adapter, infer

model, matcher = load_models('cuda:0')
model.requires_grad_(False).eval()
matcher.requires_grad_(False).eval()
adapter = load_adapter('outputs/d2_modulation/pair/best.pt', 'cuda:0')
with torch.no_grad():
    shared = model(images)  # 1×2×3×512×512，与既有数据加载器相同归一化
    prediction = infer(shared, matcher, adapter)
# points_a / points_b：原图像素坐标 N×2；confidence：N，未校准为正确率
```

不需要 H、重叠掩码或其他标签；没有调用 QRRU。父点选择沿用诊断用旧离散源父点和冻结连续D8目标中心，每对最多2048点。`none`、`self`、`pair` 检查点可用同一个推理入口。

缓存位于 outputs/d2_modulation/cache，只为本轮冻结上游试验使用。改动上游权重或选点规则后必须重新 prepare 到新的 --output 目录；不要复用旧缓存。权重和缓存不入 Git。训练组目录存在时不会覆盖，另选 --output 可启动新试验。
