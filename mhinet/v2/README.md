# MHI v2：Flow0 → D8 → Pair-D2 正式路径

当前正式入口为 `from mhinet.v2 import load_mhi_v2`，配置 `configs/mhi_v2/releases/flow_pair_v2.json`。默认组装完整 RGB 的 Flow best（第4轮）与 D2 best（第1轮），冻结 DINO，使用局部 Pair；不启用 H0/QRRU/DGFO/全图 Delta/拒配头。

[加载、推理、继续训练、权重来源和验收说明](../../docs/mhi_v2/flow_pair_release.md)。原 U0 与历史接口保留如下，其训练配置不会自动改用新权重。

# 历史 U0 路径

D16 Flow0 固定采用 **U0 第30轮**：冻结的原 DINOv3 + 已训练 MVT + 原 MatchEmbeddingDecoder。输入512×512，输出32×32逐位置坐标；不预测H0。分类增强、局部遮挡、局部残差头及其评测脚本不并入本分支，实验记录保留在 `codex/d16-local-occlusion`。

## 直接运行 Flow0

配置：`configs/mhi_v2/hangzhou_u0.json`。本机权重已独立保存为 `weight/mhi_v2_u0.pt`，包含MVT和完整Flow头；DINO路径由runtime配置提供。模型权重不提交Git，换机器时需复制该权重并修改配置路径。

```bash
python -m mhinet.v2.u0 A.png B.png --device cuda:0 --output outputs/mhi_v2/flow0.pt
```

两张图必须为512×512。Python调用：

```python
import torch
from mhinet.v2.u0 import load_u0

model = load_u0(device='cuda:0')
# images: [B, 2, 3, 512, 512]，RGB float，范围[0, 1]
with torch.inference_mode():
    result = model(images, pyramid_scales=())
```

输出字段：

- `flow16_target_norm`：目标xy，形状[B,2,32,32]。
- `flow16_norm`：目标坐标减源网格坐标。
- `flow16_matchability`：形状[B,1,32,32]，保留为分数。
- `flow16_valid`：数值有限、坐标在界内；不是可靠的遮挡判定。

坐标遵循 `align_corners=False`，`pixel=(norm+1)*256-0.5`；Flow像素位移为`flow16_norm*256`。U0未通过人工遮挡拒配门槛，不添加未经验证的拒配阈值。

## 继续训练后续模块

现有训练入口直接支持U0权重：

```bash
python -m mhinet.v2.train_hangzhou --config configs/mhi_v2/hangzhou_u0.json --device cuda:0
```

该配置冻结DINO/MVT/Flow头，训练VGG、特征解码器、D8/D2温度与QRRU；保留原小样本清单、30轮上限、余弦退火与早停。这里只提供可运行入口，不自动启动新训练。U0权重仅覆盖D16模块，不是已经训练好的完整下游匹配器。

D8→D2→QRRU沿用本分支的 `semidense.py`，输入仍仅512，最终输出默认最多2048。训练监督继续使用 `geometry_gt.py`、`data.py`、`losses.py` 与 `training.py`，保留精确合成H及杭州近似H行为。原有完整checkpoint初始化仍受支持。

### D2 目标扩窗

`SemidenseConfigV2.fine_target_side` 默认 8：direct_d8 路径保留 A 侧 4×4 D2 子点，B 侧围绕原窗口中心搜索 8×8 D2 候选（16×64 相关矩阵）。训练与推理共用该设置，越界候选屏蔽；设为 4 可复现原窗口。旧权重可直接加载，旧配置缺省时采用新默认值 8。H0 兼容路径和 QRRU 的局部窗口不变。

本次仅扩大候选范围，不包含联合训练或连续坐标解码改动；扩窗诊断不能完整解释 RGB_512 精度下降，也不代表主线已恢复原精度。
