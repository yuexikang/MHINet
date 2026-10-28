# MHI-v2 正式路径：Flow0 → D8 → Local Pair-D2

版本 `mhi-v2-flow-pair-20261028`。从 `codex/mhi-v2@acdbbfa` 选择性整合，未 merge 两条研究分支历史，未进行新训练。

## 正式路径与历史接口

原主线以 U0 D16 Flow0 为基础，`SemidenseMatcherV2` 走全局 D8、离散 D2 窗口及可选 QRRU。此前单独合入的 D2 目标扩窗仍保留。

新正式入口 `mhinet.v2.load_mhi_v2` 使用：冻结 DINOv3 → 已训练 MVT/原 Flow 头 → VGG+CGMDP → 离散局部 D8 父点选择 → 连续 D8 目标中心 → D2 双侧 8×8 上下文、A 中心 4×4 查询、B 全部 8×8 候选 → Local Pair → FP32 16×64 Dual-Softmax → 峰值邻域连续坐标。

最多 128 个父点、每父点 16 个输出，最多 2048 点。保留原父点优先、查询次序，不追加排序/阈值，不丢弃无效行以免改变历史指标分母。`valid` 供调用者显式过滤。输入原生 512×512 RGB float [0,1]，像素坐标 xy，归一化转换 `(xy+.5)/256-1`，`align_corners=False`。

正式模型只构造共享特征、两项匹配温度和局部 Pair；没有 QRRU/DGFO 子模块，也不调用 H0 fitter。Flow Head 内原 dustbin 概率仍是已训练坐标解码的一部分，没有增加 dustbin 拒配决策。Matchability 保持原 D8 引导计算，阈值为 0。

旧 `u0.py`、`semidense.py`、H0 兼容、QRRU、原训练入口及配置不改默认行为。旧实验通过原显式入口运行。新入口尚未接几何图/QRRU；后续可用 `parent`、`parent_ids`、D8 中心及匹配置信度连接几何验证。

## 权重与可迁移配置

- 正式配置：`configs/mhi_v2/releases/flow_pair_v2.json`。
- 权重清单：`configs/mhi_v2/releases/flow_pair_v2.weights.json`，记录 SHA256、字段、轮次、训练代码来源、数据规模和选模口径。
- 默认 Flow：`outputs/full_rgb_d2_20261027/flow/best.pt`，epoch 4；与 `outputs/full_rgb_d2_10epoch_20261028/flow/best.pt` 内容哈希完全相同。
- 默认 D2：`outputs/full_rgb_d2_10epoch_20261028/d2/best.pt`，epoch 1。
- 可选 `variant='latest'`：只切换 D2 第 10 轮，Flow 仍为第 4 轮 best，不能换成 Flow latest。
- U0：原 `weight/mhi_v2_u0.pt` 保留；不覆盖、不删除。

Flow 检查点严格加载 `mvt/head`；D2 严格加载 `vgg/dedode/adapter/temperatures`。不是两个完整 state_dict 的合并。加载器核对实际文件哈希、阶段、轮次及 D2 记录的 locator 来源，迁移后的同内容文件仍可使用。默认选择依据仅为 RGB 验证 P@3，不能按最终测试成绩更换 best。

源代码归档提交为 `codex/flow-d8-d2@c6d6ec4`，训练入口提交 `224fbd0`，父点优化提交 `afa1f52`。**原始检查点没有内嵌 Git SHA**，清单明确区分代码归档来源与训练进程未记录的信息，不伪造训练时 SHA。

公开默认配置没有服务器绝对路径。`assets_root`（或环境变量 `MHI_V2_ASSETS`）决定权重和 runtime 相对路径根目录，默认仓库根目录。需要本地 DINO 权重、LoMa/LoRetta 源码；配置的 `weight/dinov3.pt` 是可迁移占位路径，需指向原冻结 DINO 文件。也可以传入本机 runtime JSON 的三个源代码/权重路径。推理不要求数据集或训练 manifest。

```python
import json
import torch
from mhinet.v2 import load_mhi_v2

runtime = json.load(open('configs/mhi_v2/runtime.rgb512.nvme.server.json'))  # 本机路径配置
model = load_mhi_v2(device='cuda:0', runtime=runtime, assets_root='/home/disk1/MHINet')
# images: [2,3,512,512] 或 [1,2,3,512,512]，RGB float32 [0,1]
result = model.match(images, diagnostics=True)
a, b, confidence = result['points_a'], result['points_b'], result['confidence']
valid, parent, parent_ids = result['valid'], result['parent'], result['parent_ids']
```

`parent` 是输出所属父点的局部序号；`parent_ids` 是每个输出对应的 D8 原网格 flat ID。可选 diagnostics 返回 D16 Flow/Matchability/valid、父点和中心、原窗口描述子/掩码、候选 log-probability、调用统计。默认结果不保留大诊断张量。`match` 一次处理一对，批量数据逐对调用，与原实验 batch=1 协议一致。

## 保留的两阶段训练

入口 `python -m mhinet.v2.pair_matching.training`；算法来自原 `d2_modulation/full_rgb.py`，训练损失、选择策略、初始权重、参数组与余弦调度保持不变。离散父点选择无梯度，连续 D8、VGG/CGMDP、D2 Pair 的监督保留。DINO 始终冻结。D2 阶段冻结 Flow/MVT。无 QRRU 优化器参数。

```bash
# 本次只迁移/核对代码，没有执行以下训练命令。
python -m mhinet.v2.pair_matching.training \
  --config configs/mhi_v2/releases/full_rgb_training.json \
  --runtime configs/mhi_v2/runtime.rgb512.nvme.server.json \
  --assets-root /home/disk1/MHINet --device cuda:0 --stage all

# 从已有完整epoch继续：选择新总epoch数后，明确指定原输出目录。
python -m mhinet.v2.pair_matching.training \
  --config /path/to/continuation.json \
  --runtime configs/mhi_v2/runtime.rgb512.nvme.server.json \
  --assets-root /home/disk1/MHINet --device cuda:0 \
  --stage d2 --resume --output /home/disk1/MHINet/outputs/full_rgb_d2_10epoch_20261028
# 单独继续 Flow：将 --stage d2 改为 --stage flow。
```

单独续训直接读取该阶段 latest，不再依赖原初始 downstream/adapter 文件。恢复 latest 的模块、AdamW 状态、scheduler 步数、CPU/CUDA/Python RNG；保留 best，严格提高验证 P@3 才更新 best，早停 min_delta 与 best 保存分离。`--resume` 必须有 latest/best，不静默从头训练。D2 续训核对记录的 Flow 文件内容。恢复粒度为完整 epoch；中途被打断的 epoch 会重跑。总 epoch 改变会重算余弦周期并记录 `schedule_rebased`，不是原学习率轨迹的无缝延续。

配置默认保留历史 D2 所使用的 `frozen_flow_checkpoint`。如果未来要运行新的一整套两阶段训练并使用该次 Flow 阶段选出的 best，明确将该字段设为 `null`。数据根目录、初始 downstream/adapter 都是显式配置，不再通过 QRRU 实验加载器隐式引入旧 checkpoint。初始文件仍需按归档记录提供。

完整 10 轮实验由原早停任务续训完成，包含总 epoch 从 20 改为 10 的调度重设；不能声称是从头固定 10 轮的历史轨迹。完整历史见归档报告。

## 实验档案与研究边界

- [完整 RGB 两阶段结果](../../reports/full_rgb_d2_10epoch_20261028/README.md)：40849 train / 5106 val / 5106 RGB test / 235 杭州 test_real。
- [CGMDP A0–A4](../../reports/cgmdp_residual_research/analysis.md)：1024 train / 256 RGB test / 64 杭州；保留完整结果、负结果与旧三尺度参考。
- [父点选择优化](cgmdp_selection_optimization.md)：复用离散 D8，跳过被丢弃的旧 Fine；仍保留连续 D8 的独立计算。
- [A3 无训练诊断](../../reports/cgmdp_a3_conditioning/analysis.md)：仅归档，不将静态或动态 Delta 接入默认网络。

Self、Cumulative、A2/A3/A4、No-Flow、QRRU 四角监督、Dustbin/几何图研究仍是实验，不计为主线创新。完整 RGB 和 A0–A4 的规模/权重/测试集不同，不直接比较两张表的 P@3。研究代码留在来源分支，不随报告搬入主线。

## 验收入口

```bash
python -m pytest tests/v2 -q
# 原分支单独进程：参考32对，同GPU、同权重。先 cd 原实验分支的工作树。
CUBLAS_WORKSPACE_CONFIG=:4096:8 python /path/to/integration/tools/mhi_v2/reference_pair.py \
  --source-root /home/disk1/MHINet --output outputs/mhi_v2_integration/reference32_deterministic \
  --device cuda:2 --deterministic
# 整合分支：逐层零容差比较，然后完整测试。
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m mhinet.v2.pair_matching.evaluate \
  --runtime configs/mhi_v2/runtime.rgb512.nvme.server.json --assets-root /home/disk1/MHINet \
  --rgb-root /home/disk1/Data/datasets/MHINet_RGB_512 \
  --hangzhou-root /home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4 \
  --device cuda:2 --deterministic-reference outputs/mhi_v2_integration/reference32_deterministic \
  --historical-results reports/full_rgb_d2_10epoch_20261028/results.json \
  --output outputs/mhi_v2_integration/best --full
# 仅恢复真实状态，不更新参数：
python tools/mhi_v2/check_resume.py --assets-root /home/disk1/MHINet \
  --runtime configs/mhi_v2/runtime.rgb512.nvme.server.json \
  --output outputs/mhi_v2_integration/resume.json
```

普通 CUDA 比较可增加 `--reference outputs/mhi_v2_integration/reference32`；其参考生成命令去掉 `--deterministic`。稀疏 scatter 归约可能改变近同分父点顺序，记录顺序差异，再按整组父点对齐比对，不掩盖父点集合变化。确定性路径要求零容差且不允许换序。

最终验收数据和文件清单见 [整合验收报告](../../reports/mhi_v2_integration/README.md)。不提交权重、原图或大型逐图窗口张量。

误差和诊断可复现命令（不训练）：

```bash
python tools/mhi_v2/audit_reduction.py --assets-root /home/disk1/MHINet \
  --runtime configs/mhi_v2/runtime.rgb512.nvme.server.json \
  --rgb-root /home/disk1/Data/datasets/MHINet_RGB_512 \
  --hangzhou-root /home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4 \
  --device cuda:2 --index 2738 --output outputs/mhi_v2_integration/reduction_remaining.json
```
