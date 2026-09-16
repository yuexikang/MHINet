# 共享描述子预训练与 DINOv3 LoRA

分支：`codex/shared-descriptor-lora`。这是独立的 `shared_descriptor_v1` 训练任务，不把新增监督冒充原 MHINet v1.2 四角损失协议。

## 边界

只运行 DINOv3 → MVT → GHIM/CGMDP，输出 H0、跨图上下文和 D8/D4/D2。DINO、MVT 每对只运行一次；不构建 MHIR，不执行 D1。描述子依赖图像对，不能当作独立单图描述子缓存复用。

基线冻结 DINO；LoRA 对照冻结 DINO 原权重，仅训练零基 blocks 8–17 的 QKV LoRA：rank=8、alpha=16、dropout=0、327680 参数。原 LinearKMaskedBias 保留在 wrapper 内，不丢 K-bias mask。两组都训练 MVT、GHIM head、VGG、累计解码器；VGG BN running statistics 固定。B=0 初始化保证初始函数不变。LoRA 路径仅前八个 block 使用 no_grad，后十个 block 使用非重入 activation checkpoint。

## 数据与超参

使用 `/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1` 第一档：train 51978 对、val 5772 对；母图/地理组互斥。test 不用于选配置。

| 项目 | 两组共同设置 |
|---|---|
| 输入 | 784×784，BS1 |
| 梯度累积 | 4，有效 BS4；epoch 最后剩余2对按实际2对平均，不重复样本 |
| 时长 | 一轮，12995 次 optimizer step |
| 学习率 | MVT/head 1e-6；VGG 5e-6；CGMDP 1e-5；LoRA 1e-5 |
| 优化器 | AdamW，weight decay 1e-4，gradient clip 1 |
| 调度 | 前5%线性warmup，余弦下降至峰值的10% |
| 验证 | 每5000步及结束，全量第一档val |
| checkpoint | 每1000步及结束；同目录 latest.pt 自动恢复 |
| 初始化 | 两组相同 LoRetta/LoMa-B 预训练权重，seed0 |

总损失 `L = Lgeo + .01 Lmat + .0001 Lcls + .05 LH + Ldesc`。GHIM 沿用现有实现；描述子三尺度双向等权 InfoNCE，temperature=.1，每方向最多1024个有效查询。正样本由真实 H 投影并双线性采样，采样后 L2 normalize；负样本由其他查询的目标对应点及半径3/6特征像素的16个局部点组成。排除距离GT≤2特征像素的负样本、边界外及两侧无效支持；不构建所有像素两两相似度矩阵。

验证区分 sampled InfoNCE 命中率与 full-gallery 检索：后者每对/尺度/方向固定32个有效查询，在目标**全部特征网格**中取最大相似度，记录784输入像素坐标误差及 Recall@1/3/5px。它不是每像素穷举，也不是原生不同尺寸影像坐标的精度；D8存在网格量化误差，跨尺度比较需注意。另记录 H0 四角平均误差、拟合失败率。无 MHIR，所以不输出六轮 H 图；每次验证前三对输出 H0 叠加图和 D8/D4/D2 匹配连线图（绿GT、红预测）。

可复现设置：关闭 cuDNN benchmark，开启 cuDNN deterministic 和 PyTorch strict deterministic，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。描述子双线性采样使用与 `grid_sample(align_corners=False)` 前向/梯度对齐的 gather 实现，允许严格确定性的 Attention 反向；不通过放宽恢复误差阈值掩盖漂移。此设置有速度代价，不沿用之前非确定性冒烟的速度作正式耗时承诺。

## 启动

```bash
cd /home/disk1/MHINet
GPU_ID=1 bash scripts/train_shared_descriptor.sh
GPU_ID=2 bash scripts/train_shared_descriptor_lora.sh
```

输出分别在 `outputs/shared_descriptor_frozen_seed0` 和 `outputs/shared_descriptor_lora_seed0`。可用 `MHINET_OUTPUT_DIR` 改目录；同目录断点恢复校验配置、数据、权重及实现文件hash，不能把两种实验混在一个目录。`--max-steps`、`--limit-train`、`--limit-val` 仅用于显式诊断，不加这些参数即完整第一档训练/验证。工程冒烟目录绝不能用作正式训练目录。

同目录有 `latest.pt` 则恢复；无 checkpoint 则从头开始，旧的本入口日志/可视化移入 `previous_no_checkpoint_*` 后重新记录，其他文件不动。目录锁阻止两个进程同时写同一实验。启动前用 `nvidia-smi` 确认目标卡可用；短程峰值约10GB，不代表可与其他大显存训练挤占同一张卡。

## 供两项目复用

每次验证导出 `shared_descriptor.pt`，约418MiB；包含 MVT、GHIM head、VGG、累计解码器及可选 LoRA，不含冻结 DINO 大权重，也不含 MHIR/LoMa密集匹配头。加载器校验所需 DINO checkpoint SHA256。

在 MHINet 或相邻 LoMa 的 DINOv3/MVT 分支中，将 `/home/disk1/MHINet` 加入 PYTHONPATH：

```python
from mhinet.config import RuntimePaths
from mhinet.pretraining.model import load_shared
runtime = RuntimePaths.from_json('/home/disk1/MHINet/configs/runtime_paths.quadrant.server.json')
shared = load_shared(runtime, '/absolute/path/shared_descriptor.pt').eval()
outputs = shared(images)  # [B,2,3,784,784], RGB [0,1]
H0 = outputs['H0_norm']
D8, D4, D2 = (outputs['pyramid'][s] for s in (8,4,2))
```

两种下游分别接 MHIR 或密集匹配模块；这不等于直接覆盖 LoMa 原版 DINOv2 checkpoint。共享bundle的加载、推理接口已提供，下游各自完整训练/性能仍需单独实验。现有训练入口与相邻 LoMa 源码未修改。

## 验收与限制

工程检查包括 H/坐标、双侧mask、LoRA初始函数、第二步梯度、MVT与LoRA两条损失分支、冻结主干hash、真实小批训练/恢复和导出加载。测试通过不代表训练已经收敛；正式一轮结果必须看完整 val 检索精度、H0及下游任务迁移，不能仅看训练loss。当前不承诺LoRA必然优于冻结基线。
