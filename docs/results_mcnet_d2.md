# MCNet 解码器、D2 截断主线实验记录

当前架构 SHA256：`92095ffe16a5e650bab641df3fd17e16a0bf1f34ab47c82c59f37faa43ae092a`。
主线为 D8/D4/D2 各两轮；D1 代码保留，默认不执行。以下工程检查不能替代完整 tiny gate 或模型精度验证。

当前进展：正式训练前的四项 tiny、独立 checkpoint 重载及合并 gate 已通过；按要求停在 E00/E01 前。最新全表见 [pretraining_mcnet_d2_summary.md](pretraining_mcnet_d2_summary.md)，本页保留阶段性结果并在末尾解释完整结果。

## 六轮资源与梯度检查

输入 784×784、batch=1、RTX 4090。时间取第二个优化步骤，单位毫秒；显存采用十进制 GB。仅双步剖析，未执行正式充分预热的延迟基准。

| 训练范围 | 前向 ms | 反向 ms | 峰值 allocated GB | CGMDP 解码步数 | D1 调用数 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| heads | 257.731 | 434.172 | 4.000 | 4 | 0 | 通过 |
| joint | 265.485 | 889.338 | 10.935 | 4 | 0 | 通过 |

证据为 `artifacts/p4_profile_heads_mcnet_d2.json` 和 `artifacts/p4_profile_joint_mcnet_d2.json`。两种设置均保留六轮 H 的梯度，DINO/MVT/VGG 每步各调用一次。joint 第二步 DeDoDe/VGG/MVT 梯度范数分别为 0.361962/0.489198/34.208104，DINO 和 GHIM head 参数无梯度。零初始化导致第一步前层梯度为零，已由第二步和独立双分支审计补充检查。

## checkpoint v2 断点续训对照

配置 `configs/train_minimal_smoke.json`：D8 两轮，heads，seed=0，累计4个样本后更新，2个优化步骤。验证集仅2对图像，test 未使用。

| 运行方式 | H0 MACE px | H1 MACE px | H2 / Hfinal MACE px | 失败率 | 峰值 allocated GB | 单次前向均值 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 连续两步 | 6.611870 | 6.600273 | 6.598216 | 0/2 | 1.638155 | 83.344 |
| 一步保存后续跑 | 6.611870 | 6.600273 | 6.598216 | 0/2 | 1.638170 | 83.845 |

两次训练日志的损失、梯度范数、几何轨迹和数据游标一致。续跑恢复模型、优化器、调度器以及 CPU/CUDA RNG，游标从4推进到8。逐张量检查发现 D8 adapter projection 权重的119个元素有差异，最大绝对差为 `1.959502696990967e-06`；对应 Adam 一阶/二阶矩最大差为 `1.746286670822883e-09` / `1.0216064376834116e-16`。其余模型张量一致，scheduler、RNG 和进度一致。因此边界恢复检查通过，但逐位一致检查未通过。

本机对 correlation 的独立反向探针在 deterministic warn-only 模式下报告 `grid_sampler_2d_backward_cuda` 无确定性实现。这支持 CUDA 采样反向非确定性是差异来源的判断，但不能单凭警告证明全部差异均来自该算子。后续不宣称 bitwise reproducibility；若需要该保证，应单独实现并验证确定性采样反向。

结果目录：

- `/home/disk1/MHINet/outputs/mhinet_resume_smoke_mcnet_d2_v2`
- `/home/disk1/MHINet/outputs/mhinet_uninterrupted_smoke_mcnet_d2_v2`

| checkpoint | SHA256 |
| --- | --- |
| 断点 step1 | `b8d4ffb8b4d76015cac49c958cd2b8ac329dfbf550a4df2edd355a29fbf707be` |
| 续跑 step2 | `e9dca3f40173d1a8e252aa212e7ede91fd45ea4560a05b91213a0ae4de081681` |
| 连续 step2 | `705da3ea80ade2394f4eefb85b822d6b3f55719f92ab48f518292d02b6af1d15` |

以上只有两个优化步骤，不能据此判断泛化提升或通过 tiny-overfit。E00/E01 仍等待 D8/D4/D2/TINY-6 的新架构 gate。

## D8 32步短程诊断

单张真实图像对、32个受控平移 H0，seed=0，BF16 CNN / FP32 correlation 与几何，AdamW lr=0.001，effective batch=4，残差范围为 D8 单步上限的0.5倍；无权重平均。

| 读出 | H0 MACE px | H1 MACE px | H2 / Hfinal MACE px | 失败样本 | 拒绝更新 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始化 | 14.868890 | 14.868889 | 14.868889 | 0/32 | 0/64 |
| step32 | 14.868890 | 9.773121 | 7.634817 | 0/32 | 0/64 |

最终误差下降7.234072 px，但未达到均值低于0.1 px的门槛，artifact 状态为 `failed`，CLI exit=1 表示判据未达标。梯度均有限，未出现求解拒绝。训练循环耗时7.865 s，峰值 allocated 为568,849,920 bytes，缓存描述子后的最终评估为22.878 ms/样本；该时间不包括 GHIM/CGMDP，不能与整模型前向速度直接比较。

证据：`artifacts/p4_tiny_s_d8_mcnet_probe32.json`，SHA256 `2ad74875dcbbc4c640b17f978f4ad5bad8e210f3216cbe7b6a67e04b2e11827c`。原始 progress checkpoint 为 `/home/disk1/MHINet/outputs/tiny_overfit/mcnet_d8_probe32_progress.pt`，SHA256 `0152f35babaff7d60dca7fb4f40be3a151141290880f69ef3dd8b808629b73a2`。

32步时的证据支持进入更长预算的 D8 可学习性检查。32步预算属于 progress signature，不能改成2000步直接续跑；后续完整预算已独立命名并从零初始化。

## 完整预算结论与停止边界

| 实验 | 停止步数 | 原始终点 MACE px | 登记主读出 MACE px | 主读出来源 | 结果 |
| --- | ---: | ---: | ---: | --- | --- |
| D8 | 1664 | 0.290066 | 0.098226 | 1536步起129次参数平均 | 通过 |
| D4 | 1568 | 0.272937 | 0.099413 | 1536步起33次参数平均 | 通过 |
| D2 | 1152 | 0.099895 | 0.099895 | 原始参数，尚未启用平均 | 通过 |
| TINY-6 | 1568 | 0.209412 | 0.081194 | 1536步起33次参数平均 | 通过 |

四项最终评估均为失败0/32、拒绝更新0，梯度均有限。独立重载四个checkpoint后，H0与每轮H的均值轨迹和原始artifact完全一致。合并器核验实际权重、配置、数据、协议及指标后通过；训练入口只执行门槛验证，未构造正式训练任务。

这些结果证明当前精修链能在登记的32个受控条件上拟合至门槛。D8/D4/TINY-6的原始终点仍高于0.1 px，参数平均仅属于tiny诊断，不能自动带入正式训练。D4/D2距离阈值很近，不能据此推断跨种子稳健性。全部条件来自同一真实训练图像对，不能推断真实GHIM初始化、跨图像对或联合训练的泛化能力。

TINY-6平均读出的轨迹为14.868889 → 1.304389 → 0.208130 → 0.184050 → 0.141688 → 0.081053 → 0.081194 px。最后一轮小幅回退约0.000141 px；原始终点的中尺度也存在回退。后续正式验证应同时看整条轨迹及各地理组，而非只看Hfinal。当前各轮保留梯度并重新采样，不为改善此表而改动损失、迭代数或门槛。

D2早期约1.4 px的平台在既定预算内自行突破，并以原始参数通过。额外的纵向condition2真实相关性检查中，256维原描述子最近候选top1比例为0.755147，随机32维adapter后为0.498455，正确候选与中心的平均相关性差分别为0.020783/0.021278。这证明该条件存在可用局部信号，不能据此归因全部优化波动。证据为 `artifacts/p2_real_correlation_mcnet_d2_condition2.json`。

当前停止在正式训练前；下一阶段若启动，应使用新的E01初始化和登记配置，不能将受控H0 tiny checkpoint当正式训练初始化。先做E00/E01验证，再按实验计划比较解冻预算和结构消融。D1、Planar、FGO和额外损失保持延期。
