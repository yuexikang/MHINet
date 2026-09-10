# MCNet 解码器、D2 截断主线实验记录

当前架构 SHA256：`92095ffe16a5e650bab641df3fd17e16a0bf1f34ab47c82c59f37faa43ae092a`。
主线为 D8/D4/D2 各两轮；D1 代码保留，默认不执行。以下工程检查不能替代完整 tiny gate 或模型精度验证。

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

当前证据支持进入更长预算的 D8 可学习性检查。32步预算属于 progress signature，不能改成2000步直接续跑；完整预算需独立命名并从零初始化。未通过单尺度检查前继续保留原损失，不启动 E00/E01。
