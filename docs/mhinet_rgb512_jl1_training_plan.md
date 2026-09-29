# MHINet RGB512 合成训练与 JL1Flight 微调计划

分支：`codex/semidense-512-synthetic-jl1`。当前只完成短程吞吐基准和双卡训练入口，没有启动完整训练。

## 合成数据

`/home/disk1/Data/datasets/MHINet_RGB_512` 已完整验证：训练 40,849 对、验证 5,106 对、测试 5,106 对，总计 36 GB。训练来源为 FLAIR RGB 29,360 对、GoogleEarth Current 7,701 对、WHU 3,788 对；JL1 已从这批数据排除，留作真实图像微调。按源影像划分 split，清单记录同一源影像生成完整 A 与四边形裁剪 B、H 真值和遮挡/可见掩码。

数据没有经纬度字段，所以训练数据加载器使用稳定的 `(source_dataset, source_path)` 作为分组键，继续检查 train/val 来源隔离。

## 冻结与学习率

严格按本次要求冻结 DINO。MVT、GHIM/H0、VGG、CGMDP 和半密集/QRRU 均可训练；VGG BN running statistics 固定，D1 不参与主干。各可训练参数组使用同一学习率，首轮建议 `8e-5`。从 E 组完整半密集模型初始化，分辨率切换到 512，使用新优化器。

## 双卡吞吐测量

设备为空闲的物理 GPU 0 和 2（检查时 GPU 1 正在被其他任务占用）。两卡 DDP，每卡每个 microbatch 处理 1 对，累积 4 步；全局有效 batch=8 对。DistributedSampler 将 40,849 对均匀分到两卡，每个 rank 每 epoch 20,425 对（末尾最多补一个样本），共 5,107 个 optimizer steps。

使用真实数据、E 权重和“只冻结 DINO”的策略跑了 12 个双卡 optimizer steps，去掉前 2 步热身后，10 步平均 **2.981 秒/step**，P90 2.990 秒。按 5,107 steps 推算，训练部分约 **4.23 小时/epoch**。短测末尾在 32 对验证集上耗时约 6.5 秒，粗略放大全验证集约 17 分钟。因此完整一轮预计约 **4.5 小时**，另有 checkpoint 写入和可视化开销。该估算来自短跑外推，完整 epoch 后再用实际 wall clock 校准。

吞吐记录：`artifacts/mhinet_rgb512_ddp_throughput.json`。

## 推荐轮数

合成阶段建议先跑 **2 个 epoch**，总计约 9 小时。每个 epoch 在独立 val split 上保存 Lc/Lf/Lq 和可视化；训练完单独用合成测试 split 比较真实匹配精度。若第二轮仍有明显收益、且 val 未恶化，可从已完成权重另开一轮，避免第一轮就预设过长 schedule。

JL1Flight 的 `affine_pairs_train` 有 7,490 对（749 个 A 源影像、每源 10 个变换）；数据仓库将 `affine_pairs_test` 列作 val，但它就是之前评估的测试集，不能用于微调选择。先按 A 源影像分组，从 JL1 train 留出约 20% 源影像作为 validation，避免同一 A 的变换跨 train/val；官方 test 全程不用于调参。剩约 5,990 对训练样本，双卡 batch=8 下约 750 steps/epoch，按本短测约 40 分钟训练/epoch，建议从 **5 epoch、统一 `1e-5` 至 `2e-5` 学习率**起步，观察 source-held-out validation 并用 patience 2 早停。JL1 manifest/可见掩码转换尚未实现。

## 启动命令

完整合成训练入口已写好，但尚未启动：

```bash
scripts/train_semidense_rgb512_ddp.sh outputs/semidense_rgb512_synth_seed42
```

配置为 2 epochs、DINO frozen、其他模块统一 `8e-5`：`configs/semidense_rgb512_synthetic.json`。设备路径：`configs/runtime_paths.mhinet_rgb512.server.json`。launcher 默认使用物理 GPU `0,2`；若设备占用变化，可设置 `CUDA_VISIBLE_DEVICES` 覆盖。
