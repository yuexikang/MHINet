# 32/128训练分块短程对照

实验分支：`codex/semidense-train-chunk32-128`，基于正式推理1024版本`7cddb2a`。生产训练入口与默认分块32保持不变；实验入口只在strict加载第一档C的完整权重后，将训练window_chunk改为32或128。

## 固定协议

- 初始权重：`outputs/semidense_stable_v2_tier1_lr_c_seed0/latest.pt`，两个实验读取同一SHA。复现第三档E训练路线的前200步，不是对第三档E成品权重追加大LR训练。
- 数据：GoogleEarth_quadrant_tiers_stable_v2第三档；保留完整训练集的seed0随机排列，各训练200个optimizer steps、800对样本。
- 共享/头部学习率：8e-5 / 8e-4；新AdamW，原完整一轮8663步cosine日程，在第200步停止，不把cosine压缩到200步。
- 物理batch1、累积4；冻结DINO、MVT、GHIM/H0，VGG BN统计固定；其他损失、GT采样、QRRU轮数保持一致。
- 唯一模型执行变量：训练window_chunk=32/128；对应GPU0/1。同型号不同物理GPU的时间比较仍可能受硬件/并发环境影响，须结合之前同GPU的固定前反向benchmark。
- 每100步保存latest.pt并验证；第0步额外测相同初始权重的损失。训练记录附pair_id，完成后核对实际数据顺序。
- 固定验证：原val manifest中第三档前128对（非随机抽样、非完整val）；第0/100/200步记录Lc/Lf/Lq，最后在同一128对上统一inference_window_chunk=1024，比较coarse/fine/final精度、EPE和失败率。未使用独立test作此次筛选。
- 关闭昂贵全通道图快照，保留逐步训练日志和验证结果，用对照报告展示曲线。权重和输出独立于C/D/E历史实验。

## 路径与运行

配置：`configs/semidense_train_chunk32_short200.json`、`configs/semidense_train_chunk128_short200.json`。
入口：`scripts/train_semidense_chunk_experiment.py`；启动与后续评估：`scripts/run_semidense_chunk_short_comparison.py`。
输出根目录：`outputs/semidense_chunk32_128_short200_20260928`；注册记录：`artifacts/semidense_chunk32_128_short200_registration.json`。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/loma-repro/bin/python -u \
  scripts/run_semidense_chunk_short_comparison.py
```

启动器拒绝重复输出目录，检查GPU0/1空闲，并核验两份配置除window_chunk外完全相同。训练和评估后台子进程有独立日志，失败不会伪报完成。评估器的历史`scope=smoke`标签由`--limit`自动产生；此次实验用途明确为预注册128对短程比较，不可将该标签或结果误作完整val。

此实验只能筛查明显收敛回退，不足以确定完整训练等价。正式训练默认是否升级，需结合最终验证和后续更长对照。
