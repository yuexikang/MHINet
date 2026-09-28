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

对照报告生成：`python scripts/summarize_semidense_chunk_training.py`；页面：`outputs/semidense_chunk32_128_short200_20260928/index.html`，JSON：同目录`comparison.json`。汇总器逐步核验实际训练pair_id、学习率、初始权重身份和评估pair_id；未完成时明确标记部分结果。

此实验只能筛查明显收敛回退，不足以确定完整训练等价。正式训练默认是否升级，需结合最终验证和后续更长对照。


## 已完成结果（200步，128对固定val）

两组均正常完成200个优化器步、800对训练样本，以及128对级联评估。逐步实际pair_id、cursor、学习率一致；源权重SHA、初始matcher SHA和8663步调度长度一致。监督点数与H0有效性一致，QRRU初始扰动EPE的最大差为1.64e-7 D2像素（归约浮点差异范围）。

| 指标 | window_chunk=32 | window_chunk=128 |
|---|---:|---:|
| 稳态训练步耗时（排除前10步） | 2.116884 s | 1.871589 s |
| 启动至训练+验证完成墙钟 | 540.06 s | 475.08 s |
| 第0步验证总损失 | 0.928418 | 0.928418 |
| 第100步验证总损失 | 1.044848 | 1.120670 |
| 第200步验证总损失 | 1.103657 | 1.050833 |
| 第200步验证Lc | 0.162606 | 0.159679 |
| 第200步验证Lf | 0.246576 | 0.231894 |
| 第200步验证Lq | 0.694475 | 0.659260 |
| 最后50步训练总损失均值 | 1.127551 | 1.112533 |
| coarse EPE（原图px） | 2.345412 | 2.344334 |
| fine EPE（原图px） | 0.550252 | 0.548163 |
| final EPE（原图px） | 0.180792 | 0.165738 |
| coarse precision@1px | 10.014637% | 9.980923% |
| fine precision@1px | 89.940039% | 90.101563% |
| final precision@1px | 99.558332% | 99.598632% |
| 推理失败率 | 0 | 0 |

128的训练步耗时减少11.59%（吞吐约1.131倍），最终验证总损失低4.79%，最终EPE低8.33%，precision@1px高0.04030个百分点。本次没有最终精度回退，但并非所有阶段所有指标都改善；coarse precision略低。每步时间包含数据读取/传输、梯度裁剪和optimizer更新，排除定期保存/验证，且来自不同物理GPU；因此不能直接套用之前同GPU纯前反向benchmark的25.70%节时。

第100步128的验证损失更高，第200步排名反转；两组终点损失都高于共同起点。此时cosine学习率仍接近初始峰值，实验没有跑到原8663步日程后段。结论仅为：128有实用提速，本次短程终点的细/最终精度更好，值得做更长对照；不能声称已证明训练等价或稳定改善。正式训练默认仍为32，分支不合并。

最终权重保存在输出根目录的`chunk32/latest.pt`与`chunk128/latest.pt`；各自第0/100/200步验证明细在`validation/`，级联逐对结果在`cascaded_val128/`。权重未推送Git；配置、执行脚本、注册记录和完整逐步曲线/汇总JSON已纳入版本控制。最终权重SHA和评估实际matcher配置收录在结果JSON的evaluation_registration中。

结果记录：`artifacts/semidense_chunk32_128_short200_results.json`；曲线页面：`outputs/semidense_chunk32_128_short200_20260928/index.html`。
