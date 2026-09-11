# BS1关闭相关性重计算短测

2026-09-11。按用户要求，仅测试关闭相关性activation checkpoint，不修改正式训练配置。
这不是关闭训练checkpoint保存。物理GPU0 RTX4090顺序测off/on，新train前16对循环，
BS1×累积8、相同原始预训练初始化、frozen_dino_mvt、GHIM四项监督和500步warmup。
每组3步性能预热+6步计时；包含读取、H2D、完整前反向、优化和梯度范数审计。
不含验证、加载和checkpoint保存；16对重复读取受文件缓存影响，不代表全程ETA。

| 模式 | 平均秒/优化步 | pair/s | 峰值allocated GB | 峰值reserved GB |
| --- | ---: | ---: | ---: | ---: |
| 开启相关性重计算 | 6.8233 | 1.1725 | 8.393 | 9.783 |
| 关闭相关性重计算 | 5.0855 | 1.5731 | 14.752 | 15.680 |

GB为十进制。关闭后吞吐+34.17%，每步耗时-25.47%，allocated增加6.36GB。
两组均完成9次优化，没有OOM/非有限损失。DINO/MVT无梯度，活跃组第二步后非零，D1不执行。
16对初始H0逐对坐标完全一致。独立多步训练存在小浮点差异，不声称长轨迹逐位一致。

## 同权重梯度复核

为避免独立优化产生参数差异混淆比较，额外在GPU1加载同一个模型，先做一次更新使零初始化
的投影非零，再固定所有权重，用同一真实train pair分别开启/关闭重计算。
该核验不参与GPU0计时，GPU1作业在GPU0正式计时前完成。

- loss均为4.911818504333496。
- H0及六轮H最大绝对差0。
- 309个梯度张量全部通过 `rtol=1e-3, atol=1e-5` 检查。
- 梯度最大绝对差9.536743e-7；全梯度相对L2差8.246438e-10。

因此此次实现可以在BS1下以更多显存换速度，未发现同权重预测/损失变化或超过容差的梯度
变化。推荐下一步接入可配置开关，再在实际长训练中监测显存和吞吐；不保证完整训练同幅加速。
只测试，不自动开启正式配置，不接续/覆盖任何正式checkpoint。

## 复现与证据

```bash
cd /home/disk1/MHINet
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=/home/disk1/MHINet \
 /root/miniconda3/envs/loma-repro/bin/python scripts/probe_temporal_batch.py \
 --batch-size 1 --no-correlation-checkpoint --output artifacts/temporal4_bs1_checkpoint_off_probe.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=/home/disk1/MHINet \
 /root/miniconda3/envs/loma-repro/bin/python scripts/probe_temporal_batch.py \
 --batch-size 1 --output artifacts/temporal4_bs1_checkpoint_on_probe.json
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/disk1/MHINet \
 /root/miniconda3/envs/loma-repro/bin/python scripts/check_correlation_recompute.py
```

两份probe JSON包括真实资源、配置/架构/manifest hash及逐步结果；梯度核验在
`artifacts/temporal4_checkpoint_same_weight_gradients.json`。probe输出存在时拒绝覆盖，复测更换路径。
