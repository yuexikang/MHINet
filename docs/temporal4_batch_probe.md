# temporal4：有效batch8的真实BS1/BS2对照

2026-09-11，同一张物理GPU0（RTX4090）顺序运行，开始时GPU0/1空闲、没有MHINet正式训练。
使用新train的前16对循环，seed0、原始预训练初始化、frozen_dino_mvt模式与四项GHIM监督。
两组均有效batch8、原正式学习率及500步warmup，3步性能预热、6步计时；每组9次优化。
包含影像读取、组批、H2D、完整前反向、loss、梯度检查、clip和optimizer/scheduler。
不含模型加载、验证、checkpoint保存；小集合读取受文件系统缓存影响，不是完整训练ETA。

| 配置 | 平均秒/优化步 | 中位秒/优化步 | pair/s | 峰值allocated GB | 峰值reserved GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BS1 × 累积8 | 6.7933 | 6.8012 | 1.1776 | 8.393 | 9.783 |
| BS2 × 累积4 | 5.5323 | 5.5321 | 1.4461 | 15.555 | 19.212 |

GB为十进制。BS2吞吐提升22.79%，每步耗时减少18.56%；allocated提高约85.3%。
两组均无OOM、loss/梯度范数有限；DINO/MVT无梯度，第二步后GHIM head/VGG/CGMDP/MHIR
梯度非零。D1无调用，每个microbatch共享DINO/MVT各调用一次。未保存或使用探测权重。

## 数值对齐：仍未通过，不能把性能完成写成正式配置通过

在任何优化更新前，对同样16对影像分别按BS1/BS2前向，H0四角坐标最大绝对差
0.3831176758px，超过既有0.1px门槛；平均四角欧氏差0.11635294px，16对均拟合有效。
这证明批量算术输出差异仍存在，不证明真实精度一定下降。此次没有放宽门槛、没有修改
正式batch保护开关，也没有把BS2设为正式默认。
先前诊断将主要差异定位在BF16编码器/交互路径，当前新数据复现了现象；首个差异算子
仍需定位，不把旧定位结果当作本次重新逐层审计。

另一个需要公平性处理的问题：当前GHIM geo/mat按整个microbatch的有效点/正负样本数量
归一化，BS2合并两对并不严格等于两次BS1损失的平均。因此如果要求仅改变执行batch而不改变
目标权重，后续需统一为逐对计算再平均，并测试混合有效/无效H0样本的精修loss权重。
此次数据每步8对均有效，不涉及该无效样本分母差异；测试脚本忠实复用现有loss。

## 复现

```bash
cd /home/disk1/MHINet
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=/home/disk1/MHINet \
 /root/miniconda3/envs/loma-repro/bin/python scripts/probe_temporal_batch.py \
 --batch-size 1 --output artifacts/temporal4_ebs8_bs1_probe.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=/home/disk1/MHINet \
 /root/miniconda3/envs/loma-repro/bin/python scripts/probe_temporal_batch.py \
 --batch-size 2 --output artifacts/temporal4_ebs8_bs2_probe.json
```

输出文件已存在时拒绝覆盖，复测应使用新文件名。两份JSON包含配置/manifest/架构hash、
真实权重位置、初始逐对H0、每步损失分项、梯度范数、显存、环境和计时。
后续优先修复批量H0对齐及loss归一化一致性，再复测；如保持约1.23倍吞吐，确实可用更多
显存换速度。这里不承诺完整长训练也能同幅度加速，且没有访问val/test选择配置。
