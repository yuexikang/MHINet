# 正式推理1024与训练加速实测

2026-09-28：按用户选择，将 `SemidenseConfig.inference_window_chunk` 默认设为1024，同时用于D2细匹配和QRRU推理。`window_chunk=32`只控制训练的细窗口与QRRU查询分块；粗匹配分块 `chunk=128`不变。没有修改权重、采样数量、损失、QRRU轮数或H0残差先验路径。

旧checkpoint缺少新字段时采用1024，不需要转换。完整权重仍strict加载；跨档初始化将旧matcher配置补齐后核验，忽略纯推理分块差异，但继续拒绝其他训练参数差异。精确resume仍核验旧源码/配置身份，需使用相应历史版本。两个评估入口新增 `--inference-window-chunk` 覆盖选项，并将实际matcher配置写入registration/report；分块实验脚本也改为覆盖新字段，避免新版本下误测同一个分块。

## 推理证据

- 完整C权重第三档val 3848对：32→1024，平均3902.36→223.93 ms，17.43倍；最终EPE变化约+6.44e-7 px。10对各替换1个源点，非逐位无损。详见 `artifacts/semidense_chunk_accuracy_results.json`。
- E旧权重兼容性对比：独立合成test的索引0、666、1333，分别对32/1024预热后计时；平均3894.38→223.06 ms。三对源点集合和precision@1/3/5/10px均相同，一对排序变化，公共点最大目标位移0.00082171 px。每对12000个QRRU查询，实际调用从375次降到12次（11×1024+736）。详见 `artifacts/semidense_e_chunk1024_compatibility.json`。此项只证明有限样本兼容性，E+1024完整val尚未执行。

训练时的可视化快照也会采用新推理分块；GT解耦验证损失仍走训练分块。

## 训练：固定数据前反向对照

E权重、RTX4090 GPU0、同一第三档训练集索引0/100/1000，每档每对预热一次、测量三次。相同权重、输入和随机种子；保持实际训练的确定性设置、冻结策略、现有BF16共享网络路径和FP32匹配损失。无optimizer更新，其他GPU有独立任务，绝对耗时受环境影响。

| 训练window_chunk | 完整前向+损失+反向/对 | 共享前向 | 损失前向 | 全部反向 | 耗时减少 |
|---|---:|---:|---:|---:|---:|
| 32（保留默认） | 504.65 ms | 102.01 ms | 96.70 ms | 305.89 ms | — |
| 64 | 409.33 ms | 102.14 ms | 59.20 ms | 247.93 ms | 18.89% |
| 128（下一轮候选） | 374.94 ms | 102.13 ms | 46.66 ms | 226.10 ms | 25.70% |

此处不含读盘、数据传输、梯度裁剪、optimizer.step、定期验证和可视化，不能直接宣称整轮训练节省25.70%。各档本次峰值allocated约6.14 GiB，整体峰值由其他阶段主导，不表示匹配头本身的激活内存没有增加。

128相比32总损失最大绝对差5.96e-8，但梯度不是逐位相同。共享训练组（名称cgmdp，实际包含VGG+CGMDP）梯度相对L2差0.69%–1.02%，下游组0.0117%–0.0173%；64对应0.58%–1.09%、0.000086%–0.000604%。分块改变反向累加顺序，现有BF16路径可能放大差异；本次没有隔离证明唯一原因。不能用单次loss近似相等代替收敛验证。

复现脚本：`scripts/benchmark_semidense_training_chunks.py`；原始记录、数据/权重/代码SHA及逐组梯度差：`artifacts/semidense_training_chunk_benchmark_20260928.json`。脚本只计算梯度，不更新或保存模型权重。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/loma-repro/bin/python scripts/benchmark_semidense_training_chunks.py \
  --checkpoint outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt \
  --output /tmp/semidense_training_chunks.json
```

## 后续训练优化顺序

1. 优先对照训练window_chunk=32/128。每对只有64个fine窗口、128个QRRU查询：分块32时分别2/4次，128时各1次；再改1024没有额外合并收益。相同起点、数据顺序、学习率、seed和有效batch4，用独立配置运行短程训练，再比较固定val的Lc/Lf/Lq、梯度范数和级联精度。先允许新训练实验配置显式采用128；不要伪装成原训练精确resume。通过后再升级训练默认值。
2. 减少训练内部CUDA→CPU同步：循环内多次 `float/int/bool` 读取诊断与有效计数，可将诊断汇总后统一传回CPU。保持空监督分支和control/center各自归一化，避免改变Lc/Lf/Lq含义。收益尚未实测。
3. 测试coarse分块128→256/512，以及fine/QRRU取消部分activation checkpoint，用显存换少量重算和调度开销；保留相同正样本数和完整softmax分母。先比较梯度和峰值显存，收益尚未实测。
4. 若实测读盘限制GPU利用率，再增加DataLoader worker、pin_memory和异步传输。冻结DINO/MVT/H0的输出也可缓存，但必须绑定精确输入、预处理、权重身份；存储和读取成本需单独评估。不能缓存仍训练的VGG/CGMDP输出。

共享网络已使用BF16，不能把“开启AMP”当作尚未采用的直接收益。QRRU更低精度、torch.compile或增大物理batch会引入其他数值/执行变化，排在上述优化之后。正式训练默认本次保持32，没有启动新训练。
