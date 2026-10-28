# 受控 Delta 调制研究：A0–A4（2026-10-28）

工作分支 `codex/cgmdp-global-modulation`，独立入口；不改变 `flow-d8-d2` 的训练代码、权重或正在运行的任务。沿用 stop-that-shit 的范围约束，本轮不增加均值/方差统计变体、D2全图调制、Cumulative新训练或局部Pair联合解冻。

## 问题与当前证据

全图均值同时受场景组成、覆盖率和成像差异影响；本轮继续使用相同均值条件以隔离插入位置与幅度控制的作用，不将该方法称为辐射归一化，也不声称五项风险已被证明。弱重叠污染和局部窗口找错仍是保留的限制。

已有254对实验归档在 `reports/cgmdp_residual_research/legacy_archive.json`：None/Cumulative/Delta的原验证选点、RGB/HZ结果及32对训练样本上的旧Gamma/Beta统计。原始权重和完整结果仍在 `outputs/cgmdp_global_modulation`。这些结果只作结构参考，不能直接与1024对训练后的局部Pair判定优劣。

32对训练样本测得D8/D4/D2 decoder增量平均L2范数分别为4.305、8.552、2.694。旧Cumulative各尺度平均|Gamma|约0.546/0.728/0.612，Beta范数比约0.359/0.252/0.349；旧Delta分别约0.418/0.614/0.685和0.264/0.117/0.449。幅度不小不等于描述子塌缩，需要与区分度及候选margin一起判断。

## 分组、数据与公平性

|组|全图调制|局部Pair|本阶段更新|
|---|---|---|---|
|A0|无|无|冻结参照，仅评估|
|A1|无|现有1024对训练检查点|冻结参照，仅评估|
|A2|D8受控Delta|无|D8 MLP，10轮|
|A3|D8受控Delta|与A1相同，冻结|D8 MLP，10轮|
|A4|D8+D4受控Delta|与A1相同，冻结|D8/D4 MLP，10轮|

这不是五组从零开始、总历史更新数相同的训练比较，而是以A0/A1为共同冻结参照的增量适配实验。A2–A4同样10240次优化器更新，A3/A4共享同一份预训练Pair；本阶段没有继续更新Pair。参数量A2/A3各82496，A4为164992。A1沿用既有保存权重，不声称它是旧训练历史的绝对最优。

数据完全沿用 `outputs/d2_modulation_expanded/selection.json`：RGB训练1024、验证64、测试64+192=256；杭州64为既有 `test_controlled` 子集，**不是完整test_real，也不是新人工点标注**。RGB的64对和192对清单根路径不同，通过ConcatDataset分别读取后合并，不替换样本。杭州仅报告近似H参考一致性。

同一冻结DINO/MVT/Flow/VGG/DeDoDe及温度，初始化由原 `qrru_corners.experiment.load_models()` 提供，Flow为 `weight/mhi_v2_u0.pt`，下游为 `outputs/flow_d8_d2/training/best.pt`；局部Pair为 `outputs/d2_modulation_expanded/full_pair/best.pt`。未使用当前全量RGB续训产生的新Flow权重。QRRU不执行。

## 模型与监督

仅对当前decoder增量应用 `delta + lambda*(gamma*delta + beta)`，累计描述子及context原逻辑保留，D8的改动沿累加/上采样传向D4/D2。

- `gamma = 0.1*tanh(raw_gamma)`。
- `beta_c = 0.1*RMS_train(delta_c)*tanh(raw_beta_c)`：各尺度、各通道分别由训练集前32对标定，保存在calibration.json。0.1是预先设定的相对扰动预算，不是数据证明的最优值。
- lambda固定为1，不增加一个无约束的可学习系数；记录其值以及Gamma/Beta统计。
- 最后一层零初始化；保留decoder原BF16输出dtype，以保证初始恒等性。漂移统计与损失使用FP32计算。
- `L = L_D2_cls + 1.0*L_D8_global + 0.1*mean_s(drift_s)`。
- D8全局正对应监督复用 `joint_single.loss.global_d8`：RGB精确H标签，采样256个真值查询、对全局D8描述子做分类。它不依赖no_grad父点筛选，因此能直接训练D8调制器。
- D2仍为既有四邻域软标签dual分类；128个父窗口中随机抽64个训练。D8源点筛选和D2候选坐标不反传，窗口特征重新采样以保留D2损失至调制器的梯度。
- 漂移在D8/D4/D2累计描述子上相对同一冻结基线计算，控制破坏幅度，不作为越小越好的独立性能指标。

AdamW，lr=1e-4，weight_decay=1e-4，余弦至1e-6，10轮，无早停，seed42。仅RGB验证的完整重新匹配P@3选择best，任何严格提升均保存；另保留latest和优化器/调度器，支持在epoch边界继续。没有中途利用测试结果调整配置。

## 评价约定

每轮验证同时输出：

1. **固定窗口**：冻结基线D8源父点、连续目标中心和窗口坐标，在改动后的D2特征上重新采样，只比较描述子变化。
2. **重新匹配**：用调制后D8重新筛父点和精化目标中心，再做相同D2匹配。每对保存原父点、当前父点，以及加入/丢失的D8网格ID（`y*64+x`）。

固定窗口的D8几何指标按固定基线报告；重新匹配的D8指标反映改动后的中心。两模式均输出D8 P@3/5/10、候选覆盖、D2窗口覆盖、正候选Rank@1/5、P@1/3/5/10、EPE、共同源点上的改坏/救回及新增/丢失点数。阈值均为512原图像素；P@k分母是全部输出点；EPE/覆盖以已知可见点为分母。

D8覆盖指9×9连续搜索候选中存在距离GT不超过8/sqrt(2)像素的有效候选；D2覆盖指存在3px内候选。Rank统计以有3px内正候选的查询为分母。改坏率分母为共同源点中基线正确点，救回率分母为共同源点中基线错误且标签可见的点；新增/丢失点不混入这两个比例。

特征诊断包含分尺度Gamma/Beta、累计漂移、局部Pair的Gamma/Beta，以及窗口内不同空间位置描述子的平均余弦相似度、最强正候选减最强负候选的余弦margin。正负由3px阈值及有效候选定义；margin仅在同时有正负候选时统计。quality字段是逐影像对均值，几何精度由计数汇总。不能仅凭Beta比值上升判定塌缩。

## 验收与判断顺序

- 首先确认D8独立监督及D2监督都能到达对应调制器，Backbone/Flow/Pair保持冻结；零残差复现原D8/D4/D2描述子。
- A2对A0判断D8-only收益；A3对A1判断在已有局部Pair基础上的额外收益；A4对A3判断D4是否值得保留。
- 固定窗口提高、重新匹配下降时，优先归因为候选选择/覆盖变化，而非仅看最终损失。
- 综合RGB/HZ精度、覆盖、改坏/救回和漂移决定是否保留结构；不预设A3一定胜出，不用单seed小差额声称学术优势。
- 本轮结果若没有收益，记录负结果；不自动扩大为更多结构、联合解冻或追加训练。当前三尺度旧实验尚不满足同预算更优的证据，不追加训练对照。

## 文件与运行

新增代码仅三份：`residual.py`（受控Delta与冻结特征复用）、`research_metrics.py`（两种评价及诊断）、`research.py`（标定、归档、训练与评估），均位于 `mhinet/v2/cgmdp_modulation/`。配置为 `configs/mhi_v2/cgmdp_residual_research.json`。

```bash
cd /home/disk1/MHINet/.worktrees/cgmdp-modulation
export TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp
PY=/root/miniconda3/envs/loma-repro/bin/python
$PY -u -m mhinet.v2.cgmdp_modulation.research --device cuda:0 --smoke
# 正式进程已经启动，不要重复执行；中断后同命令会从完整epoch恢复
$PY -u -m mhinet.v2.cgmdp_modulation.research --device cuda:0
```

正式输出 `outputs/cgmdp_residual_research/`；独立smoke目录带`_smoke`后缀。每组有epoch验证、history、best/latest（训练组）、RGB/HZ测试与done。每组完成后自动更新 `reports/cgmdp_residual_research/results.json` 和 `results.md`，逐对详细数据保留在outputs，不复制进代码目录。

Smoke：训练4对、验证/测试各2对；A0/A1评估和A2–A4各1轮均通过。新增模块梯度非零、冻结模块无梯度，峰值allocated显存1.94GiB。最初发现BF16增量被FP32调制输出提升dtype，已修正并通过零残差逐位核对。随后完整基线评估发现两个RGB测试清单根路径不同，已改为分别加载后拼接，并核对索引0/63/64/255可读取。

正式训练正在进行，预计约1–2小时，实际受GPU0已有并发任务影响；尚不能给出A2–A4最终结论。原全量RGB训练进程未中断。
