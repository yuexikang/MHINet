> 研究档案：原实现与运行命令位于 `codex/cgmdp-global-modulation`；本次仅迁移报告，不将这些实验模块加入正式推理。

# A3动态条件的无训练诊断（2026-10-28）

使用 `outputs/cgmdp_residual_research/A3/best.pt`（原RGB验证P@3选出的检查点），冻结Backbone、Flow、D8调制器、局部Pair及匹配温度。无优化器、无反传、不写入原检查点。

## 统计与静态对照

- 在原1024对RGB训练样本上提取实际有界Gamma/Beta，按每通道分别对图像对和A/B两侧求均值，得到256维固定Gamma和Beta；测试集不参与均值估计。
- 固定对照的A/B两侧使用同一个静态向量，保留原lambda、数据精度、Delta插入位置、上采样累计路径和局部Pair权重。它不是“每图一个标量”的对照，也没有只固定Gamma或只固定Beta。
- 训练集、RGB测试256对、杭州test_controlled64对分别报告统计。杭州指标仍只是近似H参考一致性。
- Gamma符号比例和总均值/标准差按全部样本、两侧、通道统计；另外报告每张图内部跨通道std，避免将图像间变化误当成通道差异。
- 每通道跨图像对方差分别对A侧、B侧及两侧均值计算，population variance（ddof=0）。256个通道的完整统计保存在diagnostics.json。
- 饱和定义为`abs(Gamma)>=0.095`，即0.1上限的95%；同时报告99%门槛。正负不代表语义上的正负匹配。
- 特征条件使用D8 decoder增量的空间均值。计算A/B均值向量的L2差、相对差、余弦距离及均值向量范数，与Gamma/Beta幅度及其偏离训练均值的程度做Pearson/Spearman关联；还记录每通道参数与同通道绝对特征均值差的关联。常量变量返回null，相关不等于因果，低线性相关也不等于没有条件响应。

对同一冻结特征，重算三种情况：无全图调制基线A1、动态A3、固定平均Gamma/Beta。每种情况都测固定基线D8父点/目标窗口和重新匹配两条路径。固定窗口有助于排除候选替换的影响；重新匹配反映实际使用收益。P@3分母为全部D2输出，阈值为512原图3px；记录P@1/3/5/10及可见点EPE。

## 文件与运行

新增入口 `mhinet/v2/cgmdp_modulation/diagnose_a3.py`，不改原训练/推理默认代码。

```bash
cd /home/disk1/MHINet/.worktrees/cgmdp-modulation
export TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp
/root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.v2.cgmdp_modulation.diagnose_a3 --device cuda:2
```

可读汇总：`reports/cgmdp_a3_conditioning/diagnostics.json`。原始逐图像对Gamma/Beta/条件均值数组及逐对计分保留在 `outputs/cgmdp_a3_conditioning/`，日志为 `outputs/cgmdp_a3_conditioning.log`。使用空闲GPU2，只执行一次既有检查点推理诊断。

## 完成结果

已完成 1024 对训练统计及 RGB 256 / 杭州 64 对诊断；全程无训练。固定训练均值后，重新匹配 P@3 为 RGB 88.8887%、杭州 59.7977%；动态 A3 为 88.9114%、59.8053%。动态参数对 P@3 的额外增益分别仅 0.0227、0.0076 个百分点。详细 Gamma 分布、跨图像方差、相关性、固定窗口与重新匹配对照见 [诊断报告](../../reports/cgmdp_a3_conditioning/analysis.md)。
