# Flow 损失审查与独立消融提案

2026-10-20审计记录。审计当时没有修改当前训练。用户随后授权自动实验，新损失及独立续训入口已实现，执行协议和实时状态见 [P6自动实验](FLOW_LOSS_EXECUTION.md)；当前C/D/E仍沿用锁定的Flow来源。同权重B/C的完整5106对评价已完成，最终Pre@3差值仅0.00384个百分点，见 [证据](evidence/V2-P2-04/flow_best_prior_toggle_fullval_20261020.json)。以下诊断不代表新loss收益。

值得比较更贴近 EPE 的损失，但现有证据不能把瓶颈完全归因于损失。当前损失能改善大多数影像对，少量灾难性错配几乎不动；冻结的匹配相似度和训练预算同样需要排查。

## 已经验证的事实

**1. 优化目标与评价量不同，但没有证明损失实现错误。** [当前实现](../../mhinet/v2/losses.py)使用逐 x/y 的 Cauchy/Lorentzian：

`rho = sum_xy log1p(0.5 * (residual_normalized / 0.1)^2)`

512输入下，归一化残差乘256得到输入像素残差，因此尺度为25.6像素/轴。评价 EPE 是二维像素端点距离的宏平均。小误差区的当前惩罚近似平方误差；它与径向EPE不同。Head输出仍是normalized坐标，回传梯度包含×256，不能只看损失标量或像素单位导数就断言“定位梯度太小”。具体推导见 [单位审计](evidence/flow_loss_review_20261020/loss_units.json)。

现有形式及 c=0.1、mat/cls系数来自 [LoRetta原论文III-E与IV-A](https://arxiv.org/html/2608.04106v1#S3.SS5)，其Cauchy设计用于减轻噪声伪warp目标的影响。这提供了原设计依据；是否适合本RGB512微调目标需要实验证明。不能因平均EPE下降有限就把鲁棒损失当成错误。

**2. 当前 cls 不是可学习的位置分类分支。** CPU使用冻结v1的实际Head权重、冻结合成context完成梯度拓扑检查：

| 项 | 可达参数 |
| --- | --- |
| geo | no-match两参数、共享卷积trunk、warp输出，共8张量 |
| mat | no-match两参数、共享卷积trunk、matchability输出，共8张量 |
| cls | 只有 `no_match_logit` 这个标量 |

DINO/MVT特征冻结，temperature是buffer，分类概率在decoder之前生成。提高cls权重不能改变同一查询对各真实目标的概率排序。dustbin logit加1后真实目标间log概率比最大浮点差仅9.54e-7。若未来需要学习位置分类，应另登记可训练Q/K映射或MVT策略，不能把架构变化混入单纯loss消融。

geo和mat共享卷积trunk，确有相互作用通路；**尚未测真实样本的加权梯度范数与夹角，不能声称mat抢占了geo梯度。** 合成诊断只验证连通性，不代表真实数据梯度大小。见 [CPU证据](evidence/flow_loss_review_20261020/gradient_topology.json)。

**3. 大多数样本变好，尾部基本不变。** 以下分位数都是5106个“影像对平均EPE”的分位数，并非单点残差分布：

| 每对平均EPE，px | 原始冻结Flow | 训练后 |
| --- | ---: | ---: |
| 全集均值 | 0.79010 | 0.76118 |
| p50 | 0.56480 | 0.54471 |
| p95 | 1.03085 | 0.94103 |
| p99 | 2.24726 | 2.03888 |
| 最大值 | 152.55537 | 154.66333 |

3593对改善，1513对退化，但退化超过0.1px只有12对。训练后最坏52对全部为FLAIR，贡献总体均值约24%；这个固定集合的平均EPE为17.92936→17.93716px，基本没改善。尾部可能涉及错误匹配模式或冻结特征的信息限制，不能假定换回归函数一定修复。应补点级残差、输入影像、GT投影和高置信错误证据；这些val难例仅用于诊断，不加入训练或驱动hard mining。

总loss下降的标量分解约为：77.9%来自加权mat、21.0%来自geo、1.1%来自cls。这不是梯度贡献率。完整逐对原始数据放在outputs，版本化摘要及SHA见 [误差审计](evidence/flow_loss_review_20261020/error_distribution.json)。

**4. 尚未证明训练进入平台。** 最后epoch全val EPE从0.83992降至0.76118，下降9.37%；前两epoch又曾劣于原始冻结Flow。额外训练预算、学习率过程和损失形状存在混淆。任何新loss续训都必须有旧loss相同步数/LR的续训对照，不能拿额外训练后的模型直接对比当前checkpoint。

## 第一项候选与公平比较

先只试按输入像素计算的径向 smooth-EPE / Charbonnier：

`L_geo_new = mean_valid(sqrt(dx_px^2 + dy_px^2 + epsilon_px^2) - epsilon_px)`

epsilon初始候选固定0.5px。GT有效性、D16 cell centers、两侧mask、每pair归一化全部沿用。几何项不得乘预测confidence，避免通过降低置信度逃避位置监督。

直接以单位权重替换会同时改变整体梯度量级。先固定train-only校准集（seed42，23 FLAIR/6 GoogleEarth/3 WHU）测真实context下的共享trunk梯度，对新geo乘固定常数，使初始中位梯度范数与旧geo相近。记录该常数后再跑试验，mat=0.01、cls=0.0001先保持。不能依据val挑梯度权重。

| 支路 | 起点 | 变化 | 共同额外预算 |
| --- | --- | --- | --- |
| L0：旧loss延长对照 | 当前Flow-best，克隆同一optimizer/RNG状态 | 保持当前损失 | 1 epoch / 5107更新，LR1e-6 |
| L1：新loss | 相同checkpoint及状态 | 仅换geo函数并作train梯度尺度校准 | 同上 |

两臂使用相同第六轮样本顺序、B1×累积8、10个Head可训练张量；新runner必须显式审计optimizer、scheduler扩展与代码身份，不放宽现有训练入口的严格恢复检查。pilot尚未启动，没有可执行训练配置。若结果有价值，再从同一v1起点做同5epoch预算的独立确认，额外seed另行登记。

## 如何判定是否值得采用

- 同时报告Flow全val宏平均EPE、分来源、pair尾部，以及另测的point级误差；不能混用这些统计量。
- 当前matchability监督的是GT可见/可匹配，不等于预测端点正确。另测c≥0.25/0.5/0.75的误差与覆盖，尤其高c且>8/16/32px的大错位。
- Flow实际通过D8插值后的Gaussian软奖励起作用，sigma=4 D8即32输入像素。0.79→0.76px这类亚像素均值变化，对典型GT附近奖励很小；这是示意推断，不是实际平均reward估计。还要检查D8的GT父对应保留率、有效覆盖和最终Pre@3。
- 做旧/新Flow各自prior关闭/开启的2×2对照，冻结同一QRRU。prior关闭时最终点对应相同，作为负对照。不能仅凭EPE更低替换当前C/D/E锁定的Flow来源。
- 第一轮不同时改Cauchy尺度、cls结构、hard mining、多尺度、双向监督或confidence标签，避免无法归因。直接缩小Cauchy尺度可能更早抑制大离群点，未必有利于尾部。

本审计登记诊断和比较方法。实际执行以 [P6自动实验](FLOW_LOSS_EXECUTION.md) 及其独立状态文件为准；没有新loss精度结论。
