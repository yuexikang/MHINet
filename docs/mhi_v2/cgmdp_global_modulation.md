# CGMDP 内部全图调制验证

2026-10-27；独立分支 `codex/cgmdp-global-modulation`，基于 `codex/flow-d8-d2` 的 `c6b9af1`。工作区 `/home/disk1/MHINet/.worktrees/cgmdp-modulation`。原工作区未切换分支、未修改原模型，GPU2上的完整训练继续。

## 插入点和两种含义

真实执行路径为 `SharedFeatureProvider._decode_pyramid()`。每层产生 `(delta, context)`；累计描述子先加delta，输出本层D8/D4/D2，然后累计描述子与context分别上采样。二者是两条独立通路：原decoder的下一次计算使用context，不直接使用累计描述子。

设 `U` 为原有bilinear上采样，`delta_s` 为本层decoder增量：

- 原方法：`D_s = U(D_prev) + delta_s`。
- 累计调制：`D_s = Phi_s(U(D_prev) + delta_s)`，调制后的D_s用于本层匹配并上采样给下一层累计求和。
- 增量调制：`D_s = U(D_prev) + Phi_s(delta_s)`；继承的累计描述子不再次经过本层Phi，调制增量仍随累计结果传播。

两种方式均不修改context，不把调制后的累计描述子误称作下一层decoder的context输入。D16不调制，D8/D4/D2各一个独立MLP；不执行D1和QRRU。

Phi保留现有候选调制公式：A/B共享权重，分别全图平均池化，拼接 `[own, other, |own-other|]`，768→64→512生成gamma/beta，`F'=F+gamma*F+beta`。A/B同时使用原始统计量。区别仅在统计范围从8×8候选上下文变为整张特征图；每个通道的gamma/beta在全图共享，不是逐像素空间注意力，也不需要预测对应、H或重叠掩码。三个MLP合计247,488参数，末层零初始化。

## 独立实现

- `mhinet/v2/cgmdp_modulation/model.py`：复用 `LocalPairModulation` 参数化；独立包装冻结模型，显式实现上述两个插入点。
- `mhinet/v2/cgmdp_modulation/experiment.py`：最小验证和三组小规模训练/验证/测试。
- `configs/mhi_v2/cgmdp_global_modulation.json`：本轮唯一配置。
- 本文：执行位置、监督、结果解释和命令。

只训练新增的三个MLP，DINO/MVT/Flow/VGG/原decoder/匹配温度均冻结。独立包装使调制后的累计描述子上采样链保留梯度，避免原冻结decoder的no_grad块吞掉调制梯度。D4没有新增匹配头，接收来自D2监督的间接梯度。

候选匹配仍保留原8×8上下文、A中心4×4、B全部8×8和连续解码，但候选内调制设为恒等，不与全图调制叠加。窗口选择不反传，D2全图采样特征保留梯度。

## 小规模对照

使用原固定RGB254训练/64验证/64测试，杭州64测试作跨域观察，复用原selection。不使用正在训练的Flow新权重：固定 `weight/mhi_v2_u0.pt` 和 `outputs/flow_d8_d2/training/best.pt`，便于隔离插入点影响。

比较none、cumulative、delta；同seed42、零初始化、数据顺序和训练上限。训练监督仅为此前D2四邻域软标签dual分类，不混入新的D8监督或空间平滑。每对随机64父窗口，验证128窗口、2048点上限。

AdamW lr=1e-4、weight_decay=1e-4；余弦降至1e-6；最多20epoch；至少5轮后，连续3轮验证P@3未超过显著提升参考值0.001就早停。任何严格验证P@3提升均保存best，与早停阈值分离。none只做评估，不训练。两个可训练方案早停轮数可能不同，报告实际预算，不称作固定等步数比较。

只用RGB验证选权重，完成后在RGB64与杭州64测试；记录P@1/3/5/10、EPE、候选GT排名和覆盖率、实际输出点数。原候选双图调制的已训练权重不直接复制到全图MLP，因为条件统计分布不同；此前候选调制结果仅作历史参考，不把训练预算和参数量差异掩盖掉。

## 已完成的最小验证

- none及两个零初始化方案，三个尺度与原模型逐值一致，Flow输出逐值一致。
- 只在D8加入非零偏置，D4/D2累计描述子同步变化，证明上采样传递已接通。
- 两次优化步骤中，三个MLP末层均有非零梯度，第二步隐藏层梯度均非零；原模型所有参数无梯度。
- 累计方案两步loss为4.376、3.166；增量方案为4.422、3.473。不同步骤是不同图像，不能把它们当作收敛曲线。
- 峰值allocated显存约2.21GiB，无非有限梯度或OOM。具体值见 `outputs/cgmdp_global_modulation/smoke.json`。

三张卡已有任务；本次smoke和小规模对照只占GPU0剩余空间，没有停止其他进程，没有在GPU2追加任务。

## 运行与产物

工作区内`outputs`链接到原仓库输出根，新增内容只写独立子目录 `outputs/cgmdp_global_modulation`；`weight/mhi_v2_u0.pt`链接到原只读初始化。没有复制或修改基础权重。

```bash
cd /home/disk1/MHINet/.worktrees/cgmdp-modulation
export TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp
PY=/root/miniconda3/envs/loma-repro/bin/python
$PY -m mhinet.v2.cgmdp_modulation.experiment --smoke --device cuda:0
# 三组串行执行；已启动，不要重复覆盖目录
$PY -u -m mhinet.v2.cgmdp_modulation.experiment --device cuda:0
```

日志：`outputs/cgmdp_global_modulation/run.log`。分组产物：`{none,cumulative,delta}/{epoch_*.json,best.pt,history.jsonl,test.json,hangzhou_test.json,done.json}`；none没有训练history。最小验证通过只说明接线及梯度正确；训练和测试完成前不判断哪个插入点更准确。
