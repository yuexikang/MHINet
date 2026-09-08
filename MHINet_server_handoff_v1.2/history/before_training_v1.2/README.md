# MHINet v1：完整多尺度 Homography 迭代模型设计

Multi-scale Homography Iteration Network · 2026-09-07 · 结构设计 v1.0 · 损失修订 v1.1

**当前目标只有一个：保留现有 Stage1 和共享特征基础，实现 D8/D4/D2/D1 上连续、可训练的 H 迭代。** 本版不包含 Planar Reliability，不新增置信度预测、对应点加权拟合、GRU 或语义监督。Stage1 自身已有的概率对应和 matchability fitter 保留。

模型工作名改为 MHINet，以免名字仍暗示包含平面可靠性。历史 PRHNet v0.1 保留作为后续研究记录；本版是当前有效实施方案。

## 模型定案

| 项目 | v1 固定选择 |
| --- | --- |
| 输入 | 一对 RGB，网络坐标下均为 784×784；首版 B=1 |
| Stage1 | 原共享 DINOv3 L11/L17 → MVT → Stage1 head → 原 fitter → H0 |
| 局部特征 | 原 VGG19-BN + 累积 DeDoDe，显式导出 D8/D4/D2/D1 |
| 特征通道 | 原描述子均256；新 adapter 后64/64/32/32 |
| 迭代次数 | 每尺度2轮，总8轮 |
| 相关性 | 围绕当前 H(p)，在目标特征原坐标中进行局部采样 |
| 查询位置 | 当前尺度全部源特征网格点；分块计算，不随机抽点 |
| 搜索半径 | 4/4/3/2个当前尺度像素，对应81/81/49/25个候选 |
| 更新器输入 | correlation + 候选有效mask + 源坐标 + 当前H-flow |
| 更新器 | 每尺度一个轻量CNN，同尺度两轮共享参数 |
| 输出参数 | 4×2四角位移残差，经几何层重建H |
| 新增参数 | 2,544,160，按本设计逐层计算；不含原特征网络 |
| 监督 | 等权8轮proposal四角L1；FGO与辅助项默认关闭 |
| 推理输出 | H0、8轮H、4个尺度末H、最终H及数值/边界有效状态 |

这些是可直接据此实施的结构选择，不表示超参数已经实测最优。当前交付为模型设计、接口、配置和实现计划，尚未实现训练网络。

## 文档

- [完整模型规范](docs/01_full_model.md)：所有模块、张量、相关性和几何公式。损失以v1.1审视为准。
- [训练与验证协议](docs/02_training_and_validation.md)：冻结策略、损失、梯度、数据和验收。
- [实现接口与检查表](docs/03_implementation_contract.md)：类边界、forward流程和异常约定。
- [固定模型配置](configs/mhinet_v1.json)：模型结构完整，环境路径单独传入。
- [层级尺寸与参数表](artifacts/layer_shapes.csv)：784输入下的所有新增层。
- [几何公式数值核验](artifacts/geometry_equation_check.json)：128组NumPy/FP64合成检查；不代表网络forward/backward或训练已经验证。
- [总体架构图](figures/architecture.png) / [SVG](figures/architecture.svg)。

- [损失审视与新默认](docs/04_loss_review.md)：MCNet公式/代码差异、L1、bounded-FGO及可选辅助项。
- [损失标量数值检查](artifacts/loss_numeric_check.json)：检验公式的连续性、梯度和门限语义，不代表训练效果。

## 如何读 H 的编号

$$
H_0
\xrightarrow[2\ \mathrm{updates}]{D_8}H_{[8]}
\xrightarrow[2\ \mathrm{updates}]{D_4}H_{[4]}
\xrightarrow[2\ \mathrm{updates}]{D_2}H_{[2]}
\xrightarrow[2\ \mathrm{updates}]{D_1}H_{[1]}=H_{\mathrm{final}}.
$$

每个尺度末的 H 也可在论文总图中写成 H1/H2/H3/H4；代码中用 H_updates[:,0:8] 保存8次更新，避免把“4尺度”误当成“只有4次迭代”。

本版“多尺寸”指同一图像对的多分辨率特征精修。任意输入分辨率扩展是另外的工程任务；v1不声称原项目固定784的接口已经支持任意尺寸。

基础源码固定为 2-step 提交 `899fdb99a4076e312196bbbf99a3c329739b5d7a`；参考 MCNet 提交 `cc03479689b3cf40f0c384954f338b434765c155`。本轮沿用上一轮已完成的源码/PDF审计，没有再次假称核验远程最新提交。

设计核验已完成：新增层参数的公式计数为2,544,160；128组合成H下，四角DLT重建与尺度坐标变换的FP64数值误差均低于1e-8px。PyTorch网络、GPU精度、反向传播和实际精度仍待实现验证。
