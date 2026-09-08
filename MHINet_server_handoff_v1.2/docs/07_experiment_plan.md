# MHINet 实验表与执行优先级

所有条目均为**待执行计划**，不包含测量结果。完整机器表见 experiments/experiment_registry.csv/json；Excel 是供人工安排与记录的同源表。先工程验收，再训练；20个候选不等于必须一次全跑。

## 1. 最先完成的实验

先 E00 → E01，确认冻结特征下八轮确实改善。然后 E02–E05 用同一 E01-W checkpoint 分叉做2×2解冻比较：DeDoDe始终解冻，VGG开/关与MVT开/关。每条续训10k步，不能把 E02 再续训成 E03 后当等预算对照。E01-W 使用该种子 heads 阶段按统一 val 规则选出的固定checkpoint，保存准确步数/hash。

该四分支回答“哪些特征组值得微调”；E02相对E01的改善也含额外步数，若要单独宣称解冻DeDoDe有效，必须增加 E01-C：同一E01-W仅heads继续10k步，使用相同重启和学习率日程。E01-C是该结论的必要对照。

| ID | 优先级 | 实验变化 | 训练组 | 起点 | 新增步数 | 对照 | 启动条件 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| E00 | P0 必做 | 仅评估原Stage1 H0 | eval | 原选定checkpoint | 0 | 后续全部 | 固定val IDs；失败也记录 |
| E01 | P0 必做 | 四尺度各2轮；L1等权 | heads | 原选定checkpoint+新层初始化 | 10000 | E00 | 通过P0–P4；非no-move |
| E02 | P1 主线 | 续训DeDoDe+新层 | decoder_finetune | 同一E01-W checkpoint | 10000 | E01-W及E03–05 | 相同续训数据/optimizer重启 |
| E03 | P1 主线 | 在E02参数组上加VGG | vgg_finetune | 同一E01-W checkpoint | 10000 | E02 | 保持MVT冻结 |
| E04 | P1 主线 | 在E02参数组上加MVT | mvt_finetune | 同一E01-W checkpoint | 10000 | E02 | 两条MVT梯度通过；记录自身H0 |
| E05 | P1 主线 | DeDoDe+VGG+MVT+新层 | joint | 同一E01-W checkpoint | 10000 | E02 / E03 / E04 | 相同起点和预算；记录自身H0 |
| E06 | P1 结构 | 仅D8各2轮；共2轮 | heads | 原选定checkpoint+新层初始化 | 10000 | E00 / E01 | 同数据种子；按实际U归一化loss |
| E07 | P1 结构 | D8/D4各2轮；共4轮 | heads | 原选定checkpoint+新层初始化 | 10000 | E06 | 从头训练同预算 |
| E08 | P1 结构 | D8/D4/D2各2轮；共6轮 | heads | 原选定checkpoint+新层初始化 | 10000 | E07 / E01 | 同时报告增加D1的显存和延迟 |
| E09 | P1 机制 | 四尺度各1轮；共4轮 | heads | 原选定checkpoint+新层初始化 | 10000 | E01 | 同训练步数；补报算力成本 |
| E10 | P1 机制 | 同尺度第二轮复用首轮C和mask | heads | 原选定checkpoint+新层初始化 | 10000 | E01 | 当前H-flow仍更新；不detach首轮C |
| E11 | P2 可选 | 仅将两个H-flow输入通道置零 | heads | 原选定checkpoint+新层初始化 | 10000 | E01 | 参数数目不变；保留H引导采样 |
| L01 | P1 损失 | 仅改corner为SmoothL1 beta1；等权 | heads | 原选定checkpoint+新层初始化 | 10000 | E01 | 同单位同预算；不加grid |
| L02 | P1 损失 | 仅改gamma为0.8；仍L1 | heads | 原选定checkpoint+新层初始化 | 10000 | E01 | 序列权重归一化 |
| F00 | P2 条件 | 继续原L1；活跃组lr乘0.2 | 选定主配置 | 共同选定B checkpoint | 2000 | F01 / F02 | 标签可信且进入亚像素区 |
| F01 | P2 条件 | 移植batch均值+sticky flag FGO | 同F00 | 共同选定B checkpoint | 2000 | F00 | 记录B=1；八轮求和/均值尺度约定 |
| F02 | P2 条件 | 仅D2/D1；lambda0.04；ramp200 | 同F00 | 共同选定B checkpoint | 2000 | F00 / F01 | 标量最大梯度倍率5；相同活跃组 |
| X01 | P2 诊断 | L1+overlap网格权重0.1 | heads | 原选定checkpoint+新层初始化 | 10000 | E01 | 仅内部transfer指标暴露缺口才跑 |
| X02 | P2 诊断 | L1+aux corr权重0.02 | heads | 原选定checkpoint+新层初始化 | 10000 | E01 | GT已在窗内但相关峰不学习才跑 |
| E01-C | P1 归因 | 只续训heads；与E02等预算 | heads | 同一E01-W checkpoint | 10000 | E02 | 宣称DeDoDe解冻收益时必做 |


## 2. 最小论文证据与节省预算

第一批工程结束后只有 E00/E01；第二批 E02–E05（如主张DeDoDe收益，加E01-C）；第三批尺度链E06/E07/E08对E01、轮数E09、重采样E10。基础损失只比较 L01/L02 对 E01。E11、FGO和辅助损失保留为条件任务。

尺度实验先固定冻结特征，只改变结构；不能拿 joint 四尺度对 frozen 三尺度称为 D1 增益。已有完整E01在各尺度末的输出可先做免费轨迹诊断，但“同一模型提前停止”的结果不等于“独立训练截短网络”的消融。E06–E08才是后者。

E09四轮和E01八轮相同步数只保证训练样本与更新预算相同，不保证FLOPs相同；必须报告成本。若发表计算效率结论，可另外给等GPU时间对照，另取新ID。E10重用首轮相关性时不detach且保留同一图，改变的只有是否重新取证。

先种子0做预算筛查；选定主配置、E01和关键最强对照再做0/1/2（已有seed0复用，不重复算）。只有validation用于挑选配置和checkpoint。正式冻结后对同一test一次性评估所有预注册主对照，不用test反调窗口/损失。

## 3. FGO与扩展的公平比较

F00/F01/F02从同一B checkpoint出发，训练组不变、每组lr乘0.2、相同2k步。F01保留MCNet代码的实际microbatch均值与序列内sticky flag，移植到八轮；为与本模型L1尺度一致，主比较将八轮总loss除以8。这是“原非线性/门限语义、均值归一化”的移植，不称完整原模型复现；若严格使用官方sum，单独注明loss尺度与gradient clipping差异。

旧L00混合loss与旧L01 SmoothL1+gamma0.8仅作历史可选，不列首批；当前L01是等权SmoothL1以正交比较L1，当前L02是L1+gamma0.8。不要仅凭旧编号启动错误配方。H0 anchor不预先新增主loss；出现MVT初始化漂移时先降低MVT lr/核对预处理，再定义独立实验、明确其H0四角项权重和对照。

## 4. 每个run必须保存

run_id、experiment_id、seed、git commit、config SHA256、data/split hash、init checkpoint hash、最终checkpoint、实际optimizer步数、活跃参数组、GPU/dtype、命令和起止时间。逐pair保存：parent/group ID、stage1_valid、每轮accepted/reason、H0/每轮/Hfinal MACE、grid error、window recall、更新幅度。

结果表同时给原始Stage1 H0、当前模型自身H0及最终H。使用 train/val/test 独立标识；Excel运行记录的一行对应一个run在一个split上的汇总，同一run不同split分行。未知结果留空，不能填0或写已通过。详细失败定义、native px转换和计时规则见docs/02。

最终论文表建议：主结果（E00/E01/选定联合配置）、结构消融（E06–E10）、训练组（E02–E05及必要E01-C）、损失（L01/L02及有条件F00–F02）。不自动组合所有有效技巧，先证明各项独立贡献。
