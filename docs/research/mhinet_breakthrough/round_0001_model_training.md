# 第 0001 轮：当前 RGB512 + H0 联合方法、模型和训练审计

时间：2026-10-12（北京时间）。范围：代码、配置、已落盘训练/验证/级联诊断和测试报告的只读核验；未改模型、未占用 GPU、未训练、未用测试集选择参数。本轮检查了仓库及 `/`、`/home`、`/home/disk1` 的 `AGENTS.md`，未发现文件。

## 结论及证据等级

**最值得优先验证的是“真实上游错误下的细匹配/QRRU训练与质量判断”，同时建立真实级联验证。继续单独增强 H0 全局监督，不能保证最终精度提高。** 这是由实现和既有结果支持的研究排序，并非新实验已证实的因果结论。

1. **事实：训练中的 fine 父对应和 QRRU 初始对应均来自 GT，推理则来自模型预测。** fine 使用 GT D8 父单元；QRRU 使用 GT D2 中心加小噪声。错误父单元、真实细匹配的偏差、重复结构和错误但高置信对应没有按推理分布进入这两个训练任务。
2. **事实：QRRU 在合成验证非常准确，却在既有目标测试上损伤点级精度。** 完整合成 val 的 GT 附近噪声任务最终中心 EPE 为 0.1514 px；固定 8 对合成 val 的真实级联 P@1 为 97.06%。当前 JL 测试 fine→QRRU 的 P@1/3/5 均下降，GE 的 P@3/5 下降。这支持域/输入误差分布问题，不能单凭此断言 QRRU 存在普遍实现错误。
3. **事实：QRRU 后没有重新评估质量，最终 confidence 原样继承细匹配。** 更新门控制移动幅度，不是独立的正确率预测。默认 confidence threshold 为 0；最终仅剔除非有限或归一化坐标越界的中心。
4. **事实：H0 不直接引导 D8 配对，且细窗口中 H0 的仿射平移抵消。** H0 的真实作用是 overlap 支持和以预测 D8 目标单元为锚的局部形变；其全局定位变好不等于最终细对应变好。
5. **事实：25,535 个优化步全部触发全局梯度裁剪。** 第 1 epoch 的 H0 头梯度平方占总梯度平方的逐步中位比例为 81.83%，第 5 epoch 为 9.56%。独立 H0 loss 数值小不等于梯度影响小。**推断：**需测各损失对共享参数的方向与尺度，再决定分组学习率/裁剪/权重；现有日志不能证明冲突。

## A. 当前方法的真实调用路径

主入口是 `scripts/train_semidense_rgb512_h0.sh:14–22`，经 `scripts/train_semidense_next_tier.py:22–42` 从完整 E 组 **同时加载 shared 和 matcher**，然后使用新 AdamW 和新余弦调度器。这里没有重新随机初始化 QRRU，也没有继续 E 的优化器状态。

`SemidenseSystem.forward()` 的路径为 shared→`matcher.training_losses()`→独立 GHIM loss：`mhinet/downstream/training.py:59–93`。shared 输出包含 D8/D4/D2、H0 和可微 stage1 输出：`mhinet/models/feature_provider.py:351–428`。

`mhinet/models/modules.py` 中的 MCNet correlation decoder 和 adapters 属于另一套 `mhinet/models/model.py` / iterator 路径；当前 `SemidenseSystem` 不调用它们。不能把这些模块的搜索半径/角点上限套到当前 QRRU，或通过修改它们期待改变本轮半密集结果。证据：`mhinet/downstream/training.py:7–11,14–18`；引用入口可通过 `rg -n 'models.modules' mhinet` 核验。

当前完整训练：40,849 train / 5,106 val，5 epoch、25,535 步，microbatch=1 对、累积 8 对、shared 和 matcher 初始 LR 都为 8e-5；只冻结 DINO；VGG BN running statistics 固定；D1 停用。出处：`configs/semidense_rgb512_h0_joint_5epoch.json:2–33`，`docs/semidense_rgb512_h0_joint_5epoch.txt:25–48`。

## B. H0 的作用与梯度边界

### B1. 粗匹配：支持掩码，不是几何配对约束

推理先检查 H0 的 fit/legal 状态；失效时整对直接返回空。有效时用 H0 产生 A/B overlap masks，然后在两侧被保留的 D8 网格间做全局 cosine matching + mutual extraction。H0 没有作为候选点残差或对应评分输入。出处：`mhinet/downstream/semidense.py:161–171`，`mhinet/downstream/loma_reference/overlap_masks.py:22–36`，`mhinet/downstream/loma_reference/coarse_matcher.py:135–173,217–242`。

训练 coarse 正样本来自 GT 几何/双向 roundtrip 和真实 overlap mask；其 softmax 分母包含所有对侧格点，推理分母只包含预测 overlap 被 gather 的格点。出处：`mhinet/downstream/semidense.py:82–85`，`mhinet/downstream/supervision.py:44–61`，`mhinet/downstream/losses.py:19–34`。这是训练/推理支持和概率标定差异；不能直接断言该差异必然有害。

### B2. 精匹配：局部形变以 D8 预测目标为锚

窗口的准确公式为：

`b_k = project(H0, a_k) + grid_D8[j] - project(H0, grid_D8[i])`。

出处：`mhinet/downstream/supervision.py:64–78`。若 H0 为仿射 `H0(x)=A x+t`，则严格化为 `b_k=A(a_k-grid_D8[i])+grid_D8[j]`；**t 完全消失**。H0 是 projective 时还影响局部透视形状，不能说所有 H0 误差均抵消。

因此：若 D8 选错了目标父单元，fine 仍围绕错的 `j` 搜索；提高 H0 的整体平移精度不会自动纠正此错。单窗口源是 4×4 D2 子单元（源中心跨度 6 输入 px），目标是被 H0 局部形变的 16 个格点；每个源只在这 16 个目标格点中选取，没有在额外邻近父单元中搜索。不能将目标窗口也概括为固定 6 px，它随 H0 局部 Jacobian 改变。

训练的 `i,j` 来自 GT `coarse_labels`，即训练 fine 默认父对应正确；推理的 `i,j` 来自预测 coarse。训练 fine 的正标签再由 GT 对实际 H0-warped lattice 做 mutual nearest，并要求距离≤1.5 D2 px（512 输入下 3 px）：`mhinet/downstream/semidense.py:89–103,175–191`，`mhinet/downstream/supervision.py:81–104`。没有正标签的窗口不产生额外“不应匹配”项。

### B3. 独立 H0 监督可训练，matcher 先验仍 detach

`H0=shared['H0_norm'][batch].detach()`，fine grids/labels 在 no_grad；QRRU 根本不接收 H0。出处：`mhinet/downstream/semidense.py:87–114`，`mhinet/downstream/qrru.py:1,32–39`。

独立 `Lh0=geo+0.01 mat+0.0001 cls+0.05 H` 确实经过 stage1 和共享 MVT 反传：`mhinet/downstream/training.py:76–93`。这修复了旧 GHIM 头没有直接梯度的问题，却没有把 fine 的标签/候选选取变成连续可微的目标精度优化。shared MVT 同时受 descriptor 和 H0 loss 影响，仍有间接耦合。

`ghim_loss` 使用归一化坐标，robust_scale=0.1 在 512 目标图上对应 25.6 px 的坐标尺度；H 项是归一化矩阵前 8 项的平方误差，不是像素角点误差。fit 失败只跳过 H 项，不屏蔽整个 GHIM loss。出处：`mhinet/engine/ghim_losses.py:11–27,54–64`。该尺度不是“25.6 px 内无惩罚”，只是 robust penalty 的尺度；不能直接照搬像素损失权重。

## C. QRRU 的可恢复范围、训练分布与置信度

### C1. 真实监督任务

训练从 GT 下有效的 D2 格点抽最多 128 个查询；`pa` 是源 D2 格点，`pb=GT(pa)+clamp(N(0,0.8),-2.4,2.4)`，单位均为 D2 feature pixels。512 下每轴 σ=1.6 输入 px、每轴扰动界=4.8 输入 px，扰动向量长度可达 6.79 px。出处：`mhinet/downstream/semidense.py:104–114` 和 `SemidenseConfig:30–31`。

控制点标签在源中心±1 D2 的四个象限位置计算 GT flow；中心也单独监督。各次迭代以 γ=0.8 加权，控制/中心各自按有效数归一化，误差是逐坐标 Charbonnier 型 L1。出处：`mhinet/downstream/supervision.py:107–129`，`mhinet/downstream/semidense.py:133–145`，`mhinet/downstream/losses.py:37–48`。

推理输入却是 fine 真实预测；可能是接近 GT 的离散误差、偏置、大位移错配、重复纹理或无可匹配内容。训练没有专门的错误对应类别，也没有“不改动本已正确对应”/拒绝不可恢复对应的直接监督。小噪声任务成功不保证这些情况成功。

### C2. 局部感受与输出上界

`project` 将 D2 的 256 维降至 48 维；每对查询读取源/目标 4×4 网格（offsets 为±0.5、±1.5 D2），与 32 维隐状态直接拼接成 128 通道卷积输入。没有显式局部相关代价体、额外候选窗口或 H0。出处：`mhinet/downstream/qrru.py:16–27,37–47`。

每轮控制增量是 `sigmoid(gate)*1.5*tanh(delta)`，4 轮累加，每个控制点每个坐标的总 flow 严格小于 6 D2 px；输出中心是四控制点的均值，所以中心每轴也严格小于 12 输入 px，最大欧氏位移小于 16.97 输入 px。出处：`mhinet/downstream/qrru.py:50–55`。这是可达上界，不是实际稳定可恢复半径；缺少正确局部证据时，即便误差在界内也可能无法恢复。四象限多项式外推对 4×4 网格边缘的位移可超过控制点界，不能把中心界套到全部采样点。

把迭代数或 radius 加大并不能解决错误父单元和错误证据，只是扩大可移动范围；优先统计不可恢复误差和局部证据，再决定是否扩大搜索。

### C3. 置信度与过滤缺失

coarse confidence 是 D8 dual-softmax；fine score=`sqrt(exp(fine_logp)*coarse_confidence)`；按此排序最多取 12,000，再交给 QRRU。最终返回 `confidence=score[valid]`，无任何 QRRU 后重新评分。出处：`mhinet/downstream/semidense.py:185–211`。

QRRU gate 是坐标更新门，受坐标损失间接监督；不能作为概率校准后的正确匹配置信度使用。最终 `valid` 只有 finite 与[-1,1]，没有匹配一致性、更新前后质量比较、双向 cycle 或重新检查 predicted overlap：`mhinet/downstream/semidense.py:208–216`。

默认 `confidence_threshold=0`，coarse/fine 都接受≥0，mutual 关系能够筛去部分歧义，却不提供“没有对应”概率。正 focal **仍然通过分母含有隐式负样本竞争**，不应表述为“没有负样本”；缺少的是显式 non-match / 最终质量判别训练。出处：`mhinet/downstream/semidense.py:34–39,169–191`，`mhinet/downstream/losses.py:12–16,19–34`。

## D. 已落盘数量证据

下表只是已有模型/协议的诊断；没有据此选择阈值或更改模型。

| 已有任务 | fine | final / QRRU | 解读边界 |
| --- | --- | --- | --- |
| 合成 val，完整 5,106 对，GT+噪声训练任务 | 初始 EPE 2.000790 px | 四轮 EPE：0.459716、0.177392、0.152723、0.151394 px | 逐对 EPE 宏平均；是 teacher-forced 初始输入，不是完整级联 |
| 合成 val，最新固定 8 对，真实无 GT 级联 | P@1=38.0865%，EPE=1.187317 px | P@1=97.0615%，EPE=0.190102 px | 极小固定诊断集；不能替代完整/分来源 val |
| 当前 JL 测试 1,755 对 | P@1/3/5=7.62/40.91/62.57%，SR@3=42.85% | 7.10/37.92/59.51%，SR@3=43.65% | 点级精度下降，但全局 RANSAC 配准成功率微升；两者须同时报告 |
| 当前 GE 测试 498 对 | P@1/3/5=6.29/33.87/55.58%，SR@3=46.18% | 8.25/33.24/53.42%，SR@3=47.59% | P@1 上升、P@3/5下降；不能概括为所有精度都下降 |

来源：`outputs/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep/validation/step_0025535.json` 中 5,106 行 `qrru_epe_d2_by_iteration` 逐轮均值×2；`visualizations/step_0025535/summary.json`；`outputs/semidense_rgb512_h0joint_test512_20261012/comparison.md:27–38`。8 对诊断 `pairs.json` 展示的是固定样本，需确认来源分层后再讨论跨来源表现。

完整合成 val 的最终 loss：Lc=0.119172、Lf=0.141713、Lq=0.458855、Lh0=0.004740；H0 fit 有效率100%，fine_coverage=0.654073。验证的 coverage 分母是所有被抽 fine 窗口 token，包含边缘/无正标签位置，不能直接当作有效区域召回率。完整 val 只运行 training_losses，不运行真实 infer：`mhinet/downstream/train.py:52–95`。因此当前用于追踪收敛的验证 loss 不能回答上游错误能否恢复，也不能直接为最终匹配精度选择 checkpoint。

### D1. 所有步均全局裁剪，H0 梯度不能由 loss 数字判断

CPU 流式读取 `train.jsonl` 的25,535行，按 epoch 汇总，记录中的 `gradient_norm` 是裁剪前范数：`mhinet/downstream/train.py:294–307`。

| epoch | 优化步 | norm>1 比例 | 平均 preclip norm | H0头 norm²/总 norm² 的逐步中位数 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5,107 | 100% | 17.6823 | 81.83% |
| 2 | 5,107 | 100% | 9.0474 | 61.43% |
| 3 | 5,107 | 100% | 6.6752 | 44.13% |
| 4 | 5,107 | 100% | 4.9899 | 27.24% |
| 5 | 5,107 | 100% | 3.6881 | 9.56% |

第5 epoch：GHIM/H0头范数均值1.1542，MVT0.4615，整个 matcher1.2620，全部shared3.4164。GHIM/H0和MVT是shared组子集；不能把四个范数相加。H0 norm²比例计算的是独立头参数在全模型梯度平方中的比例，不是 H0 loss 对共享 MVT 的贡献比例。

**事实：**H0独立头的强梯度在早期主导全局裁剪。**待验证假设：**其影响可能改变 descriptor/refiner 的优化与共享特征取向。**不能由此推出：**梯度冲突已被证明，或 AdamW 的有效更新简单按1/norm等比例衰减；Adam矩估计会改变这种关系。验证时应记录各损失对同一 MVT/decoder 参数的单独梯度方向、裁剪比例和实际参数更新。

## E. 冻结与 BN 的限制应怎样判断

- DINOv3 L11/L17：冻结参数、eval、no_grad、BF16，16 px patch grid；512输入的语义格点为32×32，再经 MVT/VGG/累积 decoder 到 D2。出处：`mhinet/models/feature_provider.py:219–248,277–300,379–405`，外部 pinned实现 `/home/disk1/LoMa/experiments/loma_dinov3_l17_v1/descriptor.py:81–122,173–253`。冻结减少可训练容量/适配自由度，但 D2 已有可训练 VGG/decoder，不能仅凭16px DINO patch 宣称1px精度不可能。
- VGG卷积及BN affine可训练，BN running statistics保持固定：`mhinet/models/feature_provider.py:242–248`。小microbatch的常见稳定策略不等于已证实最佳；卷积漂移而stats固定是否造成目标域问题，应做实际分布/校准诊断。
- **该固定策略只遍历 VGG**。可训练 DeDoDe `ConvRefiner` 内部仍采用 BatchNorm2d，并随decoder `.train()`更新本地统计：`mhinet/models/feature_provider.py:248`，外部 `/home/disk1/LoMa/src/loma/descriptor/dedode.py:183–213`。单次前向是1对的两张图；梯度累积8并没有变成BN的8对batch。
- 新H0 run单卡每次1对；旧RGB512 run双卡每卡每次1对。有效batch相同仍不能保证decoder BN和rank RNG过程相同。现有新旧比较包含更多变量，既不能归因于H0，也不能归因于BN/学习率。出处：`docs/semidense_rgb512_h0_joint_test512_results.md:5–18` 和 `docs/semidense_resume_audit.md`。
- 优化组只有全shared和全matcher两个；当前两组LR相同，H0/MVT/VGG/decoder没有独立LR，warmup也未配置：`mhinet/downstream/training.py:95–97`、`mhinet/downstream/train.py:213–214`。不同子模块是否需不同LR是待测假设。

## F. 按收益与代价排序的最小验证实验

以下是待执行方案，本轮未执行。所有阈值、训练策略和checkpoint选择只使用预先固定的训练/验证集；GE/JL现有测试仅用来定位已观测现象。需要真实目标域验证时先核查来源、父图/地理分组隔离及测试重叠，再独立建立开发验证集。

### 0. 先补全“真实级联验证”与同点诊断

从完整现有val中按数据来源、变换强度、重叠率固定分层小集，保存mask/lineage、fine未过滤点、QRRU每轮中心、最终置信度。报告P@1/3/5、EPE、NCM、SR@3、有效点数和空间coverage。对于fine→final，使用同一保留源点逐点比较，避免剔点改变分母造成假改善。若当前val全部是同图合成，另建无测试重叠的真实目标域val，保留原合成val作为防遗忘检查。

同时分桶：fine初始误差≤1、1–3、3–5、5–12、>12 px；局部纹理、confidence、更新幅度、H0支持边缘；统计每轮改好/改坏及不可恢复点比例。12 px是每轴中心上界，误差桶的欧氏范围只作描述，不可将欧氏>12全算理论不可恢复；真正不可达条件是某个目标坐标误差≥12px。

**否定标准：**若真实目标val中初始化/appearance分布接近合成val、QRRU在同点比较稳定改善，且误差主要已在coarse全局错误，则降低“refiner训练分布”优先级，优先coarse/数据几何原因。当前只有小合成诊断和历史测试，尚不足完成此否定。

### 1. 最便宜的无训练检查：QRRU开/关、每轮输出和confidence有效性

在固定开发val上比较fine原始输出、迭代1/2/3/4的同点结果；绘制confidence可靠性和precision–NCM–SR曲线。先验证移除或早停是否已带来稳定收益，再讨论refiner结构。在同一预登记阈值/点数预算下比较，过滤策略还需看空间分布，不能只提高P而丢尽正确点。

**否定标准：**若匹配置信度与最终正确率已有良好校准、低分过滤不改SR，且所有迭代在目标val均提升，则“末端质量判断不足”不作首要方向。若固定选择的早停只在一个来源有效，在其他来源退化，则不采用全域统一早停。

### 2. 只改输入分布：真实预测对应训练QRRU

固定coarse/fine/shared，缓存训练集的真实fine候选及GT诊断；训练QRRU采用真实可恢复候选+GT噪声混合，覆盖近零误差、偏置、重复纹理和较大可恢复残差。先只训练现有QRRU结构，不同时改半径、损失、骨干或数据来源。不可恢复/无对应点先用于“应拒绝/保持”的任务设计，不强迫refiner学把无局部证据的点送向远处GT。用固定数据量、step budget和seed对照GT噪声训练，验证QRRU在真实候选上的增量收益。

**否定标准：**若相同预算下目标val的同点EPE/P@1和SR@3没有可重复改善，或仅减少点数导致P上升却NCM/SR下降，降低纯输入分布修复的优先级。不能只看Lq下降。

### 3. 只改父窗口：错误父对应下的fine/no-match训练与邻父候选

先用训练/val固定预测coarse审计“GT是否落入当前16格点候选”和“GT父对应下可达率”。若父错误是主因，第一实验只增加预测coarse父训练，显式标记无正确目标窗口；第二个独立实验才增加邻近父/不确定性驱动候选。保留原窗口与训练budget对照，测正确候选召回和最终精度、速度、内存。

**否定标准：**若当前目标val正确候选覆盖已高、主要错误发生在已有正确候选的排序和QRRU，扩大窗口不应优先。扩大后正确候选增加但P/NCM/SR没有改善也不能称突破。

### 4. H0“作用分解”诊断，避免盲目继续加全局loss

在开发val上做诊断矩阵：预测/GT H0仅用于support；预测/GT H0仅用于fine local lattice；固定coarse pairs以隔离局部窗口贡献。GT只作为显式oracle分析，不能混入正式推理结果。记录support recall、候选可达率与级联精度，不只H0全局RMSE。另统计H0 invalid是否是实际失败主因；合成val的fit100%不代表目标域100%。

**否定标准：**若oracle H0的support与局部lattice均不能带来明显可达率/精度收益，则继续提高H0是低优先级；应检查D8 descriptors、fine candidates或数据几何。若oracle支持改进明显而局部无收益，则只针对support做鲁棒化，无需直接去掉detach。

### 5. 梯度和优化：先诊断再做一个因素的短验证

在固定小训练集记录各损失对共享MVT/decoder梯度的范数、cosine、实际更新和全局裁剪贡献；对照单损失诊断（仅为诊断，不作最终性能结果）。之后每次仅改变一项：独立GHIM optimizer group/LR、分组裁剪、loss balancing或短warmup；同数据、初始权重、microbatch/BN、总步数和seed。若改VGG/decoder统计策略，另作独立实验。

**否定标准：**若无稳定梯度冲突、共享更新和级联val均不改善，则不以“多任务优化”作为首要解释。loss或gradient norm下降不构成成功；同一开发val的真实级联改善且其他来源不明显遗忘才可继续。

### 6. 最后才扩容骨干或替换refiner

只有当候选覆盖与真实误差训练修复后，目标val仍显示局部特征无法区分候选，才测试显式correlation/cost volume、局部更大搜索、DINO参数高效适配或D1等。各项分开，与现有D2/QRRU的同预算基线比较。当前不能依据冻结DINO或小QRRU结构就断言必须更换大模型。

## G. 本轮可复现信息

执行只使用文本读取、标准库JSON统计；未实例化模型或调用CUDA。上表训练日志均值按优化步计算；第5 epoch按每步 `records` 数量加权共40,849对，得到loss=0.592873061757，与训练文档0.592873一致。尾步只有1对，因此逐步均值0.5928385略有差异。

| 文件 | SHA256 |
| --- | --- |
| `mhinet/downstream/semidense.py` | `ff87ed824ba05084342bcec7122b451912871d38d8951d12759caa60df637e52` |
| `mhinet/downstream/qrru.py` | `6508838da7f3f35661d1aa0ec38a5d3f34ce4bc837223d10e7383fe26ff74b40` |
| `mhinet/downstream/supervision.py` | `318781b2425caffd9fc0cccd527230f7ff1d802ed0b9490cda07c4bdb44767d9` |
| `mhinet/downstream/losses.py` | `ddb47c2c474c5932f08a8549fabb355890ecaa37986ac521f815b43aa9970d2c` |
| `mhinet/downstream/train.py` | `1219df18c08cdf2cd1241eb59f2af07aac967431e5526b44939c5932695cf3cc` |
| `mhinet/downstream/training.py` | `4b1f15c15799e9cda927c2a45c4edc0ae703d345a1c0efd746a7313b3410a22e` |
| `mhinet/models/feature_provider.py` | `a32d80bf0c16f3cac4f9dd49439e3fed6153d1ba9c157a9385986072930c49f3` |
| `mhinet/engine/ghim_losses.py` | `3293553afa0b20ba9098a5c2e089df9d7c63889c3c8d6360b8c32fb72e66475b` |
| `configs/semidense_rgb512_h0_joint_5epoch.json` | `a8a89925d038198f80d01ec64402d61efbce6a3edf72c0fd1c706f2b6324e92f` |

研究限制：没有新增受控训练消融或目标开发val推理；历史测试不能承担新方案的调参/模型选择；当前报告明确区分实现事实、已有观测和未验证的机制假设。
