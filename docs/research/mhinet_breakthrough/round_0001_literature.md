# MHINet 突破研究，第 1 轮：论文机制与可证伪假设

检索与核验时间：2026-10-12 11:19（Asia/Shanghai）。研究范围：当前半密集 RGB512/H0 joint 分支、遥感跨时相匹配、精细定位、训练数据与跨域适配。只采纳论文正文、作者项目页、作者仓库等一手证据。本轮未训练、未修改模型或评测协议。

**优先突破口是“监督与推理误差分布对齐 + 跨域能力保留 + 独立标签质量”，骨干替换排在后面。** 当前 DINOv3/MVT 与最新强方法已有明显交集；单独修复 H0 梯度并没有转化为最终精度提升。下面的方案是待验证假设，不能当成已经有效的改进。

## 1. 本地证据与研究口径

已阅读 [H0 joint 完整测试报告](/home/disk1/MHINet/docs/semidense_rgb512_h0_joint_test512_results.md)、[JL 微调跨域退化报告](/home/disk1/MHINet/docs/jl1flight_finetune_transfer_analysis.md)、[半密集协议](/home/disk1/MHINet/docs/benchmark_semidense_v1.md)，并检查 [SemidenseMatcher](/home/disk1/MHINet/mhinet/downstream/semidense.py)、[QRRU](/home/disk1/MHINet/mhinet/downstream/qrru.py)、[监督构造](/home/disk1/MHINet/mhinet/downstream/supervision.py)。

- H0 joint：RGB512 40,849 对，5 epoch、25,535 更新；GE 的 P@1/SR@3 为 8.25%/47.59%，JL 为 7.10%/43.65%，与旧 RGB512 基本持平。H0 梯度修复成立，整体突破尚未出现。
- 旧 JL 域微调能将 JL P@5/SR@3 提升到 92.65%/72.36%，同时 GE SR@3 从 49.20% 跌到 8.63%。这证明当前架构具有适应 JL 的能力，也暴露单域更新严重损害旧域几何能力。
- 旧域退化已发生于 D8；DeDoDe 风格累计解码器的 BN buffers 显著改变。BN 是需要控制实验隔离的候选原因，不是已证明的唯一原因。
- GoogleEarth GT 只记录在名义对齐后的原始跨时相配对上新增的合成几何，并非独立测量的原生跨时相配准。已确认并移除 2 对错误 GT；剩余样本不能因模型误差大而自动删去。
- 当前半密集 matcher 直接读取 D8 和 D2；D4 是累计解码过程的中间特征，不能把它描述成当前 matcher 的独立 D4 匹配阶段。
- QRRU 的训练起点来自 `H_gt + clipped Gaussian noise`，默认标准差 0.8 D2 单元、逐坐标截断 2.4 D2 单元，即 RGB512 中约 1.6px 和 4.8px。推理起点却来自实际 D8→D2 fine 预测，包含错误粗父节点、重复纹理、形变与 H0 误差。训练/推理起点来源不一致是直接可见的代码事实，是否导致当前降分仍需实验。
- fine 分类训练用 GT coarse 父对应；推理用预测 coarse 父对应。QRRU 推理后的最终 confidence 沿用 fine/coarse score，QRRU 的 sigmoid 更新门不是经过几何误差监督校准的匹配正确率。

论文中的 PCK、位姿 AUC、mAA、TPS 评分，与本项目逐对宏平均 P@1/3/5 和 H-RANSAC 后网格 SR@3 **口径不同**。本轮不拿论文数字与 MHINet 主表直接排序，不承诺迁移收益。

## 2. 核验文献清单

| 文献 | 初次公开时间 / 已核验状态 | 一手来源 | 对本项目最有价值的机制 |
|---|---|---|---|
| LoRetta: A Foundation Model and Extensive Dataset for Global-Scale Remote Sensing Dense Image Matching | 2026-08-04；arXiv，注明 submitted to TPAMI，不能写成已接收 | [论文](https://arxiv.org/abs/2608.04106)、[正文](https://arxiv.org/html/2608.04106v1)、[作者项目页](https://www.siweiyu.com/work/loretta/) | 数据原生 matchability；定位后残差注册；先训定位再冻结 |
| RoMa v2: Harder Better Faster Denser Feature Matching | 2025-11-19；arXiv v2 2025-11-20，本轮未核验正式会议发表 | [正文](https://arxiv.org/html/2511.15706v2)、[作者代码](https://github.com/Parskatt/RoMaV2) | 冻结 coarse 后，以其真实输出训练 refiner；不确定度；EMA；大小基线数据混合 |
| LoMa: Local Feature Matching Revisited | 2026-04-06；v2 2026-07-06；作者仓库注明 ECCV 2026 Oral | [论文](https://arxiv.org/abs/2604.04931)、[正文](https://arxiv.org/html/2604.04931v2)、[作者代码](https://github.com/davnords/LoMa) | 匹配架构不变也可通过 GT 质量、数据多样性、训练配方改善；HardMatch 人工验证 |
| Who Handles Orientation? Investigating Invariance in Feature Matching | 2026-04-13；作者 LoMa 仓库关联 CVPRW26 / LoMa-R | [论文](https://arxiv.org/abs/2604.11809)、[正文](https://arxiv.org/html/2604.11809v1)、[作者 LoMa-R 发布说明](https://github.com/davnords/LoMa) | 描述子与 matcher 分别承担旋转不变性；旋转增强与数据规模的交互 |
| MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training | 2025-01-13 arXiv；作者仓库现在标注 TPAMI 2026，本轮未独立核验出版方 DOI | [论文](https://arxiv.org/abs/2501.07556)、[正文](https://arxiv.org/html/2501.07556v1)、[作者项目页](https://zju3dv.github.io/MatchAnything/)、[代码发布状态](https://github.com/zju3dv/MatchAnything) | 多来源数据同时混合优于依次单域训练；结构差异增强；单一权重跨域泛化 |
| MINIMA: Modality Invariant Image Matching | arXiv 2024-12-27；CVPR 2025 | [CVF 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.pdf)、[作者代码](https://github.com/LSXI7/MINIMA) | 从 RGB 几何监督继承多模态匹配标签，保持已有 matcher 的数据层改进 |
| Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed | 2024-03-07；CVPR 2024 Highlight | [论文](https://arxiv.org/abs/2403.04765)、[正文](https://arxiv.org/html/2403.04765v1)、[作者代码](https://github.com/zju3dv/EfficientLoFTR) | 先离散 MNN 定位，再在 3×3 局部相关上求亚像素期望，抑制整窗期望偏差 |
| LiftFeat: 3D Geometry-Aware Local Feature Matching | 2025-05-06 arXiv；ICRA 2025，作者仓库给出 DOI 10.1109/ICRA55743.2025.11127853 | [论文](https://arxiv.org/abs/2505.03422)、[作者代码与出版信息](https://github.com/lyp-deeplearning/LiftFeat) | 用伪表面法向监督几何特征，再与 2D 描述子融合 |
| XFeat: Accelerated Features for Lightweight Image Matching | 2024-04-30；CVPR 2024 | [论文](https://arxiv.org/abs/2404.19174)、[正文](https://arxiv.org/html/2404.19174v1)、[作者代码](https://github.com/verlab/accelerated_features) | 在 coarse 特征中保留像素偏移信息；低成本配对偏移分类；可靠性分支 |
| RoMa-Ω: What Feed-Forward 3D Models Know About Image Matching | 2026-09-08；arXiv 预印本，本轮未核验接收或可用官方权重 | [论文](https://arxiv.org/abs/2609.09507)、[正文](https://arxiv.org/html/2609.09507v1) | 3D foundation 特征需要线性探针/专用 decoder，零样本最近邻好坏不能判断骨干上限 |

“最新”指本轮检索并核验到的相关公开工作，不声称穷尽所有截至当日论文。发表状态按链接显示的证据级别记录，后续轮次需要更新。

## 3. 可迁移机制、代码差距与最小验证

### 3.1 RoMa v2 → 优先修正 refiner 起点分布

论文 §3.3 明确在 coarse 训练结束后冻结 matcher、以 inference mode 输出 coarse warp 训练 refiners；同时加入残差协方差和 EMA。其数据混合兼顾宽基线鲁棒性与小基线精细定位。[正文](https://arxiv.org/html/2511.15706v2)

当前 QRRU 学的是 GT 附近人工噪声校正，部署接真实 fine 残差；当前最终分数也没有反映 QRRU 修改后的可靠性。**最小验证**：先在训练/开发集记录同点 fine→final 残差变化，按初始误差、置信度、场景分桶；再冻结 DINO/MVT/H0/VGG/CGMDP，只用它们的真实 fine 预测训练 QRRU，并混入 GT 小噪声起点保留精细能力。超出可恢复范围的错误父节点不强行当成可回归样本，先设拒绝/不更新目标。

成本：诊断一张空闲 GPU 推理即可，坐标缓存很小；不要批量缓存完整 D2 特征。短程 QRRU-only 训练另列配置和算力预算。风险：真实预测大量错误，简单全部回归会把校正器变成补偿错误粗匹配的模型；GT 噪声会被精修放大。EMA 只可能缓解较小定位偏置，不能解释或修复数像素级系统偏差。

### 3.2 LoRetta → 匹配支持域必须区分“可见”与“可对应”

LoRetta 使用 no-match 候选和 matchability 头，数据以混合 NMI/NCC 与几何验证生成伪 matchability；定位后预测残差，分阶段冻结定位分支。LEVIR-GM 也排除不可靠区域的严格几何监督。[论文正文](https://arxiv.org/html/2608.04106v1)

当前半密集分支主要以 H0 几何重叠生成支持域，再做 MNN；跨时相变化、阴影、云、重复纹理不等于可见且可对应。**最小验证**：在开发集抽样建立变化/稳定区域标注，以双向几何一致性、描述子峰值唯一性、空间覆盖等只依赖预测的信号排序；保持固定 top-K，观察 P@1/3 与网格 SR@3 是否同时提高。训练 matchability 之前先核对已有 H0 监督中的 mask 是否真的表达稳定对应，避免只学图像边界。

成本：CPU 可先做已有预测分析；新增头和监督为中等工作量。风险：NMI/NCC 伪标签会偏爱纹理和光学同模态；不能用伪标签作为独立真值验证它自己，也不能把无匹配输出的精度改高当作召回改善。

### 3.3 LoMa → 把标注与数据配方放在模块堆叠前

LoMa 复用 DeDoDe/LightGlue，并通过 17 个数据源、GT 重建质量和训练配方推进效果；HardMatch 使用人工对应和独立标注者核验。该工作提供“架构不是唯一变量”的直接证据。[正文](https://arxiv.org/html/2604.04931v2)

当前 RGB512 仍高度依赖合成几何，JL 微调只有 674 个 parent，各 10 个 affine 变换；场景数与 pair 数应分别记录。**最小验证**：制作仅来自训练/开发 parent 的跨时相真实对应小集，记录变化区与标注不确定度；按 parent/地理场景拆分，防止同原图的多个变换跨 split。以同训练步数比较真实跨时相混合与纯合成。先用已有 LoMa-B/R 权重做统一协议的外部诊断，不重训大模型。

成本：精确人工对应比扩充合成 pair 更有价值；小集推理成本低。风险：论文完整训练用多卡 A100，其规模不能直接转化为本项目小时级任务；外部 matcher 伪 GT 会有教师偏差，不能代替人工独立验证。

### 3.4 MatchAnything → 多域同时混合，避免连续单域覆盖

正文明确比较多来源混合与逐来源顺序训练，后者较差；其数据由多视几何、视频和单图合成共同组成，再引入外观与结构差异。[正文](https://arxiv.org/html/2501.07556v1)

当前 JL 微调没有混入旧 RGB512，且共享描述子、解码器 BN 统计、MVT 全部可变。**最小验证**：先进行 BN 统计交换的离线组件实验；再在开发集比较固定 decoder BN 的 JL-only、固定 BN 的 JL+RGB512 混合、以及只更新 QRRU/轻量适配参数。记录 JL 提升和 GE 下降的 Pareto 曲线，模型选择不能只看 JL loss。数据域混合与“模块先后训练”是两条不同轴：可以冻结 coarse 后训练 refiner，同时让每个阶段保留多域数据。

成本：BN 交换无需训练；短训练只保留一项因果变化。风险：论文大规模完整预训练依赖 16×A100，不适合作为首轮复制目标；作者仓库目前写明训练代码稍后发布，不能假定可以原样训练复现。

### 3.5 Efficient LoFTR → 亚像素目标要直接监督亚像素输出

其 fine 先 MNN 像素定位，再以局部 3×3 correlation-softmax 求期望，第二阶段直接监督最终亚像素坐标；局部窗口避免整窗期望被其他峰拉偏。[正文](https://arxiv.org/html/2403.04765v1)

当前 fine 输出 D2 warped lattice 的离散 MNN，QRRU 通过四角控制场平均输出中心偏移。**最小验证**：在完全相同 fine 匹配点与 D2 特征上，比较不更新、QRRU 和局部相关峰拟合/小窗口期望；先确定哪个精修机制在真实 residual 中更稳定，再决定是否训练一个轻量亚像素头。可以在双线性 D2 特征上探测，D1 特征成本留待证据支持后增加。

成本：只验证 selected matches，无需全分辨率 dense decode；小窗计算便宜。风险：D2 插值不一定含足够精细信息；重复纹理相关分布多峰时直接期望仍会漂移；论文原版使用更细特征，不能照搬它的亚像素成绩。

### 3.6 MINIMA → 保持几何的外观/结构增强

MINIMA 将 RGB matching 数据通过生成/估计模型扩充为多模态，继承几何标签，并复用多个已有 matcher。[CVF 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.pdf)、[作者训练与数据代码](https://github.com/LSXI7/MINIMA)

**最小验证**：先用轻量且能核对几何的灰度、色彩、阴影、模糊、锐化、局部纹理缺失增强，并和真实跨时相样本混合；只改变增强，保持 parent、步数、采样和优化器一致。检测输入域与 decoder BN 分布的交互。

成本：简单增强低；CycleGAN/扩散生成高，暂缓。风险：风格转换可能改变局部结构，继承的标签就不再精确；合成 SAR/红外不是对应成像物理真值，本项目光学跨时相也不必先转向全模态。

### 3.7 Who Handles Orientation? → 用旋转分桶定位失效源

论文区分 descriptor 与 matcher 的旋转训练，显示数据规模也影响旋转泛化；在描述子提前学不变性可减少后续匹配层的工作。[正文](https://arxiv.org/html/2604.11809v1)

**最小验证**：先按真实相对旋转/尺度/重叠分桶比较 D8、fine、final；如果大角度仍主导失效，比较小步训练中仅 matcher 承担旋转与 CGMDP 也接受旋转增强。先评估 LoMa-R 是否仅在大角度桶明显优于当前模型。

成本：分桶分析低；重训完整论文模型高。风险：小数据下强不变性可能降低描述子区别性；论文的大规模结果不保证 40K pair、数百 parent 条件同样成立。

### 3.8 XFeat → 低成本精修与可靠性参考

XFeat 用 coarse 配对特征预测 8×8 像素偏移类别，并训练可靠性分支；其 semi-dense refinement 不依赖全图高分辨率特征。[正文](https://arxiv.org/html/2404.19174v1)

**最小验证**：把它作为速度/内存对照，或在相同 MHINet fine 父节点上测试小型 residual 分类头是否能胜过当前四角 recurrent update；继续保留坐标回归或更细局部相关以验证 P@1。

成本：小头和抽样计算低。风险：原始像素偏移分类不等价亚像素定位；轻量骨干可能降低跨时相鲁棒性。不能只以速度优势推荐整体替换。

### 3.9 LiftFeat → 重复纹理的几何辅助，后置验证

LiftFeat 用预训练单目深度产生伪表面法向标签，监督几何特征并融合到 2D 描述子。[论文](https://arxiv.org/abs/2505.03422)、[作者代码](https://github.com/lyp-deeplearning/LiftFeat)

**最小验证**：在重复屋顶/农田开发桶核验单目 depth/normal 是否稳定，并只加小型几何辅助投影对照；首先跑公开 LiftFeat 权重了解失效模式，不据此替换 CGMDP。

成本：伪标签预处理增加模型推理与存储；轻量投影中等。风险：俯视遥感分辨率、近正射场景与地面单目深度训练分布差异大，错误几何伪标签可能强化重复纹理歧义，优先级低于真实残差与 GT。

### 3.10 RoMa-Ω → 骨干候选需要受控线性探针

最新预印本比较 DINOv3 与 3D foundation features，指出原始最近邻效果差的后层特征，经线性探针/decoder 后仍可能含强对应信息；完整模型冻结 VGGT-Ω 并重训 RoMa v2 管线。[正文](https://arxiv.org/html/2609.09507v1)

**最小验证**：仅在前面问题隔离后，缓存少量训练/开发特征，以同一线性投影、同样 GT、同一配对和匹配预算比较 DINOv3 与候选层；若没有开发集收益就停止。禁止仅根据最近邻可视化决定骨干。

成本：该论文完整训练是多卡 A100、数十万步，远高于现有首轮预算；小探针低得多但仍要核验权重是否正式可用。风险：3D 模型天然更偏邻近视角与视频分布，未必适合跨季遥感；本轮未核验官方推理权重，不能登记为随时可跑的基线。

## 4. 优先级与停止条件

| 优先级 / 假设 | 最小独立证据 | 通过条件 | 停止或降级条件 |
|---|---|---|---|
| P0：真实 fine 残差与 QRRU 训练噪声失配，导致错误修正 | 同点 lineage 保存；每次迭代 ΔEPE；初始误差/置信度/场景分桶；no-update 对照 | 冻结 shared、真实预测起点训练后，在未参与训练的开发 parent 同时改善 P@1/3 和几何 SR，且另一域无实质退化 | 只修复 GT 合成噪声却不改善真实 fine；或收益来自删除大量困难点 |
| P0：部分指标低由原生 GT 不精确/变化区强制匹配放大 | 独立稳定点人工对应、重复标注一致性；保留原始 GT 另报独立 GT 结果 | 标签不确定度可测；明确真实模型误差与名义几何误差的差别 | 使用当前模型自己产生的 H 当 GT；以模型误差决定删样本 |
| P1：decoder BN 更新与单域训练共同造成 GE 退化 | before/after weights × before/after decoder BN buffers 的组件对照；多个训练域开发集 | 固定 BN / 混合回放在保留 JL 增益时恢复 GE；组件替换能隔离效应 | 只比较不同训练步数/不同 parent 的结果；把参数变化率直接当因果贡献 |
| P1：confidence 不反映 fine/QRRU 后正确性，输出数量掩盖质量 | 固定 top-K；空间分块均衡；预测一致性与真实错误的校准曲线 | 同预算/同支持域 P 与 SR 同升，不靠集中于小区域 | 只提升 P@5、网格 SR 继续下降；无匹配率明显增加 |
| P2：离散 D2 fine 与四角 QRRU 限制严阈值定位 | 相同 fine 点 local-correlation 与 no-update；D2 插值探针 | 小窗方法提升 P@1，且不放大多峰错误 | 未证明 feature 含有更细信息就增加 D1/大型 refiner |
| P3：新骨干能补足粗对应信息 | 冻结特征线性探针、同 GT/同参数预算 | 开发集多个域、多个难例桶一致改善 | 仅天然零样本可视化好看或外部位姿 AUC 更高 |

通过条件需要用预先冻结的开发集和配对 bootstrap 不确定区间评估；不能在 GE/JL Test_Data 上选最佳配置，再回报无偏测试成绩。全量正式测试用于固定方案后的确认。研究可以先做已有输出 CPU 分析，不应由“每小时一次”推断每小时可自由启动完整训练。

## 5. 后续轮次的具体问题

1. 保存的预测是否足够逐点重建 fine→final 的增益/伤害？若只有阶段集合指标，下一次开发集诊断必须保存同点 lineage。
2. 实际 fine 残差中有多少落在训练 noise bound 内、多少落在 QRRU 理论位移范围内、多少根本是错误 coarse parent？误差范数与逐坐标范围分别统计。
3. 模型 BN buffers 的历史变动能否解释一部分 GE 几何偏差？先采用组件对照，避免立即改所有归一化。
4. 半密集 `fine_gt_tolerance=1.5` D2 单元的标签容差和连续坐标目标，是否让模型学到 P@5 合格却 P@1 不准的对应？先对照真实精确点。
5. 预测置信度与误差/空间覆盖的关系如何？若高置信错误形成一致偏置，单纯 RANSAC 内点率不能作为真实配准成功证据。
6. 现有训练数据的 parent、地理位置、时间段、生成几何与 radiometry 组成如何？优先补真实跨时相和独立精确标签，而不是继续复制同 parent 的合成变换。

本轮建议下一小时首先完成第 1–2 项只读诊断。待证据确认后，第一项可训练的改进应是 **冻结 coarse/shared 后，用真实 fine 起点和多域混合训练 QRRU**，并把 matchability/拒绝机制与亚像素精修分开验证。
