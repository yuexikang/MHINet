# 训练与验证协议 v1.2

本版替代旧文档中“整个 Stage1 永久冻结、H0 始终 detach”和“仅解冻 DeDoDe”的训练限制。结构不变，loss_revision=1.1。步数和学习率是首轮实验预算，不是已经验证的最优值。

## 1. 数据与训练输入

主线使用可信 A→B H 标签，优先原项目精确单图合成数据。RGB [0,1]，共享入口 bicubic 784；按母图与地理区域划分 train/val/test。同一母图的合成对不得跨 split。已有反复诊断的 test 标为历史对照，正式结论使用另行封存的独立 test。真实非平面数据若无可信全局 H，不据此虚构地面精度结论。

正式训练始终从模型预测 H0 出发，各轮使用自己的前一轮 H；不向中途注入 GT。已知小残差用于 TINY 诊断，必须与正式训练分开记录。冻结阶段缓存只对固定图像、增强、模型和预处理有效；联合训练禁止缓存 MVT 输出、VGG 输出、H0 或 DeDoDe 输出。冻结 DINO 输出仅在输入也固定时可以缓存；inference-mode 张量须转换成可供下游 autograd 保存的普通张量。

## 2. 参数组与阶段

| profile | 可训练参数 | DINO / Stage1 head 参数 | 预算 optimizer steps |
| --- | --- | --- | ---: |
| heads | adapters + 四尺度 refinement decoders | 均冻结 | 10,000 |
| decoder_finetune | heads + DeDoDe | 均冻结 | 从共同 heads checkpoint 续训 10,000 |
| vgg_finetune | decoder_finetune + VGG | 均冻结 | 同起点、同预算 10,000 |
| mvt_finetune | decoder_finetune + MVT | 均冻结 | 同起点、同预算 10,000 |
| joint | decoder_finetune + VGG + MVT | 均冻结 | 同起点、同预算 10,000 |

这四条续训是实验分支，并非依次训练四遍。目标能力是 joint 可用；实际推荐配置由验证集决定。Stage1 head 参数不训练时仍可对输入 MVT 特征求导，joint 下 H0 不 detach。VGG BN 运行统计固定 eval；VGG 解冻时卷积和 BN affine 参数可训练。DINO 维持 eval/no_grad。

AdamW：新增层 lr=1e-4、DeDoDe=1e-5、VGG=5e-6、MVT=1e-6；weight_decay=1e-4。B=1、累积4个有效 microbatch，有效 batch=4；loss 按实际有效累积数归一化，全无效窗口跳过 optimizer/scheduler step。gradient norm clip=1；每阶段5% warmup + cosine 至初始 lr 的0.1；每500 optimizer steps 验证。新解冻分支重建参数组，四条分支均采用相同 optimizer 重启约定，不能有的继承动量有的重置。阶段结束再决定是否延长预算，所有对照同步。

BF16 autocast 用于特征/CNN，几何、线性求解与主要 loss reduction 用 FP32。硬件不支持 BF16 时先使用 FP32 smoke，再独立验证替代精度配置。DDP、FP16 和大 batch 不是首个可运行版本的前置要求。

## 3. 默认损失

目标角点由可信 H_gt 投影得到。u 为全局更新序号，共8轮，单位为目标784输入像素：

$$
\begin{aligned}
e_{b,u}&=\frac{1}{8}\sum_{l=1}^{4}\sum_{a\in\{x,y\}}
\left|Q_{b,u,l,a}^{\mathrm{prop,px}}-Q_{b,l,a}^{\mathrm{gt,px}}\right|,\\
\mathcal L_{\mathrm{base}}&=\frac{1}{B_v}\sum_{b=1}^{B_v}\frac{1}{8}\sum_{u=1}^{8}e_{b,u}.
\end{aligned}
$$

使用 guard 前 proposal 角点，非法 DLT 不会消除该直接监督路径。Stage1 无效样本不参与精修 loss，评估仍计失败；有限但拒绝的 proposal 不应从主 loss 丢弃。非有限 proposal 是数值错误，隔离并记录，不把 NaN 替换为零误差。架构消融有 U 轮时，对 U 轮归一化为等权平均。

默认 H0 辅助监督权重0：八轮损失已能通过 H0 和特征分支训练 MVT。保留 H0 指标监控；若明显漂移，先检查预处理、head 兼容和 MVT 学习率，再单独评估 H0 anchor，不默认叠加。其他损失详见 docs/04：FGO、overlap 和 corr 默认关闭。

FGO 条件满足时从同一选定 checkpoint 分叉2,000步，保持可训练参数组不变，所有活跃组 lr 为上述初始值的0.2（新增层2e-5），三条分支相同。继续 L1、原代码 FGO 语义、bounded-FGO 使用相同数据顺序与预算；原代码 FGO 的 batch 均值使用实际 B=1，不把梯度累积当成 B=4。

## 4. 学习性验收

TINY-S：分别验证各尺度，32个固定精确合成样本，残差处于局部可观察范围；TINY-8：32个固定样本跑八轮。最多2,000步，lr=1e-3、weight_decay=0、无 scheduler，仅训练新增层。建议诊断目标 MACE<0.1输入px；不是 loss<0.1，也不是泛化标准。未达标先分析窗口覆盖、标签、梯度、步长和 numerical guard，再继续主线。

先跑 FP32/AMP 梯度与显存 smoke，记录 step0 与至少第二个更新之后的各模块梯度。完整零初始化时上游新层第一步梯度可以为零。MVT 的 H0 分支可能第一步已有梯度，不能把两者混淆。

## 5. 评价与选择

每对图像 MACE 为四角欧氏距离的平均；报告 mean/median/P90、5×5 网格 transfer error、Pre@1/3/5、每轮误差轨迹。像素阈值主表统一为784输入px；另提供原始目标图 native px 指标，按真实 resize 矩阵换算，非正方图不能简单乘单一倍率。

对所有预注册 test IDs 记录输出，包括失败。有效对 MACE 是条件统计，必须同时报告 all-pair success@1/3/5：只有合法输出且 MACE≤阈值才成功。Pre 若采用点级精度，分母定义另列，禁止与 pair success 混名。比较时另给共同有效 ID 集上的 paired 差值；初始化失败与精修拒绝分别统计，避免筛掉难例制造均值收益。

模型选择优先看 val MACE，同时审查 P90、all-pair success 与失败率；不得以牺牲有效率换取较低条件均值。没有预设显著精度提升承诺。以母图/地理簇为 bootstrap 重采样单位，保留每对 ID，避免同母图相关样本被当作独立样本。最终选定配置与关键对照种子0/1/2报告 mean±std；样本置信区间和种子波动分开报告。

速度统一硬件、dtype、B=1，推理 warmup20次，计时至少100对，CUDA同步；报告包含 Stage1 的端到端 median/P90、各模块分解、peak allocated/reserved 显存。另计完整训练 step 含 backward 的显存/耗时。首次 profiling 后填真实预算，不预估成已测 GPU 小时。
