# MHINet v1 训练与验证协议

## 1. 训练目标和数据

当前任务是全局H精修，监督为可信A→B homography。优先复用原单图H合成数据及其明确H标签；本轮不要求地面分割、Planar标签或多平面训练。真实配准数据只有在H标签可信时才用于H监督；模型现在不作“优先选择地面”的能力承诺。

沿用按parent image与地理区域划分的train/validation/test，防止同母图的合成对跨split。旧test已被多次诊断，保留历史对照；需要新正式主结论时另封存地理独立测试集。

正式训练从模型真实预测H0开始，每轮读取上轮自己的H，和推理一致；不在后续尺度注入GT H。已知H扰动只用于几何单元测试及早期可学习性诊断，结果与正式predicted-H训练分开记录。

冻结Stage1后可以缓存H0和特征，但只对完全固定图像、固定预处理和固定模型成立。换数据增强、checkpoint或训练DeDoDe后旧缓存必须失效。路径与hash纳入缓存key。

## 2. 当前主损失：等权八轮四角L1（v1.1修订）

损失审视后，默认取消v1.0的0.5四角/0.5网格SmoothL1与gamma=0.8。当前先使用等权八轮proposal四角L1，详细理由、MCNet原式及可选辅助项见[损失审视v1.1](04_loss_review.md)。

共8次更新，每次使用guard之前的proposal角位置，与可信H_gt投影得到的角位置比较，统一到目标784输入像素。每样本、每轮先对4角×2坐标取平均：

$$
e_{b,u}=rac{1}{8}sum_{l=1}^{4}sum_{ain{x,y}}left|Q_{b,u,l,a}^{mathrm{prop,px}}-Q_{b,l,a}^{mathrm{gt,px}}
ight|,qquad
mathcal L_{mathrm{base}}=rac{1}{B_v}sum_{b=1}^{B_v}rac{1}{8}sum_{u=1}^{8}e_{b,u}.
$$

每个尺度使用同一输入像素单位，不按feature stride再加权。Stage1失败样本不用于精修训练，评估仍计失败；batch没有有效样本则跳过optimizer step。proposal角点loss保留在几何guard之前，使被拒绝的更新仍有直接监督。NaN不是零误差，必须隔离记录。

默认没有网格loss、额外offset loss、confidence/Planar loss、photometric loss或FGO。5×5网格transfer error继续作为评价指标。是否用gamma=0.8、是否用SmoothL1，作为独立对照。

精确标签且进入亚像素收敛区后，可从同一checkpoint单独开展2,000步精修：继续L1对照、MCNet原代码FGO语义、bounded-FGO仅D2/D1。保持相同可训练模块和学习率2e-5；bounded-FGO的最大标量梯度倍率5，前200步线性升权。该步骤不与解冻DeDoDe同时首次引入。

重叠网格辅助loss仅在内部几何指标暴露缺口时试验，lambda=0.1；局部correlation辅助loss仅在窗内相关性不可学习时试验，lambda=0.02。二者默认权重0，具体mask、梯度和采样规定见损失审视文档。

## 3. 梯度路径

H0和共享context由冻结Stage1生成并detach。初版DINO/MVT/VGG/Stage1/DeDoDe全冻结，但adapter与四尺度decoder可训练。

在8轮更新内部不detach H/T/correlation；后期尺度损失可通过可微采样和四角DLT回到前期更新。允许checkpoint重算中间激活，不能以detach替代。guard是确定性离散数值选择，其被拒分支主要由proposal corner loss学习。

逐级解冻DeDoDe时，冻结encoder输出可在no_grad中产生；DeDoDe和其后的adapter必须位于no_grad之外。Stage1仍取同一冻结context，解冻DeDoDe不会改变H0。VGG的BatchNorm保持eval，防止运行统计漂移。

## 4. 训练阶段

| 阶段 | 目标 | 更新参数 | 固定起点 |
| --- | --- | --- | --- |
| T0 | 几何/接口/初始no-op、显存检查 | 无 | 16个固定合成例子与少量真实输入 |
| T1a | 单尺度head可学习性 | 一个adapter和一个decoder | 已知小残差，32个固定样本，最多2k步 |
| T1b | 8轮完整链路tiny-overfit | 全部新模块 | 32个固定train样本，最多2k步 |
| T2 | 真实H0下全模型head训练 | 全部新模块，2,544,160参数 | 首轮10k步 |
| T3 | 提升特征对H任务的适配 | 新模块+DeDoDe decoder | 额外20k步；验证无收益则保留T2 |

T1a可先在D8进行，也需分别验证D4/D2/D1；固定样本残差要处在当前尺度可观测搜索范围内。tiny-overfit目标是角点平均欧氏误差<0.1输入px，不能用SmoothL1数值<0.1代替。若0.1px超过既有H标签精度，仅在精确合成诊断使用该阈值。

T2/T3使用AdamW，new模块lr=1e-4，DeDoDe解冻时lr=1e-5，weight_decay=1e-4；B=1，梯度累积4次，有效batch=4；所有loss在累积时除以4。global gradient norm clip=1，5%warmup+cosine，最小lr为初始lr的0.1。BF16 autocast，geometry FP32。

T1使用lr=1e-3、weight_decay=0、无scheduler；保存零初始化step与第2步之后的梯度记录。T2/T3每500 optimizer steps验证一次，以val平均角点误差选best checkpoint，记录P90和失败率；步数按optimizer step，不按microbatch数。正式预算需根据T0显存与耗时回填，不承诺固定耗时。

## 5. 基础验证指标

- 每对图像MACE：四个控制角的欧氏距离均值；跨图像对报告mean/median/P90。
- 5×5网格transfer error及Pre@1/3/5，均明确784输入px；另报告native px。
- H0、全部8轮及各尺度末H的误差轨迹；记录改善/持平/恶化比例。
- 每轮角更新的均值、最大值、tanh饱和比例、有效query数、proposal拒绝原因和head梯度范数。
- Stage1失败率、精修拒绝率、最终有效输出率。pair success@τ定义为最终有效且MACE≤τ；失败不可静默剔除。
- latency的median/P90、峰值allocated/reserved显存，分解Stage1/金字塔/各尺度精修，同时给端到端总计。

输入为精确H数据时MACE是主要评价；真实非平面数据没有可信全局H时，不强行用此协议证明配准优劣。

## 6. 当前阶段必要实验

| ID | 结构 | 回答的问题 |
| --- | --- | --- |
| M00 | Stage1 H0 | 初始化基线 |
| M01 | 仅D8、2轮 | 粗尺度精修是否可学 |
| M02 | D8/D4、各2轮 | 增加中尺度是否改善 |
| M03 | D8/D4/D2、各2轮 | D2的收益 |
| M04 | 完整v1四尺度、各2轮 | D1是否进一步降低误差 |
| M05 | 四尺度各1轮 | 8轮相对4轮收益是否合理 |
| M06 | 同尺度第2轮固定correlation，其他相同 | 每轮H-guided重采样是否真正有效 |
| M07 | 完整v1去掉H-flow输入，保留H-guided采样 | H-flow作为decoder输入的收益 |
| M08 | T2冻结DeDoDe vs T3解冻DeDoDe | 特征适配是否必要 |

所有比较保持相同训练数据、预处理、Stage1和种子；用相同ordered IDs做paired bootstrap。架构选择只看validation。最终完整v1至少3个种子；test一次性冻结评测。

继续条件：M04应在独立validation上优于M00，并排除no-move；D1是否保留为默认生产配置，依据M04相对M03的误差收益与额外成本决定。若第2轮不改善，先检查重采样、梯度、步长和窗口覆盖，不立即添加GRU、confidence或Planar模块。

## 7. 当前能力边界

只要H0严重错误到有效对应落在D8窗口之外，后续精修没有保证找回；需报告in-window recall和错误分组。本版没有global rescue、地面选择或显式outlier拒绝机制。网络可能通过训练形成隐式鲁棒性，但这不是已验证结论。

## 8. 损失版本与消融

当前使用loss_revision=1.1、profile=sequence_corner_l1。L00–L08按[损失审视](04_loss_review.md)单独运行；原结构消融M00–M08不同时混入未经验证的新loss。网络尚未训练，本轮通过的是公式与数值函数检查。
