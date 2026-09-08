# MHINet v1 完整模型规范

损失设计已更新至v1.1：默认等权八轮四角L1，详见[损失审视](04_loss_review.md)。下述网络结构不变。

## 1. 模块分工

模型分为 A 全局初始化、B 共享特征金字塔、C 多尺度H迭代三部分。A产生H0；B产生各尺度图像对特征；C反复读取当前H和当前尺度特征，预测残差并更新同一个全局H。

$$
\begin{aligned}
X&=\operatorname{DINOv3}_{11,17}(I_A,I_B),\\
Z&=\operatorname{MVT}(X),\\
H_0&=\operatorname{Stage1HeadAndFitter}(Z),\\
\{D_A^s,D_B^s\}_{s\in\{8,4,2,1\}}&=\operatorname{DeDoDePyramid}(\operatorname{VGG}(I_A,I_B),Z),\\
H_{\mathrm{final}}&=\operatorname{MultiScaleIterator}(H_0,\{D_A^s,D_B^s\}).
\end{aligned}
$$

DINO和MVT都只提取一次共享结果；迭代过程中重新计算的是局部correlation，图像、VGG、DINO、MVT和特征金字塔不重跑。

## 2. Module A：现有Stage1完整保留

复用 FrozenDINOv3L11L17Stage1 和 Stage1HeadFromSharedMVT。784输入生成49×49网格；DINO零基blocks 11/17的各1024维特征拼接为2048，经原MVT变为1024维pair-aware特征。Stage1尾部保留概率匹配、coarse warp、coarse matchability和原加权fitter。

初始 heads 阶段冻结DINO、MVT和Stage1头以对齐原H0；联合阶段DINO和Stage1头参数仍冻结，MVT可训练，H0保留输入梯度且数值允许变化。训练策略以v1.2协议为准。Stage1内部matchability不删除；本版只是不再增加任何新的可靠性/置信度分支。

共享RGB入口沿用现有金字塔路径：bicubic resize、RGB [0,1]；DINO内部自行ImageNet归一化，VGG沿用原接口。不可再套一层LoRetta训练loader的mean/std。比较H0时以旧共享入口为reference，不拿不同预处理的独立入口要求bitwise一致。

P0定位用户当前选定的Stage1 checkpoint，成套加载MVT+locator/head并匹配预处理。官方LoRetta权重仅是明确标注的独立初始化对照，不自动替代现有训练权重；真实路径从运行时输入。

## 3. Module B：四尺度共享金字塔

### 3.1 复用VGG/DeDoDe

VGG使用原vgg19_bn features[:40]，在四次MaxPool之前取特征。DeDoDe从scale16的MVT特征开始，逐级使用VGG跳接、context和累计描述子。

$$
\begin{aligned}
D^{16}&=\Delta D^{16},\\
D^s&=\operatorname{Up}(D^{2s})+\Delta D^s,\qquad s\in\{8,4,2,1\}.
\end{aligned}
$$

这里只用公式表示descriptions的累加；每一级的ConvRefiner同时接收该级feature map与上采样context，返回delta和新context。累计描述子保持256通道，不能把adapter压缩结果送回累计链。

| 层 | 每张图的原始输入 | 输出累计描述子 |
| --- | --- | --- |
| scale16 | MVT 1024×49×49 | 256×49×49 |
| scale8 | VGG 512×98×98 + 上级context | D8：256×98×98 |
| scale4 | VGG 256×196×196 + 上级context | D4：256×196×196 |
| scale2 | VGG 128×392×392 + 上级context | D2：256×392×392 |
| scale1 | VGG 64×784×784 + 上级context | D1：256×784×784 |

修改现有decode_cumulative_pyramid的返回接口：保存D4，取消新路径的D2 early-exit，把scale1作为正式输出。旧实验中叫D1的局部几何mask不是这里的全分辨率D1。

### 3.2 四个特征adapter

各尺度使用独立Conv1×1，无bias、无归一化层、无激活，之后按通道L2 normalize，epsilon=1e-6。同尺度A/B共享adapter权重。

$$
F_X^s=\frac{W_sD_X^s}{\max(\|W_sD_X^s\|_2,10^{-6})},\qquad X\in\{A,B\}.
$$

输出通道依次64/64/32/32；adapter使用Kaiming初始化。压缩后先在网格点归一化；对目标特征双线性采样后再作一次通道归一化，保证相关性确为采样描述子的cosine相似度。接近零范数位置显式屏蔽。

## 4. 坐标、H和四角状态

所有持久H均为normalized A→B，沿用align_corners=False。像素中心坐标到normalized坐标：

$$
N(W,H)=\begin{bmatrix}
2/W&0&1/W-1\\
0&2/H&1/H-1\\
0&0&1
\end{bmatrix}.
$$

图像控制点是输入A图像的四角像素中心，顺序固定TL、TR、BL、BR；经N_A转换得到c_l。用H0把它们投到目标normalized坐标，得到绝对目标角位置Q0和角位移T0：

$$
Q_{0,l}=\pi(H_0\tilde c_l),\qquad T_{0,l}=Q_{0,l}-c_l.
$$

T是4×2的normalized角位移状态。目标角允许位于图像外，不做[-1,1]裁剪；小重叠或透视时角点在图外可能完全合法。

每尺度的特征像素H由坐标变换得到：

$$
H^{s}=N(w_s,h_s)^{-1}H\,N(w_s,h_s).
$$

v1两侧特征尺寸相同；泛化到不同A/B尺寸时左右使用各自的N_B,s与N_A,s。不能只将H平移除以s，也不能在尺度转换时把T乘2。

## 5. Module C第一步：H-guided dense local correlation

### 5.1 完整源网格与候选

对当前尺度的每个源网格位置p_i，先投影到目标特征坐标：

$$
\hat q_i=\pi(H^{s}\tilde p_i),\qquad q_{i,\delta}=\hat q_i+\delta.
$$

候选偏移按dy外层、dx内层从小到大排列，δ=(dx,dy)。r=4/4/3/2，K=(2r+1)²=81/81/49/25。所有轮次使用当前H重新生成采样位置；相关性不能在第一轮后固定。

$$
C_s(i,\delta)=\left\langle F_A^s(p_i),\operatorname{Normalize}\big(\operatorname{Bilinear}(F_B^s,q_{i,\delta})\big)\right\rangle.
$$

不做dual-softmax、argmax、soft-argmax或对应点输出；局部correlation保留整个峰形，交给CNN直接解码全局四角残差。v1采用原始cosine，不引入温度超参。

这一算子是目标原坐标中的H(p)+δ；MCNet官方代码先warp目标特征后在warped坐标中相关，强透视下两者不等价。因此这里是明确的新实现选择，不声称直接照搬MCNet算子。

### 5.2 有效mask仅表示采样合法性

M_s(i,δ)=1需同时满足：投影分母合法、候选坐标位于目标像素中心范围[0,w_s−1]×[0,h_s−1]、源与采样目标描述子范数有效。否则该correlation置0，mask置0；不把border padding当作数据。源查询位置不做GT筛选。

mask是确定性的几何/边界标记，不是可学习inlier mask，也不表达哪个平面更可靠。

## 6. Module C第二步：Residual Geometry Decoder

### 6.1 输入通道固定

为每个源网格点构造normalized源坐标P和normalized当前H-flow U：

$$
U_i=\pi(H\tilde P_i)-P_i,
\qquad X_s=\operatorname{Concat}[C_s,M_s,P_x,P_y,U_x,U_y].
$$

U在输入CNN前裁剪到[-2,2]以限制极端外推值；此操作不修改H或角点状态。分母非法位置的U置0。输入通道为2K+4，即166/166/102/54，mask转换为与输入一致的dtype。

本版不额外输入DINO token、VGG特征或另一个Transformer：它们已经通过金字塔描述子影响correlation。P和U用于让decoder知道局部证据所在位置与当前全局变换。

### 6.2 更新器逐层结构

每尺度一个独立decoder，同尺度t=1、2共享完整权重；跨尺度不共享，因为输入K和下采样深度不同。

| 层 | 结构 | 输出 |
| --- | --- | --- |
| Stem | Conv1×1(2K+4→64,bias=False) → GroupNorm(8,64) → GELU | 64×h_s×w_s |
| DownBlock×n_s | 下述残差下采样块 | 64×13×13（784配置） |
| Spatial pooling | AdaptiveAvgPool2d(4,4) | 64×4×4 |
| Flatten | channel-first连续flatten | 1024 |
| FC1 | Linear(1024,256,bias=True) → GELU | 256 |
| FC2 | Linear(256,8,bias=True) | 8 |
| Bounded residual | reshape为4×2，逐轴b_s·tanh | 四角输入像素残差 |

DownBlock：主支Conv3×3(64→64,stride=2,padding=1,bias=False) → GN(8,64) → GELU → Conv3×3(64→64,stride=1,padding=1,bias=False) → GN(8,64)；跳支Conv1×1(64→64,stride=2,bias=False)；相加后GELU。不用BatchNorm、dropout或GRU。

| 尺度 | n_s | 空间尺寸变化 | FC2逐轴残差上界b_s，输入图像px/轮 |
| --- | --- | --- | --- |
| D8 | 3 | 98→49→25→13 | 32 |
| D4 | 4 | 196→98→49→25→13 | 16 |
| D2 | 5 | 392→196→98→49→25→13 | 6 |
| D1 | 6 | 784→392→196→98→49→25→13 | 2 |

4×4 pooling保留空间布局，FC联合读取16个分区，再输出四角；不把全图压成无位置信息的均值。全分辨率correlation先经过可训练卷积才逐步缩小，不预先将D1 correlation均值压到D8分辨率。

b_s是数值/步长设计值，不能保证真实误差每轮下降，也不能严格等同于角点处的搜索范围。局部窗口覆盖的是网格查询点；四角可能是远离重叠区的外推位置。v1先固定上述值，正式验证检查tanh饱和率与捕获范围。

FC2的weight和bias全部初始化为0，模型初始角更新严格为0，H_final与H0几何一致；四点重建可能引入浮点误差，因此以投影容差核验，不要求逐位一致。首个反传step主要更新FC2，其前层梯度可能为0；第二步及之后应出现非零梯度。其余Conv/Linear用Kaiming初始化，GN weight=1、bias=0。

### 6.3 新增参数量

每DownBlock为78,080参数；每MLP为264,456参数。

| 部件 | 参数 |
| --- | ---: |
| 4个adapters合计 | 49,152 |
| D8 decoder | 509,448 |
| D4 decoder | 587,528 |
| D2 decoder | 661,512 |
| D1 decoder | 736,520 |
| 新增总计 | 2,544,160 |

该数字由上述层定义计算，未计入复用的Stage1/VGG/DeDoDe；最终实现须用真实参数计数断言核对。

## 7. Module C第三步：四角残差更新与H重建

decoder先输出输入图像像素单位的δ_l。转换到target normalized位移，再更新T：

$$
\begin{aligned}
\delta_l&=b_s\tanh(z_l),\\
\Delta T_l&=\operatorname{diag}(2/W_B,2/H_B)\,\delta_l,\\
T_l^{\mathrm{proposal}}&=T_l^{\mathrm{current}}+\Delta T_l,\\
H^{\mathrm{proposal}}&=\operatorname{FourPointDLT}\left(\{c_l\},\{c_l+T_l^{\mathrm{proposal}}\}\right).
\end{aligned}
$$

这里W_B=H_B=784。跨尺度继续传递normalized T及H；无额外乘2、无矩阵相加、无重新设identity。

FourPointDLT采用h33=1的8×8线性系统：

$$
\begin{bmatrix}
x&y&1&0&0&0&-ux&-uy\\
0&0&0&x&y&1&-vx&-vy
\end{bmatrix}\theta=\begin{bmatrix}u\\v\end{bmatrix}.
$$

四组对应堆叠后FP32 solve_ex求解，组装3×3 H。不做加权，不加ridge，不从密集对应拟合H，不显式矩阵求逆；非退化时这是精确四点映射。为维持Stage1输出，初始H0直接取原值，不能先DLT回建H0再冒充逐位相同。

## 8. 8轮迭代与参数共享

$$
\begin{aligned}
(T^{8,0},H^{8,0})&=(T_0,H_0),\\
(T^{s,t},H^{s,t})&=\mathcal U_s(T^{s,t-1},H^{s,t-1},F_A^s,F_B^s),\quad t=1,2,\\
(T^{s/2,0},H^{s/2,0})&=(T^{s,2},H^{s,2}),\quad s=8,4,2,\\
H_{\mathrm{final}}&=H^{1,2}.
\end{aligned}
$$

这一迭代没有独立隐状态，历史由当前H/T携带。每轮使用同尺度固定F_A/F_B和最新H，重新采样、重新生成correlation、调用共享decoder。

## 9. 数值合法性与失败行为

Stage1 fit_succeeded=False时，该样本标记初始化失败，不继续将identity占位当作有效估计。有效Stage1样本进入精修。

每轮先检查至少16个源query拥有一个有效候选；不足则保持原H/T并标记no_valid_support。先检查proposal及系统finite、8×8系统条件数<1e6，仅对通过预筛的子集进行可微solve_ex；求解后再次检查成功状态，且normalized源图像内9×9检查网格的投影分母同号、绝对值>1e-4。后两阈值为v1数值保护起点，需在P0检查是否误拒合法样本。

不因目标四角越界而拒绝，也不做所谓地面/非地面选择。proposal不合法则本轮保持前一H/T，记录reason；下一轮仍按固定流程执行。v1不加入学习gate、线搜索或动态停止；这使8轮计算协议明确。

奇异系统与非法除法必须在进入可微运算前安全隔离，不能先算非法值再用where隐藏；详见docs/05。

训练保留guard之前的proposal角点，用独立角点loss给被拒更新提供梯度，避免所有拒绝样本都变成零梯度。全无有效支持样本只训练合法的proposal角点路径，可选辅助几何loss启用时按mask处理；Stage1失败样本在本版不优化精修，失败率照常计入评估。

## 10. 计算和显存实现

所有尺度使用dense源查询，但仅有局部候选，复杂度约为各尺度N_s·K_s·C_s，不建立N_s²的all-pairs体。D8/D4/D2/D1分别为9,604/38,416/153,664/614,656个query。

sampling+normalize+dot按1024个query分块，最后按原网格顺序拼成correlation；mask同序。训练对整个分块相关性计算使用activation checkpoint，避免保留全部N×K×C采样激活；不使用detach来省显存。GroupNorm依赖空间统计，因此decoder不能简单按空间块独立执行后拼接。

冻结金字塔时用no_grad生成原描述子，再离开no_grad运行可训练adapter。原始双图D1描述子BF16约600.25MiB；压缩为32维后约75.03MiB。可顺序解码并在后续不再需要时释放缓存，但训练图引用未释放前不能把理论张量大小当作真实显存节省。

主线B=1、梯度累积；AMP仅用于特征/CNN，dot accumulation和几何建议FP32，C存储可转为BF16供decoder使用。最终训练显存与时延须profiling，不在设计阶段承诺实时或某张卡一定可训练。

## 11. 原项目接入边界

复用原Stage1HeadFromSharedMVT、VGG/DeDoDe权重迁移和geometry约定；新增feature pyramid出口、4个adapter、local correlation、4个decoder和四角updater。旧matcher的全局coarse matching、LoFTR fine、RRU、DaD、LoMa Transformer与assignment head不进入本模型forward。

新模型由nn.Module持有一份共享encoder。禁止复制两个DINO/MVT实例，禁止沿用旧matcher强制全体eval的train()覆盖而使DeDoDe解冻无效。输入B=1时包装为现有2图入口，未来B>1需另改共享encoder和batch测试。
