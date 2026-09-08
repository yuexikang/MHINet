# MHINet 损失审视与修订 v1.1

2026-09-07。模型结构保持v1不变；本文件替代v1.0默认的“0.5四角SmoothL1 + 0.5网格SmoothL1、gamma=0.8”。当前推荐基础训练使用等权八轮四角L1；FGO和辅助项单独实验，不能把未验证的组合直接当作最终最优配方。

## 1. MCNet 的论文与代码分别做了什么

已重新阅读用户PDF第5页（印刷页25936）的3.3节、公式(4)(5)，并核对固定版本homo_utils.py与train.py。

论文对每轮累计四角位移T_u监督，而不是直接监督3×3 H元素。基本形式为：

$$
\mathcal L_{\mathrm{MCNet}}=\sum_{u=1}^{KQ}\left[e_u+f(e_u)\right],\qquad
f(e)=\begin{cases}-\dfrac{1}{e+\epsilon},&e<\alpha,\\0,&e\ge\alpha.\end{cases}
$$

正文把e写成L1范数；官方代码具体使用对batch、4角、2坐标一起取均值的绝对误差。默认alpha即speed_threshold=1，epsilon=0.1。代码所有轮等权相加，无gamma衰减。MACE另由最后一轮的角点欧氏误差计算，不等于上述分量L1。

FGO激活区间内，单个标量e的梯度倍率为：

$$
\frac{d(e+f(e))}{de}=1+\frac{1}{(e+\epsilon)^2}.
$$

epsilon=0.1时，其接近零误差处的倍率趋于101；在误差完全为零时，实际abs采用的零点次梯度可以是0，不意味着完美样本仍产生非零梯度。

代码存在一个需明确的实现语义：sp_flag在进入序列时初始化为0，任一轮batch均值低于threshold后置1，后续轮即使误差又超过threshold也不复位。它仅在本次loss调用内持续，不跨optimizer step。按论文每轮独立门限实现与按官方代码实现，可能产生不同梯度；复现必须注明版本。

论文：[MCNet PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_MCNet_Rethinking_the_Core_Ingredients_for_Accurate_and_Efficient_Homography_CVPR_2024_paper.pdf)。代码：[loss_function](https://github.com/zjuzhk/MCNet/blob/cc03479689b3cf40f0c384954f338b434765c155/homo_utils.py#L106)、[默认配置](https://github.com/zjuzhk/MCNet/blob/cc03479689b3cf40f0c384954f338b434765c155/train.py#L119)。

## 2. 对v1.0损失的重新判断

### 2.1 主任务应优先沿用四角序列监督

当前decoder直接输出四角残差，累计后的四角位置与H是一一对应的（非退化条件下）。主损失应直接约束每次更新后的角位置，提供最短监督路径；不必新增中间匹配head或H矩阵元素loss。

### 2.2 SmoothL1没有错误，但不宜无证据设为精度主线

beta=1输入px时，单个残差r在|r|<1区间的梯度为r；例如0.01px残差的梯度幅值是L1的约1%。这与MCNet在收敛区提高小误差权重的思路相反。它不证明SmoothL1必然更差，但说明原先选择需要实验而非默认。

精确合成H主线改为L1。若真实H存在标注噪声，SmoothL1/Huber仍是独立稳健性对照；不能为了追求细精度而对不准确标签强行做FGO。

### 2.3 网格损失与四角损失信息重叠

同一个精确H_gt决定四角和内部网格，二者不是两个独立监督来源。网格项改变的是空间误差分布与优化权重，可帮助特定透视/小重叠情况，但没有必要从一开始等权0.5叠加。

v1.0的corner使用guard前proposal，而grid使用guard后保留H；更新被拒时，grid并不能直接纠正被拒proposal。保留proposal角点主损失是合理的；grid改为可选、只作用于数值合法proposal的辅助项。

### 2.4 暂时去掉gamma衰减

四尺度更新器跨尺度不共享，过分弱化早期轮次会使粗尺度训练信号变小。gamma=0.8、8轮归一化时，前两轮合计权重约11.34%，末两轮约43.26%；这不是必须遵循的比例。

默认各轮权重1/8，先与MCNet的等权序列思路对齐。是否使用gamma=0.8作为单独消融；不同时改变loss、迭代次数和backbone。

## 3. 新默认：等权八轮proposal四角L1

设b为有效训练图像对，u为8次更新编号，Q^prop为guard前proposal目标角位置，Q^gt由可信H_gt投影得到；全部转换到目标784输入像素。

$$
e_{b,u}=\frac{1}{8}\sum_{l=1}^{4}\sum_{a\in\{x,y\}}
\left|Q_{b,u,l,a}^{\mathrm{prop,px}}-Q_{b,l,a}^{\mathrm{gt,px}}\right|,
\qquad
\boxed{\mathcal L_{\mathrm{base}}=\frac{1}{B_v}\sum_{b=1}^{B_v}\frac{1}{8}\sum_{u=1}^{8}e_{b,u}}.
$$

内层1/8为4角×2分量平均，外层1/8为8轮平均。先每个样本每轮求e，再聚合；这样未来加入非线性FGO时，不会把batch组成变成样本的权重开关。基础L1下，固定完整角标签时这些平均可交换。

所有尺度损失都用同一输入像素单位。不能在D8用feature pixels、D1用input pixels后直接相加，也不再乘1/s或s²给细尺度增权。

保留迭代间BPTT；heads阶段H0冻结；MVT联合微调阶段H0保留梯度，后续H/T始终不detach（训练v1.2）。Stage1无效样本不用于精修loss，但在评价计失败。若有效样本数为0，跳过optimizer step并记录原因；不能用clamp分母静默产生零loss假装训练正常。

proposal角点由有界decoder产生，可绕开非法DLT的数值梯度；出现NaN feature/proposal是数值错误，按样本隔离并记录，不把NaN当作0误差。

## 4. 第二阶段候选：梯度受控的FGO精修

### 4.1 为什么不直接默认照搬

原FGO有阈值处的函数跳变，其代码门限又依赖batch均值和持续flag；101倍近零梯度可能在标签噪声、AMP和梯度裁剪下产生不利影响。负loss本身不是数学错误，关键是梯度尺度、门限语义和可比较的日志。

建议先保留精确原代码语义作复现对照；另外定义一个明确命名的bounded-FGO实验。它是本方案的工程变体，不是MCNet原式，也不作为已经成立的新颖性贡献。

### 4.2 连续、非负且倍率有界的变体

设e0=1输入px，t=e/e0，tau=1，epsilon=0.1，lambda=0.04，均对每样本每轮独立计算：

$$
\begin{aligned}
\phi(t)&=t+\lambda\left(\frac{1}{\epsilon}-\frac{1}{\min(t,\tau)+\epsilon}\right),\\
\phi'(t)&=\begin{cases}
1+\dfrac{\lambda}{(t+\epsilon)^2},&0<t<\tau,\\
1,&t>\tau.
\end{cases}
\end{aligned}
$$

在t=tau处函数连续但导数有折点；实现规定增强分支只在t<tau取梯度，t≥tau使用常数饱和值。该loss在0处为0、非负且单调；增加的常数不用于跨配置比较loss数值。

最大梯度倍率为1+lambda/epsilon²=5。该上限是相对于标量e的局部倍率，不是整个网络梯度范数上限；跨轮梯度、DLT与tanh链式导数仍会影响实际梯度。

梯度倍率不等于精度收益：频繁gradient clipping可能抵消部分放大，Adam对统一梯度缩放也有一定适应性。需要验证的是样本/轮次相对权重改变后的泛化效果，而不是单看梯度变大。

只对D2/D1的后四轮启用候选精修，粗尺度保持L1：

$$
\mathcal L_{\mathrm{fine}}=\frac{1}{8B_v}\sum_b\left[
\sum_{u=1}^{4}e_{b,u}+\sum_{u=5}^{8}e_0\phi(e_{b,u}/e_0)\right].
$$

L1基线训练结束后，从同一个checkpoint分叉比较：继续L1、原FGO语义、bounded-FGO。冻结范围保持不变，新增模块lr=2e-5，预算2,000 optimizer steps；bounded-FGO的lambda在前200步从0线性升到0.04。先在独立validation确认细尺度已有足够e<1px样本和可信GT，再运行此精修。若还未进入精度收敛区，优先修复基础学习问题。

不要把开启FGO和解冻DeDoDe放在同一次未拆分实验里；否则难以判断提升来自哪里。FGO阈值统一是784输入px，不能把MCNet 128输入下的参数视为跨分辨率不变定律。

## 5. 额外损失A：重叠区域的网格几何监督（默认关闭）

触发条件：独立验证显示四角误差下降，但GT重叠区域的内部transfer error无改善或恶化；需要检查透视、外推和标签后，再考虑该项。

用GT H定义可见重叠区域，取7×7均匀源网格中的合法GT重叠点，最多49点。mask只取自GT，不随预测H裁剪，防止网络把难点投出图后逃避loss。

$$
\mathcal L_{\mathrm{overlap}}=\operatorname{Mean}_{b,u,p\in\mathcal G_b}
\rho_{\beta=1}\left(\pi(H_{b,u}^{\mathrm{prop}}\tilde p)^{\mathrm{px}}-
\pi(H_b^{\mathrm{gt}}\tilde p)^{\mathrm{px}}\right).
$$

仅使用每尺度末的u∈{2,4,6,8}，每样本每尺度先对有效网格点和坐标分量平均，再均匀平均。数值非法proposal不参与该投影loss，由四角主loss纠正；非法率单独报告，不能作为丢弃难例的成功信号。

候选lambda_overlap=0.1，主损失仍为八轮L1。它是几何权重补充，不引入新的标签来源、学习mask或Planar判断。GT很小重叠导致网格空集时该辅助项跳过该样本，主损失照常。

## 6. 额外损失B：局部相关性辅助监督（默认关闭）

触发条件：无gate的完整模型仍学习困难，诊断发现GT在搜索窗内，但相关性峰长期不朝GT靠近；尤其在训练adapter或解冻DeDoDe后。若GT在窗口外，该辅助loss无法补救捕获范围。

每尺度只在第一轮用至多1024个均匀query构造辅助相关性；以stop-gradient当前H确定窗口中心，采用可信H_gt给出的subpixel位置，向邻近4个候选分配双线性软标签y。只保留GT可见、在窗口内且软标签支撑候选均有效的query。

$$
\mathcal L_{\mathrm{corr}}=-\operatorname{Mean}_{b,s,i}\sum_{\delta}y_{b,s,i,\delta}
\log\operatorname{softmax}_{\delta}\left(C_{b,s,i,\delta}^{\mathrm{aux}}/\tau_c\right),\qquad\tau_c=0.1.
$$

辅助分支的中心和软标签都detach，梯度进入可训练的adapter/DeDoDe，以及已解冻的VGG/MVT特征分支；主分支correlation仍保留对H的梯度。实现需用H.detach()重新计算这一小组辅助correlation，而不是把现有主C全部detach。该项不新增预测head，不改变推理输出；候选权重不是Planar mask。

先每图每尺度平均再跨尺度平均，避免D1点多主导训练。候选lambda_corr=0.02，单独相对L1基线验证；不默认与overlap/FGO三项一起叠加。它直接帮助局部表示，不证明全局H一定改善，必须看最终H指标。

## 7. 暂不添加的损失

| 损失 | 当前不采用的理由 |
| --- | --- |
| 3×3 H元素MSE | H有尺度等价性，各项量纲/几何敏感性不同；已有四角参数化 |
| 额外残差ΔT真值loss | 若目标为Q_gt−Q_prev，与更新后角点误差代数重合；detach目标只改变梯度路径，非新增信息 |
| 更新量趋零正则 | 可能鼓励旧项目出现过的no-move倾向 |
| 每轮必须单调改善loss | 有限窗口、噪声和多步优化不保证每一步GT误差单调；先记录再诊断 |
| 图像photometric loss | 跨时相/跨模态外观差异与H误差混杂，当前已有明确几何GT |
| cycle consistency | 需要额外反向推理且错误变换也可互逆，暂不作为基础监督 |
| confidence/Planar/semantic loss | 超出当前仅多尺度H迭代范围 |

## 8. 损失消融与接受条件

以下L00–L08为历史配方编号；实际执行优先级及当前实验ID统一见docs/07。旧混合配方不要求首批运行。

| ID | 配方 | 目的 |
| --- | --- | --- |
| L00 | v1.0原配方：SmoothL1 corner/grid各0.5，gamma0.8 | 保留上一版对照 |
| L01 | 仅corner SmoothL1、gamma0.8 | 去掉grid的影响 |
| L02 | 仅corner L1、gamma0.8 | L1与SmoothL1的影响 |
| L03 | 仅corner L1、等权8轮 | 新默认，单独检验序列权重 |
| L04 | L03同checkpoint继续L1精修 | 控制额外训练步数与学习率 |
| L05 | 同checkpoint，MCNet原代码FGO语义精修 | 精确记录batch均值/持续flag，保持相同预算 |
| L06 | 同checkpoint，bounded-FGO仅后4轮 | 控制强梯度，测试亚像素收益 |
| L07 | L03 + overlap，其他同L03 | 内部几何权重是否必要 |
| L08 | L03 + auxiliary corr，其他同L03 | 局部表示是否是瓶颈 |

L05对MHINet使用其8轮输出，但只称“MCNet loss语义移植”，不是完整MCNet结果。梯度累积4次不等于把4个microbatch先求均值再做非线性FGO：官方语义受实际microbatch大小影响，必须记录B=1。

使用固定validation IDs与配对bootstrap，主要看最终MACE、Pre@1、P90、失败率、梯度裁剪触发率；同时记录8轮误差、有效窗口覆盖、FGO激活比例及梯度倍率分布。不能依据总loss变小决定哪种配方好，因为不同loss的数值尺度/常数不同。

若精修改善仅见于训练或tiny-overfit，不能宣称FGO提高泛化。噪声标签组中若P90/失败率恶化，即使均值改善也要单独报告。组合多个有效辅助项只在各自增益独立成立之后做验证。

## 9. 本次结论

**先用等权八轮proposal四角L1跑通完整模型；FGO是后期精度优化候选；overlap与correlation辅助项均以诊断为触发，默认权重0。** 不需要新增Planar模块，也不需要为显得复杂而堆叠损失。

本次完成公式/源码审视与数值函数核验，不包含网络训练效果验证。新增配置和训练入口应读取v1.1损失设定，不继续把v1.0混合loss当作当前默认。
