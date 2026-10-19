## 一、先锁定改造范围

我核对到了 10 月 19 日的新提交：`main` 当前为 `a1d32e3`，最近新增了 GoogleEarth 地面同名点标注工具；模型的主要半密集匹配代码仍位于 `mhinet/downstream/`。

这次执行方案只包含两项模型改造：

1. D16 Flow0 替代 H0，并在 D8 全局粗匹配时提供几何引导。 不另建 D8 Flow 更新网络。
2. 改造 QRRU：增加 Feature Discrepancy 和局部 DGFO。 保留现有 4×4 特征窗口、2×2 控制点和四轮循环更新机制。

中间的 D2 仍然承担 4×4 子网格匹配。由于它目前依赖 H0 生成目标子网格，因此还需要完成一个必要的接口适配：使用 D8 父匹配直接展开子网格，而不是继续调用 H0。这个适配不属于新增 Flow Refinement 模块。

整个新流程为：

DINOv3 + MVT → F_MVT

Flow0 Head

D16 坐标映射 + Matchability

CGMDP

D8 / D4 / D2 描述子

Flow-guided D8 Global Matching

原特征相关性 + Flow 几何软先验

D2 Fine Matching

根据 D8 父匹配展开局部 4×4 子网格

Discrepancy-guided QRRU

动态 Warp → 差异特征 → DGFO → 2×2 控制点更新

最终半密集匹配点对及置信度

最关键的设计原则是：Flow0 负责改善 D8 的候选排序，QRRU 负责改善已经形成的局部对应关系。两项改动独立实现、独立训练、独立消融。

这样才能判断提升来自粗匹配还是末端精修，而不是一次改动太多，最终无法分析原因。

## 二、改造一：D16 Flow0 Head 与 D8 全局匹配融合

### 2.1 Flow0 Head：复用已有输出，不重建 Backbone

当前 [`feature_provider.py`](https://github.com/yuexikang/MHINet/blob/main/mhinet/models/feature_provider.py?utm_source=chatgpt.com) 已经能够获得：

```
coarse_warp          # [B, 2, 32, 32]，512 输入coarse_matchability  # [B, 1, 32, 32]
```

`coarse_warp` 是归一化目标坐标。直接减去参考网格，就得到 D16 Flow：

\\[ F\_{16}(p)=W\_{16}(p)-p \\]

我建议新接口明确区分位移、目标坐标和可匹配性：

```
{    "flow16_norm": ...,          # [B, 2, H16, W16]    "flow16_target_norm": ...,   # [B, 2, H16, W16]    "flow16_matchability": ...,  # [B, 1, H16, W16]    "pyramid": ...,              # D8 / D4 / D2}
```

其中 `norm` 统一表示 `align_corners=False` 的归一化坐标。

这个阶段移除 H0 Fitter 的前向执行，但应保留旧实现作为基线模式，不要删除旧权重加载路径。

### 2.2 我建议的 D8 Flow 引导不是局部搜索，而是软几何先验

目前 [`coarse_matcher.py`](https://github.com/yuexikang/MHINet/blob/main/mhinet/downstream/loma_reference/coarse_matcher.py?utm_source=chatgpt.com) 采用全局余弦相似度和 FP32 dual-softmax，并支持分块计算。

这个机制应完整保留。

现有分数：

\\[ S\_{ij}^{app} = \frac{D_A(i)^TD_B(j)}{\tau} \\]

加入 Flow 后，设：

\\[ \hat q_i=p_i+F_8(p_i) \\]

其中 \\(F_8\\) 由 D16 Flow 插值而来。

针对 D8 候选 \\(q_j\\) 计算：

\\[ G\_{ij} = \exp\left( -\frac{\\|q_j-\hat q_i\\|\_2^2}{2\sigma^2} \right) \\]

最终：

\\[ \boxed{ S\_{ij} = S\_{ij}^{app} + \lambda c_iG\_{ij} } \\]

这里 \\(c_i\\) 来自 D16 的匹配可靠性预测。

这一形式的设计目的如下：

| 情况                | 作用         |
| ----------------- | ---------- |
| 特征匹配很强、Flow0 也接近  | 候选获得几何奖励   |
| 特征匹配很强、Flow0 偏离较大 | 仍然可以被选中    |
| Flow0 置信度低        | 减弱甚至关闭几何奖励 |
| Flow0 无效或越界       | 回退到纯外观相似度  |

我建议第一版使用有上界的正向奖励，不要直接使用很大的负距离惩罚。

后者容易把错误 Flow0 变成硬性搜索约束，重现之前 H0 错误引导的问题。

示例：Flow0 为 D8 候选增加几何分数

几何权重 λ

4.5

空间范围 σ（D8 像素）

4.0

| 候选 | 外观分   | Flow 距离 | 融合分   |
| -- | ----- | ------- | ----- |
| A  | 9.00  | 1       | 12.49 |
| B  | 11.00 | 5       | 12.65 |
| C  | 10.50 | 12      | 10.54 |

仅演示几何加分对单个查询候选排序的影响，实际 D8 还需要经过完整 dual-softmax 和双向互选。

### 2.3 修改代码时有一个容易出错的地方

几何加分必须发生在 dual-softmax 之前。

你当前的 chunked matcher 会计算：

- 每行的 `logsumexp`；
- 每列的 `logsumexp`；
- 双向最佳候选；
- 最终 dual-softmax confidence。

因此，新增 Flow 几何分数之后，所有这些计算必须使用同一个融合后的 \\(S\_{ij}\\)。

不能只在最后输出 confidence 时额外加几何分，因为那样候选选择过程和置信度计算使用的是两个不同的分数矩阵。

建议修改范围集中在 `coarse_matcher.py`，新增可选的 `flow_prior` 参数，确保 `flow_prior=None` 时保持原数值路径。

### 2.4 D8 不再使用 H0 重叠掩膜

当前 H0 可以通过正向和逆向投影构造 A、B 两侧的重叠区域。

但单向 Flow0 没有现成的逆映射，因此不能简单照搬 `predicted_overlap_masks(H)`。

第一版建议：

- D8 两侧特征全部参与候选匹配。
- 对预测位置有效的 A 像素使用几何加分。
- 低置信度位置不使用几何加分。
- 无效 Flow 不导致整对影像匹配失败。

对于 512 输入，D8 全图有 4096 个特征位置，相关性候选数量达到约 1678 万。当前分块实现可以避免一次存储完整矩阵，但取消 H0 掩膜后，计算量和耗时仍可能增加。

因此必须保留分块计算，并记录 D8 的独立耗时。

## 三、必要的 D2 适配：去除 H0，不额外引入 Flow

这里我严格遵守你的限定：Flow0 只新增 D8 引导，不额外设计 D2 Flow Refinement。

当前 [`supervision.py`](https://github.com/yuexikang/MHINet/blob/main/mhinet/downstream/supervision.py?utm_source=chatgpt.com) 的 `fine_grids()` 使用：

\\[ q_2(p)=H_0(p)+[q_8-H_0(p_8)] \\]

删除 H0 后，改为直接利用 D8 粗匹配：

\\[ \boxed{ q_2(p)=q_8+(p-p_8) } \\]

其中 \\(p\\) 是 D8 父单元内的 D2 子网格坐标。

具体来说，A 侧仍然从一个 D8 单元展开 16 个 D2 子点，B 侧则以匹配到的 D8 目标中心为锚点展开相同结构。

后续 16×16 局部相似度、dual-softmax、互选、置信度组合均保持不变。

但这个变化存在明确的性能风险。

你此前的 H0 位置消融结果为：

| GoogleEarth               | P\@3   |
| ------------------------- | ------ |
| A：原始 H0 引导                | 33.24% |
| B：保留 D8 H0 掩膜、关闭 D2 H0 引导 | 30.73% |
| D：完全关闭 H0                 | 30.63% |

这说明关闭 D2 的 H0 Warp 之后，局部匹配本身可能退化约 2.5 个百分点。这不是新 Flow Head 一定能通过 D8 改善补回来的。

因此，后面必须设置一个单独的 No-H0 基线：

\\[ \text{No-H0 + Global D8 + Direct-D8 Fine + 原 QRRU} \\]

它才是 Flow-guided D8 的正确对照。

不能只比较新方案和原始 H0 完整模型，否则无法判断性能差异来自哪里。

# 四、改造二：将 Feature Discrepancy 和 DGFO 融入现有 QRRU

这是我认为更有研究潜力、同时需要谨慎设计的部分。

你的 [`qrru.py`](https://github.com/yuexikang/MHINet/blob/main/mhinet/downstream/qrru.py?utm_source=chatgpt.com) 当前有一个不错的特点：

局部位移不是任意的 4×4 Dense Flow，而是由四个控制点决定。

我建议保留这个结构，不要直接将 QRRU 替换为 CRFT 的 5×5 Dense Flow 网络。

### 4.1 当前 QRRU 的实际计算

512 输入时，D2 为 256×256。

QRRU 每个匹配点对：

\\[ (p_A,p_B) \\]

提取两侧局部特征。

| 张量               | 形状           |
| ---------------- | ------------ |
| `wa`             | `[N,48,4,4]` |
| `wb`             | `[N,48,4,4]` |
| 隐藏状态 `h`         | `[N,32,4,4]` |
| 2×2 控制点残差 `flow` | `[N,2,2,2]`  |

当前循环：

```
gb = gb0 + interpolate_controls(flow, off)wb = sample_uv(feature_b, gb)z = self.encoder(torch.cat((h, wa, wb), 1))g = torch.sigmoid(self.gate(torch.cat((h, z), 1)))h = (1 - g) * h + g * zout = self.head(self.down(h))flow = flow + torch.sigmoid(out[:, 2:3]) * self.radius * torch.tanh(out[:, :2])
```

这里已经具备：

- 利用当前 Flow 更新 B 的采样位置；
- 使用重新采样的特征预测下一轮位移；
- 循环进行局部坐标修正。

因此 SGT 部分不需要重新实现。

### 4.2 首先引入 Feature Discrepancy

当前 `wa` 与 `wb` 各有 48 个通道。

我不建议直接照搬 CRFT 的：

\\[ R_i=1-\operatorname{Normalize}(wa-wb) \\]

因为这种表示弱化了差异的绝对大小，而且其数值不直接对应匹配可靠性。

我建议第一版改成：

\\[ \bar W_A=\operatorname{Normalize}(wa) \\]

\\[ \bar W_B^i=\operatorname{Normalize}(wb_i) \\]

\\[ \boxed{ D_i=|\bar W_A-\bar W_B^i| } \\]

得到：

\\[ D_i\in\mathbb R^{N\times48\times4\times4} \\]

然后用一个轻量网络映射：

\\[ \boxed{ R_i=\psi(D_i) } \\]

其中：

\\[ \psi=\operatorname{Conv}\_{1\times1}(48,32) +\operatorname{GN} +\operatorname{GELU} \\]

输出：

\\[ R_i\in\mathbb R^{N\times32\times4\times4} \\]

这样 \\(R_i\\) 就成为后续差异引导分支的输入。

### 4.3 不建议直接改变原 Encoder 的输入维度

当前 Encoder 输入通道为：

\\[ 32+48+48=128 \\]

如果直接拼接 \\(D_i\\)：

\\[ [h,wa,wb,D_i] \\]

通道会变成 176。

虽然可以修改首层卷积，但会破坏已有权重的形状兼容性，也不利于无训练消融。

我更推荐采用残差注入。

保留：

\\[ z_i^{old} = \operatorname{Encoder}([h_i,wa,wb_i]) \\]

新增：

\\[ z_i^{new} = z_i^{old} + \operatorname{DiffAdapter}(R_i) \\]

其中：

\\[ \operatorname{DiffAdapter}: 32\rightarrow32 \\]

建议 `DiffAdapter` 使用零初始化的 1×1 Conv。

这样刚开始：

\\[ z_i^{new}=z_i^{old} \\]

能够严格验证新模型在初始化时是否保持旧 QRRU 的输出行为。

后续训练再让差异分支逐渐学习贡献。

## 五、DGFO 怎样接入 2×2 控制点结构？

这是整个设计最重要的一步。

### 5.1 先把 2×2 控制点变为 4×4 局部 Flow

当前：

\\[ T_i^{ctrl}\in\mathbb R^{N\times2\times2\times2} \\]

通过现有 `interpolate_controls()`：

\\[ \boxed{ T_i^{local} = \operatorname{InterpolateControls} (T_i^{ctrl}) } \\]

得到：

\\[ T_i^{local} \in\mathbb R^{N\times2\times4\times4} \\]

于是能够与差异特征：

\\[ R_i\in\mathbb R^{N\times32\times4\times4} \\]

在相同的局部空间网格上进行 Attention。

注意：这张 Flow 是相对于当前匹配点中心的局部残差，而非全图绝对位移场。其坐标单位是 D2 像素。

### 5.2 用差异特征生成 Q、K

\\[ Q_i=W_QR_i \\]

\\[ K_i=W_KR_i \\]

其中：

\\[ Q_i,K_i\in \mathbb R^{N\times32\times4\times4} \\]

我建议先使用半径为 1 的局部注意力，即 3×3 邻域。

因为当前 4×4 patch 本身很小，直接使用 5×5 会使大量邻域位置落在 patch 之外。

对于每个位置 \\(p\\)，提取最多 9 个有效邻居。

\\[ \alpha\_{pq} = \operatorname{Softmax}\_{q\in\mathcal N(p)} \left( \frac{Q_i(p)^TK_i(q)}{\sqrt{32}} \right) \\]

对应张量变化：

| 计算        | 形状            |
| --------- | ------------- |
| Query     | `[N,16,1,32]` |
| 邻域 Key    | `[N,16,32,9]` |
| Attention | `[N,16,1,9]`  |
| 邻域 Value  | `[N,16,9,2]`  |
| 输出        | `[N,16,1,2]`  |

最终恢复为：

\\[ \boxed{ P_i\in\mathbb R^{N\times2\times4\times4} } \\]

这里的 Value 是：

\\[ V_i=T_i^{local} \\]

并且应对 patch 外的邻居执行显式 Mask，而不是让零填充的伪 Flow 参与 Softmax。

### 5.3 DGFO 的输出怎样回到 QRRU？

我不建议直接用 \\(P_i\\) 替换四控制点。

原因是经过 Attention 加权后的 \\(4\times4\\) 位移场未必还能被四个控制点的双线性插值精确表示。

如果硬要从它恢复四控制点，就额外引入了拟合或降维过程，改变了现有 QRRU 的几何结构。

我建议让 DGFO 提供修正信息，而不是直接改写控制点。

计算：

\\[ \boxed{ E_i=P_i-T_i^{local} } \\]

它表示：

> 根据差异特征传播后的位移，与当前位置原始位移之间有什么不同？

然后将 \\(E_i\\) 投影到隐藏状态空间：

\\[ Z_i^{flow} = \operatorname{FlowAdapter}(E_i) \\]

其中：

\\[ \operatorname{FlowAdapter}: 2\rightarrow32 \\]

同样采用零初始化。

最终 QRRU 的特征更新变成：

\\[ \boxed{ z_i = \underbrace{\operatorname{Encoder}([h_i,wa,wb_i])}\_{\text{原 QRRU}} + \underbrace{\operatorname{DiffAdapter}(R_i)}\_{\text{显式特征差异}} + \underbrace{\operatorname{FlowAdapter}(E_i)}\_{\text{DGFO 传播信息}} } \\]

后面的 Gate、Down、Head、控制点累积更新全部保持原样。

改进 QRRU 单轮计算流程

现有控制点 Flow + 4×4 双图局部特征

SGT：根据控制点位移采样 `wb`

差异特征

R_i [N,32,4,4]

DGFO 传播

P_i - T_local

原 Encoder 输出 + 两个零初始化 Adapter

原 Gate → Hidden State → Down → Head

更新 2×2 控制点 → 下一轮采样

### 5.4 为什么必须同时设置 Feature Discrepancy 的独立分支？

因为你的 QRRU 初始状态是：

```
flow = wa.new_zeros((n, 2, 2, 2))
```

第一轮：

\\[ T_0^{ctrl}=0 \\]

因此：

\\[ T_0^{local}=0 \\]

DGFO 的 Value 全部为零。

无论 Attention 权重如何：

\\[ P_0=0 \\]

所以：

\\[ E_0=P_0-T_0^{local}=0 \\]

第一轮 DGFO 实际上没有有效的位移信息可以传播。

这不是代码错误，而是当前 QRRU 初始化和 Attention 加权平均共同决定的结果。

第一轮需要依靠原 Encoder 和 Feature Discrepancy 分支产生初始控制点更新。

到第二轮，控制点开始出现不同位移：

\\[ T_1^{local}(p_1)\neq T_1^{local}(p_2) \\]

此时 DGFO 才有可能学习在不同位置之间传播残差位移。

因此，我建议：

- 保持第一轮控制点更新机制不变。
- Feature Discrepancy 从第一轮开始参与。
- DGFO 在后续轮次发挥主要作用。
- 记录每轮 `DGFO` 输出相对输入位移的变化量和梯度。

不必刻意让第一轮 DGFO 非零，否则容易引入与当前四控制点模型无关的额外机制。

## 六、推荐的 QRRU 伪代码

以下是建议实现，不是仓库当前已有代码。

```
def forward(self, feature_a, feature_b, points_a_uv, points_b_uv):    # 原有特征投影、坐标网格构造保持不变    ...    h = zeros([N, 32, 4, 4])    flow = zeros([N, 2, 2, 2])    for t in range(self.iterations):        # 1. 2x2 控制点 -> 4x4 局部位移        local_flow = interpolate_controls(flow, off)        # [N, 4, 4, 2]        # 2. SGT：根据当前局部位移采样 B        gb = gb0 + local_flow        wb = sample_uv(feature_b, gb)        wb = wb.permute(0, 3, 1, 2)        # 3. Feature Discrepancy        a_norm = F.normalize(wa, dim=1)        b_norm = F.normalize(wb, dim=1)        discrepancy = (a_norm - b_norm).abs()        R = self.diff_encoder(discrepancy)        # [N, 32, 4, 4]        # 4. DGFO：Q、K 来自 R，V 来自 local_flow        local_flow_chw = local_flow.permute(0, 3, 1, 2)        propagated = self.dgfo(            R,            local_flow_chw,            valid_mask=...        )        propagation_residual = (            propagated - local_flow_chw        )        # 5. 保留原 Encoder        z_base = self.encoder(            torch.cat((h, wa, wb), dim=1)        )        # 6. 将两个新增分支残差注入        z = (            z_base            + self.diff_adapter(R)            + self.flow_adapter(propagation_residual)        )        # 7. 原 QRRU 更新        g = torch.sigmoid(            self.gate(torch.cat((h, z), dim=1))        )        h = (1 - g) * h + g * z        out = self.head(self.down(h))        flow = (            flow            + torch.sigmoid(out[:, 2:3])            * self.radius            * torch.tanh(out[:, :2])        )    return ...
```

这里 `valid_mask` 必须由真实采样位置及 patch 邻域有效性构造，不能用 GT 重叠区域在推理时提供隐藏信息。

另外，`flow_adapter` 和 `diff_adapter` 使用零初始化可以保留旧模型的初始行为，但新分支的部分上游参数第一步可能暂时没有梯度，应检查几个优化步之后的梯度是否正常。

## 七、训练方式必须调整，否则 QRRU 可能学不到预期功能

这一点比再加一层 Attention 更值得重视。

你当前 `SemidenseMatcher.training_losses()` 中，QRRU 的训练中心是通过 GT 几何生成的：

```
truth, _ = project_normalized(grid[fi[take]], H)noise = torch.randn_like(pa) * c.noise_stdpb = (    to_uv(truth, hf)    + noise.clamp(-c.noise_bound, c.noise_bound))
```

默认：

- `noise_std = 0.8`
- `noise_bound = 2.4`
- 噪声和 QRRU 控制点修正都使用 D2 像素单位。

也就是说，QRRU 当前训练的是：

> 已知一个接近 GT 的对应位置，如何修正随机扰动？

但是实际推理中，QRRU 接收的是 D2 匹配结果。

两者的误差分布不一定相同。

D2 匹配错误可能来自错误父对应、重复建筑、局部视角差异，而不仅仅是小幅随机噪声。

因此，建议把 QRRU 的训练分成两种输入来源。

| 训练输入           | 用途          |
| -------------- | ----------- |
| GT 正确位置 + 随机扰动 | 保持原有局部精修能力  |
| 当前模型预测的 D2 匹配点 | 学习真实推理产生的误差 |

对于第二类输入，只使用可通过 GT 确定真实对应且处于合理修正范围内的点进行位移监督。严重错误的粗匹配不应强迫 QRRU 在很小的局部窗口中修正成功。

在开发阶段，可以先将两类输入按 1:1 混合做受控实验，但比例不是已经验证的最佳值。

还应按输入误差分组统计：

| D2 初始误差 | 重点观察           |
| ------- | -------------- |
| 0–1 px  | 是否破坏本来正确的匹配    |
| 1–3 px  | 是否稳定降低位置误差     |
| 3–6 px  | 是否能够恢复较难样本     |
| ＞6 px   | 是否出现错误修正或不可靠预测 |

由于当前每轮控制点增量受 `radius=1.5` 调制，总共四轮，单个控制分量的理论累积修正幅度不超过 6 个 D2 像素；这不是所有局部变形或目标中心都能成功修正 6 px 的保证。

我建议把“QRRU 修好多少点”与“QRRU 破坏多少原本正确的点”作为重点诊断。

# 八、正式执行顺序

我建议不要直接让 Codex 同时重构 Flow Head、D8 和 QRRU。

分成五个可独立验收的阶段更合适。

### 1

D16 Flow0 接口改造

保留共享网络权重和 `coarse_warp` 预测能力，取消当前新分支的 H0 拟合输出依赖。实现 Flow0 的归一化坐标转换、置信度与有效性检查。

验收：512/784 形状正确，坐标方向正确，零 Flow 为恒等映射，预测和 GT 的 EPE 能独立评价。

### 2

D8 Flow 软引导 + D2 去 H0

为 D8 分块 dual-softmax 添加可选 Flow 几何奖励；D2 改为根据 D8 父对应直接构造目标子网格；去除推理中的 H0 有效性门控。

验收：λ=0 时与 No-H0 基线数值一致；完整矩阵与分块实现结果一致；不使用 GT 作为预测掩膜。

### 3

QRRU Feature Discrepancy

新增差异编码器及零初始化 Adapter，暂不加入 DGFO。其余控制点、循环状态和 Head 保持不变。

验收：迁移旧权重后，未训练模型与原 QRRU 输出一致；训练后单独评价差异分支收益。

### 4

QRRU DGFO

将 2×2 控制点插值为 4×4 Flow，在差异特征上执行 3×3 局部 Attention，将传播残差注入原隐藏状态更新。

验收：Attention 权重归一化正确，边界 Mask 正确，梯度可回传；新增分支不能导致无效采样、NaN 或大幅显存增长。

### 5

受控训练与全量评价

Flow Head 和 QRRU 分开训练，固定共享特征及匹配器，先验证单模块收益，再做联合微调实验。

验收：相同数据清单、相同评估口径、相同输出匹配上限，并保存各阶段独立结果。

### 各阶段推荐的训练策略

| 阶段           | 可训练                              | 冻结                       |
| ------------ | -------------------------------- | ------------------------ |
| Flow0 直接测试   | 无                                | 全部                       |
| Flow Head 微调 | Flow Head                        | DINO、MVT、CGMDP、匹配器、QRRU  |
| QRRU 差异模块训练  | QRRU + 新分支                       | Flow Head、共享特征、D8/D2 匹配器 |
| DGFO 消融      | QRRU + Difference + DGFO         | Flow Head、共享特征、D8/D2 匹配器 |
| 最终联合训练（可选）   | Flow Head、匹配器、QRRU；再决定是否解冻 CGMDP | DINO，初期冻结 MVT            |

Flow Head 微调时，可以基于现有 `mhinet/engine/ghim_losses.py` 保留位置、可匹配性、匹配分类监督，并移除 H 参数监督。

但现有半密集训练入口默认冻结全局 Head，且匹配训练损失并不会自动对 Flow Prior 形成完整的可微训练链路，因此需要显式新增 Flow Head 训练配置，不能仅把参数 `requires_grad` 改为 True 就认为训练协议完整。

另外，新增 QRRU 参数意味着旧 checkpoint 的模型状态和优化器参数组不再完全一致。应该使用明确的 checkpoint 迁移函数加载已有权重，新增参数单独初始化；新实验使用版本化元数据和全新优化器，不直接恢复旧优化器状态。

## 九、我建议的消融实验矩阵

这一轮不要做太多组合，先保留以下六组。

| 编号 | D8 几何引导   | D2        | QRRU               | 目的                   |
| -- | --------- | --------- | ------------------ | -------------------- |
| A  | 原 H0 掩膜   | 原 H0 Warp | 原版                 | 历史完整基线               |
| B  | 无几何引导     | Direct-D8 | 原版                 | 纯无 H 基线              |
| C  | Flow0 软先验 | Direct-D8 | 原版                 | 测试 Flow 引导           |
| D  | Flow0 软先验 | Direct-D8 | 原版 + Discrepancy   | 测试显式差异               |
| E  | Flow0 软先验 | Direct-D8 | Discrepancy + DGFO | 测试位移传播               |
| F  | 无几何引导     | Direct-D8 | Discrepancy + DGFO | 检验 QRRU 是否依赖 Flow 引导 |

其中 B→C 才是 Flow-guided D8 的独立增益；C→D 是 Difference 增益；D→E 是 DGFO 增益。

A→E 是完整新旧架构的比较，但不能单独说明哪个模块贡献最大。

为了避免由于训练差异带来的混淆，建议 D/E 使用完全相同的训练数据、初始化检查点和优化步骤预算，并在 E 中只新增 DGFO 的可训练结构。

### 评价指标

你目前最重要的仍然是 Pre\@3，而不是 NCM。

| 层级   | 指标               | 目的           |
| ---- | ---------------- | ------------ |
| 主要   | Pre\@1/3/5/10px  | 判断匹配精度       |
| 主要   | RMSE             | 判断错误幅度       |
| D16  | Flow EPE、有效覆盖率   | 判断 Flow Head |
| D8   | Pre\@3、正确父匹配比例   | 判断 Flow 引导   |
| D2   | Pre\@3、Fine 覆盖率  | 判断是否丢失正确候选   |
| QRRU | 修正前后 EPE、改善/退化比例 | 判断精修是否有效     |
| 辅助   | NM、NCM、空间覆盖率     | 防止只通过筛点提高精度  |
| 效率   | 各模块推理耗时、峰值显存     | 判断结构成本       |

数据评价可以继续使用 GoogleEarth 和 JL1Flight 的固定测试清单。

但还应充分利用你刚加入仓库的 [`tools/gcp_annotator`](https://github.com/yuexikang/MHINet/tree/main/tools/gcp_annotator?utm_source=chatgpt.com)。

这个工具能够用人工标注的地面同名点，对预测匹配拟合得到的几何变换进行独立评价。这里的 H 只用于离线评价，不进入新模型的几何预测或训练。

人工 GCP 评价和合成 H 评价必须分开报告，不能混成一种 Ground Truth；而且只有实际完成并审核的人工标注才可以作为有效参考。

你此前已经设计的 Top-K=1024、2048、4096 消融，也适合在最终阶段复用，但不应拿它替代匹配精度评价。

## 十、代码实施清单

建议将新分支集中放到以下文件，避免在旧实现上大量写条件分支。

| 文件                                                   | 建议修改                                   |
| ---------------------------------------------------- | -------------------------------------- |
| `mhinet/models/feature_provider.py`                  | 增加 Flow0 输出接口，保留旧 H0 模式兼容              |
| `mhinet/downstream/loma_reference/coarse_matcher.py` | 添加 Flow-guided dual-softmax            |
| `mhinet/downstream/supervision.py`                   | 新增无 H 的 Direct-D8 `fine_grids`         |
| `mhinet/downstream/semidense.py`                     | 引入 `geometry_mode`，切换 Flow 引导及直接 D2 展开 |
| `mhinet/downstream/qrru.py`                          | 实现 Difference、局部 DGFO、零初始化残差注入         |
| `mhinet/downstream/training.py`                      | 独立训练组、严格的 checkpoint 迁移                |
| `mhinet/downstream/evaluate_semidense.py`            | 增加 Flow、D8 和 QRRU 分阶段诊断                |

我建议在配置中增加：

```
{
  "geometry_mode": "flow_d16",
  "flow_guidance": {
    "enabled": true,
    "mode": "soft_gaussian",
    "lambda": 3.0,
    "sigma_d8": 4.0,
    "use_matchability": true
  },
  "fine_prior": "direct_d8",
  "qrru": {
    "use_discrepancy": true,
    "use_dgfo": true,
    "dgfo_radius": 1,
    "dgfo_dim": 32,
    "iterations": 4,
    "radius": 1.5
  }
}
```

以上是建议的初始实验配置，不是经验证的最优超参数。尤其 `lambda` 与 `sigma_d8`，应在固定验证集上进行小范围搜索，最终测试集不能参与调参。

## 十一、我认为需要设置的三个停止条件

第一，Flow0 的有效区域精度不足。 如果原始 `coarse_warp` 的错误很大，而且几何加分经常偏向错误目标，就不要继续扩大几何权重。先通过独立 Flow 监督提升粗定位能力。

第二，D2 Direct-D8 损失过大。 如果 C 的 D8 比 B 明显改善，但 D2 匹配依旧大量丢失正确对应，说明瓶颈已经转移到 D2 的局部搜索网格。不能把这类问题交给 QRRU 强行修复；届时需要另立 D2 几何模型实验，而不是混入本轮。

第三，DGFO 无法提供独立增益。 如果 D 与 E 在相同训练预算下没有可重复的改善，或者只提高了匹配数量、却没有提高 Pre\@3，那么优先保留较简单的 Difference-QRRU，不必强行增加 Attention。

## 最终确定的执行方案

我建议把这次研究划分为两条彼此独立的实验线：

实验线 A：D16 Flow-guided Global Matching

\\[ \boxed{ \text{Flow16} \rightarrow \text{D8 全局匹配软引导} \rightarrow \text{D2 Direct Matching} } \\]

重点研究：低分辨率 Flow 预测能否有效提升大范围特征匹配，同时不过度排斥 Flow0 预测错误时的真实对应。

实验线 B：Discrepancy-guided Recurrent Refinement

\\[ \boxed{ \text{SGT} \rightarrow \text{Feature Discrepancy} \rightarrow \text{DGFO} \rightarrow \text{QRRU 控制点更新} } \\]

重点研究：显式特征差异和局部 Flow 传播，能否让 QRRU 更准确地修正位置，同时减少对原本正确匹配的破坏。

我建议先完成 B→C 的 D8 引导实验，再固定 C 的匹配输入完成 C→D→E 的 QRRU 消融，最后才联合训练。

这样整体架构不会失控：DINOv3、MVT、CGMDP、D8/D2 半密集匹配框架以及 QRRU 的四控制点结构都保留下来，真正新增的学习机制主要就是 Feature Discrepancy 和 DGFO。

更重要的是，它能回答两个具有独立研究意义的问题，而不只是证明“换了网络以后指标有所变化”。