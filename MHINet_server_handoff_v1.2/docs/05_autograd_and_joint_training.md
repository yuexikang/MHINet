# 全流程可微性与联合训练约定

结论：本模型可以构建为端到端训练的计算图，DINO 冻结不妨碍其后模块联合优化。但现有仓库若仅切换 requires_grad，仍不能保证该图成立；旧推理包装需要改造。包括硬阈值和有效性判断的实际程序是分段可微，不能声称每个离散决定在所有输入上可微。

## 1. 两条回传路径

$$
\begin{aligned}
Z&=\operatorname{MVT}_{\theta_M}(\operatorname{DINO}_{\mathrm{frozen}}(I_A,I_B)),\\
H_0&=g_{\theta_G\,\mathrm{frozen}}(Z),\\
F^s&=a_s\bigl(d_{\theta_D}(v_{\theta_V}(I_A,I_B),Z)\bigr),\\
\frac{\mathrm d\mathcal L}{\mathrm d\theta_M}
&=\left[\frac{\partial\mathcal L}{\partial H_0}\frac{\partial H_0}{\partial Z}
+\sum_s\frac{\partial\mathcal L}{\partial F^s}\frac{\partial F^s}{\partial Z}\right]
\frac{\partial Z}{\partial\theta_M}.
\end{aligned}
$$

公式中的偏导包含展开八轮后的下游依赖。冻结 θ_G 只是不更新 head 权重，仍需保留 head 对 Z 的导数。MVT 一旦训练，H0 也可能变化；“保留 Stage1 架构和头权重”不等于“保证 H0 永久不变”。必须同时评估微调后模型自身的 H0，以及最初冻结 Stage1 的 H0。

| 操作 | 梯度性质 | 实施要求 |
| --- | --- | --- |
| 冻结 DINO | 无须 DINO 参数梯度 | 仅该前缀可 no_grad，输出供下游正常训练 |
| MVT、VGG、DeDoDe、adapter/CNN | 常规可微 | train/参数组正确；不能被外层 no_grad 覆盖 |
| L2 normalize、点积、双线性 grid_sample | 非奇异区域可微/分段可微 | 对目标特征和连续采样坐标保留梯度 |
| 四角残差相加、tanh | 可微 | 饱和可能削弱梯度，不 detach T |
| 非退化8×8线性求解、齐次投影 | 合法区域可微 | FP32；条件数/分母保护，避免先计算非法分支 |
| Stage1 matchability 阈值、有效mask、guard | 离散判定本身无梯度 | 接受分支保留连续计算图，拒绝 proposal 保留四角监督 |

不使用 argmax、RANSAC、NumPy/OpenCV 拟合输出替代训练几何层。它们可用于离线 reference；不能放在需反传路径上。mask 无梯度不等于整条网络无法学习，但硬拒绝样本不会获得穿越该决定的梯度。

## 2. 已发现的旧代码阻断

源码路径均相对设计审计的 2-step commit，后续服务器先核对差异：

- `experiments/stage1_dedode_pyramid_hroi_v1/model.py`：构造器冻结全模型、train() 强制推理模式、extract_pyramid 使用 inference_mode。为新实验建立训练 provider，不直接套用旧推理入口。
- `experiments/loma_dinov3_l17_v1/descriptor.py` 的 Stage1 共享 descriptor：MVT 初始被冻结，contextualize_pair 有 no_grad；解冻 MVT 必须同时移除该范围的禁梯度包装。DINO 独立 no_grad 保留。
- 原 VGG/DeDoDe encoder 的 train() 有 eval 覆盖。eval 本身不关闭 autograd，但 BN 行为、requires_grad 和上层 wrapper 必须分别检查。
- `experiments/stage1_loma_shared_v1/stage1_head.py`：head 参数冻结但 forward 本身没有 no_grad，可以作为冻结的可微输入映射；不要在新 provider 中包成 no_grad。
- `experiments/loretta_stage1_h/fitter.py`：matchability>.3、至少10点、ridge=1e-4、FP32 solve_ex 和失败 identity。保留原前向语义的同时审查梯度安全；原 sqrt(clamp(weight,1e-8)) 下屏蔽点在系统中的权重并非严格数学零，不要擅改后仍声称 H0 对齐。

Stage1 fitter 的安全重构若改变正常样本 H0，应单列差异并验证；失败分支允许独立安全处理。训练 joint 不能仅依赖 fitter 的最终 success flag 掩盖奇异求解梯度。

## 3. 安全求解与硬分支

先构造四角线性系统，在 no_grad 中检查输入 finite、秩/条件数及最少支持；仅对合格子集执行带 autograd 的 solve。若求解探测仍失败，应在正式可微求解前剔除，或以有限非奇异替代系统执行并隔离其输出。不能对奇异矩阵先 solve/inverse，再 where 选旧 H：隐藏输出不保证 backward 不出现 NaN。

除法前为已判无效的分母使用有限安全替代值，并用无效状态屏蔽结果；对正常分母保持原公式。正式求解后再检查 H 与9×9网格分母，同号且绝对值>1e-4。拒绝更新保留前一合法 H/T，proposal 角点主 loss 仍可训练 decoder；mask/条件数检查不作为可学习惩罚项。

有效 Stage1 样本没有局部支持时仍计算有限 proposal 角点监督，但不接受 H 更新。初始化失败样本本版不用于精修训练；这也意味着 joint 不会直接从这些失败例获得精修 loss，需要如实记录训练覆盖率。

## 4. 必须留下的反传证据

1. 逐组输出 requires_grad、optimizer 参数 ID，排除共享模块重复注册/重复优化；DINO 和冻结 head 的 grad 为 None。
2. 使用合法小型 FP64 样例，对四角 solve 和 correlation 的坐标/特征做有限差分或 gradcheck，避开 mask 边界和整数采样拐点；再做实际784 FP32/AMP smoke。
3. 仅 final loss 下验证后尺度到前尺度 decoder/H 的梯度非零；八轮主 loss 下验证所有尺度可训练。
4. 验证 VGG 和 MVT 在 joint 下得到有限梯度，DINO 无梯度。通过一次诊断性的分支断开，分别确认 MVT→H0→迭代与 MVT→金字塔→迭代两条路径。该断开只用于测试，不进入正式训练。
5. FC2 零初始化先更新一步，再核对前层梯度；可另外用测试专用微小非零末层权重检查路径。不要把首步为零误判成断图。
6. checkpoint 与非 checkpoint 输出/梯度在容差内一致；resume 保存 optimizer、scheduler、scaler（若用）、随机状态和数据进度。不得使用 detach 代替 checkpoint。

理论依据：[PyTorch autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html)、[grid_sample](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.grid_sample.html)、[linalg.solve](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.solve.html)。最终服务器 PyTorch 版本要实测记录；CUDA grid_sample backward 可能有非确定性，种子固定不等于逐位复现。
