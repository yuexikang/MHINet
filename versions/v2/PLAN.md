# MHI v2 实验实施计划

登记日期：2026-10-19；状态更新：2026-10-20（北京时间）。本计划已按版本分档，执行入口为 [逐条验收清单](ACCEPTANCE.md)，逐项签收见 [验收登记](acceptance_status.json)。独立 v2 实现已落地，57 项 CPU 测试与 [真实 GPU 工程验收](evidence/real_feature_acceptance_20261020_final.json) 通过，Flow Head 两步实数据训练及断点恢复烟测通过。正式长程已于北京时间2026-10-20 03:24:37在 GPU0 启动 Flow Head，C/D/E 排队，实际状态以调度输出的 `campaign.json` 和阶段 `status.json` 为准；尚无正式完整训练或新精度收益结论。

当前需求修订为：**模型输入与形状验收只覆盖 512×512；增加独立最终输出上限 `max_output_matches`，默认 2048 对/影像对**。该上限在 QRRU 与最终有效性过滤后生效，按置信度稳定降序选取；同分保留原顺序，不足上限时返回实际数量，不补点、不用 GT 筛点。配置可显式覆盖为正整数，各对照须统一限额。

## 当前工程证据与首轮执行协议

- `mhinet/v2/` 已实现 Flow0、D8 soft prior、Direct-D8、Discrepancy/DGFO、严格 checkpoint 迁移、独立训练/评价和 RGB_512 数据适配。显式入口为 `python -m mhinet.v2.train`、`python -m mhinet.v2.evaluate`；[run_campaign.py](../../scripts/mhi_v2/run_campaign.py) 串行调度，[validate_real.py](../../scripts/mhi_v2/validate_real.py) 生成真实工程证据。
- [真实 GPU 报告](evidence/real_feature_acceptance_20261020_final.json) 为 `passed`：两对固定 val、三步 QRRU 反传诊断，H Fitter 调用为0，零 adapter 的四轮轨迹与旧 QRRU 一致，冻结参数/buffer 与新增上游梯度检查通过。两对结果只作工程诊断，不能当作完整精度对照。
- 正式独立配置已经登记于 [configs/mhi_v2](../../configs/mhi_v2/README.md)：先 Flow Head，再 C、D、E，各 seed42、5 epoch、25535 次 optimizer 更新。每步累积8对，`ceil(40849/8)×5=25535`；每 epoch 完整5106对 val 选模，中途64对只诊断。Head 以 Flow EPE 最低选模；C/D/E 以级联 Pre@3 最高选模。
- 只使用原 RGB_512 固定 train=40849/val=5106，不改变划分，不读取封存 GoogleEarth/JL1Flight 评价集。源图身份交叉为0；当前数据身份分组未证明地理邻近或真正空间隔离。
- 首轮仅 seed42，不能声称多 seed 稳定性。完整精度、资源对照、跨 seed 复核和封存集评价都须另有实际证据；`campaign.json` 未显示完成时不能把注册预算写成实际完成预算。
- 实现固定于 `88fee1b`；[资源验收](evidence/resource_acceptance_20261020.json) 已通过，[campaign 实时状态](../../outputs/mhi_v2/campaign_rgb512_seed42_20261020/campaign.json) 已生成。V2-P2-04 的完整 D8/D2 retention/coverage 诊断仍未签收，P5 精度验收尚未开始。

## 版本与交付范围

| 项目 | 固定约定 |
| --- | --- |
| MHI v1 代码 | 注释标签 `mhi-v1` → `522d9bd5565e88267e72866bd829fb05274762e4` |
| MHI v2 实验分支 | `codex/mhi-v2`，从上述提交创建 |
| 正式 v1 半密集权重 | `weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt`，25535/25535 步 |
| v1 权重 SHA256 | `43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4` |
| 机器可读冻结清单 | [versions/v1/freeze_manifest.json](../v1/freeze_manifest.json)，保留原文件字节与 SHA256 |
| 机器可读实验登记 | [experiment_plan.json](experiment_plan.json)，是规划规格，不能传给现有训练入口 |
| 用户原始方案 | [proposal.md](proposal.md) |
| v1 源码工作区 | `/home/disk1/MHINet-v1`，detached 于 `mhi-v1` |
| v2 新实现 | `mhinet/v2/`；专属入口 `scripts/mhi_v2/`，配置 `configs/mhi_v2/`，测试 `tests/v2/` |
| 新输出目录 | `outputs/mhi_v2/<实验ID>/<阶段>/<运行ID>/`，每次新实验使用独立目录 |

原方案引用的 `a1d32e3` 后，main 新增了标注工具更新 `522d9bd`。以本次实际 HEAD 为 v1，包含这次更新；标签建立后不移动、不覆盖。后续模型修改只进入 v2 实验分支，main 的本次基点保持不变。这里的 MHI v1/v2 是研究版本命名，仓库名仍为 MHINet；历史 `v1.2` 训练协议、`stable_v2` 数据集、`semidense_qrru_v1` checkpoint task 和 Python 包 `0.1.0` 不做全局替换。

准备阶段交付包括 v1 标签与独立源码工作区、分版本档案、实际资源预检、27项任务清单及证据登记。P1–P5 按清单逐条实施验收；准备通过不代表模型改造或精度通过。

## 文件隔离规则

v1 档案只进入 `versions/v1/`，新研究计划和证据只进入 `versions/v2/`；旧归档路径通过 [relocations.json](../relocations.json) 查询。冻结清单不因整理而重写。大权重和历史原始预测保留冻结清单中的原路径，新 v2 权重使用 `weight/mhi_v2/`，新推理/训练产物只进入 `outputs/mhi_v2/`。

原 `mhinet/models/`、`mhinet/downstream/`、旧 CLI 和 LoMa 来源核作为 v1 兼容底座保留。下面的旧文件引用用于解释现状；所有 **v2 专属实现进入 `mhinet/v2/`**，通过 wrapper、子类或显式适配复用稳定功能，不往旧文件加入 v2 分支。v1 默认入口与 checkpoint 严格加载行为不变。v1 回放必须记录实际导入源码路径并显式映射归档 runtime；独立 worktree 不会自动修复 checkpoint 内嵌的旧绝对路径。

## 研究问题和改造边界

两条实验线分别回答：D16 Flow0 能否提高 D8 全局匹配的候选排序；显式特征差异与局部位移传播能否提高 QRRU 的局部修正精度。先完成 B→C，再固定 C 的匹配输入开展 C→D→E，最后才考虑联合微调。

```text
DINOv3 + MVT → F_MVT ─→ Flow0：D16 A→B 目标坐标、位移、Matchability
                └────→ CGMDP：D8 / D4 / D2 描述子
                                   ↓
                 D8 全局匹配 + Flow 几何软奖励
                                   ↓
                 D2 Direct-D8：父对应展开 4×4 子格
                                   ↓
                 QRRU：动态 Warp → Discrepancy → 局部 DGFO
                                   ↓
                 原 Gate / Head 更新 2×2 控制点，共四轮
                                   ↓
                 半密集点对与置信度
```

保留 DINOv3、MVT、CGMDP、D8 全局 dual-softmax、D2 16×16 局部匹配，以及 QRRU 的 4×4 特征窗口、2×2 控制点、四轮循环。新增学习机制集中在 Discrepancy 与 DGFO；Flow0 先复用已有定位预测。范围不含 D8 Flow 更新网络、D2 Flow Refinement、全图逆 Flow 或替换为 Dense Flow QRRU。H0 的旧代码、旧权重加载及默认推理路径继续作为 v1 对照。

## v1 代码审计背景与 v2 适配要求

以下记录的是冻结 v1 的接口限制；对应适配现已进入独立 v2 路径，保留这些背景便于复核设计约束。实现与工程证据不等同于正式训练和精度验收。

1. `SharedFeatureProvider.forward()` 当前通过 `stage1` 输出 `coarse_warp/coarse_matchability`；`compute_stage1=False` 会连定位预测一起关闭。必须在 MHINet 内拆分定位预测和无参数 H Fitter，Flow 模式只跳过拟合。避免修改相邻 LoMa 仓库，并保持旧参数键名兼容。
2. `SemidenseConfig` 位于 `mhinet/downstream/semidense.py`，是扁平 frozen dataclass。原方案的嵌套配置尚不能直接读取。新增版本化解析器，旧配置缺省解释为 H0 + H0 residual fine grid + 原 QRRU；未知字段应明确报错。
3. 当前推理使用 `len(shared['H0_norm'])` 遍历批次，并依赖 `stage1_valid`、H 合法性和 H overlap masks。训练 fine loss 也由 `good_h` 门控。无 H 模式必须同时去除这些依赖，批次从描述子读取。
4. 当前训练入口 `mhinet/downstream/train.py` 强制冻结 Head；`SemidenseSystem` 默认训练部分共享特征和完整 matcher。P5 需要显式训练组和梯度审计，不能直接复用默认训练策略。
5. `ghim_loss` 即使 H 项权重为零也会读取、计算 H 项。Flow Head 监督必须拆出位置、matchability、分类损失，真正不读取 H 预测；GT 几何仍可用于训练标签。
6. 当前评价入口只接受 `metadata.task='semidense_qrru_v1'` 且严格加载完整状态。v2 使用独立协议 `mhi_semidense_v2` 和明确迁移函数。仅允许缺失登记过的新增参数，其他缺失、额外键或形状差异均报错；不得用全局 `strict=False` 掩盖错误。
7. main 中的 `configs/semidense_qrru_512.json` 是 quadrant tier1 模板，**不是正式 RGB512 联合训练配置**。权重元数据引用的正式 runtime/config 和部分历史实验脚本不在当前 main；冻结清单保存 checkpoint 内嵌配置、runtime、来源提交和历史报告路径。P1 必须先解决实际 runtime 解析与资源预检，不能以模板替代。Git 标签只冻结受跟踪代码，权重、数据、outputs 仍需按清单在外部保存。

## 阶段计划与验收

### P0：固定 v1、建立 v2 登记（已完成）

创建标签和实验分支；核对正式权重及两份评价清单的实际 SHA256；保存内嵌训练配置和历史评价引用；登记 A–F、主指标和停止条件。原始方案存档以便逐项核对。

验收：v1 标签指向精确基点，v2 由该基点派生；本轮没有修改模型源码；所有新实验均为 planned，精度结果为空。

### P1：D16 Flow0 接口与兼容加载（已实现，工程证据已生成）

新增文件：`mhinet/v2/config.py`、`feature_provider.py`、`checkpointing.py`、`training.py`。参照旧 provider/training 接口实现独立适配，保持已有参数键名，避免将旧模块额外包一层后改变所有 checkpoint 前缀。先解决归档 runtime 显式加载与实际源码导入路径验证。

输出契约：`flow16_target_norm:[B,2,H16,W16]`、`flow16_norm` 同形状、`flow16_matchability:[B,1,H16,W16]`，并显式输出预测有效性。方向为 A→B，归一化采用 `align_corners=False` 的 cell centers。`flow16_norm = target_norm - source_grid_norm`。确认实际 matchability 是概率还是 logits，再统一为 [0,1] 概率，避免重复 sigmoid。

先插值**位移**至 D8，再加 D8 参考网格；不直接上采样绝对坐标。距离换算为 D8 像素时 x/y 分量分别乘 W8/2、H8/2。无效值在距离和插值运算前安全处理，不能用 `0 * NaN` 屏蔽错误；有效性传播策略需独立记录。

验收：仅接受 512×512 模型输入，D16=32×32、D8=64×64、D4=128×128、D2=256×256；其它模型输入尺寸明确拒绝。零 Flow 恒等，已知平移方向及半像素坐标正确；缺失、NaN、越界、低置信度可回退；Flow 模式 H Fitter 调用次数为零；512 下 v1 模式输出和旧 checkpoint 严格加载保持一致。用独立 GT 计算 Flow EPE 和有效覆盖率，真实图像烟测仅使用 train/val。

### P2：D8 Flow 软引导与 D2 Direct-D8（已实现，工程证据已生成）

新增文件：`mhinet/v2/coarse_matcher.py`、`supervision.py`、`semidense.py`。参照旧 D8 参考核、fine grid 和 matcher 契约实现；保留旧来源核文件与哈希。

先实现 MHIv2-B 的无 H 基线，再实现 MHIv2-C 的 Flow 软先验。D8 保留两侧全图候选与 FP32 dual-softmax，对有效 A 查询加入有界正奖励：

```text
S(i,j) = dot(normalize(DA[i]), normalize(DB[j])) / tau
         + lambda * c[i] * exp(-||q[j] - q_hat[i]||² / (2*sigma_d8²))
```

初值 lambda=3、sigma_d8=4，未经验证。`flow_prior=None` 或 lambda=0 显式走原外观打分路径。full、chunked 首遍的行列 logsumexp/最佳候选，以及最终 mutual 置信度或 top-k 二次遍历全部调用一致的融合打分。无效、越界和低置信度 Flow 只取消该查询奖励，不拒绝整对影像；不从单向 Flow 伪造 B 侧 H overlap mask。

新增 `direct_d8_fine_grids()`，原 `fine_grids()` 保留：`b = grid[j] + a - grid[i]`。子格仍为 16 个 D2 点；后续 16×16 相似度、互选、置信度组合不变。推理只使用预测和边界条件；GT masks 仅能参与监督及评价。

验收：lambda=0 的 C 与 B 在相同输出限额下逐点相同；full/chunked 候选一致、置信度在预先登记的 FP32 容差内；包含空输入、并列分数、异常 Flow、无效 H0 测试；直接子格与恒等 H 的旧子格一致；训练与推理均不再通过 H0 门控 B/C。最终输出逐对不超过 `max_output_matches`，并验证空输出、不足/恰好/超过上限、同分稳定排序及显式覆盖。记录 D8 独立耗时与峰值显存，512 全图 4096² 候选，保留分块计算。

### P3：QRRU Feature Discrepancy（已实现，工程证据已生成）

新增文件：`mhinet/v2/qrru.py`，以及 v2 checkpoint 迁移与版本化配置；旧 `mhinet/downstream/qrru.py` 作为对照保留。

每轮重采样 `wb` 后，按通道归一化计算 `D=abs(normalize(wa)-normalize(wb))`；使用 1×1 Conv(48,32)+GN+GELU 得到 R。保留原 128 通道 encoder 和所有旧参数形状，采用零初始化 1×1 `DiffAdapter(32,32)` 残差注入。此阶段关闭 DGFO。

验收：同一旧权重、输入、采样与分块设置下，新增分支初始化为零时所有迭代控制点、中心、最终匹配及置信度与旧 QRRU 对齐；空匹配和边界采样无 NaN；训练若干步后 adapter 与差异编码器均得到有效梯度。通过后才报告 C→D 收益。

### P4：局部 DGFO（已实现，工程证据已生成）

实现文件：`mhinet/v2/qrru.py`、`mhinet/v2/dgfo.py`。

用现有 `interpolate_controls(flow, off)` 从 2×2 控制点构造 4×4 局部残差，Q/K 来自 R（32维），V 为局部位移（2维）；对半径1的3×3有效邻居做 masked softmax。`E=propagated-local_flow` 经零初始化 `FlowAdapter(2,32)` 加入原 z 更新，控制点仍由原 Gate/Down/Head 更新，不从传播结果拟合或覆盖控制点。

现有插值在 ±1 控制点对 ±1.5 patch 位置做双线性**外推**，不能替换成普通 `F.interpolate`。四轮 radius=1.5 的6px理论界限仅针对控制点的单分量累积，不是全部局部采样点的界限。

mask 同时检查 patch 邻域和真实采样位置；无有效邻居时令传播修正为零，避免全 -inf softmax。建议 FlowAdapter 不带 bias，使首轮零 Flow 时 DGFO 注入始终为零；首轮由原 encoder 和 Discrepancy 产生初始更新。不得用 GT overlap 构造推理 mask。

验收：有效邻域权重和为1，越界权重为0，无有效邻居稳定回退；零/常量 Flow 的传播行为正确；首轮传播残差为0；后续轮次梯度可达 Q/K。记录每轮 E 范数、控制点增量、有效邻居数、分支梯度、耗时/显存；零 adapter 的旧输出等价和数步优化后的上游梯度均须验证。

### P5：分离训练、等预算消融与最终评价（入口/烟测通过，Flow Head 已启动、C/D/E 排队）

实现文件：`mhinet/v2/train.py`、`training.py`、`semidense.py`、`losses.py`、`evaluate.py`，入口与配置分别放入 `scripts/mhi_v2/`、`configs/mhi_v2/`。显式入口及运行配置已实现；[研究登记](experiment_plan.json) 仍然不可作为训练配置执行。

| 子阶段 | 可训练 | 冻结 |
| --- | --- | --- |
| 原始 Flow0 直接测试 | 无 | 全部 |
| Flow Head 微调 | 定位/可匹配性 Head | DINO、MVT、VGG、CGMDP、D8/D2、QRRU |
| 原 QRRU 受控重训 | 原 QRRU，无新增分支 | Flow Head、共享特征、D8/D2 |
| Discrepancy 训练 | 原 QRRU + Difference | Flow Head、共享特征、D8/D2 |
| DGFO 消融训练 | 原 QRRU + Difference + DGFO | 同上 |
| 可选联合微调 | Flow Head、matcher、QRRU；CGMDP另行登记 | DINO 固定，初期 MVT 固定 |

Flow 监督使用位置、matchability、分类项，不依赖预测 H；现有离散候选选择与 coarse loss 不构成自动可微的 Flow 训练链。各组保存实际参数名单、优化器组、冻结梯度与 BN buffer 审计。

本轮 QRRU 输入目标固定为 50%“GT+有界随机扰动”与50%“当前预测 D2 对应”。预测只在 GT 可见、目标与控制点误差满足6 D2像素修正范围时入选；严重错误单列，合格预测不足时用噪声补齐，必须报告实际预测/噪声数量。噪声标准差0.8、界限2.4 D2像素。缓存同一冻结 C 来源的 D2 预测，由原 QRRU 受控重训 C、D、E 复用。F 主对照复用 E 已训练的 QRRU，仅关闭 D8 Flow prior；若另用 B 的预测输入重训 F，须单列等预算 run。按 D2 初始误差 0–1、1–3、3–6、>6px 分组统计修好/破坏比例。

C 的原 QRRU 受控重训、D、E 均从同一个 Flow Head 最佳 checkpoint 独立初始化；其中旧 QRRU 权重仍来源于冻结 v1，D 增加零初始化 Difference，E 再增加零初始化 DGFO。三组固定 seed42、训练顺序、样本数、5 epoch/25535 次 optimizer 更新和完整 val Pre@3 选模规则。C 同时保留冻结 QRRU 的推理结果：B→C 用冻结版本，C→D 用等预算重训版本，避免把重训收益归因于差异分支。正式配置已登记，实际完成量从运行状态读取；中断/诊断步数不计为完成预算。本轮为单 seed42 的首轮实验；后续多 seed 稳定性复核须另行登记并实际执行。

所有新实验只迁移模型权重，重新建立 optimizer/scheduler/scaler，不恢复 v1 optimizer 状态。真正的 v2 同协议续训再严格匹配架构、配置、数据和 RNG/游标。保存基点提交、实现提交、协议/schema、来源权重 hash、配置 hash、清单 hash、训练组、随机种子、环境和每次评价命令。

## 六组消融登记

采用完整实验 ID，避免与历史 H0 位置消融 A–D 混淆。**历史 H0-D 对应本轮 B 的结构，不是本轮 D。** 独立 v2 路径已实现无 H 模式；旧报告只保留历史来源身份，不直接作为新实现结果。

| ID | 原方案编号 | D8 | D2 | QRRU | 作用 |
| --- | --- | --- | --- | --- | --- |
| MHIv2-A | A | H0 overlap mask | H0 residual warp | 原版 | 冻结 v1 对照 |
| MHIv2-B | B | 全局纯外观 | Direct-D8 | 原版 | 无 H 对照 |
| MHIv2-C | C | 全局 Flow0 软先验 | Direct-D8 | 原版 | B→C：Flow 收益 |
| MHIv2-D | D | 同 C | Direct-D8 | Discrepancy | C→D：显式差异 |
| MHIv2-E | E | 同 C | Direct-D8 | Discrepancy+DGFO | D→E：传播收益 |
| MHIv2-F | F | 全局纯外观 | Direct-D8 | Discrepancy+DGFO | B→F、E↔F：独立性/依赖性 |

固定输入512、D8 chunk=128、训练窗口chunk=32、推理窗口chunk=1024、max_coarse=6000、confidence threshold=0、QRRU四轮和radius=1.5。新增 `max_output_matches=2048` 限制最终返回点对；原 `max_matches=12000` 明确作为 QRRU 前的内部候选预算保留，不能混用两个限额或把输出上限用于缩小 D8 搜索。学习温度从旧权重读取，不能把初始化0.05当作实际温度。A–F 主表统一最终上限2048；1024/4096仅作为单独登记的补充上限实验。

旧 v1 的 `max_matches` 在 QRRU 前截断，随后才过滤无效点。本轮新增的最终上限在过滤之后应用，因此不能把历史最终 NPZ 取 Top2048 称为“原生 `max_matches=2048` 复跑”。使用历史 A 预测时，必须先确认512输入、QRRU前候选预算和有效性规则相同，再按本轮最终2048协议离线重算全部指标（包括重新拟合 RANSAC）；标明为离线重评分。历史表格和证据保持原样，耗时比较须另作同协议实测。

B→C 先采用同一冻结权重，仅切换 prior；若微调 Flow Head，还需在同一微调后权重上成对开关 prior，避免把 Head 训练收益误算为几何奖励。C→D/D→E 是等预算训练后的结构比较，同时保留未训练零 adapter 的等价结果。E↔F 的主对照保持所有模型权重相同，仅切换 prior。A→E 只代表整体新旧比较。

## 数据、评价与选型

固定沿用正式 RGB512 的 train/val 数据：`/home/disk1/Data/datasets/MHINet_RGB_512/{train,val}/pairs.jsonl`，40849/5106对，按冻结清单核对 SHA256。当前以 `source_dataset + source_relative_path`（缺失时用 source_path）核验源图身份，train/val 交叉为0；这不代表地理邻近隔离，不能声称真正空间隔离。调参与模型选择只使用该 val；首轮使用登记的 lambda=3、sigma_d8=4，未经完整 val 证明为最优。

| 锁定评价集 | 清单 | 对数 |
| --- | --- | --- |
| GoogleEarth | `/home/disk1/Data/datasets/Test_Data/GoogleEarth/val/pairs.jsonl` | 498 |
| JL1Flight | `/home/disk1/Data/datasets/Test_Data/JL1Flight/test/pairs.jsonl` | 1755 |

GoogleEarth 的路径虽名为 `val`，在本轮仍作为锁定的外部评价集，不参与超参/模型选择。498对沿用排除原始影像/真值错误431、432后的清单；不再根据新模型结果删样本。两份清单的实际 SHA256 见冻结清单。已有历史结果属于已使用过的评价集，报告为固定基准复核，不声称是首次完全未知的测试。

主指标为逐对宏平均 Pre@1/3/5/10px，以 Pre@3 为首要选型指标；全部输出匹配进入分母，无匹配对精度计0。NM/NCM/空间覆盖率和失败率同时报告，防止仅通过少输出点提高精度。阈值统一采用存储影像的512目标像素；若回到不同原图尺寸，必须单列坐标尺度。

分阶段诊断包括 D16 EPE/有效覆盖率、D8 精度/正确父对应比例、D2 精度/Fine覆盖率、QRRU 前后 EPE/修好及破坏比例，并跟踪每轮轨迹。分别记录共享特征、Flow Head、D8、D2、QRRU 的耗时及峰值显存；统一预热和测量范围，排除 I/O、指标和哈希，速度结论应在无并行干扰的 GPU 上测量。

RMSE 明确区分：全部匹配 RMSE、预测 H 网格 RMSE、论文口径 RMSE。沿用历史论文口径时，H 仅由预测点 RANSAC 拟合（3px/10000次/置信度0.999/seed0），以GT可见20×20网格 SR@3px 判成功；成功对取GT误差≤5px正确匹配的RMSE，失败对记10，成功但无正确匹配单独记无定义。不能混用不同口径；GT只在预测完成后供评价使用。

人工 GCP 与合成 H 评价分开报告。只有实际完成审核的人工同名点能进入 GCP 参考集；随机候选、待复核点和未完成影像对不能当作真值。标注工具只提供采集能力，不代表已有有效人工基准。离线拟合 H 不进入 v2 几何预测。

v1 历史 A 的参考 Pre@3：GoogleEarth 33.24%、JL1Flight 37.92%；论文 RMSE 6.520/6.791。它们来自旧输出协议的归档消融，详见 [v1 的 H0 推理位置消融结果](../v1/reports/H0_position_ablation.md)，不能直接填入本轮最终2048主表。本轮 A 的全部指标须按相同输出协议复算；历史主表也缺少 Pre@10，未复算前保持为空。v2 各组结果保持为空，实际完成后再填。

## 停止条件与交付检查点

1. **Flow 不可靠**：若几何奖励经常提升错误候选，不扩大 lambda；先查坐标、置信度校准及独立 Flow 监督，只在 val 上作决定。
2. **Direct-D8 成为瓶颈**：若 C 在 D8 改善，但 D2 丢失大量正确候选，停止扩大 QRRU 修正范围。另立 D2 几何实验，不混入本轮。
3. **DGFO 无独立增益**：先检查本轮相同预算、seed42 的 E 与 D；无收益或收益仅是NM增加时，优先保留 Difference-QRRU并报告负结果。单seed有收益也只作为候选；多seed可重复性须另行实际复核，不能从本轮推断。

每阶段单独提交代码、工程检查和验证结果；通过验收后更新实验登记的状态与实际产物。P1–P4工程通过不等于精度提升；训练损失下降也不能替代固定清单级联推理评价。最终交付含 A–F 成对结果、错误分组、资源成本、来源清单及是否保留 Flow/DGFO 的证据结论。只有完成实际验证后才考虑 v2 发布标签或集成 main。
