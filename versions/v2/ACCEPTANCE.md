# MHI v2 逐条执行与验收清单

登记日期：2026-10-19（北京时间）。本清单面向执行与复核；研究边界见 [PLAN.md](PLAN.md)，机器登记见 [experiment_plan.json](experiment_plan.json)。

**本轮只做 P0：v1/v2 分档与只读 preflight。模型改造、GPU 推理、梯度诊断、正式训练和新精度评价均未开始。** 下列 P1–P5 均为待实现任务，列出的证据路径是交付约定，文件未生成前不得标为完成。

## 执行规则与状态

1. 按条目 ID 顺序执行；只有前置条目的证据经复核为 `pass`，才进入依赖它的条目。阶段末尾必须另存验收记录，不能只勾选 Markdown。失败、缺资源或证据缺失时登记原因、修复动作和复验结果；不得跳过前项。
2. 分开记录三种状态：`engineering`（工程）、`training`（训练）、`accuracy`（精度），取值为 `not_started / in_progress / pass / fail / blocked / not_applicable`。工程通过不代表训练完成或精度提升；损失下降不代表精度通过。
3. 每个条目证据至少包含：条目 ID、三类状态、实现提交、完整实际命令与退出码、运行时间、环境、输入/配置/权重/清单 SHA256、结果统计、通过标准及判定理由。尚不存在的命令仅放在 `planned_command`，`actual_command` 必须留空；不能写成已执行或可立即执行。
4. 数值容差、正式训练预算、模型选择规则、精度收益门槛、允许的 NM/覆盖率/失败率退化上限，必须先依据工程需要或 train/val 证据登记，再运行相应验收。未填阈值不能判 `pass`；不得看完锁定评价集结果后修改。
5. 所有新算法实现进入 `mhinet/v2/`；专属入口、测试、配置和运行产物分别进入 `scripts/mhi_v2/`、`tests/v2/`、`configs/mhi_v2/`、`outputs/mhi_v2/`。可 import v1 稳定功能；不得把 v2 专属分支写入原 v1 算法文件。若稳定接口不足，优先在 v2 wrapper/适配器中解决并登记限制。
6. `versions/v1/` 保存冻结清单与说明，不复制大权重和数据；`versions/v2/` 保存本计划及证据。历史名称中的 `stable_v2`、旧训练协议 `v1.2` 和 checkpoint task `semidense_qrru_v1` 不做全局替换。
7. `MHIv2-A` 至 `MHIv2-F` 始终用完整 ID。运行目录统一为 `outputs/mhi_v2/<实验ID>/<阶段>/<运行ID>/`，禁止覆盖历史运行。工程证据存于 `versions/v2/evidence/<条目ID>/`；P0 的汇总报告例外见下文。

初始状态：P0 待本轮真实报告验收；P1–P5 的工程、训练、精度均为 `not_started`。冻结 v1 的既有历史结果只作参考，不能冒充本轮 A 的复跑结果。

## P0：分档、固定来源与只读资源预检

### [ ] V2-P0-01：建立分档边界

- **前置**：读取现有计划和冻结清单，记录工作区已有修改。
- **动作**：将 v1 冻结登记放入 `versions/v1/freeze_manifest.json`；v2 主计划、原始方案、机器登记和验收清单放入 `versions/v2/`；修正引用，建立独立实现/配置/入口/测试/输出目录约定。用 Git diff 审查本轮变更范围。
- **证据**：`versions/v2/evidence/V2-P0-01/layout.json`（路径映射、迁移来源、Git 状态/差异摘要）；允许将同等信息纳入 `preparation.json` 并在阶段验收记录中定位。
- **通过标准**：新旧文档各有唯一正式归档路径；不存在引用已迁移旧路径的有效内部链接；v1 算法文件无本轮改动；大权重/数据未复制、移动或改写；v2 产物目录与 v1 明确分离。

### [ ] V2-P0-02：核验 v1 冻结与 v2 血缘

- **前置**：V2-P0-01 通过。
- **动作**：核对 `mhi-v1` 指向 `522d9bd5565e88267e72866bd829fb05274762e4`；核对当前实验分支及该基点为其祖先。按清单只读核验正式权重及冻结 train/val/锁定评价清单的实际哈希、条目数与可访问性；区分“已核验”和“清单中仅有历史引用”。
- **证据**：`versions/v2/evidence/preparation.json` 中 Git、冻结清单及资源检查项。
- **通过标准**：标签/祖先关系与登记完全一致；正式权重 SHA256 为 `43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4`；所有必需资源与冻结清单一致。缺失或不一致项必须登记为阻断，不能静默换用模板、其它权重或其它数据清单。

### [ ] V2-P0-03：执行唯一已落地的准备命令

- **前置**：V2-P0-02 的检查来源齐全；只读预检脚本已实现并经静态检查。
- **动作**：实际执行下列命令；脚本只读取仓库/资源并写指定报告，不加载模型、不调用 CUDA、不启动训练、不改写被核验资源。

  ```bash
  python scripts/mhi_v2/preflight.py \
    --baseline-worktree /home/disk1/MHINet-v1 \
    --verify-large-hashes \
    --output versions/v2/evidence/preparation.json
  ```

- **证据**：`versions/v2/evidence/preparation.json`，以及阶段验收中记录的实际命令与退出码。
- **通过标准**：报告是本次实际运行生成；必需检查全部通过，所有检查有明确状态与理由，必需检查为 `passed`，`known_next_task` 仅列后续尚未实现功能；分支、版本路径、资源身份和计划非可执行状态可复核；报告不含模型或训练成功的虚假状态。脚本失败则修复并保留复验记录。

### [ ] V2-P0-04：签收准备阶段

- **前置**：V2-P0-01 至 V2-P0-03 证据全部通过。
- **动作**：更新本清单与机器登记的 P0 状态，记录后续首项为 V2-P1-01；保存分档提交与报告哈希。
- **证据**：`versions/v2/evidence/V2-P0-04/acceptance.json`，引用准备报告及实际提交。
- **通过标准**：`engineering=pass`；`training=not_started`、`accuracy=not_started`；P1–P5 仍为计划。任何必需资源失败则 P0 保持未通过，并列出可执行的修复项。

## P1：独立 v2 契约、Flow0 接口与权重迁移

### [ ] V2-P1-01：登记接口与数值验收协议

- **前置**：V2-P0-04 通过。
- **动作**：审查真实 checkpoint 内嵌 runtime/config，登记 v2 schema、H0/Flow 模式、坐标定义、dtype、有效性和概率契约；预先登记 FP32 比较的绝对/相对容差及测试样本。确认 matchability 实际为概率或 logits，定义唯一转换位置。
- **证据**：`versions/v2/evidence/V2-P1-01/contracts.json`；对应可解析配置进入 `configs/mhi_v2/`。
- **通过标准**：实际 runtime 可解析且来源明确；未知配置字段报错；规划用 `experiment_plan.json` 不被当成训练配置；`flow16_target_norm`、`flow16_norm` 为 `[B,2,H16,W16]`，matchability 为 `[B,1,H16,W16]` 的 `[0,1]` 概率，并有独立 validity。

### [ ] V2-P1-02：实现严格迁移与 v1 回归基线

- **前置**：V2-P1-01 通过。
- **动作**：在 `mhinet/v2/` 实现 v1→v2 权重迁移及版本化加载；显式列举允许新增参数。构造遗漏旧键、额外键、旧张量形状错误、错误 task/schema 的拒绝用例；显式加载归档 runtime，记录实际 mhinet.__file__，避免导入另一个工作区；旧 CLI 映射保持 v1，v2 使用独立入口。记录未改造 v1 的固定输入输出供后续比较。
- **证据**：`versions/v2/evidence/V2-P1-02/migration.json`、固定输入/输出及其哈希。
- **通过标准**：v1 仍 strict 加载；仅 allowlist 中的新增参数允许初始化；其它不匹配全部显式失败，无全局 `strict=False`；旧默认 H0 路径输出在已登记容差内一致。新 checkpoint task 为 `mhi_semidense_v2`，不伪装旧 task。

### [ ] V2-P1-03：拆出 Flow0，跳过 H Fitter

- **前置**：V2-P1-02 通过。
- **动作**：通过 v2 适配接口复用定位预测，分别输出 target、displacement、matchability、validity；Flow 模式保留定位预测并跳过无参数 H Fitter；批次大小从描述子读取。用调用计数或 spy 证明执行路径。
- **证据**：`versions/v2/evidence/V2-P1-03/flow_interface.json`、H Fitter 调用追踪。
- **通过标准**：A→B 方向明确；`flow16_norm = target_norm - source_grid_norm`；Flow 模式 H Fitter 调用次数为 0，H0 模式符合原行为；Flow 推理不读取 `H0_norm`、`stage1_valid` 或预测 H 合法性作为必需输入。

### [ ] V2-P1-04：坐标、形状与异常回退

- **前置**：V2-P1-03 通过。
- **动作**：以 cell centers、`align_corners=False` 检查恒等、已知正负平移、边界与半像素；先插值位移至 D8，再加 D8 网格；x/y 距离分别乘 W8/2、H8/2。检查 NaN/Inf、缺失、越界、低置信度与非方形网格；有效性先清洗再参与算术。
- **证据**：`versions/v2/evidence/V2-P1-04/coordinates.json`、异常用例结果。
- **通过标准**：512 的 D16/D8/D2 为 32/64/256；784 为 49/98/392；恒等/平移误差在 V2-P1-01 容差内；异常不产生 NaN、不拒绝整对影像，回退与有效覆盖率计算明确；禁止以 `0 * NaN` 作为屏蔽。

### [ ] V2-P1-05：真实资源烟测与 P1 签收

- **前置**：V2-P1-04 通过。
- **动作**：只从 train/val 的预登记小样本验证真实特征和实际 runtime；独立用 GT 评估 Flow EPE、有限/边界内覆盖率和 matchability 分布，不拟合预测 H。归档 P1 工程结果。
- **证据**：`versions/v2/evidence/V2-P1-05/smoke.json`、`acceptance.json`，大体积原始输出放入 `outputs/mhi_v2/Flow0/P1/<运行ID>/`。
- **通过标准**：真实输入通过接口契约，无隐含 H 依赖；所有样本保留，指标定义和无有效点策略明确；这里只判工程可用，不判 Flow 精度提升；`engineering=pass`、正式 `training=not_started`，精度仅记录诊断值。

## P2：先建立无 H 的 B，再加入 Flow prior 的 C

### [ ] V2-P2-01：实现无 H 的 MHIv2-B

- **前置**：V2-P1-05 通过。
- **动作**：在 v2 路径保留全局 D8 外观匹配、FP32 dual-softmax 和旧 QRRU；实现 `direct_d8_fine_grids()`：`b = grid[j] + a - grid[i]`。逐项去除训练/推理中的预测 H overlap、`good_h` 与 H 有效性门控，保留 GT 仅作监督/评价。
- **证据**：`versions/v2/evidence/V2-P2-01/no_h_baseline.json`、依赖审计与固定输入。
- **通过标准**：每个 D8 父对应恰有 16 个 D2 子点；恒等 H 情况与旧子格对齐；注入无效/缺失 H0 不改变 B；推理不读取 GT；16×16 fine 匹配、原置信度组合与四轮 QRRU 保持约定。

### [ ] V2-P2-02：验证 B 的 full/chunked 与成本

- **前置**：V2-P2-01 通过。
- **动作**：覆盖 full/chunked、空输入、并列分数、边界、不同 batch；比较候选和 mutual 输出；独立测量 D8 时间与峰值显存，完整登记预热、同步、设备和测量范围。
- **证据**：`versions/v2/evidence/V2-P2-02/b_equivalence.json`、`resources.json`。
- **通过标准**：候选集合一致；并列按预先登记的确定性规则处理；分数/置信度在 P1 已登记 FP32 容差内；无 NaN；512 为 4096²、784 为 9604² 候选，保留分块，资源测量排除 I/O、指标及哈希时间。

### [ ] V2-P2-03：实现 MHIv2-C 的有界几何软奖励

- **前置**：V2-P2-02 通过。
- **动作**：在 D8 外观 logits 上加入 `lambda*c*exp(-d²/(2*sigma_d8²))`；初始 lambda=3、sigma=4 仅作未验证起点。统一 full/chunked 首遍、行列 logsumexp、最佳候选和后续 top-k/mutual 的融合打分；保留双方全图候选。
- **证据**：`versions/v2/evidence/V2-P2-03/prior.json`、逐路径比较结果。
- **通过标准**：`flow_prior=None` 或 lambda=0 与 B 逐点一致；有效奖励在 `[0,lambda]`；异常、低置信度、越界 Flow 仅取消相应 A 查询的奖励；不伪造 B 侧 H mask；full/chunked 的候选和置信度满足同一标准。

### [ ] V2-P2-04：固定权重 B→C 配对诊断与 P2 签收

- **前置**：V2-P2-03 通过；train/val 样本、prior 候选表与失败判据预登记。
- **动作**：完全相同冻结权重、温度、匹配上限和输入，仅开关 prior；记录 D16→D8→D2→QRRU 分阶段指标、正确父对应保留率、NM/覆盖率与成本。若已有 Flow Head 微调，另作同一微调后权重的成对开关，分开报告。
- **证据**：`versions/v2/evidence/V2-P2-04/b_vs_c.json`、`acceptance.json`；原始预测进入各自 B/C 输出目录。
- **通过标准**：配对输入/权重哈希一致，唯一区别是 prior；工程检查全部通过且精度指标可复算。若奖励频繁提升错误候选，停止扩大 lambda，转入独立 Flow 监督诊断；若 D8 改善而 D2 丢失正确候选，登记 D2 瓶颈，不扩大 QRRU 修正范围。不在此阶段宣布正式精度收益。

## P3：Discrepancy 与 MHIv2-D

### [ ] V2-P3-01：独立实现差异编码和零残差注入

- **前置**：V2-P2-04 通过，B/C 诊断未触发未解决的工程阻断。
- **动作**：每轮使用本轮 warp 后的 `wb`，计算通道归一化绝对差；以 1×1 Conv(48,32)+GN+GELU 得到 R，再经零初始化 1×1 DiffAdapter(32,32) 残差注入。DGFO 关闭。
- **证据**：`versions/v2/evidence/V2-P3-01/architecture.json`、参数键/形状与初始化审计。
- **通过标准**：旧 128 通道 encoder 的参数名称/形状不变；保留 4×4 patch、2×2 控制点与四轮更新；R 每轮重算；仅迁移允许的新分支参数。

### [ ] V2-P3-02：零 adapter 的旧输出等价

- **前置**：V2-P3-01 通过。
- **动作**：固定旧权重、输入、采样和分块配置，对比原 QRRU 与迁移后 D；包含空匹配和边界点，逐轮保存控制点、中心、最终点与置信度。
- **证据**：`versions/v2/evidence/V2-P3-02/zero_equivalence.json`、对比张量及其哈希。
- **通过标准**：全部输出在预登记容差内等价，无 NaN；不能只比较最终均值；任何超差先修复再进入训练诊断。

### [ ] V2-P3-03：诊断梯度与 P3 签收

- **前置**：V2-P3-02 通过；诊断用例、有限更新步数及期望梯度路径已登记。
- **动作**：在合成或 train 小样本上做反传/有限步优化诊断。逐步记录 DiffAdapter 和上游差异编码器梯度；零 adapter 首步可能阻断上游梯度，待 adapter 更新后再检查。该诊断独立记录，不冒充正式训练。
- **证据**：`versions/v2/evidence/V2-P3-03/gradients.json`、`acceptance.json`。
- **通过标准**：梯度有限，adapter 获得非零有效梯度，后续步差异编码器梯度可达；旧冻结模块未误更新。`engineering=pass`；正式训练仍未完成；C→D 精度收益留至等预算 P5。

## P4：局部 DGFO 与 MHIv2-E/F

### [ ] V2-P4-01：实现局部传播残差

- **前置**：V2-P3-03 通过。
- **动作**：复用 `interpolate_controls(flow, off)`，保留 ±1 控制点到 ±1.5 patch 的双线性外推；Q/K 来自 R，V 为 2 维局部位移；半径 1 的 3×3 邻域做 masked softmax。将 `E=propagated-local_flow` 经零初始化、无 bias 的 FlowAdapter(2,32) 注入 z，仍由旧 Gate/Down/Head 更新控制点。
- **证据**：`versions/v2/evidence/V2-P4-01/architecture.json`、外推数值用例。
- **通过标准**：无 `F.interpolate` 替代外推，无传播结果拟合/覆盖控制点；无独立 D8 flow update 或 D2 refinement；实现完全处于 v2 路径。

### [ ] V2-P4-02：mask、守恒与退化用例

- **前置**：V2-P4-01 通过。
- **动作**：测试零/常量 Flow、边界、不完整邻域、空邻域与真实采样越界；mask 同时覆盖 patch 邻域与源/目标采样有效性。
- **证据**：`versions/v2/evidence/V2-P4-02/propagation.json`。
- **通过标准**：有效邻域权重和为 1，越界权重为 0；空邻域返回有限值且修正为 0；常量 Flow 的传播残差为 0；初始零控制点的首轮 DGFO 残差为 0；不读取 GT overlap；不把“6px 单分量控制点累计界限”误当所有 patch 采样位置界限。

### [ ] V2-P4-03：等价、梯度、资源与 P4 签收

- **前置**：V2-P4-02 通过；梯度诊断和资源判据预登记。
- **动作**：验证零 adapter 与旧输出等价；有限步诊断后验证后续轮 Q/K 梯度；逐轮保存 E 范数、控制增量、邻居数、各分支梯度，并测量增量时间与显存。
- **证据**：`versions/v2/evidence/V2-P4-03/diagnostics.json`、`resources.json`、`acceptance.json`。
- **通过标准**：等价满足容差，所有张量/梯度有限，后续轮 Q/K 梯度可达，资源不超过预登记上限或超限原因被明确解决。工程通过后才准入正式 E 训练，不能以可运行替代 D→E 增益证据。

## P5：训练协议、等预算 C/D/E、锁定评价与保留决策

### [ ] V2-P5-01：落地训练/评价入口与损失隔离

- **前置**：V2-P4-03 通过；P1–P4 所有工程问题关闭。
- **动作**：实现独立 v2 入口和配置，完成 CLI/配置读取验证。Flow 监督拆成位置、matchability、分类项，完全绕过 H 损失读取/计算；按阶段显式列出可训练参数与 optimizer 组；新建 optimizer/scheduler/scaler，禁止恢复 v1 optimizer。
- **证据**：`versions/v2/evidence/V2-P5-01/interfaces.json`、参数清单与损失依赖审计。
- **通过标准**：不存在 `h_weight=0` 但仍读取预测 H 的路径；Flow Head、原 QRRU、D、E 的训练组可分别构建；冻结参数无梯度且 BN buffer 无意外改变；v2 评价严格加载正确协议。到本项通过后，相关训练命令才可登记为实际可执行命令。

### [ ] V2-P5-02：正式预算与选择规则预登记

- **前置**：V2-P5-01 通过。
- **动作**：只依据 train/val 短程诊断确定正式步数、样本数、batch/累积、optimizer、LR、seed、输入混合、可修正范围、验证频率与选模规则。冻结比较 C/D/E 的共同预算及收益/退化阈值；明确 loss 的筛点不用于推理。初始 1:1 GT+有界扰动/预测 D2 只是候选，需写入最终值。
- **证据**：`versions/v2/evidence/V2-P5-02/preregistration.json`、配置哈希、数据/来源权重/代码哈希、训练与验证分组隔离报告。
- **通过标准**：正式预算非空；C/D/E 的样本顺序、固定 seed、optimizer step 数与选择规则一致；混合比例和误差筛选阈值有值；预登记在锁定评价前完成。GoogleEarth/JL1Flight 不参与调参或选择预算。

### [ ] V2-P5-03：Flow Head 独立监督与诊断

- **前置**：V2-P5-02 通过。
- **动作**：先保留原始 Flow0 直接测试结果；若需训练 Head，只训练定位/可匹配性 Head，冻结 DINO/MVT/VGG/CGMDP/D8/D2/QRRU。按预登记预算运行，检查梯度、BN、EPE/覆盖率/置信度校准；用同一微调后权重成对比较 B/C。
- **证据**：`versions/v2/evidence/V2-P5-03/flow_training.json`；训练产物 `outputs/mhi_v2/Flow0/P5/<运行ID>/`。
- **通过标准**：实际预算、参数更新与冻结策略一致；损失不依赖预测 H；与未训练 Flow0 的区别明确；prior 收益只来自同权重开关比较。若不采用 Head 训练，记录有依据的 `not_applicable` 和固定原始 Flow0 身份，不伪造训练完成。

### [ ] V2-P5-04：冻结 C 的 D2 缓存与 C/D/E 初始化

- **前置**：V2-P5-03 通过或训练项有依据地记为 `not_applicable`。
- **动作**：固定一个 C 模型生成可追溯的训练/验证 D2 缓存，包含预测、有效性和评价标签分离；三组各自从同一旧 checkpoint 独立迁移，禁止从已训练 D 接着训练 E 冒充等预算对照。
- **证据**：`versions/v2/evidence/V2-P5-04/cache_and_initialization.json`、缓存清单/哈希；缓存放入 `outputs/mhi_v2/MHIv2-C/cache/<运行ID>/`。
- **通过标准**：C/D/E 引用同一缓存、来源哈希、seed 和样本顺序；三组初始旧参数完全相同，D/E 仅有登记新增零分支；严重父对应错误单列，不强制作为小窗口可修正标签。

### [ ] V2-P5-05：执行等预算 C/D/E 训练

- **前置**：V2-P5-04 通过。
- **动作**：运行原 QRRU 的 `C/matched_qrru_training`、Discrepancy D、Discrepancy+DGFO E；保留 `C/frozen_qrru`。阶段结束各自保存实际步数、样本数、参数更新、验证选择轨迹和最终 checkpoint；发生中断时登记实际预算与恢复依据。
- **证据**：`versions/v2/evidence/V2-P5-05/training_comparison.json`；各组 `outputs/mhi_v2/<实验ID>/P5/<运行ID>/`。
- **通过标准**：C/D/E 实际完成相同预算；冻结审计通过、无非有限值；checkpoint/config/schema/数据/RNG/游标齐备；未完成或预算不同的组不能进入主对比表。C→D 对照使用等预算 C，B→C 使用冻结 C，二者不可混用。

### [ ] V2-P5-06：验证集选型、多 seed 与阶段停止条件

- **前置**：V2-P5-05 通过。
- **动作**：按登记的 val Pre@3 选型，候选用固定 seeds（建议 42/43/44，最终列表以预登记为准）重复同预算。统计初始 D2 误差 0–1/1–3/3–6/>6px 的修好/破坏比例，并复核 Flow、Direct-D8、DGFO 三类停止条件。
- **证据**：`versions/v2/evidence/V2-P5-06/validation_selection.json`、分 seed 明细与停止条件决策。
- **通过标准**：无锁定集调参；每个 seed 实际预算齐全，按预登记门槛决定是否保留新增模块；E 未重复优于 D 或收益仅来自 NM 增加时，保留 Difference 基线并报告 DGFO 负结果。失败分支可停止，须留证据，不强行推进扩展训练。

### [ ] V2-P5-07：固定协议复核 A–F

- **前置**：V2-P5-06 通过，或已完成有证据的简化模型保留决策；选型/阈值/模型/配置全部锁定。
- **动作**：按冻结清单在 GoogleEarth 498 对与 JL1Flight 1755 对执行 A–F 固定协议评价；A 由冻结 v1 运行或有完整来源的冻结匹配离线重评分，Pre@10 未重评分前留空。F 主对照复用 E 权重，仅关闭 prior；另外训练 F 必须单列等预算运行。
- **证据**：`versions/v2/evidence/V2-P5-07/locked_evaluation.json`；各组逐对结果、匹配及完整实际命令进入独立输出目录。
- **通过标准**：全部配对身份/清单哈希固定，无按新结果删样本；宏平均 Pre@1/3/5/10 以所有输出匹配为分母、无匹配对计 0，目标尺度为存储的 512 像素；同时报告 NM/NCM、覆盖率、失败率、分阶段诊断与无并行干扰的模块耗时/显存。历史结果明确标历史，未实际运行的组留空并说明。

### [ ] V2-P5-08：RMSE/GCP 口径与最终逐项验收

- **前置**：V2-P5-07 证据齐全。
- **动作**：分别报告全部匹配 RMSE、预测 H 网格 RMSE、论文 RMSE；如采用历史论文口径，登记预测点 RANSAC（3px/10000 次/0.999/seed0）和 GT 可见 20×20 网格 SR@3 成功判定，失败对记 10，成功且无正确匹配记未定义。人工审核 GCP 独立成表；汇总三类状态、对照归因、资源成本与是否保留 Flow/Discrepancy/DGFO。
- **证据**：`versions/v2/evidence/V2-P5-08/final_acceptance.json`、`versions/v2/RESULTS.md`（执行后创建），引用全部运行与证据哈希。
- **通过标准**：每个结论能追溯逐对输出，GT 只用于训练标签或预测后的评价，不参与推理 mask；GCP 仅含完成审核点，不把候选标注当真值；工程/训练/精度分别判定，负结果保留。仅全部发布条件实际满足后再讨论 v2 标签或集成 main；可选联合微调须另行预登记，不能追加到本次等预算结果。

## 下一步入口

先完成并签收 **V2-P0-01 → V2-P0-04**。随后只开始 **V2-P1-01**；尚未实现的 P1–P5 命令不得作为本轮“可直接运行的实验脚本”展示。每项完成后，更新该项复选框与机器登记，并由证据明确决定下一项是否解锁。
