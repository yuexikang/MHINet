# MHI-v2 Flow0 → D8 → Pair-D2 主线整合验收

日期：2026-10-28。以 `codex/mhi-v2@acdbbfa` 建立 `codex/mhi-v2-flow-d8-d2-integration`，选择性移植 `codex/flow-d8-d2@c6d6ec4`。研究选择依据归档自 `codex/cgmdp-global-modulation@22a947e`，另外保留随后完成的 A3 无训练诊断。没有将实验分支整条历史合并，没有训练或改写模型权重。

## 结构与交付

原入口以 U0 为基础，后续是全局 D8、离散 D2 和 QRRU 兼容路径。新增正式 `load_mhi_v2 → model.match`：DINO/MVT → Flow0 与 VGG/CGMDP → 离散局部 D8 父点选择 → 连续 D8 中心 → 双侧 8×8 上下文 Local Pair、A 中心 4×4、B 全 8×8 → 16×64 Dual-Softmax → 连续 D2 坐标。最多 2048 点。

训练/推理共用 `pair_matching` 的 coarse、continuous、local、losses，未搬入一次性调制训练、QRRU实验或诊断脚本作为正式推理依赖。保留独立全局 No-Flow 控制函数供两阶段训练原评测协议调用，默认模型不用它。父点直接复用 local_coarse，跳过旧 Fine；没有合并离散/连续两次 D8 搜索。

默认没有 H0 fitter、QRRU、DGFO、全图 Cumulative/Delta 或新增拒配决策。原 U0、H0、QRRU 和旧训练配置保留不变。A2/A3/A4、Self、四角监督等只保留实验来源与报告，不计入主线模块。匹配输出带 valid、parent、parent_ids、confidence，供后续几何图/QRRU 接口使用。

## 权重、选模与可复现入口

默认 Flow epoch 4 best + D2 epoch 1 best；D2 latest epoch 10 可独立选择，始终配 Flow best。Flow 的两个服务器副本 SHA256 一致。加载器按模块严格装载，不把 D2 当作整网检查点；核对 hash、阶段、轮次及其记录的 locator。默认选择依据是 RGB 验证 P@3，而非 RGB/杭州测试成绩。

源实现归档提交 c6d6ec4，训练入口提交 224fbd0，选择优化提交 afa1f52。原 checkpoint 未内嵌 Git SHA，不能证明训练进程当时恰好对应某个干净提交。权重清单明确标记这一来源限制，并记录实际文件哈希与 checkpoint 字段。

模型配置、完整权重清单、推理和训练命令：[使用说明](../../docs/mhi_v2/flow_pair_release.md)。默认配置没有服务器绝对路径；权重与 DINO 留在服务器，不提交 Git。原 U0 未覆盖或删除。

## 等价性与兼容性

- 同一 GPU2、同一 Flow/D2 best、RGB 16 + 杭州 16 对：D16 Flow、D8 父点 ID/顺序、连续中心、窗口描述子/掩码、候选 log-probability、最终坐标/置信度/正确计数逐层核对。
- 确定性 CUDA：32/32 零容差通过；全部最大张量误差为 0，父点顺序一致。
- 普通 CUDA：1/32 父点顺序改变，集合一致；按整组父点对齐后所有张量最大误差为 0，正确计数一致。原因定位为 sparse scatter 归约影响近同分父点排序。正式路径未强制改用确定性模式。
- 第一版参考采集曾为记录父点再次调用选择函数，导致 1 对参考元数据与已采样窗口顺序不一致；已改为直接从原窗口源坐标确定实际父点，并保留真实模型输出不动。最终零误差结论基于修正后的对应关系。
- 90 项 v2 测试通过：新加载/无 GT/输出形状/空匹配/父点关联/独立续训初始化，以及原 H0、Flow、QRRU 等测试。
- 修正一项既有陈旧测试：原测试在主线已改默认 8×8 后仍假设目标 4×4。现在显式测试 target_side=4 的历史等价性，同时新增默认 8×8 中心子窗口与旧 4×4 的一致性；没有删除测试或改回主线扩窗。
- 真正的 Flow、D2 epoch 10 latest 可恢复 AdamW 张量状态、参数组、调度器状态（102130 step）及 CPU/Python RNG；本项在 CPU 检查，CUDA RNG 保存字段及恢复代码保留，未执行 CUDA 恢复后的优化步。优化器步数实际执行 0。
- U0 实际权重 MVT/head 严格加载通过。D2 latest 通过 RGB/杭州各 1 对真实无 GT 前向及正确计数比较，D16 Flow 与 best 版本逐位相同；没有重跑 latest 的完整测试，完整旧结果已归档。

## 完整测试回归

实际重新执行 RGB/test 5106 对、杭州/test_real 235 对，全部使用新正式默认模型。单位为 512 原图像素，P@k 分母为所有 D2 输出，EPE 仅用可见已知点。杭州是近似 H 参考一致性。

| 数据 | 输出点数 | P@1 | P@3 | P@5 | P@10 | EPE | 用时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB 5106 | 10457088 | 90.202349% | 98.338132% | 98.798729% | 98.887090% | 0.697979866 | 679.8s |
| 杭州 235 | 481280 | 22.301155% | 62.682015% | 78.652136% | 89.572390% | 11.695726241 | 32.0s |

峰值 PyTorch allocated 显存约 1.65GiB。GPU0/1 原任务保持运行；此用时是本机此次含逐对数据加载的测试记录，不是独占环境吞吐承诺。

**全部 5341 对的 n、visible、correct1/3/5/10 与原逐对报告完全一致**，不仅是聚合百分比一致。

### EPE 标量差异的具体核查

RGB 共 9 对、杭州 3 对的 FP32 `error_sum` 差 0.000061–0.000122px。合计造成 RGB EPE +2.95×10⁻¹¹px、杭州 −2.64×10⁻¹⁰px；不是精度计数变化。逐对差异全部保存在 JSON。

对这 12 对固定共享特征重复推理各 32 次：按源点对齐后的坐标始终逐位相同；规范顺序 FP64 累加的误差和始终唯一。11 对观察到了父点换序、不同 FP32 误差和，并实际重现旧报告的数值，定位到父点顺序及后续 FP32 累加。

剩余 RGB 索引 2738，在额外 32 次交替调用原 infer 内核/正式包装时，坐标仍完全一致，FP64 误差和稳定为 1060.9183525955305，当前 FP32 为 1060.91845703125；历史标量 1060.9183349609375 未再次出现。**这一历史 FP32 标量没有逐位复现**；原报告未保存该对当时的完整坐标和归约顺序，不能倒推出其精确舍入路径。当前新旧实现输出一致，差异仅留在历史 FP32 误差和的末位；保留这项限制，不宣称全量 EPE 逐位复现，也没有更改阈值或指标来隐藏差异。

## 训练复现与后续边界

保留原两阶段损失、冻结规则、参数组、best 选择、余弦调度和完整 epoch 恢复。独立 `--resume --stage flow/d2` 直接使用该阶段 latest，不再要求已经不参与恢复的初始 downstream/adapter 文件；D2 仍需对应 Flow 和冻结 DINO。原任务有 20→10 epoch 调度重设，完整历史如实归档。修改总 epoch 是显式新调度预算，不能称为原调度无缝续跑。

当前没有人工点/遮挡 GT，也未改 Matchability 标签或训练损失。QRRU/几何图如何接入、拒配阈值如何选择，都属于后续研究；本次不自动开启。A0–A4 数据和权重与全量实验不同，不以其 P@3 直接比较完整模型。

复现命令、运行条件见使用说明；逐层核对、恢复、完整计数、归约审计文件均在本目录。大窗口参考张量与逐对全量输出留在本地 `outputs/mhi_v2_integration/`。

## 文件清单

以下文件相对于原主线迁移/新增或修改；历史实现文件 feature_provider.py、config.py、semidense.py、u0.py 未改动。

完整逐文件列表见 [files.txt](files.txt)。核心变更：

- `mhinet/v2/pipeline.py`
- `mhinet/v2/pair_matching/`
- `configs/mhi_v2/releases/`
- `docs/mhi_v2/flow_pair_release.md`
- `mhinet/v2/README.md`
- `tests/v2/test_pair_pipeline.py`
- `tests/v2/test_flow_matching.py`
- `tools/mhi_v2/`
- `reports/full_rgb_d2_10epoch_20261028/`
- `reports/cgmdp_residual_research/`
- `reports/cgmdp_selection_optimization/`
- `reports/cgmdp_a3_conditioning/`
- `reports/mhi_v2_integration/`
