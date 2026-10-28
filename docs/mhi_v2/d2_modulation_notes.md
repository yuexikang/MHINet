# D2 调制实验执行 notes（2026-10-26）

后续完整RGB两阶段训练及三路径速度实测见 [full_rgb_d2_notes.md](full_rgb_d2_notes.md)（2026-10-27）。

分支：`codex/flow-d8-d2`。独立模块和训练入口，未修改既有主线默认模型、匹配配置或损失。按 stop-that-shit 范围约束实现当前需求，未增加通用实验框架、哈希清单、额外网络模块或新数据标注。

## 新增文件清单

|文件|作用|
|---|---|
|`mhinet/v2/d2_modulation/__init__.py`|导出模型、匹配和加载入口|
|`mhinet/v2/d2_modulation/model.py`|等大8×8上下文、共享残差MLP、4×4→8×8匹配、无标签推理、权重加载|
|`mhinet/v2/d2_modulation/experiment.py`|冻结特征准备、三组训练/评估、早停、余弦调度、检查点和日志|
|`mhinet/v2/d2_modulation/check.py`|恒等初始化、双向对称、边界掩码、两步梯度检查|
|`mhinet/v2/d2_modulation/report.py`|从已完成结果生成表格、逐对证据及训练曲线|
|`mhinet/v2/d2_modulation/README.md`|执行及无标签在线推理示例|
|`docs/mhi_v2/d2_local_modulation.md`|详细模型、数据、监督、参数和验收计划|
|`docs/mhi_v2/d2_modulation_notes.md`|本执行记录|
|`reports/d2_modulation/results.md`、`curves.png`|结果说明和曲线|
|`reports/d2_modulation/summary.json`、`history.json`|配置、聚合结果和逐轮历史|
|`reports/d2_modulation/data.json`、`pairs.jsonl`|实际数据根目录/划分ID及逐对评估计数|

原有源代码没有修改。输出缓存、训练日志和权重在 `outputs/d2_modulation`，不加入Git；其他任务的已有未跟踪文件没有修改或纳入本次提交。

## 参数和数据

RGB_512：254训练/64验证/64测试，精确合成H。杭州64测试仅作跨域观察。RGB三划分的pair_id和原始source_path无交集。采用前轮固定上游初始化。

A/B均8×8上下文；A中心4×4查询、B8×8候选；128父窗口，每对2048查询。共享768→64→512 MLP、82,496参数、末层零初始化。仅训练MLP，Flow/DINO/MVT/VGG/DeDoDe/D8/温度均冻结，QRRU不运行。

AdamW lr=1e-3、weight_decay=1e-4；余弦退火至1e-5；每步1对、随机64父窗口；裁剪norm=1；最多30epoch；验证P@3连续5轮未提升0.1个百分点早停。初始恒等模型允许入选。损失只有精确H四邻域软标签dual分类。

## 运行命令

在仓库根目录执行，Python环境为 `/root/miniconda3/envs/loma-repro/bin/python`：

```bash
python -m mhinet.v2.d2_modulation.check
python -m mhinet.v2.d2_modulation.experiment prepare --device cuda:0
python -m mhinet.v2.d2_modulation.experiment train --mode none --device cuda:0
python -m mhinet.v2.d2_modulation.experiment train --mode self --device cuda:0
python -m mhinet.v2.d2_modulation.experiment train --mode pair --device cuda:1
python -m mhinet.v2.d2_modulation.report
```

self/pair 两命令可放在两个终端并行执行。本轮GPU0/1可用，GPU2已有其他任务，未占用。执行环境限制导致普通沙箱无法访问CUDA，已在获准的GPU执行环境完成。临时目录根分区空间不足，实际设置 `TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp`，未删除用户文件。

现有组目录不可覆盖。重跑应为 prepare/train/report 一致指定新的 `--output`/`--root`；本轮已完成，无需重跑。

## 耗时与资源

按现有产物时间估算：冻结特征提取446对约52秒；自调制从初始验证完成到最终评估约207秒，双图调制约251秒。两卡并行，正常环境预期约5–10分钟，受存储和GPU共享情况影响。两组实际提前停止，因此不是满30轮耗时承诺。

FP32局部特征缓存约5.6GB；单模块权重约0.33MB。训练进程只加载窗口和MLP，不加载原网络。两组完整训练及评估无CUDA OOM、非有限损失/梯度或Traceback。没有记录训练峰值显存，故不提供未经测量的GB数字；显存可运行性由GPU0/1上的完整执行证明。

## 验证与验收结果

- PASS：同一入口内恒等初始化相似度、坐标逐值相同。
- PASS：A/B交换对称；无效上下文值不影响有效位置调制。
- PASS：两步分类反传后两层MLP梯度有限且隐层梯度非零。
- PASS：加载真实已训练检查点，在真实缓存数据上输出128×16个有限坐标。
- PASS：完整训练、日志、验证、保存/加载best、两域最终测试已执行，超过最小smoke要求。
- self：22轮早停，最佳第17轮；pair：27轮早停，最佳第22轮。实际训练更新分别5588、6858次。
- 日志：`outputs/d2_modulation/{prepare,none,self,pair}.log`。
- 权重：`outputs/d2_modulation/{self,pair}/{best,last}.pt`。

|方法|RGB验证P@3|RGB测试P@3|杭州测试P@3|
|---|---:|---:|---:|
|不调制|82.38%|81.66%|58.07%|
|自调制|88.42%|87.69%|59.60%|
|双图调制|88.59%|87.92%|59.66%|
|同点直接Flow|87.20%|85.86%|58.91%|

双图调制在RGB测试上相对基线+6.26个百分点、相对Flow+2.06、相对自调制+0.23。输出点数相同，未靠筛点抬高P@3；具体净救回、候选排名和EPE见报告。验证样本结果不代替测试结论。杭州是近似H参考一致性。

当前none与前轮诊断在131072个RGB验证查询上相差1个阈值正确点；本次三组统一以当前none比较。不能将这一微小跨入口差额解释为模型收益。

## 尚未做

未做多seed重复、扩大训练集、其他真实数据独立人工真值验证、D8调制、原网络联合解冻或部署性能评测；未将本模块和新权重合入上一主线。这些不属于本次独立模块落地及最小验证的必需项。跨图条件比自调制仅小幅提高，需要后续重复才能判断稳定性。

## 后续训练必须注意：早停与最优权重分离（2026-10-27）

用户明确要求：本次只记录，不执行截图中的再训练任务，不修改当前训练代码。

现有训练把早停的 min_delta=0.001 同时用于 best.pt 保存，导致已保存 best 不一定对应训练历史中最高的验证 P@3；早停也未证明已经充分收敛。之前报告中的 best 应理解为“现行阈值规则选中的权重”，不能称作历史最高或最终最优结果。

下次训练应将两项逻辑分开：验证 P@3 严格超过历史最大值就保存 best；早停另用自己的 min_delta/patience。优先安排相同固定预算（例如先完整30轮）的对照，再根据验证曲线判断是否延长；不能用测试集调整早停。仅以RGB验证选权重，确定后再统一测试。旧检查点没有优化器和调度器状态，不能把从旧last出发的新训练称为严格无缝恢复。保留原结果，不覆盖旧实验。以上为后续执行约束，本次未实施或启动训练。

## 取消 Flow 的推理对照（2026-10-27）

此次仅评估，不训练、不修改早停或主线默认行为。上文“尚未做”指原254对实验完成时的状态；1024对扩量结果已在 `reports/d2_modulation_expanded/results.md`。本次统一使用扩量 full_pair 的已保存 epoch21 权重，不将其称为历史最高验证权重。

核对结果：`mhinet/v2/coarse_matcher.py` 已支持 `flow_prior=None`，其无先验全局 dual-softmax 与 LoMA 参考实现兼容，因此两份 coarse_matcher 均无需改动。独立入口直接复用现有实现，且在每个评估划分首对图像上比对 LoMA 的源/目标索引和置信度。

三组保持同一套 DINO/MVT/VGG/DeDoDe、温度、双图 D2 调制权重，QRRU 禁用：

- `local_flow`：原有 Flow 局部 D8 搜索和连续坐标解码，原 D2 调制推理。
- `global_flow`：完整4096×4096 D8候选，互选后最多128父窗口；加入既有 Flow Gaussian 奖励。
- `global_no_flow`：同样的全局 D8 搜索，不计算 Flow、不使用 Matchability 或几何先验。保留 MVT 特征。

两种 global 使用相同的峰值周围3×3连续坐标解码，再进入原8×8上下文、A中心4×4/B全部8×8 D2匹配，最多2048点。未使用H选择窗口或筛选输出；H只用于最后计分。全局与局部组的搜索范围、父点选择和候选网格不同；不能把两者全部差额仅归因于取消Flow。global两组的差异才直接反映该全局路径下Flow先验的作用；即便点数相同，源点集合仍可能不同。

指标使用512原图像素。P@1/3/5/10以全部输出点为分母，无可见对应的点不计正确；EPE仅统计真值有效且可见的点。杭州仅为近似H参考一致性。该结果是对既有权重做推理消融，不代表无Flow模型重新训练后的上限。

### 文件和执行命令

- 新增 `mhinet/v2/d2_modulation/no_flow.py`：独立冻结评估、LoMA对照、无Flow执行检查、聚合及逐对计数。
- 修改 `docs/mhi_v2/d2_modulation_notes.md`：本记录及前述用户要求的早停提醒。
- 结果 `reports/d2_modulation_no_flow/`：结果表和三个划分的完整JSON计数。
- 本地日志/单对smoke：`outputs/d2_modulation_no_flow/`；smoke日志为 `outputs/d2_modulation_no_flow_smoke.log`。

仓库根目录运行；三条完整评估命令可分别在三个终端执行：

```bash
export TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp
PY=/root/miniconda3/envs/loma-repro/bin/python
$PY -m mhinet.v2.d2_modulation.no_flow --split test --device cuda:0 --limit 1 --output outputs/d2_modulation_no_flow/smoke
$PY -m mhinet.v2.d2_modulation.no_flow --split test --device cuda:0
$PY -m mhinet.v2.d2_modulation.no_flow --split extra_test --device cuda:1
$PY -m mhinet.v2.d2_modulation.no_flow --split hangzhou_test --device cuda:2
```

默认 selection 为 `outputs/d2_modulation_expanded/selection.json`，adapter 为 `outputs/d2_modulation_expanded/full_pair/best.pt`。上游初始化沿用 `qrru_corners.experiment.load_models()`：原清理杭州检查点、`weight/mhi_v2_u0.pt`定位器和 `outputs/flow_d8_d2/training/best.pt` 的VGG/DeDoDe/温度。复现需本地权重及对应数据。重跑可用 `--output` 指定新目录。

### 本次结果与验收

P@k 的阈值和 EPE 均为512原图像素；P@k分母为全部输出，EPE只统计有效可见真值。每组每对均2048点，源点集合不一定相同。杭州指标只反映近似H的一致性。

|数据|推理路径|P@1|P@3|P@5|P@10|EPE(px)|
|---|---|---:|---:|---:|---:|---:|
|RGB 原测试64对|现有 Flow 局部 D8|39.29%|89.35%|97.21%|99.61%|1.550|
|RGB 原测试64对|全局 D8＋Flow 先验|27.35%|85.02%|96.60%|99.49%|1.839|
|RGB 原测试64对|全局 D8，不预测 Flow|26.37%|82.83%|94.99%|98.52%|1.906|
|RGB 扩展测试192对|现有 Flow 局部 D8|37.77%|88.10%|96.57%|99.30%|1.608|
|RGB 扩展测试192对|全局 D8＋Flow 先验|26.83%|84.50%|96.20%|99.34%|1.872|
|RGB 扩展测试192对|全局 D8，不预测 Flow|25.92%|82.36%|94.60%|98.35%|1.936|
|杭州64对|现有 Flow 局部 D8|22.71%|59.85%|73.10%|87.40%|11.874|
|杭州64对|全局 D8＋Flow 先验|25.35%|58.08%|72.14%|86.74%|11.905|
|杭州64对|全局 D8，不预测 Flow|25.38%|58.01%|72.22%|86.54%|12.144|

## 结论

RGB合并256对：无Flow P@3 82.47%，全局＋Flow 84.63%，现有局部Flow 88.41%。在相同全局路径下，Flow先验增加约 2.15 个百分点。

杭州取消Flow仍能匹配，但P@3从现有局部方案59.85%降到58.01%。全局路径加Flow仅提高约0.07个百分点。杭州无Flow的P@1反而高于局部方案，因此不能声称局部Flow在每个精度阈值都更好。

现有局部方案与全局方案还改变了搜索范围、父点选择和候选采样网格；它们的全部差额不能归因于是否预测Flow。适配器原本在Flow局部窗口上训练，本轮是固定权重的推理消融，不能据此断言无Flow重新训练后的上限。

## 验证与资源

单对smoke通过；三个完整进程均退出0，输出坐标有限，无OOM。每个划分首对与LoMA无先验D8实现的源/目标索引及置信度一致。每对无Flow前向均禁止调用Flow decoder，并检查flow_head=0；QRRU调用被禁止。现有local_flow复算的三个划分P@3正确点数与原缓存全部一致，EPE差异小于3e-8px。

- RGB 原测试64对：评估循环 13.07s，进程峰值 allocated 显存 1.720GiB。
- RGB 扩展测试192对：评估循环 33.38s，进程峰值 allocated 显存 1.724GiB。
- 杭州64对：评估循环 14.81s，进程峰值 allocated 显存 1.737GiB。

上述耗时不含模型加载，显存包含同进程的三组对照，不能当作单独无Flow部署开销。运行命令及文件说明见 `docs/mhi_v2/d2_modulation_notes.md`。未重新训练、未合并主线，本轮请求已完成。
