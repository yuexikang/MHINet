# D2 调制实验执行 notes（2026-10-26）

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
