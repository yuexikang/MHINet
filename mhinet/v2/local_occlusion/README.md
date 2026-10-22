# D16 Flow0 局部遮挡实验

分支 `codex/d16-local-occlusion`，基于 `codex/d16-correspondence-ablation` 的 `ba28a1b`。用户方案保存在 [PLAN.md](PLAN.md)。旧 U/B/C 结果及网络文件不修改。本目录仅调用 DINO → MVT → Flow0，`pyramid_scales=()`，不运行 D8/D2/QRRU，也无 H0 预测。

## 阶段一

`diagnose.py` 加载旧 U/B/C 的 **latest.pt / epoch 30**，输出所有64对、每对1024查询的NPZ；保存Top10、概率熵、正区域概率质量、边界距离、预测偏差、H理想概率输入Decoder后的坐标。`summary.json` 按排名、置信度、边界、误差分箱。`report.py` 同时比较三组共同Rank=1查询，避免子集不同造成混淆。

H引导的分布是诊断性反事实输入，不是新增真实标签，也不用于推理。旧Decoder具有7×7 D16源网格感受野（3层3×3卷积），理想概率输入改变了邻域与概率分布，结果不能直接归因于单点Decoder。

固定3对G3保留在正式统计；另给排除后的诊断统计、原始H/尺度旋转元数据、掩膜和叠图。没有独立标注，不能只凭H一致性将错误自动认定为标签问题。

## 阶段二：配对实验

|实验|清洁损失|增强损失|GPU安排|
|---|---|---|---|
|N1|旧U原损失|局部遮挡|0|
|N2|旧B分类增强损失|同N1|1|
|U0|旧U原损失|相同增强输入，损失乘零|2|
|B0|旧B分类增强损失|相同增强输入，损失乘零|2，U0之后|

GPU2先对旧U/B/C第30轮做同一套局部遮挡验证。四个新训练从原预训练MVT/定位头初始化，**不从U/B/C第30轮继续训练**；DINO只加载原权重并冻结。初始化来源记录在每组initialization.json。

seed=42、相同254训练/64验证清单、30轮、累积4、AdamW decay=.01、裁剪1、MVT LR=1e-5 / Flow Head LR=5e-5、余弦退火到1e-6。每对clean加一次增强前向，所有组都有每轮508次训练前向和64次更新；旧实验为318次前向，因此新实验比旧实验增加59.7%训练前向，不能把计算增益全归于负例。各组记录实际次数和墙钟时间，没有宣称测量硬件FLOPs。每轮首对样本记录各项未裁剪、累积前的MVT/head梯度L2范数；控制组也执行相同梯度探针。

保存best_p3（按清洁P@3）、middle（15轮）、latest；正式对照统一第30轮。没有按验证情况改动训练集。

## 矩形与标签

- 仅B，1个矩形；面积在5–10%、10–15%、15–20%三个层等概率随机，长宽比对数均匀0.5–2。黑白各50%，原RGB中0/255。加载器产生RGB[0,1]，遮挡发生在编码器归一化之前。
- seed + pair_id + epoch确定随机序列（CRC32仅用于稳定随机种子，不用于文件校验）。一半尝试以H可见查询为中心，一半图内均匀放置。矩形可贴边，但不出图；实际整数像素面积记录。
- 每个目标可信支持框半径=8px D16半单元 + soft_radius_B_px + 2px缓冲。查询原本known/visible，支持框完整位于B图内且所覆盖的原始B有效掩膜像素全有效，才可赋增强标签。
- 支持框完全在矩形外为Positive；完全在矩形内为Negative；相交、出界、原始不可信为Ignore。没有把黑白块附近全部标成负例。
- 至少4个可信负查询；最多8次矩形重采样，仍不足则跳过本次增强监督（所有增强查询Ignore），记录原因。clean照常训练，不扩大H可信范围。

## 损失及训练前选择

`L = L_clean + .01 L_occ + .01 L_keep + 1.0 L_keep_geo + .005 L_keep_match`。

- `L_occ = mean_N(-log dustbin) + .1 mean_N(-log(1-matchability))`。
- `L_keep = mean_P(-log 正区域概率总质量)`。
- keep_geo使用原鲁棒Geo公式及杭州H不确定死区，但仅针对增强后Positive；keep_match仅针对Positive。
- 各项分别按有效查询数平均；Negative/Ignore没有Geo监督。

方案文字要求未遮挡正查询保留坐标/Matchability，因此在其简写的L_keep分类公式之外，显式列出keep_geo与keep_match，避免隐藏权重。清洁损失保持所属基线原样。训练前首样本梯度检查发现occ/keep权重为.1时对MVT梯度明显高于U clean，因此降到.01；正式四组均使用固定后的同一配置，未根据验证结果调权。

N1相对U0不仅加入负例，也加入了增强正区域的保持损失；若有效，仍需单独移除keep/occ的后续消融才能将收益精确归因。

## 评价与预登记门槛

每轮clean64对及独立固定seed=20261023的局部遮挡64对；mask epoch固定0，因此模型/轮次之间严格配对。0/15/30轮额外用同一64对评估5/10/20%黑、白六种条件和10%未见灰色。黑白同面积使用相同矩形，记录actual_area。全量统计包含重采样失败记录，Positive/Negative指标仅在实际可监督查询上计算，按查询数聚合。

报告clean P@1/3/5/10、EPE/长尾、Rank@1/5/10、分组和误拒绝；增强负例dustbin Top1、负例Matchability分位数、Positive P@3/EPE/误拒绝、边界误差、AUROC。误拒绝主指标是dustbin Top1或matchability<.5，同时提供两项拆分。AUROC以可观察Positive为正类，score=matchability，忽略Ignore。

门槛在训练前固定：clean P@3下降≤0.5pp；负例dustbin Top1比对照至少增加10pp；clean及未遮挡Positive误拒绝增加≤0.5pp；EPE>32px增加≤0.5pp；G1/G3各P@3下降≤1pp。这些是诊断门槛，不是统计显著性声明。

`report.py` 在最后一个控制组完成后自动生成RESULTS.md、acceptance.json及逐对差值。任何未通过者不并主线；不自动启动N3。即使通过，也须另设未参与选模的测试数据并复核额外seed，才可选最终方法。R1/R2保持独立，当前没有改Decoder或引入合成训练。

## 运行与输出

环境 `/root/miniconda3/envs/loma-repro/bin/python`；从仓库根目录运行：

```bash
python -m mhinet.v2.local_occlusion.test_occlusion
python -m mhinet.v2.local_occlusion.diagnose --variant U  # B/C同理
python -m mhinet.v2.local_occlusion.evaluate_baselines
python -m mhinet.v2.local_occlusion.train --variant N1  # N2/U0/B0同理
python -m mhinet.v2.local_occlusion.report
```

输出 `outputs/d16_local_occlusion/`：

- `diagnostics/README.md`：阶段一报告；U/B/C子目录包含全查询NPZ、JSON、局部/审计图。
- `original_baselines/{U,B,C}/epoch_030.json`：旧模型在新固定遮挡条件上的对照。
- `{N1,N2,U0,B0}/`：实际配置、初始化、逐轮CSV、配对clean CSV/occlusion JSONL、训练标签/损失/梯度记录、状态、权重。
- 0/15/30轮图包含原始A/B、修改B、三态查询掩膜、dustbin、Matchability、正查询Flow残差与EPE。
