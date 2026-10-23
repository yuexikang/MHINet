# PCF-01：杭州 Partial Correspondence Flow0

原始任务见 [PLAN.md](PLAN.md)。实验独立于正式 U0，原默认推理不变。

## 数据和边界

- 唯一新增母图来源：杭州原始 `HangZhou_cloud0_rgb` 中的天地图 RGB 2048² TIFF。没有 GoogleEarth。
- 只取清理后 RGBMatch_v4 训练清单中仍有配对使用的母图；读取实际 GDAL mask，不相信残留的第 4 波段标签。母图全部有效、纹理非恒定、完整落在一个 8192m 地理块内。
- 70 个独立地理块，各取一张母图，先划分 56/7/7（80%/10%/10%），再生成裁剪。校准、评价各 7 块排除旧 U0 实验记录的所有杭州地理块；训练集可能包含旧训练块，逐母图记录。此隔离针对已记录实验，不宣称 DINO 预训练影像完全未见。
- 每张母图：一个同几何全重叠对、四个 20/40/60/80% 部分重叠对、四个对应纹理覆盖版本。全部输出 512²；部分重叠旋转 0–15°，缩放 0.9–1.1。裁剪 H 来自生成参数，两个方向都计算标签。
- 同 split 的另一地理块提供纹理，覆盖 B 的 5–20%；禁止跨划分借用供体。几何非重叠仍保留自然影像，完整 16×16 支持域和变换后支持域判断正负，边界保守忽略。
- 负例来源：`negative_kind=1` 为几何图外，`2` 为原本完整可对应的支持域被纹理覆盖，`3` 为反向供体覆盖了原几何边界。纹理专项指标只把 `2` 作为负例；完整统计保留全部已知负例。
- 实际重叠率按逐像素映射落入目标范围的有效面积计。保存双向 H、裁剪变换、双向有效/重叠掩膜、供体、覆盖区域。
- 旧杭州 G0–G3 开发图对只作参考诊断：`gt_state=UNKNOWN`，精确 GT 坐标/EPE 为空，近似 H 指标单独放 `reference_*`。无真实遮挡精度声明。

## 模型与预算

- 原 `u0.load_u0`，DINO/MVT/MatchEmbeddingDecoder/Dustbin 全冻结，`pyramid_scales=()`。
- 输入特征：L2 归一化的 MVT A 查询特征 + Dustbin 概率、Matchability、最大空间 logit、Top1–Top2 logit 差、空间条件熵，共 1029 维，全部 detach。logit 由 softmax 比值与原 Dustbin logit 精确恢复。
- 独立 MLP：1029→128→LayerNorm→GELU→1。分别归一化正负 BCE，UNKNOWN 忽略；没有坐标、循环、平滑或下游损失。
- 共享 FP32 冻结特征缓存，AB/BA 都执行原 U0。训练不重复执行 U0；T1/T2 每 epoch 使用同样数量的缓存图对、分类头前向和优化步数。
- T1 全部用 clean partial；T2 每 epoch 一半用相应纹理版本，下一 epoch 交换，两个方向同版本。相同初始化，AdamW 1e-3/0.01，batch 8，最多 15 epoch。
- 两组共同校准损失的均值用于 patience=3 早停，并共同取同一个最佳 epoch，避免不同训练预算干扰归因。该方式是对方案公平预算要求的具体实现。

## 校准与验收（运行前固定）

- A0=`p_dustbin`；A1=`1-matchability`；A2=`1-matchability*(1-p_dustbin)`，固定组合，不增加拟合参数。
- 每种方法都在校准集全重叠、几何、纹理的真实正例上求 99%/99.5% 上分位点，再分别取三类最大值作为 low/high。high 向上移动一个 float，保守处理分数相等。分数饱和时可能不产生 NO_MATCH，不隐瞒该失败。
- 真实正例的 `NO_MATCH + UNCERTAIN` 均算误拒。无对应召回只计 `NO_MATCH`。
- 阶段 A 仅看校准：若有既有方法几何召回≥95%、纹理召回≥50%，且各类误拒≤1%，不训练。否则启动计划内两组分类头。
- 最终独立评价：各类正例误拒≤1%；纹理召回≥50% 且比最强同协议基线高≥15pp；几何召回不低于 A1 超过0.5pp；原始坐标 P@3 完全一致；各类预测 UNCERTAIN 比例≤5%。原方案的“不得大面积不确定”具体化为最后一个门槛。
- 所有误差均为 512 原图像素。评价集只作最终报告和验收，不改阈值或追加组合。

## 使用

环境：`/root/miniconda3/envs/loma-repro/bin/python`，新增生成 TIFF 数据的依赖为 rasterio。

```bash
python -m pytest mhinet/v2/partial_flow/test_partial_flow.py -q
python -m mhinet.v2.partial_flow.dataset
# 三个独立进程，分别使用三张卡
python -m mhinet.v2.partial_flow.diagnostics --shard 0 --shards 3 --device cuda:0
python -m mhinet.v2.partial_flow.diagnostics --shard 1 --shards 3 --device cuda:1
python -m mhinet.v2.partial_flow.diagnostics --shard 2 --shards 3 --device cuda:2
# 三个缓存任务都完成后
python -m mhinet.v2.partial_flow.evaluate --stage-a
python -m mhinet.v2.partial_flow.train
python -m mhinet.v2.partial_flow.evaluate
python -m mhinet.v2.partial_flow.report
```

数据、缓存、逐查询 CSV、权重和全部实验结果位于 `outputs/pcf01/`。中文报告、核心指标、逐对结果与图片同时整理到 `reports/pcf01/`。

`inference.predict` 返回 `(conditional, dense_u0)`。`conditional` 只发布 MATCH 坐标；`accepted_indices` 是 `batch*1024+row*32+column` 的全局索引，可还原批次。三态数值 MATCH=0 / NO_MATCH=1 / UNCERTAIN=2。原稠密结果仍在 `dense_u0` 中用于诊断。本轮不减少原 Decoder 运算量。
