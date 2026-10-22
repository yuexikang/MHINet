# D16 对应判别实验

分支：`codex/d16-correspondence-ablation`。只运行 DINO → MVT → Flow0；不执行、修改或训练 D8/D2/QRRU。

|组|GPU|训练内容|监督|
|---|---|---|---|
|A|三组 epoch 0 共用初始化评估|冻结定位器基线|不训练|
|U|0|预训练 MVT、MatchEmbedding/Flow Head 微调|原 flow_head_loss 原样使用|
|B|1|同 U|原 Geo + 区域位置分类 + Top16 难负候选 + 正查询可匹配性|
|C|2|同 B|再加入完全擦除 B 图的受控无匹配标签|

三组使用同一初始化、seed=42、清理后的 254 训练/64 验证影像对。DINO 冻结；MVT LR=1e-5，头 LR=5e-5，AdamW decay=.01、累积4、裁剪1、余弦退火至1e-6。固定30 epoch，不采用各组独立早停，保证比较预算一致。按共同验证 Flow EPE 保存 best；同时保存 latest 和第15轮 middle（仅 MVT/head，不含冻结模块）。

B/C 位置权重 .1、难负权重 .05、可匹配性 .01。难负项是 Top16 错误候选 log-mean-exp 与最大正候选的 softplus margin，margin=log(2)。杭州近似 H 使用候选16px单元与不确定性区域相交形成正集合，最大概率候选排名也是相对于这个集合。精确 H 使用四邻域双线性软标签。Geo 调用旧函数且保留旧有效域，便于隔离分类变化。

杭州没有人工真值点、真实遮挡标签或真实稠密 Flow。真实图像中 H/overlap 失败的位置在 B/C 中忽略，不当作确定无匹配；有效图像边缘不因位于最外 D16 中心之外而直接作为负查询。U 故意保留原监督作为对照，所以 U/B 的差异还包括忽略规则，不能把全部变化归因于分类权重。

每四个正常训练样本，三组额外读取同一个完全擦除 B 的输入；U/B 的这次前后向损失乘零，C 训练 dustbin 与不可匹配。正常样本照常训练。这个 C 测试只代表容易辨认的受控空图无匹配，不能证明真实遮挡性能。验证空图固定每组首对，共4对。

逐轮输出 point-weighted EPE、Flow-P@1/3/5/10、正区域 Rank@1/5/10、难负混淆率、EPE>8/16/32 和分组统计。legacy_epe 保留旧中心边界、逐对平均，用于和之前35.93px比较；新EPE的有效域/聚合方式不同。杭州所有这些指标仅代表近似 H 一致性。

第0/1/15/30轮固定每组首个验证样本输出 G1/G2/G3 图和最差8个有效查询的 Top10 候选、概率、正区域排名。残差箭头绘在 B：起点是参考 H 位置，终点是预测位置，单位1:1像素；红色候选在正区域外，绿色在区域内。这些是固定样本，未按成功结果挑选。

运行（仓库根目录，环境 loma-repro）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m mhinet.v2.d16_experiment.train --variant U
CUDA_VISIBLE_DEVICES=1 python -m mhinet.v2.d16_experiment.train --variant B
CUDA_VISIBLE_DEVICES=2 python -m mhinet.v2.d16_experiment.train --variant C
```

输出：`outputs/d16_correspondence_20261022/{U,B,C}`。`epoch_000.json` 是 A 基线；`status.json` 看进度，`epoch_NNN.json` 看完整验证指标。原 `mhinet/v2/losses.py` 及正式训练入口均未改动，精确合成 H 的现有行为保留。
