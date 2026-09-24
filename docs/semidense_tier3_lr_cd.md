# 第三档 C/D 学习率对照（2026-09-24）

两组直接继承第一档 C 最终第12995步完整模型，包括共享特征、匹配温度与 QRRU。使用 stable_v2 第三档，seed0、物理BS1×累积4，各训练一轮34652对/8663步，完整val3848对。冻结 DINO、MVT、GHIM/H0 head；训练 VGG、CGMDP、匹配温度和 QRRU，监督 Lc+Lf+Lq。

| 组 | GPU | CGMDP/VGG 初始LR | 下游初始LR |
|---|---:|---:|---:|
| C | 0 | 2e-5 | 2e-4 |
| D | 1 | 4e-5 | 4e-4 |

两组均新建 AdamW 和余弦调度；完整继承模型参数，未继承第一档已归零的调度。相同数据顺序、验证样本、训练步数和超参，仅学习率不同。每100步保存，5000步和结束做完整验证，每1000步输出详细可视化。

配置见 configs/semidense_tier3_lr_c.json 和 semidense_tier3_lr_d.json。启动命令、PID、源权重SHA及数据SHA见 artifacts/semidense_tier3_lr_cd_registration.json。输出各自位于 outputs/semidense_stable_v2_tier3_lr_{c,d}_seed0；中文看板增加两组入口。跨档数据不同，不直接用不同档的loss判断收益。
