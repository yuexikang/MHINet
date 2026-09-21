# 稳定几何第一档低学习率适应登记（2026-09-21）

目的：观察已有共享网络在稳定几何第一档上的损失和验证指标是否继续改善；不是从头预训练，也不是升级匹配头。原权重作为A对照，B初始化同一权重后更新共享网络。

## A：训练前完整val基线

原权重SHA256 `5a9ed14cc32a1a4ff3a843b737410da13d795b79c42a9d33d82a068226e30386`，文件`outputs/shared_descriptor_frozen_tier3_seed0/latest.pt`。

| 档位 | val对数 | 总loss | descriptor loss | H0重叠区均值px | 四角MACE px |
|---|---:|---:|---:|---:|---:|
| 1 | 5772 | 0.131815872 | 0.126020921 | 0.930410331 | 2.707986128 |
| 2 | 3848 | 0.133039119 | 0.127274857 | 0.930194257 | 2.933929338 |
| 3 | 3848 | 0.141973942 | 0.132764357 | 1.420393846 | 4.832650851 |

三个完整基线均已完成，机器归档`artifacts/shared_stable_v2_baselines.json`。三个档位数据不同，不能用表内横向差异判定训练收益。没有test选模。

## B：适应设置

- 数据`/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2`，只第一档train51978；val5772完整，不截取。
- train manifest SHA `c48eb269be7e96a6be8d4eb43b668035d758f0aaf4268d9f6abb8e337e3aad23`，val SHA `9e69f1af29dca1073ca3f12a618d61b52b66a76e67346d2ab39830de17ec5f44`。
- 冻结DINO，不启用LoRA；MVT、GHIM head、VGG、CGMDP训练，VGG BN统计固定；不运行MHIR或密集下游，不计算D1。
- 峰值LR：MVT/head1e-7、VGG5e-7、CGMDP1e-6，上一档峰值的1/5。新AdamW状态，weight decay1e-4，clip1，BS1×累积4，seed0，queries1024。
- GHIM四项与三尺度descriptor损失沿用原设置，不改损失来掩盖问题。
- 预登记预算2000 optimizer steps，即8000对（约第一档train的15.4%），500步存latest；结束后完整val。不是一整轮。
- 复用已有调度：cosine horizon仍为完整一轮12995步，warmup650步；--max-steps仅截断运行，不压缩cosine。进度条分母12995，正常应在2000停止；这不是未跑完。2000后若要继续，需新决策，不自动延长。
- 输出`outputs/shared_stable_v2_tier1_adapt_seed0`；相同目录恢复，新目录新实验。

执行入口`GPU_ID=1 bash scripts/train_shared_stable_v2_tier1_adapt.sh`。先8对train/2对val两步工程冒烟，loss0.149783/0.130817、峰值9.96GB；不同batch不构成下降证据。正式记录以run.json、train.jsonl和最终完整validation为准。

## 判读规则

训练loss按窗口均值查看，不能对单步或不同batch直接比较。主要比较A/B同一第一档完整val的总损失、descriptor loss、D8/D4/D2检索、H0重叠区误差/四角误差和失败率。若训练loss下降而val不改善，不宣称适应有效。若第一档改善，仍需回测第二/三档以检查遗忘，不能立即替换共享主权重。单seed筛查不是普遍结论。

## 下游术语

LoMa迁入分支统一称“H0引导的粗细匹配”：D8粗匹配，D2局部/warped细匹配，返回A/B原生像素坐标和每对置信度。P3返回离散D2网格中心，P6因H-warped采样可返回连续B坐标；二者当前均未额外接RRU或LoFTR-style可训练亚像素模块，也未自动拟合最终H。不把它称为已迁入可训练LoFTR头。
