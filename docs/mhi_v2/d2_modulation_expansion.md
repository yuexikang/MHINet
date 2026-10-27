# D2调制扩量与重复实验（2026-10-27）

沿用已实现模型及损失，不解冻Flow/MVT/VGG/DeDoDe/D8/温度，不增加D8调制，不改主线默认行为。新增内容只涉及训练seed、数据选集、每验证周期训练对数及最终分组评估。

## 预先固定的试验

- 254训练对：已有seed42结果保留，self/pair各补seed43、44；相同64验证对选模型。
- 1024训练对（原1024包含原254）：self/pair各做seed42。分为每验证周期随机抽254对的预算对照，以及完整遍历1024对的扩量训练。
- 两种扩量均最多30周期，余弦退火和连续5周期早停不变。前者最大7620次更新，后者最大30720次更新；报告实际更新与所选best更新数。相同最大预算及早停协议不等于各组实际更新次数完全相同。
- 自调制/双图调制使用相同seed与样本/窗口采样顺序；其余参数保持原方案。
- 原RGB测试64对不变；另选192对，按source_dataset平衡，从此前登记的D16测试256对之外抽取，不用新增测试选模型。
- RGB训练/验证/原测试/新增测试在pair_id和source_path上无交集。仅保证登记选集隔离，不宣称地理区域或其他预训练经历绝对隔离。
- 杭州64测试保持不变，近似H参考一致性，仍不参与训练/调参。
- 新增测试按来源、min(A/B重叠率)及max(s,1/s)尺度分组；来源包括RGB_512已有的flair_rgb、googleearth_current、whu，不加入其他外部数据。

所有冻结特征只提取一次，原缓存按pair_id硬链接复用。扩量目录独立为outputs/d2_modulation_expanded，原实验结果不覆盖。无需重新做模型结构测试：只检查参数化入口及真实首轮日志，完成后评价各组best。

## 文件与命令

修改 `mhinet/v2/d2_modulation/experiment.py`：增加 `--seed`、`--run-name`、`--samples-per-epoch`、`--selection`、`--reuse-cache`、`--preload-train`；默认值保留原seed42行为。新增 `scale_study.py`：固定数据选集、补充测试评估和汇总。结果见 `reports/d2_modulation_expanded`。

```bash
# 定下数据列表，只首次运行一次；不要在训练期间改列表
python -m mhinet.v2.d2_modulation.scale_study plan
python -m mhinet.v2.d2_modulation.experiment prepare --device cuda:2 --output outputs/d2_modulation_expanded --selection outputs/d2_modulation_expanded/selection.json --reuse-cache outputs/d2_modulation
# 以下mode分别取self、pair，seed分别取43、44（四组）
python -m mhinet.v2.d2_modulation.experiment train --mode self --seed 43 --run-name self_seed43 --device cuda:0
# 1024对相同最大更新预算，mode分别self/pair（两组）
python -m mhinet.v2.d2_modulation.experiment train --mode self --seed 42 --run-name matched_self --samples-per-epoch 254 --output outputs/d2_modulation_expanded --device cuda:2 --preload-train
# 1024对完整遍历，mode分别self/pair（两组）
python -m mhinet.v2.d2_modulation.experiment train --mode self --seed 42 --run-name full_self --output outputs/d2_modulation_expanded --device cuda:0 --preload-train
# 全部done后补评原小样本权重在新增测试上的结果；不更新任何参数
python -m mhinet.v2.d2_modulation.scale_study evaluate --device cuda:0
python -m mhinet.v2.d2_modulation.scale_study report
```

GPU0/1先各自完成小样本重复，再各自运行完整扩量；GPU2先提取特征，再顺序执行两组预算对照。每次启动均核实GPU状态；不因一次监测超时重启作业。预计10–25分钟，依早停与存储负载而变，以done.json实际耗时为准。

## 判据

小样本报告三seed均值/标准差及同seed pair-self差，不能只挑最好seed。扩量先看相同最大预算对照，再看完整遍历，区分数据量与更多更新。原测试和新增测试同时报告，按同点直接Flow比较；不删除查询以提高P@3。扩量只seed42，未覆盖扩量后的多seed重复；若pair没有稳定优势，优先保留self作为更简单方案，不自动合并未验证的新策略。

扩量的1024对FP32训练缓存约16.13GiB，可用 `--preload-train` 一次载入24GB显卡以减少反复磁盘读取与传输；算法与特征精度不变。done.json记录峰值allocated显存和耗时。新增测试原始清单图片路径相对于test目录，实际extra_test根目录为RGB_512/test；该路径修正未修改数据集文件。

数据来源计数：1024训练对为flair_rgb 342、googleearth_current 341、whu 341；原64验证/测试分别22/21/21，新增192测试每来源64对。此分层保持来源平衡，不能据此假定不同来源难度相同。

## 执行结果

八个新增训练任务均已完成，原seed42三组结果保留复用。self_seed43/44分别16/18周期早停，pair_seed43/44分别27/23周期早停；1024预算对照self/pair均27周期，完整扩量self/pair分别20/26周期。每组done.json记录实际更新、耗时与峰值显存。

小样本补跑单组约221–359秒，GPU allocated峰值约71MiB（不含CUDA上下文/驱动占用）；预加载扩量单组约249–318秒，allocated峰值约16,586MiB。队列在GPU0/1运行小样本后转完整扩量，GPU2先准备缓存后执行预算对照。缓存提取中发现一次新增测试相对路径前缀错误，已修正并只补做未完成测试缓存；训练没有重启、权重没有覆盖。

所有loss有限；所有组在相同测试划分上的查询数、可见性、Flow正确点数和Flow误差总和完全一致，因此指标差异来自新调制器，不是换源点或改上游。未出现OOM。

结果与配置汇总：`reports/d2_modulation_expanded/results.md`、`summary.json`；逐对及分组：`pairs.jsonl`、`strata.json`；测试元数据：`test_selection.json`。

新增192对上，小样本三seed的self/pair均值分别86.363%/86.588%；1024预算对照为86.864%/86.872%；完整扩量为87.491%/88.103%，同点Flow为84.507%。主测试64对上完整扩量pair为89.349%。精确值以summary.json为准。

最佳完整扩量权重：`outputs/d2_modulation_expanded/{full_self,full_pair}/best.pt`；预算对照为`{matched_self,matched_pair}/best.pt`。这些是独立调制权重，仍用原冻结上游并通过现有load_adapter/infer入口加载，不替换主线默认权重。

边界：新增测试没有高重叠≥0.7或尺度比<1.25的样本；扩量只有seed42。杭州仅近似H评估，未增加人工真值。更大数据、多seed扩量及主线接入未在本轮进行。
