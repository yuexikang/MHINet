# 真实影像精度评估

按用户明确规定：训练用train（36000对），训练中验证用val（2500对），独立评估包括E00用test（1000对），test不用于选择超参或checkpoint。

独立E00初始化基线使用 `bash scripts/test_e00.sh`：默认卡1、完整test，读取runtime登记的原始预训练GHIM，不需要训练checkpoint，不执行CGMDP/MHIR，不产生七轮图。输出 `outputs/E00_h0_seed0/report.md` 和 `metrics/`。`DRY_RUN=1 bash scripts/test_e00.sh`可先查看命令；默认不设置max-pairs上限。

`scripts/test.sh`现在是模型精度评估入口，不再是单元测试。`scripts/unit_tests.sh`单独运行工程检查。正式精度必须使用实际训练checkpoint，脚本不会自动挑选“最新”权重，也不会在缺少权重时退回随机初始化。

```bash
cd /home/disk1/MHINet
# 将位置参数换成实际训练checkpoint；默认物理GPU1，全量test
bash scripts/test.sh /absolute/path/to/trained_checkpoint.pt
# 小规模检查（不能代替全量test评估）
GPU_ID=0 bash scripts/test.sh /absolute/path/to/trained_checkpoint.pt --max-pairs 16
# 配置/权重锁定后才执行test；禁止用test选择checkpoint或超参
bash scripts/test.sh /absolute/path/to/locked_checkpoint.pt --split test
```

默认输出 `outputs/evaluation/<UTC时间>/`；可指定 `--output-dir DIR`，已有非空目录默认拒绝覆盖。默认读取runtime指定数据根目录下的 `test/pairs.jsonl`，不限制对数。训练过程中调参用val；评估脚本仅forward，不更新权重。

- `report.md`：H0、H1～H6的汇总表。
- `metrics/summary.json`：完整汇总、checkpoint hash、数据manifest hash、配置与资源信息。
- `metrics/pair_metrics.csv` / `.jsonl`：每个真实影像对、每轮几何与更新诊断，失败行不删除。
- `visualizations/pair_0000/`：H0及六轮预测叠加图（绿GT、红预测），共7张；`--visualization-pairs N`输出前N对。

MACE是四个源图角点经预测H与GT投影后的平均欧氏距离；网格误差使用5×5规则网格。两者同时记录784输入像素与原始目标影像像素单位。报告中的误差均值是有效输出上的条件统计，必须同时读取失败率。

每个H均输出成功率@1/3/5输入像素：有效且MACE≤阈值的数量/全部影像对数。AUC为归一化经验召回曲线的精确积分：

`AUC(t) = sum(valid_i * max(0, 1 - MACE_i/t)) / N`，t=1、3、5。

AUC范围[0,1]，失败输出贡献0但保留在N中。这是明确登记的经验CDF面积定义，不声称等同于其他论文使用的梯形插值实现。更新被拒绝后保留合法H，该状态仍评估；拒绝率另报。单对同步计时与包含影像I/O/绘图的墙钟均摊分开，不作为统一warmup/重复计时的正式性能基准。

当前指标版本 `real_image_v2_ecdf_auc_quantile_median`：中位数采用0.5分位数，偶数样本取中间两项均值；修正此前torch.median取较小中间项的行为。旧报告保留原值，不与新中位数口径混用。

本轮实际入口检查使用现存的 `outputs/engineering_visualization_seven_smoke/checkpoints/step_0000002.pt`，只训练过两步，不是正式训练模型。评估前16对真实val，最终输出 `outputs/real_image_accuracy_val16_verified`；这是带真实影像/标签的准确率测量和入口验证，不是最终test结果，也不证明正式训练精度达标。当前工作区未发现E01正式训练checkpoint，后续请显式提供训练得到的权重路径。
