# 单母图监督与无真值视觉测试

2026-09-15 起停用跨母图单位变换标签。旧 temporal4 及合成 GoogleEarth test
结果保留归档，其真实精度结论失效，不作为新训练恢复来源。

新生成目录：`/home/disk1/Data/datasets/GoogleEarth_single_parent_v2`。
输入：`/home/disk1/Data/datasets/GoogleEarth/training_data`。
每张母图生成普通/较难各一对；past/current各自独立生成，不交叉。
沿用地理分组9:1划分；同一区域两个时相不跨train/val。
计划 train=34652、val=3848；不是恢复旧版不安全的随机切分。
复用 `/home/disk1/LoMa/generate_pairs.py` 的裁剪、变换和光度增强函数。
历史模块名 generate_temporal 保留兼容，不再产生跨时相监督。

```bash
conda activate loma-repro
bash scripts/generate_temporal_dataset.sh # 只看计划
bash scripts/generate_temporal_dataset.sh --generate
```

生成器拒绝覆盖已有目录。训练启动检查拒绝旧 temporal4 数据。
train_frozen_dino.sh 和 train_frozen_dino_mvt.sh 默认切换新runtime、新输出目录；
旧超参配置文件名暂保留，实际数据以runtime和checkpoint元数据为准。
新数据未完整生成前不能训练。不得继续旧输出目录的跨时相训练。

原始测试：`GoogleEarth/evaluation_data/test_pairs.csv` 的500对，A=Target、B=Source。
不裁剪、不合成几何变换，不使用CSV未核实的标注字段。
仅在网络读取时按既有输入约定resize到784；原始文件不改变。
每对保存H0和六轮共七张预测叠加图，明确NO GT，不画真值框。
不计算MACE、成功率或精度AUC，不进入精度评估总表。

```bash
GPU_ID=0 bash scripts/test.sh /绝对路径/checkpoint.pt --output-dir outputs/original_visual_test
```

新test清单仅引用原图，无H标签。通用evaluate入口禁止GoogleEarth test精度评估；
合成单母图val仍正常计算精度，用于模型选择。
旧test_e00.sh的精度入口因此停用；未来有可靠真值的外部数据需独立接入。
