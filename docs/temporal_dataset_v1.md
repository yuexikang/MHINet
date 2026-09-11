# 双时相四对生成方案

用户确认原始 past/current 母图“基本严格配准”。因此跨时相标签使用母图间基础变换为
单位矩阵的假设，但明确标记为近似共坐标标签，不声称其物理配准真值完全精确。

## 每组母图的四对输出

| pair_kind | A母图 | B母图 | 采样 |
| --- | --- | --- | --- |
| same_past | past | past | 同图合成 |
| same_current | current | current | 同图合成 |
| cross_past_current | past | current | 独立normal采样 |
| cross_current_past | current | past | 独立hard采样 |

两对同时相使用一个normal、一个hard，按组的稳定随机种子交换分配，避免固定给某个时相
更难的增强。全组共2个normal、2个hard。跨时相两对不是把已生成的一对交换顺序复制，
而是独立生成。四对均继承原生成器的几何范围和独立外观增强。

## 划分与实际数量

扫描实际 past/current 的 Train、Val 影像目录，按完全一致的文件stem配对，合并原母图池，
而非按CSV的重复行增加样本。发现9,625组。缺失另一时相或重复母图时直接报错，不静默丢弃。
先按0.01度地理格分组，以seed42选择最接近10%母图数的完整格集合为val；随后生成四对。

| split | 母图组 | 地理格 | 计划输出影像对 |
| --- | ---: | ---: | ---: |
| train | 8,663 | 2,564 | 34,652 |
| val | 962 | 286 | 3,848 |

val占9.9948%，整数与地理分组约束下接近1:9，不拆母图或地理格凑比例。
原test/1000通过新目录下的test符号链接复用；生成器不写入链接目标，也不重新划分test。
母图分配写入 `parent_split_manifest.json`；每对元数据记录母图、pair_kind、group、seed、
增强参数、H、mask以及标签来源。

## 命令

仅查看计划（不创建数据目录）：

```bash
cd /home/disk1/MHINet
conda activate loma-repro
bash scripts/generate_temporal_dataset.sh
```

实际生成：

```bash
bash scripts/generate_temporal_dataset.sh --generate
```

计划输出 `/home/disk1/Data/datasets/GoogleEarth_temporal4_v1`。
可用 `MHINET_DATA_OUTPUT` 指定另一个新目录；非空目录拒绝覆盖。生成失败时不会伪造完整状态
或静默减少每组四对的数量。脚本默认仅计划，避免误触长时间/大规模生成。
各split保存最多20份抽样可视化，复用LoMa原有绘图函数。

## 标签与兼容性

继续复用 `/home/disk1/LoMa/generate_pairs.py` 的尺寸采样、几何变换、一次warp、光度增强、
H组合、重叠mask与可视化；LoMa源码未改动。包装入口在
`mhinet/dataio/generate_temporal.py`，由MHINet仓库版本控制。
同图标签为精确合成几何；跨时相标签为 `T_B @ inverse(T_A)`，隐含母图基础变换为单位矩阵。
CSV中含义未确认的仿射参数不擅自混入标签。几何自洽检查不验证真实地物是否精确配准。
overlap mask只代表几何可见区域，不是建筑变化/遮挡的语义有效掩码。

新数据仍兼容MHINet的pairs.jsonl、H方向、坐标与overlap mask读取。完成全量生成和审计后，
再创建新runtime配置指向新根目录；本次不自动切换运行中的训练或修改旧runtime。
由于重分了母图池，旧E01训练集可能包含新val母图：不能拿旧E01继续训练后称新val是独立验证。
新实验应从原始预训练初始化，并使用新的输出目录。

## 验证边界

`--smoke --generate`只生成每个split各1组、共8对，明确标记smoke_only，不可当正式数据集。
三项分组/配方测试覆盖四种配对、seed复现、地理隔离和配比。
实际小样本结果位于 `outputs/diagnostics/temporal4_generation_smoke_v2`。
全量38,500对尚未生成；统计是根据实际母图清单计算的计划，不是已完成产量。
