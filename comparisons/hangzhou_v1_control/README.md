# 杭州 v1 对照实验

目的：在同一杭州数据、同一训练预算下比较 v1 和 v2，观察差距是否仍存在。v1 原有源码与冻结档案不改动；实验入口和输出单独存放。

## 相同条件

- 直接读取 v2 正式实验的 `selection.json`，训练 256 对、验证 64 对，训练顺序使用同一 seed/epoch 规则。
- 512×512 输入、最多 2048 对输出、D8 查询 512、D2 窗口 64、QRRU 查询 128、四轮迭代。
- 只加载原始 DINOv3 并冻结，其他活跃模块新初始化；同一共享网络构造方法，没有加载旧 v1/v2 权重。
- AdamW、学习率 2e-4、weight decay 0.01、累积 4、梯度裁剪 1、余弦退火到 1e-6。
- 最多 30 epoch、至少 5 epoch，验证 loss 连续 6 轮未改善 0.001 则早停。最佳模型按各自 loss 选择。

## 保留的 v1 差异与解释边界

- 使用未修改的 `mhinet/downstream/semidense.py` 和原生 QRRU，预测 H0 限定重叠区域并生成 fine 窗口。
- 定位头使用原生 `Stage1HeadFromSharedMVT`，H 拟合采用 v1 provider 本身使用的 Safe fitter，初始权重来自新初始化模块。
- 保留 `Stage1HLoss`（含 0.05 的 H 参数损失）以及原生 Lc/Lf/Lq。v1 原生损失将提供的近似地理 H 当成目标，包括控制点；这不是新增真实标注，也不表示这些目标可靠。
- v2 使用新增容差、D2 多正例和中心监督，训练损失定义不同。不能横向比较总 loss，也不能把方案差异全部归因于模型结构。
- 验证逐样本固定随机种子；原生 v1 的 fine 分支仍取决于预测 H0 的有效性，QRRU 查询随机数也可能随分支改变。因此 v1 的验证 loss 不是完全固定查询的损失对照。共同评价以完整 64 对验证集的 Pre@1/3、输出点数、Flow EPE 为准。
- 相同配置从头训练不是复现旧 v1 的完整预训练流程。如果两者都差，仍不能排除小样本量和初始化、优化限制，不能直接认定数据错误。

**本数据没有人工真实对应点和真实非刚性 Flow。所有精度均为近似地理参考一致性。**

## 运行和结果

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TMPDIR=/home/u2ssd2_TQL/mhi_v2_tmp/qrru-pilot \
/root/miniconda3/envs/loma-repro/bin/python -m comparisons.hangzhou_v1_control.train \
  --config comparisons/hangzhou_v1_control/config.json
```

输出：`outputs/hangzhou_v1_control/fresh_20261022/`，包含每轮验证记录、latest/best 检查点和最终中文 `REPORT.md`、共同指标 `comparison.png`。当前入口不支持恢复未完成运行；已有输出不会被覆盖。

抽样脚本：`sample_pairs.py`。从实际子集中每个 G 组随机抽一对，seed=20261022；保存原始 A/B 图片、传感器和日期，不添加假设对应点。抽样清单位于 `outputs/hangzhou_v1_control/samples/selection.json`。

已完成两次真实优化更新和两对验证检查，活跃模块梯度有限、DINO 无梯度。没有运行无关的大型测试。
