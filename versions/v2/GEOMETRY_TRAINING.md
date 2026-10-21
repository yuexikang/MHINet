# 杭州主线：统一几何监督、DINO-only 初始化

## 当前数据实际具备的监督

**本轮杭州数据没有人工标注的真实对应点，也没有真实非刚性 Flow。** 实际使用的是 `geo_approx` 近似地理 H，Flow、D8/D2、QRRU 的目标均来自这个参考。`dense_flow` 和 `sparse_points` 仅为接口支持能力，本轮没有对应数据，也没有使用这些监督分支。接口中的查询点不等于人工真实标注点；增加容差不能提高标签本身的真实性。

本轮指标只能衡量与近似地理参考的一致性，不能证明真实像素级匹配精度。实验结果见 [杭州小规模训练结果](HANGZHOU_FRESH_RESULTS.md)。

## 当前执行路径

| 路径 | 用途和状态 |
|---|---|
| `train_hangzhou.py → batch_loss(stage=joint)` | 当前主线。Flow + D8/D2 + mixed-QRRU 联合训练 |
| `train.py → batch_loss(stage=flow_head/qrru)` | 上一轮 RGB_512 分阶段训练，保持兼容，已暂停 |
| `pilot.py → weak_qrru_loss` | 上一轮只训练 QRRU 的 128 步小试，已完成，当前不使用其权重 |
| `SemidenseMatcherV2.training_losses(..., geometry=...)` | 统一监督的另一入口，调用相同 matching/mixed-QRRU 函数 |
| `SemidenseMatcherV2.training_losses(shared, H, masks...)` | 原有精确 H 联合损失路径，保留原采样和公式 |
| `mhinet/downstream/supervision.py` | v1 和精确 H 兼容实现，合法的 GT 使用，不是预测 H0 依赖 |

当前推理仍为 `flow_d16 → D8 → D2 → 四轮 QRRU`，512 输入，最多 2048 对；不预测或拟合 H0。GT 只进入损失和事后评测。

## 几何接口

`mhinet/v2/geometry_gt.py::GeometryGT` 表示一对图像的标签。内部统一为 **normalized xy、像素中心、align_corners=False**。像素坐标转换：`2 * (pixel + 0.5) / [W,H] - 1`。`hw_a/hw_b` 明确记录分辨率；模型当前仍只接受 512。

`query(source, reverse=False)` 返回：

- `target`：目标绝对坐标，而不是位移。
- `known`：该位置确实有监督。缺失位置不能被当作 no-match 负例。
- `visible`：标签对应是否有效、可见，包含图内与掩膜检查。

| kind | 查询与标签规则 |
|---|---|
| `homography` | 给定 H 的任意点投影；可合法求逆得到 B→A；精确 H 保留原公式 |
| `dense_flow` | 采样真实绝对坐标映射场；已标注支持区采用保守双线性有效性；只使用显式提供的 B→A，不反演非刚性场 |
| `sparse_points` | 仅匹配实际标注的源点，浮点容差 1e-6 normalized；不将邻点位移推广到未标注区域 |
| `unlabeled` | 所有查询均 unknown；监督损失为零，用于推理或另行定义的自监督任务 |

真实稠密场只在已知、可见处监督 D16。真实稀疏点在标注坐标采样预测网格；matchability、分类、位置损失都只使用标注查询，不给其他网格位置生成负例。D8/D2 的稀疏分支只使用实际标注点对进行描述子对比，不把标注坐标四舍五入后扩成密集标签。

QRRU 中心与 2×2 控制点通过适配层取得目标。精确 H 仍使用原控制点公式；真实 dense-flow 可以独立查询五个位置；稀疏中心只监督中心，四个控制点不补造标签。四轮、半径和采样结构保持不变。真实预测候选的可监督性、误差和修正范围由相应几何标签决定。

## 数据入口

`GeometryPairDataset` 加载新标签；`RGB512Dataset` 和其继承加载器保留原行为。旧精确 H 行不需要改格式；杭州 `gt_type=geo_approx` 会明确转换为 `homography, exact=False`。

新增标签行示例（NPZ 文件相对 dataset root）：

```json
{"geometry": {"kind": "dense_flow", "path": "labels/pair.npz", "coordinates": "pixel", "representation": "absolute"}}
```

稠密 NPZ：`map_ab` 为 `2×Ha×Wa`，值是 B 的绝对 xy；可提供 `known_a`、`visible_a`，以及显式的 `map_ba/known_b/visible_b`。`coordinates` 可为 `pixel` 或 `normalized`。不能把位移场直接冒充绝对映射；未标注场元素用 known 掩膜或非有限值标出。

```json
{"geometry": {"kind": "sparse_points", "path": "labels/pair.npz", "coordinates": "pixel"}}
```

稀疏 NPZ：`points_a/points_b` 为对应的 `N×2`，可选 `visible`。`{"geometry":{"kind":"unlabeled"}}` 显式表示无可靠几何监督。当前杭州训练入口只针对已有地理 H 数据；未知标签不用于监督优化。

## 杭州弱监督策略

当前清理后的数据为 train 79760、val 1670。固定 seed 42，从现有 train 按 G0–G3 各取 64 对，共 **256 对**；val 各取 16 对，共 **64 对**。选中的训练与验证空间块不相交，不使用 test。

杭州 H 是 `geo_approx`，不是精确非刚性真值。本轮不伪造 dense-flow 或稀疏点标注：

- Flow 坐标误差使用 `soft_radius_B_px` 对应的容差；D16 分类仍是粗尺度的双线性软目标。
- D8 使用弱地理粗格对应；D2 将容差圆内所有有效候选作为正例集合，对集合概率求和，不强迫一个最近点成为精确真值。
- QRRU 仅使用带容差的中心监督；近似 H 不用于强制四个控制点的精确位移。
- 验证 Pre@1/3、Flow EPE 和 RMSE 表示地理参考一致性，不能称真实 GCP 精度。

## 初始化和训练

配置：[hangzhou_fresh_joint.json](../../configs/mhi_v2/hangzhou_fresh_joint.json)。不加载旧 v1/v2/C-best/Flow-best/QRRU 小试权重。

`build_fresh_shared()` 只读取官方来源中的 DINOv3 张量，并冻结它。MVT、VGG、CGMDP、Flow Head、D8/D2 温度参数、QRRU 使用新初始化。Flow/QRRU 输出头的零初始化是结构默认，其他权重为随机初始化；位置编码频率重新生成。新 VGG 的 BN 统计允许更新。约 **1.093 亿**活跃参数参与训练；DINO 的 **2.276 亿**参数冻结。兼容的 D1 分支不执行，也不参与优化。

AdamW，LR 2e-4，batch=1、累积 4，梯度裁剪 1。最多 **30 epoch / 1920 次更新**，每轮余弦退火，最小 LR 1e-6。验证总 loss 至少改善 0.001 才重置早停计数；连续 6 轮不改善则停止，最少运行 5 轮。早停验证使用固定 GT 查询和固定噪声，不混入随模型变化的真实预测候选；真实预测候选表现另行观察。

普通配置、日志和断点，不做代码树哈希登记。`latest.pt`/`best.pt` 包含所有非 DINO 权重、优化器、余弦调度器和 RNG。恢复时只允许相同配置，从最近完整 epoch 继续；中断轮的 `interrupted.pt` 单独保留，不当作完成轮。首次短运行用于检查接口，因弱标签集合监督和固定验证采样调整，已保存在 `hangzhou_fresh_joint_20261021_initial_trial`，不计入正式结果；正式训练重新初始化。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TMPDIR=/home/u2ssd2_TQL/mhi_v2_tmp/qrru-pilot \
/root/miniconda3/envs/loma-repro/bin/python -m mhinet.v2.train_hangzhou \
  --config configs/mhi_v2/hangzhou_fresh_joint.json
# 仅恢复同一轮正式运行：添加 --resume
```

## QRRU 前、中、后观察

每轮在固定 8 对验证影像（每组 2 对）记录第 **0/1/2/3/4 轮**的 EPE、Pre@1/3、改善比例、损伤比例和最大控制点位移：

1. 固定地理参考查询与相同扰动：隔离 QRRU 本身的学习变化。
2. 当前模型实际预测候选：反映部署路径；不同 epoch 的候选集合不同，不冒充固定点对照。

运行结束后自动选取 epoch 0、实际结束区间的中间 epoch、最终 epoch，生成 `qrru_before_middle_after.json`、`qrru_summary.json`、`qrru_progress.png` 和 `REPORT.md`。`best.pt` 用于选模；after 图表示最后实际训练状态，两者可能不同。

## 已做的必要检查

- 精确 H：Flow 损失、指标和三个输出分支的梯度与旧实现差为 0（含 B=2 和掩膜空洞）。
- 非刚性 dense map：支持区空洞忽略；缺失反向映射不被补造。
- 稀疏：只有已知中心受监督，控制点无标签；Flow 只使用标注点。
- unlabeled：监督 loss 为 0。
- 原训练兼容检查 14 项通过；原来拒绝 joint 的检查按新需求改为接受 joint、仍冻结 DINO。
- 两次真实联合更新成功，所有活跃模块都有有限梯度，DINO 无梯度，峰值显存约 5.7 GiB。
