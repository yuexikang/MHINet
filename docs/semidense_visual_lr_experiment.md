# Tier3共享预训练 → Tier1半密集下游：学习率对照

2026-09-22。使用完成8663步及最终3848对val的 `outputs/shared_stable_v2_full_tier3_seed0/shared_descriptor.pt`，在 stable_v2 第一档51978对train上训练一轮（BS1、累计4，12995优化步），验证5772对val。未使用test选参数。

| 对照 | GPU | CGMDP峰值学习率 | 匹配温度/QRRU峰值学习率 |
|---|---|---|---|
| A | 1 | 1e-5 | 1e-4 |
| B | 2 | 5e-6 | 5e-5 |

相同seed0、初始化、数据顺序、预算、cosine调度、损失Lc+Lf+Lq。冻结DINOv3、MVT、GHIM/H0预测头；VGG BN运行统计固定，D1不激活。H0不会参与QRRU采样。两组 `run.json` 记录初始matcher SHA，用于证明新模块初始化一致。

每100步保存latest和刷新曲线，每5000步/结束完整val；每1000步/结束输出12对固定验证样本诊断。第0步先存checkpoint并生成诊断，绘图完成后开始优化。全通道图库在第0步、中点附近的诊断步及结束生成；其余为固定32通道。初始化诊断耗时不计为训练卡住。

可视化文件位于各输出的 `visualizations/index.html`，包括：

- GT解耦损失、梯度、温度、学习率、耗时，以及固定val纯预测级联的coarse/fine/final曲线。
- D8/D4/D2和Q48的固定共同PCA基、固定色阶通道图与原始范数/离散程度。
- 16个固定全图查询的cosine和完整分母dual-softmax概率，GT/预测位置、Top5和过滤原因。
- 4个当前选中coarse窗口的16×16相似度/概率、GT与预测选择、H0位置及H残差位置。
- 最多8条QRRU的第0～4轮坐标、控制流、更新门控、采样网格；第一条另有GIF。
- NPZ原始分数/坐标/选择链，JSON样本与单位、特征裁剪率和每轮误差。

固定查询、动态选中例子明确区分；动态例子不能冒充同一个点的训练轨迹。支持区外固定查询仍保留失败状态。概率使用固定对数色阶1e-6～1，原始概率不重新归一化。PCA基和色阶在同一对图的第0步共同标定，以后复用，不能把颜色变化归因于每次重新PCA。

val样本按manifest等距索引固定选择，跨A/B共享选择规则；不是根据误差挑选成功例子。诊断是固定子集，不等于全量验证。QRRU前后同时报告同一保留点集合的误差，以及出界/剔除数量；fine完整结果不被QRRU剔除子集替代。

启动器：`scripts/launch_semidense_lr_compare.py`；机器可读登记：`artifacts/semidense_tier1_lr_compare_registration.json`。重复启动拒绝已有目录/登记。运行状态以进程、日志和checkpoint为准，登记的launched不是完成证据。

## C组：更大学习率追加对照（2026-09-23）

用户授权新增一组更大学习率。C使用GPU1，CGMDP学习率2e-5、温度/QRRU学习率2e-4，均为A的2倍。重新从同一第三档共享预训练导出开始，完整训练stable_v2第一档一轮，12995步；不从A训练后的权重续训。其余配置、seed、冻结策略、可视化和验证频率保持一致。

配置：`configs/semidense_tier1_lr_c.json`。登记含来源SHA、命令、进程及输出：`artifacts/semidense_tier1_lr_c_registration.json`。输出：`outputs/semidense_stable_v2_tier1_lr_c_seed0`。启动状态不代表完成或更优；结束后按同样的完整val监督损失与独立级联诊断比较。
