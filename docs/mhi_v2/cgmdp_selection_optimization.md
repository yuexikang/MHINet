> 研究档案：原实现与运行命令位于 `codex/cgmdp-global-modulation`；本次仅迁移报告，不将这些实验模块加入正式推理。

# 跳过被丢弃的旧版 D2 匹配（2026-10-28）

分支：`codex/cgmdp-global-modulation`。

## 正式修改

`mhinet/v2/d2_modulation/model.py::extract_windows()` 删除旧版 `discrete_infer` 的导入与调用，直接使用 `flow_prior_from_shared` + `local_coarse` + `source_grid` 获取前128个D8父点。保留原来的可匹配性阈值0、互选、置信度阈值、排序和父点上限；后续连续 `coarse()`、8×8上下文采样和Pair-D2不变。权重不变，不重新训练。

没有合并两次D8搜索：`local_coarse`在离散网格上选择源父点，`coarse`在连续Flow中心采样并精化目标位置，两者不等价。正式路径没有新增禁止调用钩子。旧版 `flow_d8_d2.matching.infer` 及共享 `fine_scores` 仍由其他旧实验使用，未删除这些有效调用所需的实现。

## 验收方式

独立脚本 `mhinet/v2/cgmdp_modulation/benchmark_selection.py`，旧窗口代码读取git提交 `ba1ea84`，全局对照读取已上传提交 `224fbd0` 的 `no_flow.py`。没有把历史旧逻辑重新接入正式推理。复现需要这两个git提交和既有本地数据/权重。

固定 `outputs/d2_modulation_expanded/selection.json` 的RGB test前32对，FP32匹配、同一GPU0、batch=1，每路径5次预热，每对重复3次且轮换执行顺序。分别验证原候选Pair适配器，以及本支线cumulative/delta的best检查点；CGMDP两组沿用原实验的identity局部适配器，不叠加候选调制。完整训练任务继续运行，没有停训练或更改权重。

- 等价验收使用确定性CUDA归约与cuBLAS工作区，对128父点顺序、连续D8目标中心、全部窗口特征/掩码、D2坐标/置信度进行零容差核对。
- 默认非确定性模式另核对父点集合、按源坐标对齐后的D2坐标和P@1/P@3，记录近同分排序差异；不把确定性测试结果冒充默认CUDA逐位稳定性。正式代码不改变确定性开关。
- 计时使用默认CUDA内核和CUDA Events。局部总耗时包括先验构建、父点选择、连续D8和Pair-D2；全局Flow总耗时同样包括先验构建。所有路径共用预先算好的特征，因此均不计DINO/MVT/Flow/VGG等特征提取、数据读取和模型加载。
- 分段 `parent_ms`：旧路径包含被丢弃的Fine，新路径只有 `local_coarse`；`continuous_coarse_ms`为连续D8；`pair_d2_ms`包含窗口采样、Pair-D2及最终置信度。总时间还含先验、源坐标构造等开销，不应强行等于三段之和。
- `global_no_prior`只表示匹配不使用Flow先验，本报告不能据此宣称整个网络省掉Flow头的端到端耗时。
- GPU0存在并发训练；结果是该环境下交替测量的实测值，不是独占GPU的部署吞吐。全局路径的候选策略不同，单列精度，不能把路径间速度差额全当作无损优化。
- 脚本临时禁止旧Fine/QRRU用于验收，退出后无副作用；正式推理不加这些钩子。

## 命令

在该支线工作树根目录执行：

```bash
cd /home/disk1/MHINet/.worktrees/cgmdp-modulation
export TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp
PY=/root/miniconda3/envs/loma-repro/bin/python
for mode in candidate cumulative delta; do
  $PY -u -m mhinet.v2.cgmdp_modulation.benchmark_selection --device cuda:0 --mode "$mode"
done
```

输出：`reports/cgmdp_selection_optimization/{candidate,cumulative,delta}.json`，包含32对ID、精度计数、逐对等价检查以及每次分段耗时。日志位于本地 `outputs/cgmdp_selection_*.log`。下方结果表由上述JSON汇总。

## 实测结果

单位ms，均值；P@1/P@3分母为全部D2输出，阈值为512原图像素。每模式32对×2048点。

|模型|旧局部|优化局部|匹配耗时减少|全局+Flow|全局无先验|
|---|---:|---:|---:|---:|---:|
|candidate|34.710|13.313|61.6%|10.819|7.072|
|cumulative|35.016|10.827|69.1%|9.498|5.446|
|delta|35.070|10.795|69.2%|8.607|5.272|

|模型|优化local_coarse|连续coarse|Pair-D2（含窗口采样）|
|---|---:|---:|---:|
|candidate|9.200|0.793|1.494|
|cumulative|7.299|0.759|1.031|
|delta|7.485|0.779|1.016|

|模型|局部P@1（前后相同）|局部P@3（前后相同）|全局+Flow P@3|全局无先验P@3|
|---|---:|---:|---:|---:|
|candidate|40.427%|89.828%|85.297%|83.156%|
|cumulative|34.659%|87.456%|86.136%|84.074%|
|delta|34.874%|86.943%|85.138%|81.183%|

### 等价性边界

- candidate：确定性核对32/32通过，D8中心和D2坐标最大差0。默认模式1/32对父点顺序不同，集合不同0对；按源点对齐后D2坐标最大差0px，P@1/P@3正确点数相同。
- cumulative：确定性核对32/32通过，D8中心和D2坐标最大差0。默认模式0/32对父点顺序不同，集合不同0对；按源点对齐后D2坐标最大差0px，P@1/P@3正确点数相同。
- delta：确定性核对32/32通过，D8中心和D2坐标最大差0。默认模式0/32对父点顺序不同，集合不同0对；按源点对齐后D2坐标最大差0px，P@1/P@3正确点数相同。

结论：第一步删除无用旧Fine计算有效，已完成32对严格对照并覆盖两个CGMDP检查点；未合并离散与连续D8搜索。默认CUDA浮点归约仍可能让近同分父点交换顺序，本次没有修改算法去强行稳定排序。速度结论仅适用于匹配部分及当前并发环境，不等同端到端提升。未重新训练、未扩大测试集。
