# QRRU 局部精修发布

版本 `mhi-v2-flow-pair-refined-20261104`，默认入口 `load_mhi_v2()`。

## 最终方案

Flow0 → 局部D8 → Pair-D2 → QRRU局部精修。
精修模块计算双图4×4局部特征、2×2控制点偏移及最终目标点；对外直接返回最终坐标。
保留D2加深和可选互选过滤，最多2048点，坐标使用512原图像素。

模块96,339参数，训练冻结所有上游，仅使用RGB精确H：4096对训练、512对验证，10 epoch。
损失为平均端点误差加平均正向增误差惩罚（λ=1），只监督有效可见位置。选模仅依据RGB验证EPE，最终为epoch10。

## 完整测试

| 测试集 | 路径 | P@1 | P@3 | EPE(px) |
|---|---|---:|---:|---:|
| RGB 5106对 | 精修前 | 91.742% | 98.360% | 0.666314 |
| RGB 5106对 | 当前主线 | 92.336% | 98.390% | 0.630161 |
| 杭州235对 | 精修前 | 22.572% | 63.036% | 11.684399 |
| 杭州235对 | 当前主线 | 23.445% | 63.383% | 11.665272 |

这些为已完成训练的完整测试成绩，本次迁移不重新训练。杭州H为近似参考。

## 迁移验收

- 11项定向CPU测试通过：权重布局对应、坐标计算、空输入、零偏移初始化、投影梯度、定位及改坏损失、现有主线和互选协议。
- GPU2上RGB16对＋杭州16对实数据，分别检验默认和互选输出。
- 与已验收计算的最大坐标差0，P@1/3/5/10计数及输出数量一致。
- 峰值分配显存1.777GiB（验收中同时持有参考模块，不是独立速度基准）。
- 详见 `verification.json`。复现：`PYTHONPATH=. python tools/mhi_v2/verify_local_refinement.py`。

## 文件

- `mhinet/v2/local_refinement.py`：直接精修计算与定位/改坏损失。
- `mhinet/v2/pipeline.py`：组装、权重加载、精修前诊断坐标。
- `configs/mhi_v2/releases/flow_pair_v2.json`：主线默认配置。
- `configs/mhi_v2/releases/flow_pair_v2_deep.weights.json`：Flow/D2来源及精修权重条目。
- `weight/mhi_v2/qrru_local_20261104.pt`：已导出的独立精修权重，不包含优化器或实验配置；权重不提交Git。
- `tests/v2/test_local_refinement.py`、`tests/v2/test_pair_pipeline.py`：定向验收。
- `tools/mhi_v2/verify_local_refinement.py`：真实输入迁移对照。
- `docs/mhi_v2/flow_pair_release.md`、项目及模块README：正式接口和方案说明。

原始权重来源：`/home/disk1/MHINet/outputs/qrru_monotonic_20261104/runs/monotonic_best.pt`。
运行仅需发布权重，不依赖研究分支、排队脚本或实验输出目录。
