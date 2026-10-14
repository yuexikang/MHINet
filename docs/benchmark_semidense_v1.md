# 历史实验总结与对比基准 v1（2026-09-29）

文档状态（2026-10-14）：2026-09-29 的 stable_v2 第三档历史基准。保留其 C/D/E 对照、固定回归集和当时协议，不能用它覆盖当前 RGB512 主路线的训练冻结范围、权重阶段或双数据集指标。当前定义见 [D8→D2 主路线](mainline_d8_d2.md)，现行基线证据见 [RGB512 H0 联合测试](semidense_rgb512_h0_joint_test512_results.md)及 [H0 推理位置消融](h0_inference_position_ablation_results_20261013.md)。历史 784 合成点匹配协议与当前 512 双数据集配准协议分别报告。

## 1. 已有结论及适用范围

截至 2026-09-29，本节合成第三档对照中的最佳候选是半密集第三档 E：GHIM/CGMDP → D8粗匹配 → D2细匹配 → 4轮QRRU。它不是早期的 MHIR 六次单应更新实验。以下数字来自 artifacts/semidense_cde_timed_test_results.json，均为同一合成 test 2000 对、逐对宏平均；本节不提供不同真实数据集的完整排名证据。

| 第三档方案 | VGG/CGMDP LR | 温度/QRRU LR | 完整val解耦loss↓ | final P@1px↑ | final EPE↓ |
|---|---:|---:|---:|---:|---:|
| C | 2e-5 | 2e-4 | 0.771101 | 99.471052% | 0.121309742 px |
| D | 4e-5 | 4e-4 | 0.716338 | 99.558959% | 0.114445443 px |
| E | 8e-5 | 8e-4 | 0.640607 | 99.636735% | 0.103906393 px |

E 的 P@3/5px 为 99.785257%/99.797001%；2000对均有有效EPE。C/D/E推理失败率均为0，至少20个5px正确匹配的成功率均为100%，此成功指标已经饱和。E不是每个中间阶段都最优。结果只有seed0，不作统计显著性声明。历史测试使用分块32；当前代码默认1024，复现必须显式指定。

## 2. 数据固定

| 用途 | 数据/档位 | 对数 |
|---|---|---:|
| train | GoogleEarth_quadrant_tiers_stable_v2 第一/二/三档 | 51978 / 34652 / 34652 |
| val | 同版本第一/二/三档 | 5772 / 3848 / 3848 |
| 历史最终test | GoogleEarth_test_single_tier3_v1/test/pairs.jsonl 第三档 | 2000 |
| 真实跨时相定性检查 | stable_v2/test 原始图像 | 500，无精确GT |

第一档几何，第二档增加辐射扰动，第三档增加遮挡；输出比例0.8/0.6/0.4。划分以母图/地理区域隔离，禁止重新按合成pair随机切分。stable_v2的几何安全约束改变了数据分布，不能把旧v1和stable_v2的指标直接比较为方法提升。

固定test manifest SHA256：`8dd705d520d7753cd9e83442a99b10c8a7f984db220be628d2db49c3dad78fcd`。每次运行记录train/val/test manifest和母图划分哈希、实际pair_id；更换数据必须另立协议。历史test已经被观察，后续只作固定回归集，不能反复据其结果调参；正式新结论应在val锁定方法后使用新的、按地理母图隔离的封存测试集。真实500对只能定性展示，不能填入合成精度表。

## 3. 最优训练链与公平对照

实际来源：早期 shared_descriptor_frozen_tier3_seed0/latest.pt → stable_v2共享第一/二/三档各一轮（12995/8663/8663步）→ 半密集第一档C一轮12995步 → 半密集第三档E一轮8663步。E没有经过半密集第二档A，也不是D续训。

共享阶段冻结DINO，训练MVT/GHIM head/VGG/CGMDP；峰值LR分别为1e-7/1e-7/5e-7/1e-6，每档新优化器，5% warmup+cosine。半密集阶段冻结DINO/MVT/GHIM，固定VGG BN统计，训练VGG/CGMDP/温度/QRRU。第一档LR为2e-5/2e-4，第三档E为8e-5/8e-4；AdamW+cosine，每档重置优化器/调度器；BS1×累积4，seed0，Lc+Lf+Lq权重均1。训练window_chunk=32，iterations=4，radius=1.5，gamma=0.8；粗查询1024，细窗口64，QRRU训练查询128。完整默认值以对应配置及checkpoint metadata为准，禁止凭摘要重新猜测。

推荐先做“第三档受控对照”：所有方法从同一第一档C完整权重开始、第三档34652对一轮8663步、相同冻结范围及有效batch。E为主基线，C/D为低LR对照；每次仅改变一个因素。不要让基线从第一档C开始，而新方法从已经训练完的E开始。

种子使用0/1/2，分别报告及均值±样本标准差。共享同一第一档C时，称为“固定预训练来源的第三档随机性”，不称为全流程三seed。全流程比较需每个seed重训第一档和第三档，记录早期共享预训练成本；外部方法单列预训练来源与适应预算。

历史复现使用每档最终checkpoint。新实验若选择最佳checkpoint，必须所有方法统一按完整val的final宏平均P@1px选择，同分用EPE；验证频率事先固定。训练内GT解耦loss只作诊断，不能替代纯预测级联精度。模型结构消融不直接绕过现有checkpoint兼容性校验，应另建明确的权重迁移入口。

第三档复现训练（新输出目录，用户自行选择空闲GPU）：

```bash
cd /home/disk1/MHINet
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/loma-repro/bin/python \
  scripts/train_semidense_next_tier.py \
  --config configs/semidense_tier3_lr_e.json \
  --checkpoint outputs/semidense_stable_v2_tier1_lr_c_seed0/latest.pt \
  --output outputs/benchmark_v1_e_seed0_train
```

## 4. 固定评分规则

沿用 mhinet/downstream/metrics.py，预测结束后才读取GT评分，不使用GT初始化H0、候选筛选或QRRU。

- 坐标为原图像素中心；误差 e=||H_GT(a)-b||，单位为B原图像素。不要混用392输入像素或归一化坐标。
- P@τ：双侧共同可见重叠mask内且e≤τ的点数 / 所有输出匹配数；阈值1/3/5/10px。空输出precision=0；最终对全部影像对宏平均。额外的overlap_precision分母不同，不能替代P@τ。
- EPE：每对有效重叠匹配的平均误差，再对有定义的影像对宏平均；无有效点记缺失，不能填0。必须同时列epe_pairs/总对数、失败率和匹配点数，防止仅输出少量容易点虚增精度。
- NCM@阈值、平均匹配数、16×16空间覆盖率同时报告；success定义为每对至少20个5px正确匹配，空输出失败。
- coarse/fine/final分别输出；fine使用QRRU过滤前完整集合。跨方法可另报固定top-K匹配预算（例如K=1000，按各自置信度排序），并保持原生输出结果；不得把top-K结果混入历史表。
- 推理failure_reason、非有限点、出界及QRRU拒绝数单列；全体pair必须出现，进程中断算未完成，不能把部分输出标为全量。
- legacy_RMSE_correct5_or_failure10是正确子集条件指标，不是配准RMSE。本节的合成对照没有拟合单应矩阵的精度证据；后续 RGB512 双数据集测试已另外记录 RANSAC 和网格配准 SR 协议，不能与这里的点匹配P@1或至少20个正确点的success混为同一指标。

## 5. 两条执行轨道

A 精度复现：显式inference_window_chunk=32，完整第三档val 3848对及固定test 2000对；历史E权重SHA256为 `b5995d184240be0be70d24a44dfbdc6e6f79050c735b0f8e7547542b40d97f50`。

B 部署速度：相同权重改1024，在同一空闲物理GPU顺序测试；先在完整val核对E的精度、失败率和点集变化，再固定用于后续比较。C的完整val已经验证1024平均223.930ms/对（相对32约17.43倍），但有10对源点集合变化；E只有3对兼容性核对，不能把C速度直接当作E速度。

现有计时入口排除读图、传输前等待、预热、GT评分、绘图和写盘；CUDA event模块区间包含CPU提交间隙。报告wall mean/p50/p95/p99、模块耗时、peak allocated、GPU型号/UUID、软件版本和精度设置。历史C/D/E不同时段/物理GPU的耗时不能做速度排名。速度正式比较同卡同设置顺序重复3次，沿用第一对单独预热，并声明此预热协议；若新增多次预热则所有方法统一且另标计时版本。

可直接运行脚本：

```bash
# 默认先评完整val；GPU_ID需设为空闲卡，目录必须尚不存在。
GPU_ID=0 bash scripts/benchmark_semidense_v1.sh \
  outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt \
  outputs/benchmark_v1_e_val32 val 32
# 方法及checkpoint锁定后，再做历史测试回归。
GPU_ID=0 bash scripts/benchmark_semidense_v1.sh \
  outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt \
  outputs/benchmark_v1_e_test32 test 32
```

输出registration.json、report.json和逐对pairs.jsonl；EPE等详细字段在pairs.jsonl。报告模板：方法、训练来源、seed、数据SHA、checkpoint SHA、chunk、P@1/3/5、EPE及有效对数、NCM/点数/覆盖率、success、failure、mean/p95 ms、显存。保留git commit及未提交diff、命令和依赖版本。新方法与E按相同pair_id作配对比较；置信区间按母图/地理组bootstrap，不能把同母图多个pair当成独立样本。

本次仅制定基准与入口，未启动训练或GPU评测。
