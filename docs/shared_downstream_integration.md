# 共享网络双下游：登记与迁移进度（2026-09-21）

## 接口与 H0 路径

共享网络一次计算 GHIM/CGMDP，输出预测H0（A→B、align_corners=False归一化坐标）、fit有效位和[B,2,256,H,W]描述子。两个可选消费者：原MHIR六轮H精修；新增LoMa H引导零样本D8→D2匹配。后者不是原src/loma的DaD/LoMaB关键点匹配器，也不是RRU或LoFTR fine训练头，当前不声称后两者已迁移。

参考源：`/home/disk1/LoMa/experiments/stage1_dedode_pyramid_hroi_v1`，git `899fdb99a4076e312196bbbf99a3c329739b5d7a`；6个核函数文件逐字节固定SHA，登记 `artifacts/loma_downstream_source.json`，迁入`mhinet/downstream/loma_reference`并保留MIT LICENSE。不复制旧model的强制eval/train(False)与共享层冻结包装。

H0贯穿三个步骤：

1. 在D8与D2栅格中心，用H0及其逆投影构造预测双向重叠支持；不是使用GT overlap选择匹配。
2. D8在支持区内做cosine相关性与FP32 dual-softmax，筛选粗对应。
3. D2细匹配利用 `q_i = H0(p_i) + [b8 - H0(a8)]`：P3展开局部窗口；P6使用H-warped 4×4对称采样。不是只用H0裁剪一次然后丢弃。

迁移主对照P3：dual_topk、9×9窗口、local_mutual、温度0.05、粗上限6000/最终12000；P6：mutual粗筛＋H-warped sym4，其余使用原登记默认。没有比较精度后择优改默认。D8=98、D2=392、输入784固定；不消费D4，但共享累计解码仍经过D4；不计算D1。

`HGuidedDenseDownstream`只接受共享输出，不接受GT，且不持有共享模块。它明确是no_grad推理消费者，硬筛选不是可微训练头；非法H0在逆/投影前隔离并记录失败。不把这层no_grad带回GHIM或共享训练，也不修改现有MHIR梯度路径。

## 登记的运行

共同checkpoint：`outputs/shared_descriptor_frozen_tier3_seed0/latest.pt`，SHA256 `5a9ed14cc32a1a4ff3a843b737410da13d795b79c42a9d33d82a068226e30386`。数据：stable_v2/val，SHA `9e69f1af29dca1073ca3f12a618d61b52b66a76e67346d2ab39830de17ec5f44`。test未参与配置选择。

| 实验 | 数据与范围 | 状态 |
|---|---|---|
| shared_tier3_stable_v2_baseline_tier1 | 第一档完整val5772 | GPU0后台启动，待report.json确认 |
| shared_tier3_stable_v2_baseline_tier2 | 第二档完整val3848 | GPU1后台启动，待report.json确认 |
| shared_tier3_stable_v2_baseline_tier3 | 第三档完整val3848 | GPU2后台启动，待report.json确认 |
| dense_loma_p3_smoke | 第一档前2对，仅工程冒烟 | 通过 |
| dense_loma_p6_smoke | 同上 | 通过 |
| loma_downstream_parity | 同一对真实共享特征和H0 | 通过，详见下文 |

运行前登记见各输出registration.json；集中快照`artifacts/shared_downstream_registration_v1.json`中的running为登记时状态，不是实时状态。完成以对应report.json为准。评估脚本新增显式--data-root，不会静默沿用checkpoint内旧数据路径，拒绝覆盖非空输出。

P3/P6两对冒烟均无失败，平均匹配数12000/5969.5，GPU3峰值allocated约2.335GB；只用于确认连通，不用于精度排名。计时含首次调用，不是稳定基准延迟。

## 验证证据与限制

`scripts/check_dense_migration.py`对原LoMa的完整下游_match编排注入同一份共享特征/H0，与新消费者比较。P3/P6分别12000/5939对应，按坐标排序后A/B点差均0，置信度最大差7.15e-7/0。起初按输出顺序逐项严格比较失败；复核为近似并列置信度导致排序不同，不是对应点集合不同。记录排序方式与容差，不称逐比特完全一致。此检查隔离下游迁移，不是完整旧共享权重端到端复现。共享调用计数DINO=1/MVT=1/VGG=1，D1=0。证据`artifacts/loma_downstream_parity.json`。

旧版4组匹配/几何测试迁入并仅修改import，36项通过；新增安全拒绝、原文件SHA、评价空集/区外点测试3项通过，总39项1.86秒。loma-repro补装pytest8.3.5（iniconfig2.3.0/pluggy1.6.0/tomli2.4.1），无模型依赖升级。

## 评价口径

旧LoMa评价来源为scripts/evaluate_googleearth_scale_pairs.py::pair_metrics，源SHA已登记。保留NCM@1/3/5/10、全部输出点作分母的precision、overlap_precision、至少20个5px正确点的SR。原生目标像素坐标，不能与共享描述子784坐标指标直接比较。双侧GT mask只用于评价，不提供给匹配器。

旧RMSE仅在GT判定正确点上计算，且失败填10；迁入命名`legacy_RMSE_correct5_or_failure10`，不是估计H误差或全部点误差。新旧评价在对齐样本的各项数值一致。额外记录有效重叠内有限点EPE、非法匹配数、输出匹配数、失败率、延迟、显存。暂未完成空间覆盖率、置信度曲线、匹配拟合H的几何评价；不可将当前移植阶段宣称为完整评价体系。

## 运行入口和下一阶段

原MHIR入口不变；新增 `python -m mhinet.cli evaluate-dense`，或 `GPU_ID=3 TIER=1 RESULT_ID=C3-P3 MHINET_OUTPUT_DIR=outputs/your_new_name bash scripts/validate_dense_loma.sh`。默认对新版val的指定tier全量验证；--limit只能显式冒烟。输出目录已存在即拒绝覆盖。正式test需要显式指定对应test manifest，不把val冒充test。

下一阶段：等三档完整共享基线结束归档；补齐拟合H/覆盖率评价；完整验证P3/P6并登记，不用两对冒烟选择配置；再审计RRU/LoFTR fine的checkpoint和训练协议后决定可训练下游。当前未训练或解冻任何新下游。
