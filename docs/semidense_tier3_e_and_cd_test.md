# 第三档 E 训练及 C/D 独立测试（2026-09-26）

E 从第一档 C 最终完整权重初始化，第三档一轮8663步，CGMDP/VGG 初始LR 8e-5、下游8e-4，为第三档 D 的两倍。GPU0；新 AdamW 和余弦调度，冻结 DINO/MVT/GHIM-H0，其他训练条件与 C/D 保持一致。

第三档 C、D 最终权重分别在 GPU1、GPU3 上用相同独立测试集评估：GoogleEarth_test_single_tier3_v1/test/pairs.jsonl，共2000对，精确同母图合成真值。stable_v2/test 的500对原始跨时相图像没有GT，不适合直接计算精度，因此本次不把合成测试称为真实跨时相配准测试。

使用现有 evaluate_semidense 完整预测，GT 仅用于预测之后评分。逐对输出含 coarse/fine/final 的匹配点数、precision、重叠区 EPE、成功率与推理失败原因；不宣称拟合单应矩阵的精度。完整测试结果可用于报告，后续训练设置仍应依据验证集。

`scripts/summarize_semidense_cd_test.py --watch` 自动汇总至 outputs/semidense_tier3_cd_test_comparison/index.html 与 comparison.json。只有两组各2000对、最终report存在、pair_id及manifest一致时标记完成。各阶段EPE有效样本数单独报告，缺失不填0。并行不同GPU的耗时仅供参考。

测试目录：outputs/semidense_tier3_lr_c_test2000 和 outputs/semidense_tier3_lr_d_test2000。权重SHA、测试manifest SHA、命令和PID见 artifacts/semidense_tier3_e_and_cd_test_registration.json。中文训练看板已加入E组与测试对照链接。
