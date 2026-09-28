# D覆盖重测与E组测试（2026-09-28）

用户要求D重新测试并覆盖上次异常耗时结果，随后加入E组测试。D使用GPU1（与C此前的物理GPU相同）；E使用GPU3。两组是不同物理卡上的独立进程，不共享GPU。C结果保持不变。

D旧测试输出和日志已删除，原目录写入本次重测。D监管入口 scripts/rerun_semidense_d_timed.py 会校验C全部文件哈希保持不变，并自动更新原C/D记录及文档。E使用 scripts/run_registered_semidense_test.py 启动。两组每30秒记录GPU利用率、显存、功耗、温度、频率及同卡进程，无法覆盖采样间所有短暂负载。

E已完成第三档8663步训练，3848对完整验证总loss为0.640607（Lc 0.102965、Lf 0.166873、Lq 0.370769）。此次测试与C/D完全相同：独立同母图合成第三档2000对，完整纯预测级联，GT只用于评分；不是原始跨时相精度。

评估代码与权重不变，保留所有模块耗时及分位数；每组第一对再次做计时/不计时预测逐位一致校验。C/D/E汇总由 scripts/summarize_semidense_cde_test.py 自动刷新，完成时核对全部pair_id、manifest SHA及计时记录数量。

页面：outputs/semidense_tier3_cde_test_comparison/index.html；汇总：artifacts/semidense_cde_timed_test_results.json。D/E启动记录分别为 artifacts/semidense_cd_timed_test_registration.json 和 artifacts/semidense_e_timed_test_registration.json。D/E的GPU遥测分别为 outputs/semidense_tier3_lr_d_test2000.gpu.jsonl 和 outputs/semidense_tier3_lr_e_test2000.gpu.jsonl。
