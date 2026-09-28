# 推理分块完整验证（2026-09-28）

修改前完整工作区已保存为 be9025c，远端标签 baseline/pre-chunk-accuracy-20260928。包括当时未提交代码、记录与文档删除；未将outputs权重纳入Git。

同一第三档C最终权重，在 stable_v2 第三档完整val 3848对上，对每对分别评估32、128、256、512、1024分块。仅更改实例的window_chunk；不改训练配置、权重、默认模型源码、12000点上限、4轮QRRU或H0先验。相同影像对保持在同一张GPU；32作为基准先运行，其他分块顺序按样本轮换。

GPU0跑偶数索引1924对；GPU1等待D重测结束/显存低于256MiB后自动跑奇数索引1924对。GPU2已被其他任务占用。每对必须完成全部五种分块后才写入一条记录。第一次对每个分块分别预热。

统计各阶段precision@1/3/5/10px、重叠EPE和有效样本数、匹配点数、成功率/失败率；源点集合、顺序、新增/丢失源点分开记录，按公共源点对齐后记录目标坐标漂移。重复源点不强行对齐；相同精度均值不自动等同于逐对无退化。也记录模块耗时、总耗时和peak_allocated。

启动：scripts/launch_semidense_chunk_accuracy.py；评估：scripts/evaluate_semidense_chunk_accuracy.py；汇总：scripts/summarize_semidense_chunk_accuracy.py。入口拒绝覆盖已有目录和登记。三个CPU比较测试覆盖排序变化、集合变化、坐标偏移及空集/重复点处理。

结果页 outputs/semidense_chunk_accuracy_val3848/index.html，原始分片pairs.jsonl与完整comparison.json在同目录。artifacts/semidense_chunk_accuracy_registration.json记录排队/启动/结束状态，artifacts/semidense_chunk_accuracy_results.json持续更新。只有两分片均完成、3848个唯一pair_id/索引、权重/manifest/脚本SHA一致时标记完整。运行期间不据部分样本宣布精度无损，也不自动修改生产默认值。
