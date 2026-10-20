# RGB_512 读取性能诊断与预读

2026-10-20：Flow Head 原训练 PID 32188 持续运行；没有修改训练源码、配置、模型、优化器、随机流、batch、学习率或数据划分。实验只增加独立的只读文件预读进程。

数据在 HDD/ZFS 上，ZFS ARC 上限为8 GiB。两名原 DataLoader worker 曾同时处于磁盘等待，主进程等待数据。固定32对 CPU 装载首次冷读19.142秒；同一批热读0.477秒。热缓存结果只证明CPU解码有余量，不能外推为全量训练加速。

最终预读包括随机 JSONL 行所在块和四个 PNG，使用与训练完全相同的 CPU torch.randperm(seed+epoch)，按 epoch/cursor 提前128对，8线程。不会解码图片或写入源数据，也不修改系统缓存上限；epoch末停止提前读下一epoch，等待训练游标推进。预读程序为 scripts/mhi_v2/prefetch_rgb512.py，实际状态在 outputs/mhi_v2/engineering/io_prefetch_complete_20261020/status.json。

| 正常优化步窗口 | 平均秒/8对 | 中位秒/8对 | P95秒/8对 | 对/秒 |
| --- | ---: | ---: | ---: | ---: |
| 预读前19800–19989，190步 | 2.6223 | 1.9717 | 4.9659 | 3.0508 |
| 预读后20124–20223，100步 | 0.5261 | 0.3846 | 1.1297 | 15.2074 |

该连续窗口的平均优化步耗时改善约4.98倍。它是顺序前后观察，含不同训练样本；日志中的优化步计时不含保存和验证，不能当作整个剩余任务的精确加速或ETA。独立20秒GPU采样平均利用率19.25%，说明B1串行前后向和标量同步仍有优化空间。保留当前B1×累积8轨迹；真实batch8需要逐pair损失归一化和BF16误差单独验收，QRRU当前不能直接套用batch8。

证据：io_benchmark_first_pass.json、io_benchmark.json、prefetch_acceptance.json。首次大窗口只预读PNG效果较弱；原始尝试状态/对应脚本快照保存在 outputs/mhi_v2/engineering/io_prefetch_20261020/ 与 io_prefetch_small_20261020/，不覆盖历史尝试。最终方案同时预读清单块与PNG。

后续C/D/E的NVMe字节一致副本正在独立准备；复制校验通过前，不切换训练配置或把数据副本标成可用。封存评价集不进入复制范围。
