# MHI v2 独立入口

| 入口 | 用途 |
| --- | --- |
| [preflight.py](preflight.py) | 只读复现 P0 提交的分档/资源预检；保留历史 planned 状态断言 |
| [validate_real.py](validate_real.py) | 真实共享特征、Flow/B/C、零 adapter 轨迹、三步 QRRU 梯度及冻结审计 |
| [prefetch_rgb512.py](prefetch_rgb512.py) | 独立只读预读未来清单块与PNG，按运行seed/epoch/cursor，不改变训练进程 |
| [cache_rgb512.py](cache_rgb512.py) | 只复制train/val引用文件到新NVMe目录，逐文件双SHA验证，完成后发布manifest |
| [run_campaign.py](run_campaign.py) | 单 GPU 串行调度 Flow Head → 等预算 C → D → E，阶段成功后才继续 |
| `python -m mhinet.v2.train` | 独立运行配置、训练、完整 val 选模、checkpoint 与严格续训 |
| `python -m mhinet.v2.evaluate` | v2 checkpoint 在指定 RGB_512 清单上的级联评价，可同权重切换 prior |

真实验收已生成 [passed 报告](../../versions/v2/evidence/real_feature_acceptance_20261020_final.json)；重复验收须指定新的 `--output` 文件名以保存历史。Flow Head 的两步实数据烟测与断点恢复已通过，Flow Head已完成5 epoch/25535步，完整5106对 val 最佳EPE为0.7611818px；C在GPU0约800/25535步，D/E排队。实时进度以 `campaign.json`、各阶段 `status.json` 和日志为准。

`run_campaign.py` 要求 `--output`、`--engineering-report`，`--gpu` 选择物理 GPU。它检查真实工程报告为通过，并按 [注册配置](../../configs/mhi_v2/README.md) 串行运行；不会跳过失败阶段。阶段输出固定在各实验的独立 P5 目录；恢复只允许符合元数据、源码/数据身份与 RNG/游标检查的同协议 checkpoint。

正式注册：seed42；Flow Head 和 C/D/E 各5 epoch、25535更新；仅原 RGB_512 train=40849/val=5106；512输入、最终2048、QRRU前12000。每 epoch 完整 val 选择 Flow EPE 最低的 Head 和级联 Pre@3 最高的 QRRU。中途64对验证仅诊断。只使用 train/val，封存评价集不参与训练或选型。源图身份交叉为0，但未证实地理邻近隔离；单seed首轮不能支持多seed结论。

查看已实现入口参数，不启动训练：

```bash
python -m mhinet.v2.train --help
python -m mhinet.v2.evaluate --help
python scripts/mhi_v2/run_campaign.py --help
python scripts/mhi_v2/validate_real.py --help
```

2026-10-20 性能诊断：机械盘/ZFS随机读取受8GiB ARC限制，新增独立预读后连续100步平均耗时约2.62→0.53秒；训练源码/配置/状态不变。NVMe副本183822文件、32157538405字节已通过逐文件双SHA验证。C继续原路径；仅排队D/E改用 `runtime.rgb512.nvme.server.json`，变更及当前配置哈希见 [执行修订](../../versions/v2/evidence/performance_20261020/nvme_activation/activation.json)，原预登记哈希保留。数据字节、数学、预算与seed不变；不同存储的墙钟时间不能直接作模型速度对照。详见 [实际性能证据](../../versions/v2/evidence/performance_20261020/README.md)。
