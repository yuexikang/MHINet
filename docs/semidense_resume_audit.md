# 512 双卡训练与续训审核（2026-09-30）

审核范围：`mhinet/downstream/train.py`、检查点 I/O、可视化调度、下一档权重启动入口，以及此前临时抽样验证的恢复方式。Lc/Lf/Lq、D8/D2、H0 残差先验和 QRRU 算法没有改动。

## 发现与处理

| 优先级 | 原问题及影响 | 修改 |
| --- | --- | --- |
| P1 | 加载 RNG 后再次 `manual_seed(seed+rank)`，抵消随机状态恢复；只保存 rank 0 RNG，另一 rank 不可能精确恢复 | 保存每个 rank 的 Python/NumPy/Torch/当前 CUDA RNG；全部初始化完成后恢复，不再覆盖 |
| P1 | rank 0 验证 5,106 对，rank 1 在 NCCL barrier 等待超出 600 秒，导致整组退出 | 验证按无重复索引分到各 rank；诊断与 I/O 等待使用 CPU/Gloo 控制组，梯度通信保留 NCCL 及原有短超时 |
| P1 | `broadcast_buffers=False`，各 rank BN 统计不同；原检查点只留下 rank 0 的统计 | 新检查点保存各 rank buffers；分布式验证临时使用 rank 0 buffers，完成后恢复各 rank 原值 |
| P1 | 抽样验证只在已经删除的临时脚本中，重新启动会失效；直接导入 trainer 还会漏掉下一档权重加载适配 | 正式 `--validation-pairs` 参数，策略写入检查点和 `execution.json`；下一档入口显式传入 source loader；续训直接加载完整训练检查点 |
| P2 | 每个优化步三次 barrier；周期保存与可视化保存重复；验证前不一定保存 | 普通训练步无额外 barrier；合并保存条件，在验证/可视化前统一保存一次 |
| P2 | 续训先读取并拷贝源权重，再被断点权重完全覆盖；重复读取大检查点 | 续训只构造模型后加载一次 mmap 检查点；仍核对来源、数据、配置和实现标识 |
| P2 | 每次恢复都额外生成一套全通道特征图；每个展示样本都重新读取整份训练日志并绘制概览 | 恢复不自动补画非计划图；每组 snapshot 完成后只生成一次概览 |
| P2 | 回退检查点需人工截断日志，重复步数影响曲线，原记录丢失 | 在获得任务锁后原子整理日志，超出检查点的记录和诊断结果归档到 `resume_history` |
| P2 | 每个进程捕获所有可见 GPU RNG，会接触不属于自己的设备；运行设备字段之前被无条件忽略 | DDP 仅捕获当前 CUDA stream；设备兼容例外限于 CUDA 编号变化，其他 runtime 字段仍严格检查 |

## 算法边界

`training_losses()` 中的 H0 被 `detach()`，且细匹配窗口和标签在 `no_grad` 中生成。D8/D2 由独立描述子支路产生，因此仅使用 Lc/Lf/Lq 时，GHIM 的 H0 预测头没有直接梯度；将它设为 `requires_grad=True` 不能改变这一点。共享 MVT 仍可以通过描述子损失更新，并间接改变 H0 输入。本次保留这一原有先验设计，没有擅自给 H0 增加监督或移除 detach。

训练行中的 `records` 仍是 rank 0 的本地 microbatches，现明确标注 `records_scope`；不是两卡所有样本损失的均值。梯度通过 DDP 归约。

## 旧检查点兼容与指标口径

- 旧检查点缺失其他 rank RNG/buffers，无法事后恢复丢失的信息。迁移时标记 `legacy_missing_rank_rng`，rank 0 恢复已有状态，其他 rank 使用明确的确定性种子；之后的新检查点支持逐 rank 接续。
- 只允许审核前已知的 `train.py`、`visualize.py` 散列迁移。模型、损失、匹配器、数据、学习率、总步数、world size 等仍严格匹配。迁移记录保存在 `resume_events.jsonl`。
- 新验证使用固定清单索引、按样本索引设定随机种子，与 rank 数目无关。报告包含实际索引、样本总数、`rng_policy=per_manifest_index_v1`；旧版按整次验证设种子的损失不能直接视作严格配对结果。
- `--validation-pairs 512` 使用均匀分布的固定 512 对，`0` 表示完整验证集。续训省略该参数会沿用新检查点记录的策略。旧临时脚本的抽样策略没有写入检查点，首次迁移必须显式传入 512。

## 验证

- CPU：检查点、真实双进程恢复、验证分片、错误传播、日志归档、QRRU 损失/梯度、512 坐标及可视化测试，共 28 项通过。
- GPU：在独立 GPU 2、3 上验证双 rank 随机 dropout + BN + AdamW/调度器中断接续，下一步权重和 buffers 与不中断路径逐值一致；分片验证与串行验证逐样本一致。
- 控制组测试：诊断等待超过梯度进程组的超时后，下一次 backward 仍正常完成。
- 真实权重：在独立 GPU 2、3 上从第 5,750 步完整恢复模型、优化器及数据进度，新入口退出码 0；临时测试目录自动清理。
- 性能改善先按减少的调用量描述，不将不同样本的单步计时当作严格加速对照。

## 使用

同一任务目录存在 `latest.pt` 时，启动器自动恢复，默认使用检查点保存的训练配置与来源：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/train_semidense_rgb512_ddp.sh \
  outputs/semidense_rgb512_synth_seed42_5ep --validation-pairs 512
```

第一次启动新任务仍可指定 `TRAIN_CONFIG` 与源权重。正式训练不再依赖 `/tmp` Python 启动脚本。
