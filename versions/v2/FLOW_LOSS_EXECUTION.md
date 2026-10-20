# Flow 损失自动实验：P6

本实验在 `codex/mhi-v2` 独立实施，承接 [损失审计](FLOW_LOSS_REVIEW.md)。v1 固定在 `mhi-v1`；当前 C/D/E 继续使用锁定的 Flow-best 与候选缓存。所有新训练、验证和工程记录均放入 `outputs/mhi_v2/Flow0/P6/flow_loss_pilot_rgb512_seed42_20261020/`。

## 已登记的公平对照

| 项目 | L0：额外预算对照 | L1：候选损失 |
| --- | --- | --- |
| 初始状态 | 同一 Flow-best 模型、AdamW moments/step、全部 RNG | 相同 |
| 几何损失 | 原 Cauchy，调用原函数 | `alpha × mean(sqrt(dx_px²+dy_px²+0.5²)-0.5)` |
| 辅助项 | `0.01 mat + 0.0001 cls` | 相同 |
| 额外预算 | 1 epoch，5107 次更新 | 相同 |
| 学习率 | 固定 `1e-6`，显式登记终点延长日程 | 相同 |
| 数据顺序 | seed47，第六遍 train40849，B1×累积8 | 相同 |
| 训练参数 | 原有10个 Flow Head 张量 | 相同 |
| 输入与输出 | 512输入，最终匹配上限2048 | 相同 |

源权重 SHA：`26a24e5239a1b7a3719c638ec7b040a8490c226a7341a0606137d25f0c63c33a`。源检查点完成25535步且对应完整val EPE=0.7611818104px。两臂新增步数均为5107，绝对终点30642。源 cosine 已结束，延长实验保留优化器 moments，另建相同 constant-LR scheduler。

`alpha` 只用 [预登记32对训练样本](evidence/flow_loss_pilot_20261020/train_calibration_ids.json) 校准：23 FLAIR、6 GoogleEarth、3 WHU。真实冻结 context 下，分别计算旧/新几何损失在4个共享卷积 trunk 张量上的梯度范数，令 `alpha=median(old)/median(new)`；记录加权mat/cls梯度、夹角及dustbin梯度。校准前固定IDs；不使用val选择权重。

## 自动执行顺序与验收

1. 等待物理 GPU1 空闲：连续三次检查显存<512MiB、利用率<5%。同时核验冻结的代码、配置、数据及源检查点SHA。该检查不依赖进程列表，以覆盖容器外不可见任务。
2. 真实32对训练样本校准，保存系数和梯度诊断。任何非有限值、源身份变化、意外H拟合或冻结buffer变化均停止队列。
3. 真实工程验收：L0连续2更新，对照L0训练1更新后重启到2；模型、AdamW、scheduler、progress、RNG及训练pair顺序必须完全一致。L1独立1更新，必须与L0同源、同首批输入，所有冻结张量保持不变。工程输出独立于正式实验。
4. 新目录执行正式L0、L1各5107更新，每250步存档；每1000步固定64对诊断。终点完整5106对val完成后才标记完成；最终验证中断后可恢复。主要比较固定预算的 `last.pt`。
5. 用统一新评价入口重测source、L0、L1的完整Flow统计：pair宏平均与分位数、point分位数/尾部、c≥0.25/0.5/0.75的误差与覆盖、实际D8插值先验有效性。固定 [52个val难例](evidence/flow_loss_pilot_20261020/hard_cases.json) 保存逐点NPZ及GT边界距离，仅作诊断。
6. 冻结同一QRRU，先做64对prior-off负对照，再完整执行L0/L1各自prior关/开，共4×5106对匹配。最终关闭prior的点与置信度还须在完整5106对上逐元素相等。
7. 登记匹配父对应保留率：使用现有 `coarse_labels` 的最近格点、逆投影roundtrip、双侧mask定义GT父对应；对比相同最终2048预算下的D8输出父对应，分别报告pair宏平均和GT父对应微平均。该统计不等同于D8插值端点EPE。
8. 自动出具 `decision.json`。L1需同时优于L0的Flow EPE，最终Pre@3提升≥0.2个百分点，双侧覆盖下降≤2%，匹配数≥95%，失败率增加≤0.5个百分点，且负对照通过，才标记值得登记从v1初始化的相同5epoch确认实验。本轮止于pilot验收；确认实验及额外seed仍须另有完整配置和身份登记，未完成确认不得宣称稳定收益或替换C/D/E源。

非有限损失、损坏checkpoint、身份/代码变化、恢复不一致或不完整val均使队列失败，保存日志和partial artifacts。训练自身只从完整optimizer边界恢复；局部验证重跑时保留上一尝试。所有新代码置于 `mhinet/v2/flow_loss/`，不修改正在运行的旧训练入口及其源码身份。

## 运行入口

```bash
/root/miniconda3/envs/loma-repro/bin/python scripts/mhi_v2/run_flow_loss_experiment.py \
  --registration versions/v2/evidence/flow_loss_pilot_20261020/registration.gpu1.runtimefix.json
```

队列用独立tmux会话常驻。重启入口增加 `--resume`，已完成阶段按artifact SHA核验后跳过；正式训练沿用各自latest检查点。当前状态以输出根目录 `status.json` 为准，登记文件不是完成结果。

数据仍是原RGB_512 source-image隔离划分及逐字节验证的NVMe缓存，没有证明地理邻近隔离。评价属于raster-mask诊断；GoogleEarth498、JL1Flight1755不参与训练、选择或本轮评价。

2026-10-20 执行修订：按用户要求将尚未开始任何校准/训练的等待队列从GPU2移到GPU1。原登记和等待记录保留；新登记见 `evidence/flow_loss_pilot_20261020/registration.gpu1.json`，迁移证据见 `gpu1_relocation.json`。两卡均为RTX4090；新子进程仅可见物理GPU1，内部逻辑设备仍为cuda:0，单GPU RNG流、损失、数据顺序与预算不变。

GPU1首次校准揭示多进程AF_UNIX临时socket路径长度限制，未执行任何优化更新。已保留该尝试并将queue的TMPDIR缩短到NVMe专用目录；真实76字节socket创建与6项相关回归验收通过。当前可执行登记为 `registration.gpu1.runtimefix.json`，仅更新queue临时路径及对应测试哈希；loss、训练源码与预算不变。
