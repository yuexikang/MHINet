# 正式训练前 tiny 检查汇总

由 `scripts/summarize_pretraining.py` 根据原始 artifact 生成。
32个受控 H0 样本，seed=0；主读出可能包含预先登记的参数平均，原始终点始终单列。
延迟仅为缓存描述子后的精修，不含 GHIM/CGMDP。未使用 test；未启动 E00/E01。

| 实验 | 状态 | 步数 | H0→各轮H MACE px | 原始终点 px | 主读出 px | 失败/样本 | 拒绝更新 | 显存 GB | 缓存前向 ms |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| D8 | passed | 1664 | 14.868890 → 0.291660 → 0.098226 | 0.290066 | 0.098226 | 0/32 | 0 | 0.572 | 56.555 |
| D4 | running | — | — | — | — | — | — | — | — |
| D2 | running | — | — | — | — | — | — | — | — |
| TINY-6 | running | — | — | — | — | — | — | — | — |

通过要求为平均最终 MACE < 0.1 px、零失败、零拒绝更新，并满足梯度和协议审计。
尚有缺失、进行中或未通过项；正式实验门槛保持关闭。

## 证据

- D8：`/home/disk1/MHINet/artifacts/p4_tiny_s_d8_mcnet.json`；SHA256 `7c2f1fae0cf576c7b753fce0c2875c88746101b8b78d3723b85127adc386f294`；主读出 `equal_weight_parameter_average`。
