# 正式训练前 tiny 检查汇总

由 `scripts/summarize_pretraining.py` 根据原始 artifact 生成。
32个受控 H0 样本，seed=0；主读出可能包含预先登记的参数平均，原始终点始终单列。
延迟仅为缓存描述子后的精修，不含 GHIM/CGMDP。未使用 test；未启动 E00/E01。

| 实验 | 状态 | 步数 | H0→各轮H MACE px | 原始终点 px | 主读出 px | 失败/样本 | 拒绝更新 | 显存 GB | 缓存前向 ms |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| D8 | passed | 1664 | 14.868890 → 0.291660 → 0.098226 | 0.290066 | 0.098226 | 0/32 | 0 | 0.572 | 56.555 |
| D4 | passed | 1568 | 7.434444 → 0.191940 → 0.099413 | 0.272937 | 0.099413 | 0/32 | 0 | 1.982 | 64.281 |
| D2 | passed | 1152 | 2.787924 → 0.212890 → 0.099895 | 0.099895 | 0.099895 | 0/32 | 0 | 2.821 | 109.516 |
| TINY-6 | passed | 1568 | 14.868889 → 1.304389 → 0.208130 → 0.184050 → 0.141688 → 0.081053 → 0.081194 | 0.209412 | 0.081194 | 0/32 | 0 | 3.099 | 163.817 |

通过要求为平均最终 MACE < 0.1 px、零失败、零拒绝更新，并满足梯度和协议审计。
四项与合并审计均已通过，合并结果绑定当前四份artifact。按用户要求停止在正式训练前，未执行 E00/E01。
合并gate SHA256：`13719169eaed0d1d887a18d846f4a824525231f427dc8ec3d3456edad6a60ea3`。
四个checkpoint独立重载均匹配资源签名及保存的轨迹；序列化、加载和前向后权重身份一致。

## 证据

- D8：`/home/disk1/MHINet/artifacts/p4_tiny_s_d8_mcnet.json`；SHA256 `7c2f1fae0cf576c7b753fce0c2875c88746101b8b78d3723b85127adc386f294`；主读出 `equal_weight_parameter_average`。
- D4：`/home/disk1/MHINet/artifacts/p4_tiny_s_d4_mcnet.json`；SHA256 `962b3874ea296074e34b2306ebee967a1715921670a7efb466868d69e656243e`；主读出 `equal_weight_parameter_average`。
- D2：`/home/disk1/MHINet/artifacts/p4_tiny_s_d2_mcnet.json`；SHA256 `e35ed757acc633b30dad9af5a791cb17a846352f57f366d3e8911a3a72ef305c`；主读出 `raw_parameters`。
- TINY-6：`/home/disk1/MHINet/artifacts/p4_tiny_6_mcnet.json`；SHA256 `0f60415c4f3c03f235249f2b0cf6eb23a4fdd6ff4c3ec43bc09e9ef2a1546ed2`；主读出 `equal_weight_parameter_average`。
