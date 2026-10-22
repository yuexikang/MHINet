# 冻结既有 MVT / MatchEmbedding 的杭州对照

本轮在上一轮基础上只改变初始化及冻结范围：两版均从 `weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt` 加载 MVT 和完整 MatchEmbedding 定位头（含位置编码缓冲区、no-match 参数）并冻结，DINO 继续使用原始预训练权重并冻结。VGG、CGMDP、匹配温度和 QRRU 重新初始化训练，不加载上一轮杭州权重。

相同的 256 对训练、64 对验证样本，直接复用上一轮 selection.json。其余超参数不变：512 输入、2048 输出上限、学习率 2e-4、累积 4、余弦退火、最多 30 epoch、patience 6。v1/v2 保留各自的推理和损失方案；依然只有近似地理 H 标签，没有真实点或非刚性 Flow 标注。

- v1：物理 GPU0，配置 `frozen_v1.json`，输出 `outputs/hangzhou_frozen_locator/v1_20261022`。
- v2：物理 GPU1，配置 `frozen_v2.json`，输出 `outputs/hangzhou_frozen_locator/v2_20261022`。
- 日志：`outputs/hangzhou_frozen_locator/v1.log` 与 `v2.log`。

两版各完成两次短运行优化更新：MVT/定位头梯度为零，DINO 无梯度，其余活跃模块梯度非零且有限。短运行权重不用于正式训练。

本轮 Flow 分支完全冻结，因此 Flow EPE 应基本保持固定；关注训练后的匹配精度和 QRRU 收益。两版可能不同时间结束，对照报告是生成当时的快照，可在都结束后重新运行：

```bash
/root/miniconda3/envs/loma-repro/bin/python -m comparisons.hangzhou_v1_control.report \
  outputs/hangzhou_frozen_locator/v1_20261022 outputs/hangzhou_frozen_locator/v2_20261022
```
