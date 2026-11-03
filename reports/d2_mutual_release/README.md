# D2 局部互选合入验收（2026-11-03）

在正式主线 `5bc3497` 的加深模型上接入可选 D2 输出互选。默认 `mutual=False`，不重训、不改权重，不引入研究分支的低 Matchability 源点过滤。

## 双方代码核对

来源为 `/home/disk1/MHINet/tools/diagnostics/qrru_transfer.py` 的2026-10-31无训练诊断。诊断只从已有候选矩阵筛选输出；主线的 `match_windows` 已提供所需 logp、valid、坐标，因此无须增加特征提取或解码器。核心规则统一到 `pair_matching.local.mutual_filter_mask`，正式 `match/match_shared` 与独立 `infer` 共同调用。本地旧诊断脚本也已改为调用此函数。

Jacobian研究工作区在 `pipeline.py/local.py` 另有未提交的D8源点过滤改动。本次直接在干净的正式主线选择性实现D2过滤，没有整体合并该工作区，因此未带入它的默认行为修改。此前D8父点互选保留；D2互选作用于不同尺度，并非重复执行同一个筛选。

未纳入与本次无关的未跟踪V1对比脚本或报告。Self/Pair调制、连续D8、局部Pair、D2两残差块和训练损失均保持原样。

## 验证

- 32对实图：从原诊断RGB256对和杭州235对中分别等间隔选16对。
- 主线默认计数、过滤后计数均与原诊断逐对一致。
- 开启互选后全部坐标/置信度/有效标记/父点关联，严格等于未过滤结果的原顺序子集。
- 正式入口与独立Pair-D2入口结果逐元素一致。
- CPU新增测试覆盖冲突候选、并列分数、无有效候选、越界/非有限坐标、空父点、关联同步及公开match接口透传。
- 首次沙箱测试中5项旧测试因在只读工作树创建临时目录失败；允许工作树写入后重跑，最终 **93项全部通过（4.89秒）**。未修改这些旧测试。

实图记录为同目录 `verification.json`。这次是迁移验收，不是新一轮训练或完整测试集评估。原诊断结果与正确点数量代价见正式使用说明。

## 使用与复现

```python
model = load_mhi_v2(runtime=runtime, assets_root='/home/disk1/MHINet')
result = model.match(images, mutual=True)
```

```bash
# 在正式主线工作树运行
OMP_NUM_THREADS=2 python -m pytest tests/v2 -q
CUBLAS_WORKSPACE_CONFIG=:4096:8 python tools/mhi_v2/check_d2_mutual.py \
  --assets-root /home/disk1/MHINet \
  --reference /home/disk1/MHINet/outputs/qrru_transfer_20261031 \
  --runtime /home/disk1/MHINet/configs/mhi_v2/runtime.rgb512.server.json \
  --output reports/d2_mutual_release/verification.json --device cuda:2
```

改动文件：`mhinet/v2/pair_matching/local.py`、`mhinet/v2/pipeline.py`、`tests/v2/test_d2_mutual.py`、`tools/mhi_v2/check_d2_mutual.py`、`mhinet/v2/README.md`、`docs/mhi_v2/flow_pair_release.md`及本目录。权重文件留在本机资产目录，不提交Git。
