# D2 加深版合入 MHI-v2（2026-10-30）

正式分支 `codex/mhi-v2` 默认采用加深实验第10轮。来源 `codex/d2-capacity@766f445`，仅移植 deep 组的两个64通道残差块及其权重，不引入 local/wide 实验结构。

残差块插在 CGMDP D2 的 hidden residual 之后、输出投影之前，每块为 1×1 → GELU → depthwise 3×3 → GroupNorm → GELU → 1×1，新增18,176参数。Flow、D8、原局部 Pair、2048点输出协议和 QRRU 关闭状态保持原设置。

## 已完成实验结果

4096对 RGB 训练、512对验证，固定10轮，无早停；仅训练原 D2 和新增模块。基础组采用相同训练预算；best 按 RGB 验证 P@3 选取，两组均为第10轮。以下为此前已完成的完整测试，本次合入没有重新训练或重跑全部测试。

| 模型 | RGB P@1 | RGB P@3 | RGB EPE | 杭州 P@1 | 杭州 P@3 | 杭州 EPE |
|---|---:|---:|---:|---:|---:|---:|
| 同预算基础组 | 90.826% | 98.346% | 0.6858 | 22.268% | 62.671% | 11.6957 |
| 加深组（当前默认） | 91.742% | 98.360% | 0.6663 | 22.572% | 63.036% | 11.6844 |

RGB test 5106对，杭州 test_real 235对。P@k 分母为全部输出点，EPE 在真值可见点上统计，单位为512原图像素。杭州指标是对数据集 H 的一致性。主要收益是 RGB P@1；只有一个训练种子，不据此宣称跨种子显著提升。完整加深组指标见 [results.json](results.json)。

## 合入验证

- 既有 `tests/v2`：90项通过。
- RGB16对 + 杭州16对：主线部署模型与原实验模型的所有 state tensors 完全一致；Flow、D8父点/中心、D2候选 log-probability、最终坐标/置信度/有效掩码逐元素完全一致。
- 32对逐图 P@1/3/5/10 计数与已归档实验结果一致，记录见 [verification.json](verification.json)。
- 本次没有新训练，没有测推理加速。

## 文件清单

- `mhinet/v2/pair_matching/deep_d2.py`：独立运行的加深模块。
- `mhinet/v2/pipeline.py`：正式加载入口；`pair_matching/training.py`：支持加深权重初始化及保存结构标记。
- `configs/mhi_v2/releases/flow_pair_v2.json`、`flow_pair_v2_deep.weights.json`：当前默认配置/权重清单；`flow_pair_v2_base.json`：保留原未加深模型。
- `tools/mhi_v2/export_deep_d2.py`：完整部署包导出；`check_deep_d2.py`：实验模型与主线对照。
- `tools/mhi_v2/check_resume.py`、`audit_reduction.py`：原基础模型诊断显式使用旧配置。
- `mhinet/v2/README.md`、`docs/mhi_v2/flow_pair_release.md`、本目录：入口说明与结果。

## 运行命令

在主线工作树执行，环境为 `/root/miniconda3/envs/loma-repro/bin/python`。下列路径是本机资产位置。

```bash
python tools/mhi_v2/export_deep_d2.py \
  --base /home/disk1/MHINet/outputs/full_rgb_d2_10epoch_20261028/d2/best.pt \
  --deep /home/disk1/MHINet/outputs/d2_capacity_20261029/runs/deep/best.pt \
  --output /home/disk1/MHINet/weight/mhi_v2/flow_pair_d2_deep_20261030.pt

CUBLAS_WORKSPACE_CONFIG=:4096:8 python tools/mhi_v2/check_deep_d2.py \
  --experiment-root /home/disk1/MHINet/.worktrees/d2-capacity \
  --run /home/disk1/MHINet/outputs/d2_capacity_20261029/runs/deep \
  --runtime /home/disk1/MHINet/configs/mhi_v2/runtime.rgb512.server.json \
  --rgb-root /home/disk1/Data/datasets/MHINet_RGB_512 \
  --hangzhou-root /home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4 \
  --output reports/d2_deep_release/verification.json --device cuda:2

python -m pytest tests/v2 -q
```

正常推理仍调用 `load_mhi_v2(runtime=runtime, assets_root='/home/disk1/MHINet')`；回到原基础模型时额外传入 `config='configs/mhi_v2/releases/flow_pair_v2_base.json'`。部署包包含完整 D2 组件，不包含优化器；不能用其声称无缝恢复原实验训练。权重留在本机资产目录，不进入 Git。
