# 半密集 QRRU 下游：实现与运行

本入口执行第二阶段 `D8 coarse → H残差引导D2 fine → QRRU`。训练仅使用 Lc/Lf/Lq；QRRU 接口没有 H0，使用两侧规则4×4窗口、四象限累计控制流与连续特征重采样。训练三分支用 GT 解耦，推理严格级联。设计约定见 [执行计划](semidense_qrru_execution_plan.md)。

## 运行

第一阶段训练完成后，显式提供它的 checkpoint；不会自动读取或修改正在训练的目录。支持共享网络训练 checkpoint、共享导出 bundle，以及显式提取完整 MHINet checkpoint 的 feature_provider。新模块随机初始化，不迁入旧优化器。

```bash
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/loma-repro/bin/python \
  -m mhinet.downstream.train \
  --config configs/semidense_qrru.json \
  --checkpoint /absolute/path/to/first_stage.pt \
  --output outputs/semidense_qrru_run
```

同样可使用 `python -m mhinet.cli train-semidense`。配置中的 runtime/data_root 需要与目标数据版本一致；默认配置指向 quadrant_tiers_v1，不能把它冒充 stable_v2 实验。

第一版冻结 DINO/MVT/GHIM，训练 VGG、累计解码器、温度和 QRRU。默认4轮、每轮控制分量范围1.5 D2像素；外圈双线性延拓可能超过控制点范围。第三输出通道是更新门控，不是匹配置信度。

训练目录不得非空覆盖；续训显式提供相同配置、第一阶段来源和 resume：

```bash
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/loma-repro/bin/python \
  -m mhinet.downstream.train \
  --config configs/semidense_qrru.json \
  --checkpoint /absolute/path/to/first_stage.pt \
  --output outputs/semidense_qrru_run \
  --resume outputs/semidense_qrru_run/latest.pt
```

恢复核验配置、数据身份、第一阶段SHA、源码身份，恢复优化器/调度器/RNG/数据位置。`--max-steps` 仅提前停止，不改变调度总步数。`--limit-train/--limit-val` 改变数据范围，属于新的工程协议，恢复时须保持一致。

## 评估

```bash
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/loma-repro/bin/python \
  -m mhinet.downstream.evaluate_semidense \
  --checkpoint outputs/semidense_qrru_run/latest.pt \
  --manifest /absolute/path/to/val/pairs.jsonl \
  --tier 1 --output outputs/semidense_qrru_run/full_val
```

同样可使用 `evaluate-semidense` CLI 子命令。`--limit 2` 仅用于冒烟。输出原图坐标的coarse/fine/final指标、覆盖率、出界统计、匹配图、延迟和显存。fine指标使用细化前的完整点集合，不能用QRRU过滤后的子集替代。训练内验证仅测GT解耦损失，不代表级联准确率。

A/B原图尺寸分别应用 `(u+0.5)*W/392-0.5`。最终匹配置信度沿用coarse/fine评分，QRRU门控不参与置信度排序。当前没有新增置信度校准头或拟合H评价；后者不是GHIM H0误差。

## 工程检查

```bash
/root/miniconda3/envs/loma-repro/bin/python -m unittest \
  tests.test_semidense_qrru tests.test_dense_downstream tests.test_entrypoints -v

CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/loma-repro/bin/python \
  scripts/check_semidense_resume.py \
  --config configs/semidense_qrru.json \
  --checkpoint /absolute/path/to/first_stage.pt \
  --output outputs/semidense_qrru_replay
```

检查覆盖坐标/控制场、全量与重计算softmax梯度、空监督、非法H0、第二步上游梯度、小窗口过拟合、CPU精确恢复。GPU脚本分别训练连续两步和断点两步，比较模型、优化器、调度器、RNG、数据位置。选择空闲GPU执行，不改变其他训练任务。

## 验证范围

工程冒烟使用已有 `shared_descriptor_frozen_tier3_seed0/latest.pt`，不是用户当前训练尚未完成的新权重。2步更新/2对val只是链路证据。未启动正式第二阶段训练，尚无完整验证集的QRRU收益结论。当前参考实验中的fine误差约0.61–0.85原图像素，2步后的QRRU变化很小，不能宣称精度提升。
