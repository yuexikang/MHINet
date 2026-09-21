# 第二档共享描述子续训与 H0 有效重叠误差

2026-09-17。用户选择不继续LoRA实验，采用第一档冻结DINO版最终checkpoint初始化第二档训练。不是从随机权重训练，也不是接着第一档已经结束的scheduler继续走。

## 新指标

实现：`mhinet/pretraining/overlap_metrics.py`。考虑784输入图A的全部像素中心，用GT H投影到B，只保留A有效overlap mask、GT投影合法且位于B像素中心边界内、B overlap mask双线性采样完全有效的位置。支持集完全由GT和mask决定，与预测无关。

逐点误差为 `||pixel(H0*x) - pixel(Hgt*x)||2`，方向A→B，单位为重采样后B图像像素。预测出界不剔除；非法投影或GHIM拟合失败计入失败，保留在Recall@1/3/5的分母。没有有效GT支持的影像对单列，不记作零误差。均值对有限预测统计，必须同时阅读invalid_projection_rate/failed_fit_pairs。保存pair均值/中位数/P90、像素加权均值、point recall；旧四角MACE继续保留，不删极端样本。

每次新的训练验证自动写`summary.json:H0_overlap`和`pairs.jsonl:H0_overlap`。旧checkpoint另用`scripts/evaluate_shared_overlap.py`补评估，只执行GHIM，不解码CGMDP，不覆盖之前的验证记录。

## 第二档设置

| 项目 | 设置 |
|---|---|
| 初始化 | `outputs/shared_descriptor_frozen_seed0/latest.pt`，第一档12995步 |
| 初始化SHA256 | `df3ec90b9bc53fd983cc1945a3c028ccdf9bf688325a60055101232c509cc25e` |
| 数据 | GoogleEarth_quadrant_tiers_v1，仅tier2：几何+辐射变化 |
| train / val | 34652 / 3848对，母图及地理分组交集均0 |
| 冻结 | DINO本体，无LoRA；VGG BN running statistics固定 |
| 训练 | MVT、GHIM head、VGG、CGMDP；不执行MHIR/D1 |
| 峰值LR | MVT/head 5e-7，VGG 2.5e-6，CGMDP 5e-6 |
| 优化 | 新AdamW状态，wd1e-4，clip1，BS1×累积4 |
| 预算 | 一轮8663步；warmup约433步，cosine降至峰值10% |
| 验证 / 保存 | 第5000步及8663步完整val；每1000步及结束保存 |
| 输出 | `outputs/shared_descriptor_frozen_tier2_seed0` |

峰值LR为第一档峰值的一半，以适应新增辐射变化并控制对已有描述子的扰动；相对于第一档末尾cosine最低LR，它仍是一次带warmup的重新启动。保持同一损失，无额外损失、无第三档混入。新输出目录仅继承模型权重，optimizer/scheduler/data cursor重新开始；该新目录中断后仍按latest.pt自动恢复。

## 已执行检查

131项单元测试通过，新增4项覆盖恒等、已知位移、双侧mask及非法预测不能改变分母。真实第二档8对train/2对val两步冒烟成功：loss0.190485→0.167080，峰值约9.96GB，实际加载第一档最终权重。2对val新增metric覆盖788167个支持像素，无非法预测；仅工程证据，不作为正式第二档精度结果。

## 运行及日志

```bash
GPU_ID=1 bash scripts/train_shared_descriptor_tier2.sh
tail -n 30 -f /home/disk1/MHINet/outputs/shared_descriptor_frozen_tier2_seed0.console.log
```

后台启动时console.log记录stdout/stderr，训练每50步额外输出一行`TRAIN step=...`，可直接tail。完整逐步记录另在输出目录的train.jsonl。

第一档最终模型在第一、第二档val的H0补评估输出到`outputs/shared_descriptor_frozen_seed0/h0_overlap_v1`，日志`outputs/shared_descriptor_frozen_h0_overlap.log`，用作已有模型与第二档训练后结果的对照。未使用test选参。
