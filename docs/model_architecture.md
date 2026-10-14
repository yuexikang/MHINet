# MHINet 当前架构：D8→D2 密集匹配下游

文档状态：当前规范，更新于 2026-10-14。路线决策见[主路线](mainline_d8_d2.md)；原六轮 MHIR 架构说明已从工作树移除，可通过 Git 历史追溯。

## 功能接口

| 模块 | 输入 | 当前输出与作用 |
| --- | --- | --- |
| GHIM | 图像对 `(I_A, I_B)` | 预测 A→B 归一化 H0、拟合有效位、两图 MVT 上下文 |
| CGMDP | 图像局部 VGG 特征与 MVT 上下文 | 共享累计金字塔，向匹配器提供 D8、D2 描述子 |
| D8→D2 匹配 | 共享描述子、预测 H0 | 粗父对应及其局部细匹配；推理不接收 GT |
| QRRU | D2 特征与预测细匹配起点 | 4 轮局部匹配点精修；不接收 H0 |

```text
图像对
  └─ GHIM：冻结 DINOv3 → MVT → H0、跨图像上下文
       ├─ H0 ──────────────────────────┐
       └─ CGMDP + VGG                  │
            D16 → D8 → D4 → D2        │
                   │         │         │
                   └─ D8粗匹配 ← H0支持区
                             └─ D2局部细匹配 ← H0支持区与残差采样
                                      └─ QRRU ×4
                                           └─ 匹配坐标与置信度
```

评估需要配准 H 时，另用预测匹配拟合 H，再读取 GT 评分。QRRU 的四轮输出是局部点更新，不是 MHIR 的 H1…H6 单应更新。

## 描述子与坐标

`Ds` 是图像 `1/s` 尺度的融合描述子，不是 DeDoDe 原始层名称。共享 DINO/MVT/VGG 对每对图像只计算一次，CGMDP 累计解码至 D2，匹配阶段消费 D8、D2。D4 参与共享累计解码；没有独立 D4 匹配轮次，也不请求 D1 全分辨率解码。

| 描述子 | 当前 512 输入的形状 | 784 兼容输入的形状 |
| --- | --- | --- |
| D8 | `B×2×256×64×64` | `B×2×256×98×98` |
| D4 | `B×2×256×128×128` | `B×2×256×196×196` |
| D2 | `B×2×256×256×256` | `B×2×256×392×392` |

两图维度为 2。H0 使用 A→B、`align_corners=False` 的归一化坐标；D8/D2 栅格中心经动态网格转换后返回各自原图像素坐标。当前 RGB512 配置显式设置输入 512，类默认值及缺字段旧检查点仍按 784 解释。

## 级联推理

1. 默认 A 配置根据预测 H0 构造 D8 双向 overlap 支持区；在支持区内进行 cosine 相关性、双向归一化和 mutual 粗筛。支持区不把每个查询限制为 H0 附近的局部候选。
2. 每个预测 D8 父对应展开 4×4 D2 子格；目标格通过 H0 残差先验采样，结合 D2 预测 overlap 支持，进行窗口内双向归一化与 mutual 精筛。
3. 粗细分数组合后排序，截断至 `max_matches`，再进入 QRRU。当前默认 `max_coarse=6000`、`max_matches=12000`、置信阈值 0；学习温度从 checkpoint 加载，不把初始化温度当作最终值。
4. QRRU 使用 D2 局部描述子迭代 4 次，精修目标匹配位置；最后拒绝非法/越界目标中心。返回置信度沿用精修前的粗细联合分数，尚无精修后质量重估。

“密集匹配下游”指这条规则网格粗细匹配路线。源码名 `SemidenseMatcher` 和协议名 `semidense_qrru_v1` 保持兼容；互选、合法性过滤和截断意味着输出不是完整逐像素稠密 warp，也没有独立关键点检测器。

H0 位置消融只作用于推理：D8 开关控制 overlap mask，D2 开关同时控制 overlap mask 与 H0 warp；QRRU 从不接收 H0。A 为默认，B/C/D 为对照。细节见[固定权重消融](h0_inference_position_ablation.md)。

## 训练与梯度

当前系统为 `SemidenseSystem`。联合损失：

```text
L = Lc + Lf + Lq + Lh0
Lh0 = geo + 0.01·mat + 0.0001·cls + 0.05·H
```

DINOv3 冻结、LoRA 不启用；MVT、GHIM/H0、VGG、CGMDP、匹配温度和 QRRU 可训练。VGG BN running statistics 固定，BN affine 仍可训练；不将这一策略解释为所有解码器 BN 均冻结。

Lc 使用 GT 几何粗对应；Lf 使用 GT 粗父对应与 **detached 预测 H0** 构造细窗口，并由 GT/可见掩码生成标签；Lq 使用独立 GT D2 中心加有界噪声作为起点。推理使用预测级联起点，因此训练损失验证不能替代真实级联精度评价。

Lc/Lf/Lq 不给 GHIM H0 头提供直接梯度。当前通过独立 Lh0 监督 GHIM 头，保持细窗口先验 detach；`freeze_h0=false` 需要正的 H0 监督权重。旧 MHIR 文档中的“H0 不 detach、通过六轮 H 更新回传”不适用于此匹配路径。

## 源码与证据入口

- [匹配与推理](../mhinet/downstream/semidense.py)、[QRRU](../mhinet/downstream/qrru.py)。
- [联合系统及 H0 监督](../mhinet/downstream/training.py)、[训练配置与恢复](../mhinet/downstream/train.py)。
- [当前配置](../configs/semidense_rgb512_h0_joint_5epoch.json)、[已完成训练](semidense_rgb512_h0_joint_5epoch.txt)、[两数据集测试](semidense_rgb512_h0_joint_test512_results.md)。

早期 `HGuidedDenseDownstream` 是只读零样本消费者；六轮 `MHIR` 是另一套单应精修消费者。二者保留实现与历史证据，现行主路线的架构、训练入口和验收以本页及[实验计划](experiment_plan.md)为准。
