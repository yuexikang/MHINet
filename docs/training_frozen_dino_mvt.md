# 冻结 DINO/MVT 的完整 D2 主线训练

这是用户指定的新实验，不冒充原 E02/E03 等预算对照；不覆盖已完成的 E01。
初始权重由 `configs/runtime_paths.server.json` 指定：GHIM/DINO 使用原 LoRetta，
VGG/累计解码器使用 LoMa-B；新增 Adapter/MHIR 初始化，MHIR 输出投影为零。
默认不是从 E01 checkpoint 继续训练。

## 运行

```bash
cd /home/disk1/MHINet
conda activate loma-repro
GPU_ID=1 bash scripts/train_frozen_dino_mvt.sh
```

默认输出 `outputs/GHIM_joint_frozen_dino_mvt_seed0`。再次运行相同命令自动恢复该目录最新
checkpoint（含 optimizer、scheduler、RNG、数据游标）；新实验改 `MHINET_OUTPUT_DIR`。
不要把旧 E01 目录作为新实验输出，也不要把 smoke 目录作为正式训练输出。

## 参数与监督

| 参数组 | 是否训练 | 峰值学习率 |
| --- | --- | --- |
| DINOv3 | 冻结且 eval | — |
| MVT | 冻结且 eval；共享前向一次 | — |
| GHIM head | 解冻，包括匹配解码与 no-match 参数 | 1e-6 |
| VGG | 解冻；沿用固定 BN running statistics，affine 可训练 | 5e-6 |
| CGMDP 累计解码器 D16/D8/D4/D2 | 解冻 | 1e-5 |
| Adapter/MHIR D8/D4/D2 | 解冻 | 1e-4 |
| D1 兼容分支 | 不执行、不参与 optimizer | — |

可训练参数共 22,579,690。真实 BS1，累积4，有效 batch4；AdamW，weight decay 1e-4，
全局梯度裁剪1；10,000 optimizer steps，500步 warmup，cosine 降至峰值的10%。
这是一份明确的10k步完整训练预算，不意味着多轮数据 epoch 或收敛已被验证。

总损失为六轮 proposal 四角像素 L1 加 GHIM 四项监督：
`L_total = L_MHIR + 1.0*(L_geo + .01*L_mat + .0001*L_cls + .05*L_H)`。
GHIM 复用 LoMa 的公式：归一化坐标鲁棒几何误差（scale=.1）、平衡 matchability BCE、
双线性软标签与 no-match 分类、H/h33 八元素 MSE。L_H 不是四角误差，FGO 未启用。
权重为旧公式初始设置，尚未证实是该训练模式的最佳比例。

读取真实 `mask_A_overlap` 并结合 GT 投影入界条件构造监督；H0 拟合失败的样本仍接受
coarse 监督，但不使用占位 H 计算 L_H 或精修监督。无效 H 在除法前隔离。
H0/H/T 不 detach。日志分别写 `loss_total`、`loss_sequence_corner_l1_px`、
`ghim_losses.{geo,mat,cls,H,total,valid_H_pairs}`；进度条显示总损失。

训练使用按地理区域过滤后的 train/35996；每500步全量 val/2500，保存 checkpoint 和
固定首对影像的 H0/H1–H6 七张叠加图。测试只由独立 test 入口执行，不能用于选择配置。

## 已验证与边界

108项单元测试通过，新增组所有权、配置、各项梯度、无效奇异 H 隔离检查。
`scripts/check_frozen_dino_mvt.py` 进行真实权重和 train 影像两步梯度审计；结果见
`artifacts/frozen_dino_mvt_two_step.json`。DINO/MVT 零梯度，head 有梯度，第二步 VGG/CGMDP
非零梯度，D1调用零，共享DINO/MVT各一次；峰值 allocated 约8.29GB（BS1审计）。
首次无warmup/head LR1e-5探测第二步 loss 138.607，不能作为稳定性通过证据；
改为head LR1e-6并应用正式warmup，同一影像两步总loss为4.74614、4.72522。
这只是局部启动/梯度证据，不是完整训练稳定性或精度验证。

已有 P4 tiny gate 是原 MHIR 主线的证据，不代表新增 GHIM 损失已完成独立 tiny-overfit。
本次仅启动短 smoke，不自动启动10k步长训练。
