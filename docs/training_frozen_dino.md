# 仅冻结 DINOv3：temporal4 等预算训练

用户要求解冻MVT，保留DINOv3冻结。新增frozen_dino profile，与原frozen_dino_mvt隔离。
从原始LoRetta/LoMa预训练初始化，不继承旧实验训练权重；真实BS1、累积4、20k步，
共80k样本次数，与已完成BS4对照等预算。seed0，1000步warmup，1000步验证/保存，
cosine至峰值LR的10%，AdamW wd1e-4，梯度裁剪1。

| 参数组 | 状态 | 峰值LR |
| --- | --- | ---: |
| DINOv3 | 冻结、eval | — |
| MVT | 解冻、train | 1e-6 |
| GHIM head | 解冻 | 1e-6 |
| VGG | 解冻，BN运行统计沿用固定策略 | 5e-6 |
| CGMDP D16/D8/D4/D2 | 解冻 | 1e-5 |
| Adapter/MHIR D8/D4/D2 | 解冻 | 1e-4 |

D1保持注册但不执行、不进入optimizer。总可训练参数109,996,778。
六轮proposal四角L1 + GHIM四项监督，权重维持total=1、mat=.01、cls=.0001、H=.05；
FGO关闭。DINO/MVT每microbatch共享一次，H0/H/T梯度保留。
脚本默认关闭相关性activation checkpoint；模型checkpoint保存不受影响。

```bash
cd /home/disk1/MHINet
conda activate loma-repro
GPU_ID=0 bash scripts/train_frozen_dino.sh
```

配置 `configs/train_frozen_dino_temporal4_ebs4_20k.json`，runtime使用
`configs/runtime_paths.temporal4.server.json`，train34652/val3848，输出
`outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0`。
启动前核查数据完整性。有checkpoint同目录续训，无checkpoint从头并备份/替换旧运行记录。
每次验证输出固定首对的H0和六轮H共七张图。独立test仍须单独运行，不用于选择配置。

## 已完成检查

116项单元测试通过；新增profile参数组与配置检查。真实权重两步梯度审计：
DINO梯度0，MVT/head第一步非零，第二步VGG/CGMDP非零；D1调用0。
第二步MHIR损失经H0回到MVT context梯度范数0.06225586，经CGMDP回到context为
2.354383e-6，两条分支均非零。两步loss为4.926276、4.894512。
审计含额外分支反传，峰值allocated17.754GB、reserved18.902GB；这不是长训练显存上限保证。
证据 `artifacts/frozen_dino_two_step.json`；复现脚本 `scripts/check_frozen_dino.py`。
仅准备配置并进行短检查，不将其标记为长训练收敛或泛化通过。
