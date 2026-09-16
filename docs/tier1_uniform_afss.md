# 第一档：Uniform vs MHINet Pair-AFSS v2

仅冻结DINO；MVT、GHIM head、VGG、CGMDP、MHIR联合训练；D1关闭。
两组原始预训练初始化及新增层种子相同，末层零初始化，不继承旧跨时相训练权重。
run.json记录initial_model_sha256以核对初始完整权重；GPU训练不承诺逐位确定性。
第一档train51978/val5772，loader按tier字段过滤，绝不使用test评分或选模型。

| 设置 | A | B |
|---|---|---|
| 采样 | 每轮全量打乱 | 2轮全量后AFSS v2 |
| 总进度 | 5轮 | 5个基准轮次 |
| 实际步上限 | 64975 | 64975（一般更少） |
| 余弦实际步规划终点 | 64975 | 43534 |
| LR线性warmup | 1000实际步 | 1000实际步 |
| 验证/保存 | 每5000实际步及结束 | 每5000实际步及结束 |

BS1×累积4；每轮不足4个样本时确定性重复少量选中样本补齐，不跨轮累积。
因此每全量轮12995步，5轮64975步。
MVT/head LR1e-6，VGG5e-6，累计解码器1e-5，新增模块1e-4；
AdamW默认betas(.9,.999)、eps1e-8、weight_decay1e-4，clip1，最低LR为峰值0.1。
GHIM四项+六轮四角监督保持；FGO关闭。关闭相关性激活重计算。
warmup_fraction保留旧协议字段，但这两组明确由warmup_steps=1000覆盖。

## AFSS适配

直接复用LoMa pair_afss的config/state/controller/scheduler，放在mhinet/engine/pair_afss。
MHINet adapter评分为min(几何precision, recall, H0-AUC, H6-AUC)，而非仅H0。
对应正确阈值为784输入像素1px，matchability>=.3；H-AUC是0.5–3px六阈值梯形积分/2.5，
在GT共同可见网格上计算，不等同于评估总表的四角误差ECDF AUC。
首次完整训练集刷新后按10%/80%分位校准固定阈值，后续每5轮完整刷新；
EMA .5。LoMa实现只在刷新时更新分数；本次两轮预热后剩3轮，无第二次刷新。
困难/未知100%、中等50%、简单5%，每轮至少45%；保留原3轮覆盖/10轮复查机制。
分数并列时分类占比不保证恰好10/70/20，实际counts记录在日志。
43534=ceil(25990+0.45*(64975-25990))；实际步超过余弦终点则保持最低LR。
刷新只用train，no_grad+eval，不改变冻结训练策略；之后恢复model.train。
刷新耗时计入训练总时间；不是等算力对照，也不是原AFSS论文原样复现。

## 运行

```bash
conda activate loma-repro
cd /home/disk1/MHINet
GPU_ID=0 bash scripts/train_tier1_a.sh 2>&1 | tee -a outputs/tier1_a_console.log
GPU_ID=1 bash scripts/train_tier1_b.sh 2>&1 | tee -a outputs/tier1_b_console.log
```

分别在两个终端执行。同目录自动续训；默认输出outputs/tier1_a_seed0和tier1_b_seed0。
可用MHINET_OUTPUT_DIR指定全新目录。checkpoint包含优化器、LR、RNG、抽样列表/游标、
评分EMA/阈值/难度/复查状态，检查配置hash、pair顺序和train/val清单hash。
best_validation.json仅依据val平均最终四角误差记录最佳checkpoint路径，不复制大权重。
日志train.jsonl含实际步与基准进度；afss_refresh.jsonl记录评分耗时/阈值/类别分布。
验证输出H0+六轮七张图；结束时额外完整val。未开启正式训练前先完成小样本短测。
