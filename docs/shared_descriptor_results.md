# 共享描述子预训练：实施验收记录

日期：2026-09-16。以下全部是工程诊断，不是正式训练/测试集排名。正式两组训练尚未自动启动。

## 已完成检查

| 检查 | 实测结果 | 证据 |
|---|---|---|
| 工程回归 | 127项通过（含新增7项） | `bash scripts/unit_tests.sh`，1.991s |
| 双线性参考 | 与align_corners=False grid_sample前向、feature/坐标梯度一致 | `tests/test_shared_pretraining.py` |
| LoRA初始函数 | H0、D8、D4、D2最大绝对差均0 | `artifacts/shared_pretraining_real_tiny20_strict.json` |
| LoRA第二步 | A梯度范数0.001812、B梯度0.167984 | 同上 |
| 两条梯度分支 | GHIM及CGMDP均对MVT与LoRA B产生非零有限梯度 | 同上，逐步记录 |
| DINO冻结 | 20步前后base参数SHA完全相同 | 同上 |
| 共享调用 | DINO=1、MVT=1、累计解码4级、D1=0 | 同上 |
| 真实单对20步 | Ldesc 1.167799→0.315688（下降72.97%）；LGHIM 0.040362→0.006571 | 同上；256查询/方向，不是正式1024查询训练 |
| 基线断点恢复 | 模型与optimizer最大差均0；checkpoint SHA相同 | `artifacts/shared_frozen_resume_audit.json` |
| LoRA断点恢复 | 模型与optimizer最大差均0；checkpoint SHA相同 | `artifacts/shared_lora_resume_audit.json` |
| 基线bundle重载 | 全部806项state匹配；H0/D8/D4/D2差0 | `artifacts/shared_frozen_bundle_audit.json` |
| LoRA bundle重载 | 全部826项state匹配；H0/D8/D4/D2差0 | `artifacts/shared_lora_bundle_audit.json` |

恢复对照统一是8对train、2对val、2个optimizer step。正式训练一轮是51978对train/5772对val，不能用此处2对val比较方案优劣。

20步单对审计为BS1、无累积/无warmup，固定256查询以验证可学习性；不把它当作正式训练曲线。

## 严格确定性版本的短程数据

| 方案 | 第二步loss | 第二步训练秒数，不含保存/验证 | 峰值显存，十进制GB | 2对val H0 MACE，输入px | H0拟合失败率 |
|---|---:|---:|---:|---:|---:|
| 冻结DINO | 2.025328 | 2.347 | 9.949 | 201.394 | 0/2 |
| DINO LoRA | 2.024950 | 2.617 | 10.104 | 202.417 | 0/2 |

H0拟合成功不等于几何精度合格：这里H0误差很大，不能宣称模型已验证成功。两步的时间、精度也不支持判断LoRA优劣或可靠预测完整epoch耗时；数据缓存、验证、写盘及GPU占用均影响总时间。

源记录：`outputs/diagnostics/shared_{frozen,lora}_strict_split/train.jsonl` 和 `validation/step_000002/{summary.json,pairs.jsonl}`。每对另记录三个尺度双向full-gallery检索误差及Recall@1/3/5px。

示例图目录：

`outputs/diagnostics/shared_frozen_strict_split/validation/step_000002/past_tier1_0__34.801100126.376330/`

含H0叠加图，以及D8/D4/D2真实匹配连线图。未运行MHIR，因而没有六轮H图。

## 未通过项及修复轨迹（不隐藏）

1. 早期非确定性版本的连续/恢复比较未过：基线state最大差0.030666、LoRA最大差0.022022，最大差出现在decoder BN running variance。仅开启warn_only仍保留非确定性Flash Attention。
2. 修复为strict deterministic + gather双线性采样，保留原采样公式，前向及梯度reference测试通过。重跑后两组模型/optimizer差均为0，未通过调大容差掩盖问题。
3. 初次基线bundle复核及后续启动曾因卡1被外部进程占约21GB导致OOM。未中断外部进程；基线诊断改在空闲卡3完成，正式脚本仍默认卡1。

早期诊断目录保留用于追溯，但不应续训为正式实验。GPU上的诊断任务均有明确步数上限；正式输出目录与这些目录隔离。

## 固定资源

- 环境：`/root/miniconda3/envs/loma-repro`，Python3.10.20，PyTorch2.11.0+cu128，RTX4090。
- 数据：`/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1`；第一档母图/地理组交集均0。
- LoRetta/DINO/GHIM权重SHA256：`09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e`。
- LoMa-B金字塔权重SHA256：`3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758`。
- 完整train manifest SHA256：`6587d3be58dfc8b1601244bcf7fa05e3cb5dba1b14b7a49871dcb1fe3ef3276c`。
- 完整val manifest SHA256：`c4763636e2ca9914c62aec45f5f6f238e0cfa9269343f27571e6c0acdb3d01ee`。
- 两步基线checkpoint SHA256：`02125544071e25e9fe5c310b3dfe42dfcf14949175c373f69bdeb52af97ddc6a`。
- 两步LoRA checkpoint SHA256：`6112cd0c1404a1425522b08b3638ed3486756d4ec2cefb67aa83bf09678e6308`。

## 复核命令

```bash
bash scripts/unit_tests.sh
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 /root/miniconda3/envs/loma-repro/bin/python -m scripts.audit_shared_pretraining --steps 20 --output artifacts/shared_pretraining_real_tiny20_strict.json
OMP_NUM_THREADS=4 /root/miniconda3/envs/loma-repro/bin/python -m scripts.check_shared_resume --split outputs/diagnostics/shared_lora_strict_split --continuous outputs/diagnostics/shared_lora_strict_continuous --output artifacts/shared_lora_resume_audit.json
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 /root/miniconda3/envs/loma-repro/bin/python -m scripts.check_shared_bundle --run outputs/diagnostics/shared_lora_strict_split --output artifacts/shared_lora_bundle_audit.json
```

正式入口、参数与两项目共享加载接口见 `docs/shared_descriptor_pretraining.md`。
