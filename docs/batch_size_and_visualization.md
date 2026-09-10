# 单卡 batch、验证叠加图和正式训练启动

## 当前建议

正式 E01 继续采用真实 BS=1、梯度累积4次、有效 batch=4。新增真实 BS=2/4 支持，但不将其标记为已通过旧输出对齐：两对真实训练影像上，BS=2 对比串行的 H0 四角最大绝对差为0.367676 px，超过本次工程检查预设0.1 px。批内顺序反转差为0，未发现跨图像对串扰。原始失败记录保留在 `artifacts/batch_forward_isolation.json`，不得改写成通过。

逐段检查发现 DINO 特征最大绝对差0.927975，MVT context最大绝对差1.25。固定串行 DINO/MVT 结果后，仅组批 GHIM head 的四角差降到0.004883 px；仅把 head 改为逐对运行不能修复上游误差。差异定位在 BF16 编码器/交互路径的批量算术；具体首个产生差异的算子尚未定位。因此不通过放宽容差来替代修复。BS=1 的旧提取路径原样保留，正式配置不启用实验性批量路径。

## 短测结果（不是精度实验）

RTX 4090，PyTorch 2.11.0+cu128，conda `loma-repro`，heads profile；2次warmup + 5次计时更新，4个固定真实train pair，每次更新有效batch=4。包含前向、loss、反传、clip、AdamW；CPU预取影像，不包含磁盘读取、验证和checkpoint写入。未访问test，未保存探测权重。

| 真实BS | 累积 | GPU物理编号 | 峰值allocated GB | 峰值reserved GB | 中位吞吐 pair/s | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 3 | 4.001 | 5.098 | 0.779 | 7次更新完成；正式推荐 |
| 2 | 2 | 3 | 7.112 | 9.028 | 1.111 | 7次更新完成；串行H0对齐未通过 |
| 4 | 1 | 0 | 13.361 | 16.624 | 0.934 | 7次更新完成；非正式配置 |

服务器同期有其他GPU作业，BS=4也不是同一张卡，因此不能把这些吞吐之比解释为可靠加速比。显存更高不保证更早训练完成。首次BS=2训练入口冒烟在GPU3上因空闲显存不足OOM，日志保存在 `outputs/engineering_batch_visualization_smoke_start.log`；这不覆盖之前短测成功记录。不同profile显存不同，上表不适用于joint。

复现实例（输出文件必须不存在）：

```bash
cd /home/disk1/MHINet
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli batch-probe --runtime configs/runtime_paths.server.json --batch-size 2 --output artifacts/batch_probe_heads_bs2_gpu1_repeat.json
```

## 训练入口与验证图

`mhinet/engine/train.py` 支持真实组批；损失按有效图像对数归一化，遇到无效GHIM样本补足有效batch，而非误把微批次数当样本数。BS>1必须显式填写 `allow_experimental_batch: true`，避免误把未通过对齐的路径用于正式主线。DINO/MVT共享输出供两个分支使用；D1保持不执行。

训练和验证均有tqdm进度条。默认 `visualization_pairs=1`：每次验证固定第一个验证pair，输出H0和每轮一张图，共7张；设置N则输出7N张。不是只验证这一对：正式配置仍计算前2500个验证pair的完整指标。输出位置为：

```
outputs/E01_heads_seed0/visualizations/step_0000500/pair_0000/
  H0_initialization.png
  D8_round1_H01.png
  D8_round2_H02.png
  D4_round1_H03.png
  D4_round2_H04.png
  D2_round1_H05.png
  D2_round2_H06.png
  manifest.json
```

A按每轮预测H投影到B坐标系并半透明叠加；绿色是真值投影四边形，红色是预测四边形。图上的尺度标签是迭代使用的descriptor尺度，展示统一使用784输入坐标，避免不同画幅造成视觉误判。保存pair_id、像素H、更新接受标志；非法变换不warp，仍输出明确标注的图。绘图不计入单对模型前向延迟，但计入整次验证墙钟时间。

## 推荐超参和卡1命令

`configs/e01_heads_v1.2.json`：784×784；D8/D4/D2各两轮；AdamW lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight decay=1e-4；10000次optimizer更新；500步warmup后cosine到1e-5；clip=1；seed=0。真实BS=1，累积4次。每500步验证/存checkpoint，验证2500对、展示固定1对。六轮proposal四角L1等权；FGO、额外loss、Planar、D1均关闭。不加载tiny-overfit权重。

E01只训练D8/D4/D2 adapter和MHIR解码器（833222个参数）；DINO、MVT、GHIM head、VGG、CGMDP累计解码器冻结。后续联合阶段冻结GHIM head参数不切断其输入梯度，H0/H/T不detach。

先在同一验证子集登记E00（无反传），再运行E01。这里仅提供命令，不自动启动正式评估或训练。请在tmux终端执行；若输出目录已存在，先确认其用途，不直接加overwrite。

```bash
cd /home/disk1/MHINet
conda activate loma-repro
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi -i 1

python -u -m mhinet.cli evaluate --runtime configs/runtime_paths.server.json --split val --max-pairs 2500 --h0-only --output-dir outputs/E00_h0_seed0

python -u -m mhinet.cli train --runtime configs/runtime_paths.server.json --config configs/e01_heads_v1.2.json --tiny-gate-artifact artifacts/p4_tiny_gate_mcnet_d2.json --output-dir outputs/E01_heads_seed0
```

`CUDA_VISIBLE_DEVICES=1`后，配置里的`cuda:0`正确对应物理卡1，不要再改为cuda:1。BS=1建议至少留6 GiB空闲，仍应为其他进程/临时分配留余量。

目录整理后，完成E00即可直接运行 `bash scripts/train_e01.sh`，其默认参数与上面的E01命令一致。`DRY_RUN=1 bash scripts/train_e01.sh`只显示命令；`bash scripts/unit_tests.sh`运行CPU单元测试。`bash scripts/test.sh CHECKPOINT`执行真实影像精度评估，默认完整val，详见 `docs/real_image_evaluation.md`。新代码位置见README目录树；统一CLI命令保持不变。

续训使用完全相同的配置和输出目录，额外传 `--resume outputs/E01_heads_seed0/checkpoints/step_0000500.pt`（换成实际最后完成的checkpoint）。从头初始化与resume不可混用；配置hash不同会拒绝恢复。

## 三个解码/预测部分

- GHIM head：继承现有LoRetta权重的定位尾部，由位置匹配、match-embedding decoder产生粗对应与可匹配性，再由无学习参数的加权单应拟合器得到H0。不是新增的直接8维四角MLP。权重来源是runtime登记的`loretta.pth`。
- CGMDP累计解码器：由现有LoMa-B权重初始化的DeDoDe式coarse-to-fine decoder，把MVT粗上下文与VGG局部特征逐级融合/上采样/累计，得到D8/D4/D2匹配描述子。`loma_B.pt`的粗尺度14权重迁移到当前D16；不输出四角位移。
- MHIR解码器：新增MCNet-style局部相关性解码器，输出四角的8个位移分量；累计到当前四角后通过受保护DLT更新H。它就是本模型学习四角残差的部分，未加载MCNet预训练权重。

下一步先定位并修复批量BF16算子漂移，再在同一空闲GPU重测BS=1/2/4；通过对齐、梯度、真实入口/续训检查后才升级正式batch配置。当前工程冒烟不等于模型泛化验证成功。
