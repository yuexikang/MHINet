# D16 最后一轮执行记录

依据 `../local_occlusion/FINAL_ROUND_PLAN.md`，分支不变，所有新输出位于 `outputs/d16_final_round/`。旧模型及结果不覆盖。未改变D8/D2/QRRU或引入H0。

## 已固定的数据与实际偏差

- 历史415个杭州样本覆盖168块；按块/base_pair_id与footprint交叉清理后，旧val仅剩75对。因此校准128对来自从未参与旧实验的原train113＋val15；工作点评价128对来自原train121＋val7。两集合各29块且互不交叉，新增保留区不进入训练。训练仍是旧254对，开发仍是旧64对。
- 杭州最终256对来自test_controlled；RGB最终256对来自原磁盘独立test，不从反复使用的val拆出测试。
- RGB训练1024、开发256；源影像身份互斥。H组合关系、逆变换、48对图像/掩膜加载约定检查已完成。未声称RGB地理邻近独立或DINO外部预训练完全未见。
- 清单 `selection.json` 在模型评价前固定。`data_audit.json` 和 `geometry_checks.json` 保存实测证据。测试只查看元数据，不在配置锁定前运行模型。

## 阶段A

`calibrate.py` 不创建优化器、不更新参数。128对校准选每模型一个Matchability阈值，在另外128对上报告独立误拒绝、负例召回、覆盖率/接受后精度、面积/颜色分层、29块bootstrap。阈值对清洁与增强正点分别满足校准经验误拒绝≤1%，测试时不重调。

保存每查询npz、完整PR/ROC点、Dustbin分布、初始化/15轮/30轮no_match_logit，以及固定32训练对的梯度反向重放。重放明确不是历史梯度，旧training.jsonl实际记录另存。`report_a.py` 生成 `A_DIAGNOSIS.md`。

当前U0/B0/N1/N2均未达到≥50%负例召回工作点，不据此扩展新Dustbin模块。

## 阶段B

`train_e.py` 保留旧N2执行路径、seed、数据、增强、预算，仅选择损失项：E1保持keep/keep_geo/keep_match，E2保持occ，其余乘零。条件追加E_geo/E_cls_mat已经有入口，但必须等待`keep_gate.json`，不能无条件执行。

同初始化、固定30轮、两次前向/对、总15240次训练前向与1920次更新。E组固定最终第30轮比较，模型各自独立校准，不能选择有利epoch混比。

## 阶段C

`CoordinateModel` 冻结整个B0共享网络，保留它的原Matchability；独立训练坐标分支。旧坐标头复制B0，局部残差头末层置零。3×3窗口以detach后的空间峰定位，窗口内概率归一化；边界不重复候选。A/B上下文特征先L2归一化，输入2052维（A特征、局部加权B特征、局部概率质量/峰值/峰与窗口外峰差/空间归一化熵），MLP 2052→256→2，ReLU，±8px tanh残差。

四组固定B0特征：D00旧头/杭州，D01旧头/混合，D10新头/杭州，D11新头/混合。每轮254主样本；第二个杭州样本固定偏移127索引，RGB样本在固定1024池中轮转。每样本Geo占1/2，保留原鲁棒Geo及弱H死区；分类/Mat权重为0且冻结概率/Matchability不变。各组LR=5e-5，cosine30轮到1e-6，AdamW .01，累积4，裁剪1。原Geo最外D16中心边界有效域保持不变；评价仍覆盖所有参考有效查询并单独可分析边界。

同样15240训练前向/1920更新。旧头有预训练、新头零残差的初始差异通过epoch_000记录。参数量/时延在最终比较中报告，不能隐藏冻结旧Matchability头的开销。

## 执行检查与选择

- 原Decoder复制输出等于B0；两种头训练时共享网络无梯度，Matchability不变；新头初始残差0。
- 局部概率角点不重复质量、残差上限、梯度有限性测试通过。
- `decide.py` 只在完整30轮结果存在后应用登记门槛，写配置锁定文件。D01旧架构混合数据的候选条件按保守解释，也要求相对D00两个域各+1pp；不以其中一域退化换另一域大涨。此解释在D01启动前固定。
- 仅通过的候选与对应基线追加seed43/44。无候选则回退U0，不额外训练。最终测试只能在该选择冻结后打开；失败不能回头再调参数。

## 当前命令

```bash
python -m mhinet.v2.final_round.calibrate --variant U0  # B0/N1/N2/E1/E2同理
python -m mhinet.v2.final_round.train_e --variant E1   # E2；条件满足后E_geo/E_cls_mat
python -m mhinet.v2.final_round.train_d --variant D00  # D01/D10/D11，--seed 43等仅用于确认
python -m mhinet.v2.final_round.decide
python -m mhinet.v2.final_round.report_a
```

每个训练目录有status.json、配置、初始化/基准权重身份、逐轮指标、配对记录和latest/best/middle。状态以实际进程与这些输出结合判断，不能仅凭tmux会话名认定仍在训练。

## 开发裁决及封存测试

六组均已完成30轮、15240训练前向/1920更新。`keep_gate.json`不触发追加分解：E1达到101.0% N2 EPE改善，但P@3高0.94pp，超过预登记绝对差≤0.5pp。`decision.json`锁定无候选、回退U0，不追加seed43/44。E2拒配工作点合格但清洁定位明显退化，不保留Occ。

追溯原预训练链发现，初拟RGB测试中的85个GoogleEarth源文件名全部出现在历史GoogleEarth共享描述子train/val中，且当时MVT/头可训练。`seal_test.py`在任何封存推理之前保留原selection不改，另写test_selection（128 FLAIR＋128 WHU）。这是来源驱动的计划修正，不是根据测试表现抽样；限制最终RGB结论的域。完整证据见test_provenance.json。

`final_test.py`只读固定第30轮权重、读取已固定校准阈值，保存所有查询、逐对/G组/边界错误、阈值前后精度、图例和同卡时延。U0及所有对照在GPU0顺序执行，10次预热、30次计时，输入一对512图。`report_final.py`生成配对差值/空间块bootstrap、完整曲线、报告和最终模型接口配置。全部测试完成后还需汇总独立测试配对区间和一页结论；当前不是完成声明。

```bash
python -m mhinet.v2.final_round.seal_test  # 一次性；已有清单不得重抽
python -m mhinet.v2.final_round.final_test --variant U0
python -m mhinet.v2.final_round.report_final
```
