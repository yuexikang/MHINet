# 完整 RGB512：Flow 预训练 → 冻结定位分支 → D2 匹配训练

日期：2026-10-27。分支：`codex/flow-d8-d2`。独立入口，原主线默认行为不变，QRRU 不参与训练或推理。

## 三条路径的速度

RTX 4090，GPU2，batch=1；同一批32对RGB图像，每组预热5次，重复3次，轮换测量顺序。输入已在GPU，CUDA同步墙钟计时，不含读盘、CPU预处理、传输和权重加载；包含DINO/MVT/VGG/DeDoDe、所需Flow和D8/D2匹配。采用原有混合精度设置，匹配计算FP32。GPU0/1有其他任务，未占用它们。

|方案|总耗时均值|P50|P95|特征/Flow部分|匹配部分|吞吐|
|---|---:|---:|---:|---:|---:|---:|
|现有Flow＋局部D8|43.89ms|43.98ms|46.12ms|28.32ms|15.57ms|22.78对/s|
|全局D8＋Flow先验|34.46ms|34.23ms|35.36ms|28.26ms|6.20ms|29.02对/s|
|全局D8，不预测Flow|31.64ms|31.33ms|32.63ms|27.86ms|3.78ms|31.61对/s|

这是当前实现的速度。局部入口在选择父窗口时调用了包含D2计算的旧推理，再运行连续D8和调制D2；因此存在重复计算，不能认为局部搜索在算法上必然比全局慢。本次不混入推理优化改动。原始逐次计时见 `reports/d2_modulation_no_flow/speed.json`。

## 数据范围

|划分|对数|用途|
|---|---:|---|
|RGB512/train|40,849|两阶段各完整遍历，不抽样缩小|
|RGB512/val|5,106|每轮验证和选模型|
|RGB512/test|5,106|训练结束后的完整测试|
|杭州/test_real|235|完整原生影像测试，独立报告|

根目录见配置。采用原清单，不改变划分；已核对RGB三个划分源影像不相交，全部训练/验证/测试所引用图像和掩码存在。按用户异步确认，最终两个测试集为RGB/test与杭州/test_real；test_controlled不纳入本次最终自动测试。杭州H是近似参考，不代表人工逐点真值。

## 两阶段安排

1. **完整数据上的Flow预训练/适配**：从现有 `weight/mhi_v2_u0.pt` 的MVT和完整Flow头初始化，冻结DINO；训练MVT（87,417,088参数）和Flow头（2,957,572参数）。这是已有权重上的全量数据训练，不是随机初始化。只计算D16，不运行VGG/D8/D2。保留原Flow坐标、Matchability、位置分类损失及其权重（1、0.01、0.0001），不增加T2或预测H头。
2. **冻结完整定位分支后训练匹配到D2**：加载第一阶段RGB验证最优Flow权重，冻结DINO/MVT/Flow；训练VGG、DeDoDe、D8/D2温度和82,496参数的双图调制MLP。下游从既有 `outputs/flow_d8_d2/training/best.pt` 与扩量 `full_pair/best.pt` 初始化。损失为已有全局D8分类＋局部D8分类/坐标损失＋调制D2四邻域软标签分类；权重各1。训练随机选64父窗口，推理最多128窗口/2048点。

第二阶段的离散父窗口选择、D2采样坐标不反传；D8自己的监督更新D8，D2损失通过采样特征和调制模块更新VGG/DeDoDe。没有声称D2坐标损失跨越离散选择直接反传到D8位置，更不会更新已冻结的Flow。

两阶段分别最多20epoch；AdamW、weight_decay=0.01，前500次优化器更新从0.1倍线性预热，之后余弦降到0.01倍。Flow峰值LR：MVT=1e-6、head=1e-5；匹配峰值LR：VGG/DeDoDe=2e-5、温度=8e-5、调制=1e-4。单对前向、累积4对，末尾不足4对按实际数量归一化，梯度裁剪1。

**启用严格早停，但不用min_delta限制best保存**。每阶段至少训练5轮，连续3轮验证P@3相对最近一次显著提升的参考值未提高超过0.001（0.1个百分点）则停止；小幅进步累计超过阈值可重置耐心计数。每次RGB验证P@3严格提高就保存best，epoch0允许入选；Flow按有效可见D16点的P@3选模型，D2按全部输出点P@3选模型。EPE均为512原图像素，保留P@1/3/5/10、可见计数及逐对结果。保存优化器、调度器和随机数状态，支持从完整epoch检查点恢复；中断于epoch内部时重跑该epoch。

第二阶段结束后自动使用两阶段各自最优权重，在RGB完整test和杭州test_real计算：D16 Flow，以及同权重的局部Flow、全局Flow、全局无Flow三种D2路径。测试结果不参与选模型。主要训练路径为现有局部Flow方案，没有同时训练三份模型。

## 验证和当前状态

- 8对训练、2对验证、两域各2对测试的串联smoke已完成，两个阶段各1epoch，保存/加载best和最终三路径评估通过。
- 最初无预热、偏大学习率的smoke暴露Flow更新过大；已保留日志，改为上述较低LR和预热后复测通过。修正后的Flow小样本验证EPE由1.789降至1.650px；不能将这个小样本结果解释为全量训练收益。
- 修正后Flow训练loss=0.00517，匹配训练loss=5.25043；梯度有限。DINO/QRRU始终无梯度；第一阶段MVT/head有梯度；第二阶段它们无梯度而VGG/DeDoDe/调制有梯度。
- 修正后smoke峰值显存：Flow约2.883GiB、匹配约3.425GiB。单epoch共8对的训练循环分别0.688s、0.996s，不含完整数据I/O与每轮5106对验证，不能直接当作长程耗时承诺。
- 完整任务已安排在GPU2，先Flow，再自动切换D2，再自动测试；GPU0/1正在执行其他任务。完整精度结果尚未产生。

原30轮预算曾估计2–4天；现改为各最多20轮并允许早停，实际耗时取决于验证收敛速度。实际以完整训练前100对及首轮日志吞吐修正，受共享存储、GPU频率和验证开销影响。无需手工在两阶段之间换权重。

## 文件与命令

新增：`configs/mhi_v2/full_rgb_d2.json`、`mhinet/v2/d2_modulation/full_rgb.py`、`mhinet/v2/d2_modulation/speed.py`、本notes、`reports/d2_modulation_no_flow/speed.json`。另更新 `docs/mhi_v2/d2_modulation_notes.md` 索引。之前无Flow评估入口及结果仍保留。

```bash
export TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp
PY=/root/miniconda3/envs/loma-repro/bin/python
# 三路径速度，单独占用同一张卡
$PY -m mhinet.v2.d2_modulation.speed --device cuda:2
# 最终预热配置的smoke；重跑需使用新的输出目录
$PY -m mhinet.v2.d2_modulation.full_rgb --device cuda:2 --smoke --output outputs/full_rgb_d2_smoke_new
# 两阶段完整任务，已启动，不要重复执行
$PY -u -m mhinet.v2.d2_modulation.full_rgb --device cuda:2
# 仅在进程中断后恢复
$PY -u -m mhinet.v2.d2_modulation.full_rgb --device cuda:2 --resume
```

正式输出 `outputs/full_rgb_d2_20261027/`：`run.log`、`data.json`、`config.json`、`flow/`、`d2/`；每阶段包含 `best.pt`、`latest.pt`、`history.jsonl`、`epoch_*.json`。最终输出 `test_test.json`、`test_hangzhou_test_real.json`，全部结束后才写顶层 `done.json`。

smoke保留在 `outputs/full_rgb_d2_20261027_smoke_warmup/`，日志同名前缀 `.log`。正式日志正在写入，训练和完整测试尚待长程任务完成。

### 正式启动确认

GPU2进程PID 228320；完整5106对初始验证已结束，Flow有效可见点P@3=80.648%，EPE=2.196px（不是训练后结果）。已进入第1轮并完成至少800对，累计训练loss=0.002751，峰值allocated显存2.884GiB。前800对耗时44.80s；当前稳态约19对/s，第一阶段每轮训练约36分钟，另加完整验证与保存开销。第二阶段仍排队等待第一阶段最优权重。进度以run.log为准。

### 20轮与严格早停变更（2026-10-27）

用户要求缩短预算。两阶段上限均20轮，min_epochs=5、patience=3、min_delta=0.001；best仍按任何严格P@3提升更新，早停参考值和计数独立保存到检查点。恢复旧检查点时从既有验证记录重建早停状态。余弦总长度压缩到20轮，保持已完成优化器步数和动量，不从零重置。

变更时Flow第1轮尚未结束，已训练超过2万对。为保留当前轮进度，一次性交接进程等待epoch_001验证及latest检查点写完，再自动停止旧进程并以`--resume`启动新设置。新配置已写入仓库，原进程内存中的30轮配置会在此次轮末交接时替换。状态见 `outputs/full_rgb_d2_20261027/policy_handoff.json`，进度继续写同一run.log；没有并行启动第二份训练。

CPU逻辑验证通过：小于min_delta的进步仍保留best；累计显著进步重置patience；连续3轮不足阈值会触发早停（同时满足至少5轮）；修改调度周期保留优化器步数。本次未改变网络、监督、数据划分或启动额外训练实验。

### 只读实时曲线

入口 `mhinet/v2/d2_modulation/live_curves.py`，读取现有run.log、history.jsonl、epoch_000.json，不调用GPU，不修改训练。页面每5秒刷新，Flow/D2可切换；展示本轮累计训练loss、完整轮loss、完整验证P@3/EPE、学习率、早停计数和进程状态。训练loss每100对更新，完整验证每轮更新。Flow P@3分母为有效可见D16点，D2为全部输出点。图中验证曲线从epoch1开始，epoch0基线在图内单独注明。

服务仅绑定 `http://127.0.0.1:8766`。远程访问需将训练环境8766端口转发到本地（Docker场景需转发到实际容器监听端点）；没有开放外网端口。默认全接口绑定曾被自动审批拒绝，已按建议改为本机绑定。运行命令：

```bash
/root/miniconda3/envs/loma-repro/bin/python -m mhinet.v2.d2_modulation.live_curves --port 8766
/root/miniconda3/envs/loma-repro/bin/python -m mhinet.v2.d2_modulation.live_curves --snapshot reports/full_rgb_d2_20261027/live_snapshot.png
```

页面、两个阶段SVG和JSON指标接口已实测返回200；中文使用现有完整Noto字体，快照已检查。快照仅表示生成时状态，不会在聊天中自动刷新；实时状态以页面为准。服务日志：`outputs/full_rgb_d2_20261027/live_curves.log`。

### 2026-10-28：关闭早停，两阶段续训至总计10轮

按用户要求，Flow从latest第7轮恢复并完成8–10轮，冻结Flow→D2从latest第5轮恢复并完成6–10轮；不是额外训练10轮。独立输出目录 `outputs/full_rgb_d2_10epoch_20261028/` 复制原best/latest及验证历史，原实验保留。两阶段均 `early_stopping=false`；best仍按RGB验证P@3的严格提升保存，另保留第10轮latest并分别进行完整RGB/test（5106对）与杭州/test_real（235对）评测。

D2继续冻结原实验Flow best（第4轮）：`outputs/full_rgb_d2_20261027/flow/best.pt`，不在续训时更换上游定位器。恢复优化器、调度器和随机状态，余弦周期从20轮压缩为10轮，保留已完成步数；因此属于缩短周期后的续训，不等同于从零配置10轮。网络、损失、数据划分及QRRU关闭设置不变。

文件：新增 `configs/mhi_v2/full_rgb_d2_10epoch.json`；更新 `mhinet/v2/d2_modulation/full_rgb.py`（独立阶段恢复、固定Flow来源、best/latest分别测试）、`mhinet/v2/d2_modulation/live_curves.py`（显示关闭早停）、本notes。

实际运行命令（已经启动，请勿重复启动）：

```bash
export TMPDIR=/home/disk1/MHINet/outputs/d2_modulation/tmp
PY=/root/miniconda3/envs/loma-repro/bin/python
$PY -u -m mhinet.v2.d2_modulation.full_rgb --config configs/mhi_v2/full_rgb_d2_10epoch.json --stage flow --resume --device cuda:2 >> outputs/full_rgb_d2_10epoch_20261028/run.log 2>&1 && \
$PY -u -m mhinet.v2.d2_modulation.full_rgb --config configs/mhi_v2/full_rgb_d2_10epoch.json --stage d2 --resume --device cuda:2 >> outputs/full_rgb_d2_10epoch_20261028/run.log 2>&1
# 曲线服务（已切换至新目录，仍使用原端口）
$PY -m mhinet.v2.d2_modulation.live_curves --root outputs/full_rgb_d2_10epoch_20261028 --host 127.0.0.1 --port 8766
```

启动验证：GPU2已从optimizer_step=71491恢复，Flow第8轮前300对正常完成，累计loss=0.0005494，峰值allocated显存2.867GiB；D2排队自动接续。按照原运行速度，剩余Flow约2小时、D2约6小时，另加两阶段best/latest完整测试时间。验收为两阶段done记录均epochs=10且early_stopped=false，历史完整、best与latest测试文件齐全；目前尚未完成续训，不能据此判断最终精度变化。
