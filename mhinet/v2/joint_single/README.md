# 连续Flow→D8→D2→单次QRRU联合训练试验

当前分支独立入口；原离散推理及主线损失保持不变。设计来源：docs/mhi_v2/joint_flow_d8_d2_single_qrru.md。

实际首轮为两个**分域训练**试验（GPU0 RGB_512、GPU1杭州），用于分别验证精确H与近似H上的表现，不是设计稿后续的混合域联合训练。每域254训练/64验证，均采用qrru_corners相同的选择；结束后分别在两域各64测试对上评估，测试不参与选权重。

初始化：原Flow、杭州D8/D2第11轮最佳权重、同seed42全新QRRU末层零初始化。DINO冻结。前3轮冻结MVT/Flow，仅适配VGG/解码器/温度/QRRU；第4轮起解冻MVT与Flow。学习率分别为1e-5、1e-5、2e-5、2e-5、8e-5；AdamW，累积4对，余弦退火至1e-6，最多30轮，至少8轮、连续6轮验证最终P@3未提高0.1个百分点则早停。初始模型可入选best。

连续路径：Flow位移插值产生p0；围绕p0构建间距8px的9×9 D8候选，以局部最高峰周围3×3候选的概率加权得到p8；源端4×4 D2子点、目标端连续p8周围4×4候选，局部dual-softmax及峰值邻域解码得到p2；执行一次QRRU得到final。候选采样统一align_corners=False。峰值索引不可微，概率、坐标、采样可微，不能宣称离散候选切换本身可微。

每批均匀抽64个D8源查询（1024个D2子查询）训练。推理遍历4096个D8源查询，按D8分数取128个源窗口，产生2048对输出。**首版不执行旧路径的跨源互选和目标去重**；因此旧离散路径的输出集合不同，不能只拿两者P@3直接归因于坐标解码。每次评估同时报告相同源点的Flow/p8/p2/final指标，以及完整1024网格Flow指标；初始和最佳测试均附同权重旧离散D2对照。推理与训练共享相同连续坐标函数，无训练软坐标/推理硬坐标替换。

损失：原Flow辅助＋广域D8分类＋连续局部D8分类/定位＋连续D2分类/定位＋单次最终中心定位。局部标签在no_grad中产生，预测坐标保留图；窗口未覆盖真值时局部项忽略，不制造最近正例，记录覆盖率，广域/Flow辅助继续监督。新增坐标损失按8原图像素归一化后用log1p平方鲁棒惩罚；杭州有容忍死区；精确H用四邻域固定插值标签，杭州局部分类对可信区域求概率和。无四角直接损失、无单调性或新拒配头。

RGB_512保留原flow_head_loss数值定义。杭州无可靠真实负对应标签，本试验只对参考可见位置监督Flow坐标及D16可信候选区域，**没有直接Matchability损失**，不制造严格负例；共享特征更新仍可能间接改变Matchability。Matchability作为D8先验权重时detach，训练和试验推理都不按其阈值删查询。

验收：--smoke在真实数据上完成两次仅最终中心损失的反传，检查p0/p8/p2与Flow定位头及共享特征参数梯度，再做一次完整联合损失反传检查。已在两域通过。此为梯度连通验证，不代表精度收益。正式任务从干净初始化重新加载，不使用smoke更新后的权重。

```bash
python -m mhinet.v2.joint_single.train --domain synthetic --device cuda:0
python -m mhinet.v2.joint_single.train --domain hangzhou --device cuda:1
```

输出目录outputs/joint_single/{synthetic,hangzhou}：配置、epoch_000及逐轮验证、历史、best/last权重、训练前和最佳模型在两域测试结果、完成状态。测试指标中old_discrete_d2指**同一次评估的当前特征权重**下的旧离散解码；原冻结基线见initial_test_*.json。报告必须区分同点比较、不同选点集合比较和完整Flow网格比较。

逐项同点归因入口：`diagnose_attribution.py`。固定权重对比旧/新父中心、硬/软解码、互选和 4/8 目标窗口；完整 64 对验证结果见 `reports/joint_single/attribution/results.md`。

D2 8×8 排序与解码诊断：`diagnose_d2_ranking.py --domain synthetic --device cuda:0`；报告见 `reports/joint_single/d2_ranking/results.md`。不训练、不调用 QRRU，固定源点比较原始/双向相似度、硬/软解码及真值峰值反事实。
