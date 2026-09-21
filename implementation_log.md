# MHINet implementation log

## 2026-09-21：第一档适应后最差5对回归诊断

按用户“检查恶化最大的样本”执行只读诊断。新增scripts/diagnose_adaptation_regressions.py（GPU2、H0-only，跳过CGMDP，原GPU0/1工作不动）与scripts/refit_regression_diagnostics.py（CPU，读取已存coarse数组）。前后权重SHA及val SHA登记在artifacts/stable_v2_adaptation_worst5.json，输出outputs/diagnostics/stable_v2_adapt_worst5_v2/01–05，含原图对、前后预测叠加、拟合点空间图、coarse数组。首次绘图因非连续numpy布局失败，修正ascontiguousarray后新目录重跑，旧部分输出未删除；补充探针最初缺LoMa导入路径，增加make_loma_importable后通过。未改模型/数据/训练。

结论：5对重叠误差分别5.284→43.355、1.475→31.619、12.723→39.887、0.890→26.852、1.640→26.665px。旧坐标+新权重重现退化，新坐标+旧权重保留原精度，归因主因是matchability筛选/加权，不是有效区坐标整体退化。只保留旧筛选集合并使用新坐标/权重可回到5.514/1.344/12.989/0.874/1.593px，新增过0.3门限点影响占主导；可见区外入选点31→123、5→90、29→101、12→91、26→82，按训练目标定义的negative误入选也显著增加。覆盖面积/点数增大，正规矩阵条件数约11–18，不是奇异求解或覆盖崩溃。GT-only可见区过滤诊断误差0.888/1.326/0.400/0.552/0.696px，不作可部署结果。已目检第一对原图、拟合点图和after叠加。详细表与损失尺度分析见docs/stable_v2_adaptation_regressions.md；没有自动实施修复，没有用GT修改实际匹配。

## 2026-09-21：合并共享描述子与可选下游分支至main

按用户要求准备并执行codex/shared-descriptor-lora→main合并，LoRA暂不启用。常规shared_descriptor_frozen、tier2/tier3和stable_v2_tier1_adapt配置均为lora=false；LoRA代码、历史权重及显式独立入口保留，不自动运行。远程main在合并前为d312c0a，源分支为cfb61ea，main为其祖先，无内容冲突。

使用独立干净worktree `/home/disk1/mhinet-main-merge-BoyC8Y/tree`，只合并已提交内容，不包含原工作区6项已修改和9项未跟踪内容；未stash/丢弃原改动。先merge --no-ff --no-commit，再用loma-repro运行 `OMP_NUM_THREADS=4 python -m pytest tests -q`，已提交版本179项通过（3.41秒）。此前原工作区181项包含未跟踪测试，本次179项是干净仓库验证，不混淆。README明确当前LoRA关闭状态。合并不改训练器/config或运行中的训练进程。

## 2026-09-21：第一档稳定数据低学习率适应启动

用户要求在第一档小学习率训练观察损失，并澄清下游应称“H0引导的粗细匹配”。已在docs/shared_stable_v2_adaptation.md明确D8粗→D2细→原生A/B坐标与置信度；P3离散中心、P6连续warped位置，不是已接可训练LoFTR/RRU，未自动拟合最终H。

三档训练前共享完整val基线已完成并归档artifacts/shared_stable_v2_baselines.json：第一档5772对loss0.131815872、desc0.126020921、H0_overlap0.930410331px、corner2.707986128px；第二档3848对0.133039119/0.127274857/0.930194257/2.933929338；第三档3848对0.141973942/0.132764357/1.420393846/4.832650851。按各自数据集登记，不跨档推断训练收益。

新增configs/runtime_paths.stable_v2.server.json、configs/shared_descriptor_stable_v2_tier1_adapt.json、scripts/train_shared_stable_v2_tier1_adapt.sh。初始化outputs/shared_descriptor_frozen_tier3_seed0/latest.pt，SHA256 `5a9ed14cc32a1a4ff3a843b737410da13d795b79c42a9d33d82a068226e30386`。冻结DINO，无LoRA，其他共享层训练、VGG BN统计固定，MHIR/密集下游不执行，D1不算。LR scale0.1即MVT/head1e-7、VGG5e-7、CGMDP1e-6；BS1累积4、新AdamW wd1e-4 clip1，原损失不改。正式限制2000优化步/8000对，seed0，500步存checkpoint；cosine保持一轮12995步horizon与650步warmup，进度条2000/12995停止是预登记预算，不是中断故障。结束时完整第一档val5772，不截取；不自动延长训练。

工程冒烟命令 `GPU_ID=1 MHINET_OUTPUT_DIR=outputs/diagnostics/stable_v2_tier1_adapt_smoke bash scripts/train_shared_stable_v2_tier1_adapt.sh --max-steps 2 --limit-train 8 --limit-val 2`，反传和验证通过，loss0.149783/0.130817、显存9.96GB；不同batch不作为loss下降证据。环境沿用loma-repro，未改训练器实现。

正式后台启动 `GPU_ID=1 bash scripts/train_shared_stable_v2_tier1_adapt.sh`，Popen(start_new_session=True)，PID3829239，输出outputs/shared_stable_v2_tier1_adapt_seed0，日志同名.console.log；未覆盖旧权重。实际train/val路径为stable_v2，manifest SHA同前验收记录。比较同一第一档完整val和训练窗口均值；如有收益再回测二/三档，不把本次训练启动或冒烟标成模型改善成功。

## 2026-09-21：登记新版共享基线与LoMa H0引导下游迁移第一阶段

用户批准共享网络并列下游方案并强调保留LoMa的H0依赖。实际定位为 `/home/disk1/LoMa/experiments/stage1_dedode_pyramid_hroi_v1`，git899fdb99a4076e312196bbbf99a3c329739b5d7a，源工作区无改动。不是直接移植src/loma关键点网络。登记 `artifacts/loma_downstream_source.json`，逐文件SHA固定6个纯匹配模块，保留MIT许可证；迁入mhinet/downstream/loma_reference。P3预测H0双向支持→D8 dual-softmax粗匹配→H残差D2局部匹配；P6为H-warped sym4对照。RRU、LoFTR fine仅定位，尚未迁入/训练，不混称为已完成的密集训练头。

新增HGuidedDenseDownstream无共享模块所有权，接收现有共享输出，非法H在逆/投影前拒绝；GT不进入匹配。此消费者明确为no_grad零样本推理，未复制旧共享网络train(False)包装，未改MHIR/共享训练梯度。DINO/MVT/VGG各1次，累计解码4步至D2，D1调用0。新增CLI evaluate-dense和scripts/validate_dense_loma.sh，默认指定tier完整val、显式--limit才冒烟，不覆盖已有输出。

共同checkpoint `outputs/shared_descriptor_frozen_tier3_seed0/latest.pt` SHA256 `5a9ed14cc32a1a4ff3a843b737410da13d795b79c42a9d33d82a068226e30386`；stable_v2 val SHA `9e69f1af29dca1073ca3f12a618d61b52b66a76e67346d2ab39830de17ec5f44`。共享完整val baseline用scripts/evaluate_shared_full.py新增--data-root，启动前写registration.json，GPU0/1/2分别第一/二/三档，PID3828039/3828040/3828041，输出outputs/shared_tier3_stable_v2_baseline_tier{1,2,3}和同名.log。运行命令 `CUDA_VISIBLE_DEVICES=GPU OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/loma-repro/bin/python -u -m scripts.evaluate_shared_full --checkpoint outputs/shared_descriptor_frozen_tier3_seed0/latest.pt --tier TIER --data-root /home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2 --output outputs/shared_tier3_stable_v2_baseline_tierTIER`。集中登记artifacts/shared_downstream_registration_v1.json，快照running不代表完成。交付时需看report.json，不提前宣称无遗忘或基线精度。

GPU3先P3再P6各2对真实val冒烟，输出outputs/diagnostics/dense_loma_{p3,p6}_smoke，均无失败。P3平均12000点、P6平均5969.5点，峰值allocated约2.335GB；计时含首步，不作稳定速度/精度排名。新增metrics保留LoMa NCM/precision/overlap_precision/SR，旧RMSE明确命名legacy_RMSE_correct5_or_failure10（仅正确匹配且失败常数10），不当作全匹配或估计H误差。单位为原生目标像素。拟合H评价、覆盖率、置信度曲线尚未完成。

实际对齐 `CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/loma-repro/bin/python -m scripts.check_dense_migration --output artifacts/loma_downstream_parity.json`：原_match编排注入相同GHIM/CGMDP输出，与新下游对齐。最初逐顺序严格相等失败，复查匹配集合坐标差均0，P3置信度最大差7.15e-7造成近并列排列变化；采用坐标规范排序及atol1e-7/rtol1e-6后通过，P6置信度差0。P3/P6原评价数字完全一致。仅一对下游隔离对齐，不冒充完整旧权重端到端验证。

测试：迁入4组原pytest测试（仅改import）36项；新引用SHA/非法H/空集与区外点3项；全套pytest共181项通过3.23秒。环境loma-repro新装pytest8.3.5、iniconfig2.3.0、pluggy1.6.0、tomli2.4.1，未升级模型依赖。详情及阶段状态见docs/shared_downstream_integration.md。未启动新训练。

## 2026-09-21：旧数据删除完成与新版极端外推复核

按用户授权删除 `/home/disk1/Data/datasets/GoogleEarth_scale_pairs`（删除前约70G）及 `/home/disk1/Data/datasets/GoogleEarth_temporal4_v1`（约68G）；使用明确路径的rm -r，未跟随temporal4/test软链接遍历额外目标。2026-09-21复核两个目录均不存在、删除进程结束；未进入回收站，无法直接恢复，需重新生成。原始GoogleEarth、quadrant_tiers_v1、stable_v2、test_single_tier3_v1均保留。删除前生成摘要保存在artifacts/deleted_googleearth_legacy_20260920.json；历史配置中旧路径保留作溯源，已不能直接用于运行，未擅自指向新版。

完整外推检查：scripts/audit_stable_extrapolation.py，报告artifacts/stable_geometry_v2_extrapolation.json，说明docs/stable_geometry_v2_extrapolation.md。实际命令 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/loma-repro/bin/python -u -m scripts.audit_stable_extrapolation --root /home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2 --output artifacts/stable_geometry_v2_extrapolation.json`。train121282/val13468，全部双向，784像素半像素重采样坐标；角坐标绝对值>10000px、无穷远线距离<10px均0，无分母变号。train/val最大角坐标绝对值2342.551/2323.659px，最小无穷远线距离199.450/215.059px，最大角点局部放大率8.944/9.044。仍有正常外推，角坐标绝对值>2000px为690/69对，坐标不是预测误差。

从每个split选双向角投影最远3对，共6对、正反12组，各100次128点GT几何对应的sigma0.1px高斯扰动拟合：重叠误差中位数0.02021–0.02254px、四角误差中位数0.04721–0.20497px、P90 0.07438–0.40258px。仅这些样本该噪声条件下的控制实验，不含遮挡mask/真实误匹配，也不是模型预测。全套142项测试通过（1.993秒），包括新恒等/反向尺度/分母变号拒绝测试。重新校验当前train/val manifest SHA与外推报告一致。本次不加载checkpoint、不启动训练，不把真值几何安全表述为模型精度验证成功。

## 2026-09-20：按批准清理工程权重并验收稳定几何数据

最终验收完成：`artifacts/stable_geometry_v2_verification.json` status=passed，耗时1022.38秒。全量134750对、1078000个PNG头/尺寸通过，train/val各90对解码和双向mask重建通过（共180对，按tier×ratio取前10对，非随机抽样）。全量H组合误差最大0；归一化Jacobian上界train9.9999209/val9.9948778≤10，中心投影范围2.4948557/2.4714207≤2.5，最小无穷远线距离0.2544594/0.2748707≥0.1。母图划分与v1严格相同，train/val的母图、geo、pair ID交集均0，test清单逐字节不变。train manifest SHA256 `c48eb269be7e96a6be8d4eb43b668035d758f0aaf4268d9f6abb8e337e3aad23`；val `9e69f1af29dca1073ca3f12a618d61b52b66a76e67346d2ab39830de17ec5f44`。未做全量PNG像素解码/CRC验证，未评估模型精度，未切换训练配置、未启动训练。

按用户明确批准的第一批清单删除47个工程测试.pt（54076860939字节，50.36GiB）及11个Python缓存目录；删除前验证每个权重精确路径/文件大小、缓存仅含.pyc、无训练/生成/评估进程。删除后47个权重全部不存在，四个正式shared_descriptor输出中的latest.pt和shared_descriptor.pt仍存在。未移动至回收站，无法直接恢复工程checkpoint，测试可重跑；日志/JSON/图像/脚本和正式权重不删。outputs由约152G降至102G。清单和执行状态在artifacts/cleanup_candidates_20260920.json、docs/cleanup_candidates_20260920.md；后续Python运行会正常重建缓存。

新版数据生成摘要completed，非smoke，输出 `/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2`（约313G）。实际train三档51978/34652/34652、val5772/3848/3848。新增scripts/verify_stable_dataset.py：全manifest唯一ID/母图/geo划分、双向H安全界、T_B@inv(T_A)组合、逆矩阵、元数据JSON、全部PNG文件头与尺寸；各split按tier×ratio各取前10对，共90对做完整图像解码和mask重建，明确不是全量像素解码。128线程读取小文件，CPU几何校验OPENBLAS_NUM_THREADS=1；先前串行和24线程只读扫描因IO慢终止，未修改数据，然后从头重跑完整检查。

实际命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/loma-repro/bin/python -u -m scripts.verify_stable_dataset --root /home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2 --old-root /home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1 --output artifacts/stable_geometry_v2_verification.json`。139项单测通过（2.047秒），包括缺失PNG和尺寸不符拒绝检查；smoke数据作为完整验收输入时被正确拒绝。完整三档预览位于outputs/previews/stable_geometry_v2_complete，已目检第一、第三档。

## 2026-09-18：稳定几何 v2 试生成及全量启动

用户授权执行新版数据生成。新增 `mhinet/dataio/stable_geometry.py`、`scripts/audit_stable_geometry.py`、`scripts/generate_stable_three_tiers.sh`、`tests/test_stable_geometry.py`；原生成器增加显式 --stable-geometry，不改变旧版默认几何。约束与保留率表见 `docs/stable_geometry_v2.md`，统计文件 `artifacts/stable_geometry_v2_retention.json`。归一化双向H：无穷远线距离≥0.1、中心投影范围≤2.5、整域Jacobian保守上界≤10；除法前隔离分母不合法情形。纯几何旧val保留率58.99/60.73/62.86%，重新采样补足数量，未读取预测选择阈值，未使用test精度。阈值偏保守、改变透视难度分布，不宣称只移除极少数异常。

实际命令：`bash scripts/generate_stable_three_tiers.sh --smoke --generate --output-dir /home/disk1/MHINet/outputs/stable_geometry_smoke_v2`，28对生成并逐对复查通过；预览 `outputs/previews/stable_geometry_smoke_v2`，已查看第三档。`/root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -q`：138项通过，2.062秒。环境沿用loma-repro Python3.10.20/OpenCV4.13.0；本任务CPU生成，不加载checkpoint，不改模型权重和训练配置。

现源数据重新划分与旧完整parent_split_manifest内容严格相等：train8663/val962母图组，预计121282/13468对。数据源 `/home/disk1/Data/datasets/GoogleEarth`，复用 `/home/disk1/LoMa/generate_pairs.py` SHA256 `4a4aea0fc2dffa6739d72700da3cf1dbba4d25b8ccb4f0a5086eff754ca2087a`。新输出 `/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2`，保留旧v1；磁盘空余15TB。输出dataset_summary记录实现hash和协议。

后台命令 `bash scripts/generate_stable_three_tiers.sh --generate`，通过Popen(start_new_session=True)启动PID3798971，日志 `/home/disk1/MHINet/outputs/generate_stable_three_tiers_v2.log`。已确认进程存活、进入train生成。未启动训练；全量生成尚未结束，必须完成全量数量/几何/标签/mask/划分检查后才能用于训练，不将冒烟通过标记为数据全量或模型验证成功。

## 2026-09-16：按用户要求撤销AFSS实验接入

两组正式训练已按用户要求中断，约434步，未到首次保存点，无正式checkpoint。
撤销pair_afss模块、A/B实验配置及启动脚本、专用测试和当前实验说明；训练循环恢复普通
均匀采样、按max_optimizer_steps训练，原有warmup+cosine保留。旧sampling_strategy配置
明确拒绝，防止拿旧AFSS配置误跑。保留tier筛选和quadrant runtime，数据生成不变。
历史日志/实验记录不删除；下面AFSS条目仅为历史，不是当前实施方案。
本次删除的受版本控制源码可由git历史恢复；未覆盖用户已有工作树修改。

## 2026-09-16：第一档 Uniform / AFSS-v2 双组训练入口

按用户确认只冻结DINO、同预训练初始化、seed0、BS1×累积4；验证/保存改每5000实际优化步，
结束补完整val。新增train_tier1_a/b.sh，默认GPU0/1，独立outputs/tier1_a/b_seed0。
configs/runtime_paths.quadrant.server.json指向实际三档数据；loader tier=1过滤，
确认train51978/val5772；其他档和test不参与训练/评分/选模型。
A五轮64975步；B五基准轮，前2轮全量25990步，规划cosine终点43534（按LoMa取整公式）；
显式warmup1000实际步。学习率/损失/冻结及关闭相关性重计算沿用已确认配置。

复用LoMa config/state/scheduler/controller四文件；LoMa HEAD
899fdb99a4076e312196bbbf99a3c329739b5d7a；config hash
7d3c2feb1d94d4b1cdbd8d72a39fc258b0d8984b8f0ba18976672fe23814faae，
state 9bef7194525ecc725b875fcd07f37af511bc95676e3eed2b8141d6b798c1e5d0，
scheduler 68368fc602eed60bd190dffd4d5e1c8963417c4f76abd62586bef195180f9ca0，
controller 783099ac1a022ff2ea9b941959f790c61e4ed60895e2cb39bd9fd33beca90e8b。
新增MHINet adapter以min(P,R,H0-AUC,H6-AUC)评分，坐标784px；详细定义见docs/tier1_uniform_afss.md。
分别读取A/B配置、seed0重新构建真实模型并逐张量计算完整state_dict哈希，两组均为
d18d443b8cdd81f317fa5cbf069a1ca065c6e1067f694ca031533c39be2c67c8，确认初始化一致。
原实现只在全量刷新更新EMA，本次5轮仅第2轮结束一次刷新；不声称长期防遗忘已验证。

120项测试通过；包括初始两轮相同索引、AFSS恢复后的索引序列一致、指纹拒绝、H6退化降低评分。
真实权重8train/2val缩小测试：A前4步、B先5步后同目录续到7步完成，
B连续7步也完成；第2轮结束评分8对，后续每轮选4对，保存最终7张可视化及best_validation.json。
命令：`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli train --runtime configs/runtime_paths.quadrant.server.json --config configs/diagnostic_tier1_b.json --output-dir outputs/diagnostics/tier1_afss_resume_v2 --tiny-gate-artifact artifacts/p4_tiny_gate_mcnet_d2.json --no-correlation-checkpoint --stop-after-optimizer-step 5`；续训去掉stop参数。
暂停/续训日志outputs/tier1_afss_smoke_v2.log；连续运行outputs/tier1_afss_continuous.log。
恢复后最终checkpoint SHA256 7efd2d788bab56542820c99e206c083cbd8d27431ad6baead11c72923742b4b5；
连续最终300fc571954e314ac51efc1819bdbed0062898a1d77926784208f16f752307bd。
短测不代表收敛；跨独立GPU运行首步损失一致，更新后有小幅数值差异，未证明训练逐位可复现。
采样状态序列恢复测试与神经网络数值逐位一致是不同结论。新运行另记录initial_model_sha256。
未启动正式长训练；保留现有脏工作树中的用户脚本与自动评估表变更，不合并无关内容。

## 2026-09-16：原始测试母图各自生成第三档两对

新增scripts/generate_test_tier3.sh与mhinet/dataio/generate_test_tier3.py。
读取GoogleEarth/evaluation_data/test_pairs.csv，仅用路径配对信息发现母图；
Source/Target按相对路径去重，各自单图生成2对tier3，禁止跨母图单位H假设。
种子20260916；尺寸比.8/.6/.4循环覆盖；复用已测quadrant_tiers生成函数与可见性掩码。
原始500对视觉测试、既有train/val完全不改。新输出GoogleEarth_test_single_tier3_v1。
先运行 --generate --smoke --output-dir outputs/diagnostics/test_tier3_smoke，4对生成、
loader、同母图与几何/可见重叠检查通过。随后后台启动完整生成，PID3765023，
日志outputs/test_tier3_generation.log；启动不等于完成，摘要completed后才可验收。
本次不接入精度评估入口，也不训练；新数据仅可评价合成几何/辐射/遮挡鲁棒性，
不能据此声称真实跨时相精度。AFSS训练预算未修改。

## 2026-09-15：三档四象限数据生成后台任务

新增quadrant_tiers.py和generate_three_tiers.sh，配方见docs/quadrant_three_tiers.md。
118项单元测试通过，含9种tier/ratio组合、H组合及双侧遮挡可见性掩码精确核验。
真实母图smoke：outputs/diagnostics/quadrant_tiers_smoke，train14/val14生成及loader读取通过。
命令：`bash scripts/generate_three_tiers.sh --generate --smoke --output-dir /home/disk1/MHINet/outputs/diagnostics/quadrant_tiers_smoke`。
smoke后补充跨图H/逆H分母同号检查，完整单测重跑通过。
随后后台启动 `bash scripts/generate_three_tiers.sh --generate`；完整生成尚未完成。
生成摘要保存旧LoMa函数来源hash、驱动与quadrant实现hash；本任务无需模型checkpoint。
用户明确授权删除GoogleEarth_scale_pairs_geometry_train_val_backup（实目录约66G），
已执行精确路径rm -r，未触及其他数据；删除进行中，非回收站操作，不保证恢复。

## 2026-09-15：撤销跨时相监督，恢复单母图合成与无真值视觉测试

不同母图未严格配准，旧temporal4及旧合成test的单位母图变换不成立；旧精度结论暂停使用。
未删除旧数据、结果或权重。generate_temporal历史入口现仅生成same_past/current各normal/hard两对，
每对两侧来自同一母图。地理分组9:1不变。新输出GoogleEarth_single_parent_v2；
dry-run核实train34652/val3848；原始test CSV500对直接引用，无H标签、不额外合成。
check_ready拒绝旧跨时相数据。冻结训练脚本切换新runtime和独立输出，超参保持。
test.sh改为visual_test，原始图仅网络输入resize；七张图仅红框预测，NO GT；
无精度指标，不写精度总表。evaluate CLI禁止GoogleEarth test精度评价，val不受影响。

环境沿用loma-repro；LoMa生成器SHA256
4a4aea0fc2dffa6739d72700da3cf1dbba4d25b8ccb4f0a5086eff754ca2087a。
原始CSV SHA256 f7214f38a37347bc1ba3d44b62e39b0f2b0e9cb95ad37d15b0197e4d631b1b72。
生成冒烟：`bash scripts/generate_temporal_dataset.sh --generate --smoke --output-dir /home/disk1/MHINet/outputs/diagnostics/single_parent_v2_smoke`，
train4/val4成功，loader读取全部8对通过；test清单500对无H。冒烟后只调整了摘要pairs_per_image=2和注释。
视觉冒烟：`GPU_ID=0 bash scripts/test.sh outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0/checkpoints/step_0017000.pt --max-pairs 1 --output-dir outputs/diagnostics/original_visual_test_smoke`成功。
checkpoint SHA256 1920ff8ab8059801f7c79e58330fb1834ef91199dfe53c661613c0319013fbce。
旧checkpoint仅用于验证可视化入口，不代表认可旧训练监督。未启动完整数据生成或训练。
详细新协议与命令见docs/single_parent_v2.md。
117项单元测试通过（包含无GT叠加与单母图配方检查）；首次测试发现旧test.sh精度契约断言，
已更新为无GT可视化契约并恢复DRY_RUN支持，重跑全部通过。

## 2026-09-13：仅冻结DINOv3的完整训练准备

新增frozen_dino profile，解冻MVT并保留GHIM head/VGG/CGMDP/Adapter/MHIR训练，
D1依旧不执行。新增 `configs/train_frozen_dino_temporal4_ebs4_20k.json`、
`scripts/train_frozen_dino.sh`，BS1×累积4、20k步、80k样本预算、seed0，
MVT LR1e-6；其他LR及四项GHIM监督沿用原BS4对照。默认关闭相关性重计算。
输出 `outputs/GHIM_joint_frozen_dino_temporal4_ebs4_20k_seed0`；从原始预训练开始。
runtime/权重/新train-val沿用 `configs/runtime_paths.temporal4.server.json`；
新训练配置SHA256 `9e24d2c22c45172e0b4fa2d1c6821d1bf2f680d71ca3d5c6d6f1ac8875a261bc`。

116项单元测试通过，DRY_RUN确认新配置与独立输出。
命令 `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=/home/disk1/MHINet /root/miniconda3/envs/loma-repro/bin/python scripts/check_frozen_dino.py`
真实两步审计通过：DINO无梯度，其他活跃组第二步均非零；MVT的H0/descriptor两路
MHIR损失梯度分别0.06225586/2.354383e-6。参数109,996,778，审计峰值allocated17.754GB，
reserved18.902GB。证据 `artifacts/frozen_dino_two_step.json`。
完整训练入口另在 `outputs/diagnostics/frozen_dino_trainer_smoke` 使用
`GPU_ID=0 MHINET_OUTPUT_DIR=/home/disk1/MHINet/outputs/diagnostics/frozen_dino_trainer_smoke bash scripts/train_frozen_dino.sh --stop-after-optimizer-step 2`
做两步冒烟；不是正式20k长训练。说明见 `docs/training_frozen_dino.md`。
冒烟正常结束（status=paused），保存 `outputs/diagnostics/frozen_dino_trainer_smoke/checkpoints/step_0000002.pt`，
SHA256 `49eec986f9d17460baee6ddbd1f2ee2b46754c29e525495493b24c7b7da9e7eb`。

## 2026-09-11：卡0有效BS4等样本预算对照

用户同意增加有效BS4对照。新增 `configs/train_frozen_dino_mvt_temporal4_ebs4_20k.json`：
真实BS1×累积4，20k优化步，80k样本次数，与BS8×10k等样本预算；seed0及初始化、冻结组、
损失和峰值学习率保持一致。warmup_fraction=.05对应1000步，每1000步验证和保存，对齐
BS8实验的4000样本warmup及每4000样本验证。使用独立输出
`outputs/GHIM_joint_frozen_dino_mvt_temporal4_ebs4_20k_seed0`，GPU_ID=0，传
`--no-correlation-checkpoint`。优化器更新次数增加，因此AdamW动量/衰减累计也随之变化，
这是不同有效batch训练方案的比较，不声称完全相同优化轨迹。仅准备配置/命令，未启动训练。

## 2026-09-11：接入已验证的关闭相关性重计算训练开关

训练CLI新增 `--no-correlation-checkpoint`，对HGuidedLocalCorrelation设置
activation_checkpoint_training=False；默认不传仍保持原设置。metadata记录实际模式。
这是已通过短测的执行/显存策略，不改变参数结构或损失配置，允许原同目录checkpoint续训；
训练checkpoint保存仍正常。用户命令维持BS1×累积8，输出目录保持原temporal4_ebs8目录。
本次只接入命令，不启动正式训练。

## 2026-09-11：BS1关闭相关性重计算测试

保持BS1×累积8，独立物理GPU0 RTX4090顺序off/on，同数据/初始化/损失/优化器，
3步预热+6步计时。新增probe诊断开关，未改正式训练默认。
off平均5.08554秒/步、1.57309pair/s、allocated14.752GB、reserved15.680GB；
on平均6.82331秒/步、1.17245pair/s、allocated8.393GB、reserved9.783GB。
吞吐+34.17%，步耗时-25.47%；两组无OOM，16对初始H0一致。
GPU1另做同权重、一次更新后的梯度复核：loss均4.9118185，H0-H6差0，309梯度张量
通过rtol1e-3/atol1e-5，最大绝对差9.54e-7、相对L2差8.25e-10。
结果见 `docs/temporal4_correlation_recompute_probe.md` 与三份temporal4 checkpoint probe JSON。
新增 `scripts/check_correlation_recompute.py`；脚本py_compile通过。首次probe导入类名错误
已改为实际HGuidedLocalCorrelation后重跑成功。此次没有训练checkpoint，只有诊断JSON；
不把短测通过当作长训练精度/稳定性验证，也未启动正式训练。

## 2026-09-11：BS2×累积4性能与数值对照

用户要求测试以更多显存换速度。新增独立 `scripts/probe_temporal_batch.py`，同一物理GPU0
顺序测BS1×8和BS2×4；新train前16对、真实预训练、frozen_dino_mvt、GHIM四项监督，
3步性能预热+6步计时，含读取/完整前反向/优化，不触碰正式输出、不存checkpoint、不访问test。
实际记录 `artifacts/temporal4_ebs8_bs1_probe.json`、`artifacts/temporal4_ebs8_bs2_probe.json`。
BS1平均6.7933秒/步、1.1776pair/s、allocated8.393GB；BS2平均5.5323秒/步、1.4461pair/s、
allocated15.555GB、reserved19.212GB。吞吐+22.79%，步耗时-18.56%，无OOM与非有限梯度。
DINO/MVT无梯度，第二步后活跃解冻组非零，D1不执行，共享调用每microbatch各一次。
16对初始H0跨BS最大四角坐标差0.3831177px，既有门槛0.1px未通过；没有放宽门槛或
改正式默认batch。另记录geo/mat跨microbatch的归一化权重差异，后续公平性修复需单独测试。
结果表、命令、边界与建议见 `docs/temporal4_batch_probe.md`。没有改变正在使用的训练配置。

## 2026-09-11：无checkpoint同目录从头训练

按用户明确要求修改启动规则：同目录存在checkpoint则自动续训；没有checkpoint则
`restart_no_checkpoint`，重新初始化模型/optimizer/scheduler/数据流，从第0步训练，
替换run.json/train.jsonl。为避免丢失证据，旧运行记录及validation/visualizations先移入
同目录 `previous_no_checkpoint_*` 备份；无关文件不动。替换推迟到模型、配置、数据初始化成功后。
新增临时目录回归检查：新目录、无checkpoint重启、有checkpoint续训、显式resume优先、
旧记录可恢复、无关文件保留。未启动用户训练，原103步记录此刻尚未覆盖；下次运行时生效。

## 2026-09-11：有效 batch 翻倍至8

新增 `configs/train_frozen_dino_mvt_temporal4_ebs8.json`：真实BS1、累积8、有效batch8；
独立experiment_id/output使用 `GHIM_joint_frozen_dino_mvt_temporal4_ebs8_seed0`。
原有效batch4配置不修改。学习率、冻结组、监督与10k optimizer步保持不变，因此总训练
样本次数由40k变为80k，不是等样本预算对照，预计训练计算时间增加。
TrainConfig.validate及脚本DRY_RUN检查通过；本次未启动训练、未新增checkpoint。

## 2026-09-11：训练入口切换新 temporal4 数据集

用户正在生成新数据，要求训练脚本准备就绪。新增独立runtime与training config：
`configs/runtime_paths.temporal4.server.json`、`configs/train_frozen_dino_mvt_temporal4.json`。
`scripts/train_frozen_dino_mvt.sh` 默认物理卡1、数据根
`/home/disk1/Data/datasets/GoogleEarth_temporal4_v1`，输出
`outputs/GHIM_joint_frozen_dino_mvt_temporal4_seed0`。原预训练权重、冻结组、损失、学习率、
有效batch4、10k步不变；max_val_pairs=null使用全部新val，不固定2500。
新配置experiment_id带temporal4，旧checkpoint配置hash不匹配，避免误接旧数据实验。
脚本独立构造命令，不改用户已修改的train_e01.sh。

新增启动检查 `mhinet/dataio/check_temporal_ready.py`：必须completed且非smoke，
核对train/val四对组成、实际数量与摘要、影像/mask存在、母图/地理组无交集和test入口存在。
检查只读，不启动GPU、不修改数据。当前实际目录仍处于planned，仅train已创建；实测检查
正确退出2，提示等待生成完成。DRY_RUN确认新runtime/config/output三处路径正确。
新增配置/脚本默认、未完成/smoke拒绝、完整清单/缺图拒绝测试。未启动任何训练。

## 2026-09-11：双时相四对数据生成与分组9:1

用户要求每组past/current母图生成same_past、same_current各1对及跨时相独立2对，
val与train约1:9；用户确认母图基本严格配准。新增 `mhinet/dataio/generate_temporal.py`、
`scripts/generate_temporal_dataset.sh`，复用LoMa实际生成函数，不修改LoMa或旧数据。
跨时相按近似identity母图关系组合H，并记录近似标签来源；不宣称严格真实配准标签。
母图根 `/home/disk1/Data/datasets/GoogleEarth/training_data`，原Train/Val池合并后按stem配对
9,625组，先做完整0.01度地理格划分：train8,663组/34,652对，val962组/3,848对。
test不改，复用 `/home/disk1/Data/datasets/GoogleEarth_scale_pairs/test` 符号链接；manifest SHA256
`3db0eca8c64a23cb1000d6905cbcf82dd221f653033e3b8463cd9484601dfb06`。
LoMa生成器SHA256 `4a4aea0fc2dffa6739d72700da3cf1dbba4d25b8ccb4f0a5086eff754ca2087a`。

默认命令 `bash scripts/generate_temporal_dataset.sh` 仅计划，不写目录；显式 `--generate`
才生成到 `/home/disk1/Data/datasets/GoogleEarth_temporal4_v1`，不覆盖非空目录。
小样本命令 `MHINET_DATA_OUTPUT=/home/disk1/MHINet/outputs/diagnostics/temporal4_generation_smoke_v2 bash scripts/generate_temporal_dataset.sh --smoke --generate`。
环境沿用conda loma-repro。分组/配方新增3项单元测试通过；真实母图小样本验证与产量清单
保存在smoke目录。此处没有新模型checkpoint，也没有执行全量生成或切换训练数据。
重分母图意味着旧E01不能作为新val独立性前提下的继续训练起点。说明见
`docs/temporal_dataset_v1.md`。
最终复核：111项单元测试通过。smoke的train/val各4对、各4张预览；现有MHINet loader
可读取全部8对影像、H及overlap mask（784输入）。地理格交集为空，test manifest hash未变。
合成矩阵最大自洽误差分别4.10e-13、6.92e-13px；不将该自洽值等同真实跨时相标注精度。

## 2026-09-11：新增只冻结 DINO/MVT 的完整训练入口

用户指定覆盖原默认冻结策略：GHIM head、VGG、CGMDP 活跃累计解码层、Adapter/MHIR
全部可训练；仅冻结 DINO/MVT，D1 仍作为不执行的兼容分支。新增
`configs/train_frozen_dino_mvt.json`、`scripts/train_frozen_dino_mvt.sh`，默认物理卡1，
新目录 `outputs/GHIM_joint_frozen_dino_mvt_seed0`，原始预训练初始化，不继承 E01。
完整配置和冻结/学习率表见 `docs/training_frozen_dino_mvt.md`。

修复原 provider 强制冻结 head 和旧 head.train() 强制 eval 的包装限制；head 参数组
独立学习率1e-6，保留原 profiles 的冻结行为。总可训练参数22,579,690。
加入 `mhinet/engine/ghim_losses.py`：LoMa 四项公式，权重 total=1、mat=.01、cls=.0001、H=.05；
真实 overlap mask 来自 `/home/disk1/Data/datasets/GoogleEarth_scale_pairs/train`。
非法拟合在归一化除法前隔离，失败样本仍训练 coarse 分支；记录各项损失，H0/H/T 不detach。
安全输入对照 `/home/disk1/LoMa/experiments/loretta_stage1_h/losses.py`，五项值差均0。
训练仍是train，验证全量val/2500，test不参与选择；每500步验证并输出七张图。

资源继承 `configs/runtime_paths.server.json`：LoMa `/home/disk1/LoMa`，LoRetta权重blob
SHA256 `09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e`，
VGG/decoder `/root/.cache/torch/hub/checkpoints/loma_B.pt`。环境 `loma-repro`，
Python3.10.20、torch2.11.0+cu128。新配置SHA256
`1e9b5d0f3f37052e84224babd7f99959991d81117f602012941e30aac17f4648`。

验证命令：`OMP_NUM_THREADS=1 /root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -q`
（108项通过）；`DRY_RUN=1 bash scripts/train_frozen_dino_mvt.sh`；
`CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/disk1/MHINet /root/miniconda3/envs/loma-repro/bin/python scripts/check_frozen_dino_mvt.py`。
结果 `artifacts/frozen_dino_mvt_two_step.json`：DINO/MVT无梯度，head有梯度，第二步VGG/decoder
梯度非零，共享各一次，D1零调用，峰值allocated8,294,526,976字节。
首次无warmup/head LR1e-5探测两步loss4.746→138.607；这是不稳定警示，不标记精度通过。
正式head LR1e-6 + 500步warmup复查，同一训练样本两步loss4.746→4.725。

正式入口冒烟：`GPU_ID=1 MHINET_OUTPUT_DIR=/home/disk1/MHINet/outputs/diagnostics/frozen_dino_mvt_trainer_smoke_v2 bash scripts/train_frozen_dino_mvt.sh --stop-after-optimizer-step 2`。
真实BS1累积4，两步共8对，成功保存checkpoint SHA256
`b5fb5cb4567670c6ee06a29c276cf07995b114c89ce9f3ef5f1df74f4e1c53ba`。
同目录再执行 `--stop-after-optimizer-step 3`，确认 `resume_mode=auto_latest`，从step2
恢复到step3，总loss1.85819，checkpoint SHA256
`537d11d4b11ca1bd75c5dbfece9b1bfa1498127270bc277186f8b554ae0d46db`。
此前v1 smoke目录保留，添加显式学习率配置前的记录不能用于新配置续训。
未启动正式10k长训练；原P4 tiny gate仅证明原精修模块，不作为新增GHIM监督收敛证据。

原E01独立test已完成1000对，输出 `outputs/E01_heads_test_step10000` 并自动进入评估总表：
H0 MACE15.41256→最终13.58886px，最终成功@5px26.2%，几何无效0；
val成绩不能当作test泛化通过。本次配置来自用户冻结/损失指令，不根据test调权重。

This is an evidence log, not a claim that the model has been validated.  The
active requirements are the server handoff package training protocol v1.2 and
loss revision 1.1.  Files under `history/` were not used as implementation
instructions.

Canonical model terminology from this point forward is GHIM (global
homography initialization, producing `H0` and `F_MVT`), CGMDP (MVT-guided
multi-scale descriptor pyramid, producing the fused matching descriptors
`D8/D4/D2` on the current mainline) and MHIR (six-update homography refinement,
producing `H_final = H6`). D1 code remains registered but inactive and is not
decoded, optimized, or executed by default. Historic `stage1_*` names below
quote legacy code, checkpoints or artifacts and remain compatibility aliases;
they do not denote an additional functional module.

## 2026-09-08 — design package and repository

- Project/repository: `/home/disk1/MHINet`; initialized with `git init -b main`.
  The first implementation/evidence milestone through the D8/D4 tiny gates is
  commit `5ff680b` (`Implement MHINet v1 through P4 D8/D4 gates`).
- Design package: `/home/disk1/MHINet/MHINet_server_handoff_v1.2`.
- Read order followed: `START_HERE.md`, docs `01`, `03`, `05`, `06`, `02`,
  `07`, `08`, then all files under `configs/`.
- Package integrity command:

  ```bash
  conda run --no-capture-output -n loma-repro \
    python MHINet_server_handoff_v1.2/verify_package.py \
    MHINet_server_handoff_v1.2
  ```

  Result: 31 files checked, zero missing or hash errors.
- Architecture config:
  `/home/disk1/MHINet/MHINet_server_handoff_v1.2/configs/mhinet_v1.json`,
  SHA256 `4a859f413ee4be59acb1dc6c0cf189e80b8f4e7b3f54dc9f117acd1ac5cfe128`.
- Server runtime config written to
  `/home/disk1/MHINet/configs/runtime_paths.server.json`; model structure has no
  hard-coded server paths.

## P0 — real resources and legacy alignment: passed

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loma-repro python -m mhinet.preflight \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_preflight.json --overwrite
```

Evidence: `/home/disk1/MHINet/artifacts/p0_preflight.json`, artifact SHA256
`92edb6d2e79691df0d44d119ea13c249fe9347acb750c202fc9d9ca4685ce5cc`.

Resolved source and weights:

- LoMa: `/home/disk1/LoMa`, clean commit
  `899fdb99a4076e312196bbbf99a3c329739b5d7a`, branch
  `codex/stage1-dedode-pyramid-hroi-v1`.
- LoRetta source: `/home/disk1/LoMa/third_party/LoRetta`, clean commit
  `e274bb5628e6324dbe36a1590d88e5adbfe1d06e`.
- Selected shared DINO/MVT/Stage1 checkpoint (resolved blob):
  `/home/disk1/LoMa/outputs/loretta_stage1_h/cache/huggingface/hub/models--BRoss123--LoRetta/blobs/09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e`;
  1,294,957,584 bytes; SHA256
  `09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e`.
- LoMa-B VGG/DeDoDe checkpoint:
  `/root/.cache/torch/hub/checkpoints/loma_B.pt`; 757,888,113 bytes; SHA256
  `3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758`.
- Data root: `/home/disk1/Data/datasets/GoogleEarth_scale_pairs`.
  Train/val/test contain 36,000/2,500/1,000 manifest rows and all referenced
  images/labels exist.  Manifest SHA256 values are respectively
  `3eed3d3d425caf8465afb5121dfd85960f00932fccc343f37187e14ce27d499c`,
  `a5e6f58bc46cef2dcc49db4e859e2a845755d6aee13be88c998f14a0386b5503`,
  and `3db0eca8c64a23cb1000d6905cbcf82dd221f653033e3b8463cd9484601dfb06`.

Environment (`conda` env `loma-repro`): Python 3.10.20, PyTorch
2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, cuDNN 91900, NumPy
2.2.6, SciPy 1.15.3, OpenCV 4.13.0, Kornia 0.8.2, Einops 0.8.2.  Server has
four NVIDIA RTX 4090 GPUs (24,564 MiB each), driver 565.57.01.  `pytest` and
`openpyxl` are absent; the repository therefore uses the standard-library
`unittest` runner.  Direct `timm` import currently fails because a transitive
wandb path references removed NumPy 2 `np.float_`; MHINet does not depend on
that import path.

Legacy-wrapper audit:

- old full matcher `train()` forces eval;
- old pyramid extraction is wrapped in `inference_mode`;
- old MVT contextualization and correlation sampling use `no_grad`;
- the Stage1 head forward itself accepts input gradients;
- the old fitter can solve a whole mixed batch before masking an invalid item.

The new provider reuses lower-level compatible modules, not those inference
wrappers.  DINO alone stays under no-grad.  The frozen Stage1 head executes in
autograd, and the safe fitter isolates eligible samples before any
differentiable solve.

Live alignment command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loma-repro python -m mhinet.alignment \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_p1_alignment.json --overwrite
```

Evidence: `/home/disk1/MHINet/artifacts/p0_p1_alignment.json`, SHA256
`fa1acf5ea8039b478c6a139a026c965b986d17ed06e461f01b613e227e0c1505`.
On pair `000_pair0`, safe/new H0, D8 and D2 are exactly equal to the legacy
legal path; Stage1 support is 1,491 in both.  Shapes are H0 `1x3x3`, context
`1x2x49x49x1024`, and D8/D4/D2/D1
`1x2x256x{98,196,392,784}x{98,196,392,784}`.  DINO, MVT, VGG and the DeDoDe
full pyramid each ran exactly once.  The one-off unwarmed 530.2 ms and
3,416,032,768-byte peak are engineering observations, not a latency claim.

Grouping audit: `/home/disk1/MHINet/artifacts/grouping_audit.json`, SHA256
`5f6483972ead7897945982b67b57bd1a3c2346ef8a60fe57539ed1c4bacec8a4`.
Parent groups are disjoint.  A 0.01-degree geographic audit found one
train/val cell collision; the deterministic policy holds val fixed and removes
four affected train pairs (35,996 safe train pairs).  Test is sealed and is not
used for exclusions or selection.

## P1/P2 — feature ownership, geometry and correlation: passed at unit/reference level

Implemented shared provider, trainable adapters, align-corners-false coordinate
conversions, TL/TR/BL/BR h33=1 four-point DLT, pre-solve safety isolation,
9x9 denominator checks, H-guided local correlation, dy-outer/dx-inner
candidate ordering, sampled-feature renormalization and chunking.  H, T,
sampling grids and feature tensors are not detached in the production path.

Reference tests include identity/translation/perspective, FP64 autograd
gradcheck, a mocked assertion that `solve_ex` never sees the singular member of
a mixed batch, direct-loop correlation equality, chunked/un-chunked output and
gradient equality, and activation-checkpoint output/gradient equality.

Real D8 correlation diagnostic:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n loma-repro python \
  -m mhinet.real_correlation_audit \
  --runtime configs/runtime_paths.server.json --pair-index 0 --scale 8 \
  --seed 0 --output artifacts/p2_real_correlation_d8.json --overwrite
```

On 4,726 observable queries, a random 64D adapter places the nearest exact-H
integer candidate top-1 for 69.11% of queries and its argmax is within 1.5 D8
pixels for 95.18%; raw 256D values are 78.16% and 98.88%.  This is a fixed
feature correlation-signal diagnostic, not validation accuracy.

## P3 — exact parameter count and zero-init full forward: passed

Command:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n loma-repro python -m mhinet.engineering \
  --runtime configs/runtime_paths.server.json --pair-index 0 \
  --output artifacts/p3_zero_init_smoke.json --overwrite
```

Evidence SHA256:
`2c003cd950bffafc0d0f8b8fbd904924dbe5ba490d070b48ed98a036d32fc7b0`.
All four FC2 weight/bias tensors contain zero nonzero elements; exact new
parameter count is 2,544,160.  A real eight-update forward accepted all eight
updates and changed H0 corners by at most `6.103515625e-05` input px
(`<1e-3`).  Shared components each ran once.  Peak allocated memory was
3,426,232,320 bytes.  The unwarmed 5,833.6 ms includes a contended server and
is not a latency benchmark.

Failures found and fixed before this pass:

1. A first run on physical GPU 3 failed before iteration because unrelated
   processes left insufficient memory for a 602 MiB D1 allocation.  The target
   was moved to a GPU with verified free memory; no result was fabricated.
2. The next run exposed BF16 pyramid vs FP32 adapter weights outside autocast.
   Adapter/decoder CNN boundaries now use CUDA BF16 autocast while correlation,
   DLT and loss geometry stay FP32.
3. A later fail-fast adapter finite check attempted a scalar read on a meta
   tensor.  It now runs on materialized CPU/GPU tensors and the full-shape meta
   contract test remains valid.

## P4 — loss/trainer/gradient engineering: passed; tiny gate unresolved

Default loss is the uniform mean of the eight pre-guard proposal-corner
coordinate L1 values.  FGO, overlap, auxiliary correlation and Planar heads are
off.  An all-invalid batch produces a finite graph scalar and explicitly asks
the trainer to skip optimizer and scheduler.  Atomic checkpoints contain model,
optimizer, optional scheduler/scaler, optimizer/microbatch/data progress and
Python/NumPy/Torch CPU/CUDA RNG state; restricted `weights_only=True` loading is
tested.

Gradient-audit command:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n loma-repro python -m mhinet.gradient_audit \
  --runtime configs/runtime_paths.server.json --pair-index 0 \
  --output artifacts/p4_gradient_audit.json --overwrite
```

Evidence SHA256:
`8501fb9ed2dd9f4e837d2c85ad91966625d8e0a6c56526197da1a8bebb85c799`.
On the first zero-init step, every FC2 has finite nonzero gradient while all
four adapters and decoder-upstream groups have exactly zero gradient.  After
one optimizer update, all those upstream groups have finite nonzero gradients.
A final-H-only loss reaches all four scales.  MVT has finite nonzero gradients
independently through H0 and through the pyramid; VGG and DeDoDe have finite
nonzero pyramid-route gradients.  DINO and Stage1-head parameter gradients are
all `None`.  Peak allocation was 4,630,508,032 bytes for cached-feature eight
round backward and 5,493,430,272 bytes for the real joint branch diagnostics.

Full 784 training profiles (two optimizer steps, one pair per step):

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.profile --runtime configs/runtime_paths.server.json \
  --profile heads --optimizer-steps 2 \
  --output artifacts/p4_profile_heads_784.json --overwrite

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n loma-repro \
  python -m mhinet.profile --runtime configs/runtime_paths.server.json \
  --profile joint --optimizer-steps 2 \
  --output artifacts/p4_profile_joint_784.json --overwrite
```

- `heads`: passed; artifact SHA256
  `c71d52d313a396c5072bc3f1ba82daba8365c86b4b846c3ac594c8616cc21932`;
  peak allocated/reserved 5,593,135,616/5,796,528,128 bytes.  Step 0
  forward/backward was 3.0465/2.8476 s and step 1 was 1.9259/2.6159 s.
- `joint`: passed on one 24 GiB RTX 4090; artifact SHA256
  `abfcc2a454a18c05d1dfef886f0b4b252889c377434e25c8cecbb0dd11cdc0f7`;
  peak allocated/reserved 14,186,542,080/14,512,291,840 bytes.  Step 0
  forward/backward was 2.4575/3.2071 s and step 1 was 2.2304/2.8643 s.
  At zero init, DeDoDe/VGG gradients were zero as expected; after the first
  optimizer update both were finite and nonzero.  MVT was already nonzero at
  step 0 via H0.  DINO and Stage1-head gradients remained absent.  Shared
  DINO/MVT/VGG/full-pyramid calls were one per step in both profiles.

After D2 timing exposed excessive Python overhead from constructing one
activation-checkpoint context per 1,024-query chunk, correlation was refactored
to checkpoint the complete chunk loop once.  Sampling still occurs in the same
1,024-query chunks, with identical ordering and no detach; the existing direct,
chunked/un-chunked and checkpointed output/gradient tests all pass.  Fresh
two-step 784 profiles for the current implementation both passed:

The current path was first exercised by a 32-residual, one-step D2 integration
with finite backward and zero rejected updates (expected `failed` solely due to
the one-step budget); peak allocated/reserved memory was
2,761,892,352/3,011,510,272 bytes.  Artifact SHA256
`805142ff226a1047aa1f9bf7d17de713e912adafaf1e111abc13bfc458281a87`.

- `heads`: peak allocated/reserved
  8,731,754,496/9,053,405,184 bytes; artifact SHA256
  `c6657331a21076f54da91dfbe7374214d105827afc4a41a941ba1d182c96a37b`.
- `joint`: peak allocated/reserved
  17,336,022,016/17,792,237,568 bytes on the 24 GiB RTX 4090; artifact SHA256
  `7dac145bf10f0e8d7bd582ddf2b96c6915f708bbb54cdf3bd253aca4b59ef9c6`.
  At step 1, DeDoDe, VGG and MVT gradients were all finite and nonzero; DINO
  and Stage1-head parameter gradients remained absent, and shared call counts
  remained one.  The optimization trades about 3.1 GiB extra peak allocation
  in `joint` for lower checkpoint-management overhead and remains below the
  measured device limit; both the old and current profiles are retained.

Minimal trainer smoke (D8, two rounds, two optimizer steps, B1 x accumulation
4) used `configs/train_minimal_smoke.json`, SHA256
`6946de077ed11b5f3a5dcf495003fe073a7c7e424815597b1cd716a66f0239c4`:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.train --runtime configs/runtime_paths.server.json \
  --config configs/train_minimal_smoke.json \
  --output-dir outputs/mhinet_minimal_smoke --overwrite
```

It completed two steps with zero invalid attempts and zero shared-call
violations.  The losses were 4.722061 and 2.598176 px.  Its checkpoint is
`/home/disk1/MHINet/outputs/mhinet_minimal_smoke/checkpoints/step_0000002.pt`,
906,932,089 bytes, SHA256
`fb3f0bdd23c9b8807965ddee7471e13d0b4d234f7f8094946e80a83831c75a75`.
The two-pair validation smoke had H0/final mean MACE 6.61187/6.65973 px;
this is explicitly an entrypoint check, not an accuracy pass.  Validation
summary SHA256:
`113d0c2abdde39dadcabee5dc1f4e7fe6a008b53fd33f4a75db5ecbd9a39be9f`.

Pause/resume was then exercised at an optimizer boundary with
`--stop-after-optimizer-step 1` and the same config.  Step 1 saved data
position 4; resume restored model, optimizer, scheduler, CPU RNG and CUDA RNG,
then completed step 2 at data position 8.  The step-1 and resumed step-2
checkpoint hashes are respectively
`5a17a0d4582b478fbe22c3081044ae2a08aab69c5ac68bcae72618cb5b2155f3`
and `5b339a7a8f81c77b2998ce078ce79dcc6d3a0d68f6f817ea5a1ccb3ee534021a`.
Compared with the uninterrupted checkpoint, 963/964 model tensors were
bitwise equal; the D8 adapter differed in 134 elements with maximum absolute
difference `8.298084e-7`, consistent with the CUDA warnings for nondeterministic
grid/adaptive-pool backward.  Progress and scheduler state were exact.
Evidence: `/home/disk1/MHINet/artifacts/p4_resume_smoke.json`, SHA256
`d9ae16eb2d7a6e8125f76115efaee9ce327c0cb7384af46d1b7c7e69879e1a33`.

Current test command and result:

```bash
conda run --no-capture-output -n loma-repro \
  python -m unittest discover -s tests -v
```

Result at this point: 65 tests passed.  `compileall`, every CLI `--help`
entry, and the 31-file design-package verifier also passed.  The regression
suite now explicitly checks that an isolated non-finite decoder output is
reported instead of being hidden by its guarded zero placeholder, and that
the repeated-pair GPU-preload path can execute both raw and parameter-averaged
tiny evaluations with the correct call contract.  It also asserts that all
1,024-query chunks use one activation-checkpoint context and that optimizer
protocol drift makes the tiny gate fail closed.

### Tiny-overfit attempts

The implemented CLI uses B=1, only new layers, AdamW lr `1e-3`, weight decay
zero, no scheduler, gradient-norm clip 1, uniform proposal L1 and at most 2,000
steps.  It records H0/every update/final MACE, rejection/failure rates, peak
memory, per-pair inference time and atomic checkpoint SHA256.  Controlled H0 is
diagnostic-only and is never used by formal training.

- A 2-pair/2-step integration smoke completed all data, shared-feature,
  backward and checkpoint paths.  It is deliberately marked failed because it
  is not the required 32-sample test.  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_integration_smoke.json`.
- The first 32-distinct-pair D8 run was stopped at step 704 after discovering
  the missing protocol gradient clip; its 5.16 px observation is not a final
  result.
- The corrected 32-distinct-pair D8 stress run (clip 1, controlled residuals
  about 5.6–8 input px) ran all 2,000 steps.  Final mean MACE was 1.431142 px
  with zero rejected updates: a real improvement but a failed `<0.1 px` gate.
  Artifact: `/home/disk1/MHINet/artifacts/p4_tiny_s_d8.json`; checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d8_seed0.pt`, SHA256
  `1f974ab3f9d52e8cb5d015e69420470305f731320b2e03185b737d988acdac84`.
- A one-exact-pair/32-H0-residual diagnostic using the same small D8 residual
  regime collapsed near 6.30 px through step 832 and was stopped for diagnosis.
  The real correlation audit showed that the small residual occupied only
  about 0.63 D8 pixels and the nearest-vs-center correlation margin averaged
  0.015.  A rerun now uses 0.7–1.0 of the declared half-bound (still no more
  than half of the decoder update bound), or about 1.4–2 D8 pixels.
- The corrected locally observable translation run used one exact train image
  pair and 32 deterministic H0 residuals, B1 x accumulation4, clip 1 and the
  required fixed optimizer recipe.  It completed 2,000 steps with zero failed
  pairs and zero rejected updates, but raw final mean/median/P90 MACE was
  0.190075/0.184014/0.243916 px, so it is a failed `<0.1` gate rather than a
  pass.  Artifact SHA256:
  `33f885db0d08cc85c518c03f2adfd5b83e03ed6ff9bdc62daaf4ef72e3cf0df4`;
  checkpoint SHA256:
  `43d71164b68ead7c94b4bb5ed4eeffc8c804420c28241af5844d8d969c82f576`.
- Failure isolation found no guard/window failure and no BF16 floor: the same
  checkpoint measured 0.190075 px with BF16 CNN autocast and 0.195496 px with
  FP32 CNN execution.  A single fixed residual crossed the threshold at step
  440 (0.084624 px), proving that the differentiable path can reach the target
  but not constituting the required 32-sample pass.  A deliberately
  off-protocol tenfold-lower-lr continuation reached 0.077395 px in 20 steps;
  it is diagnosis only because it exceeds the budget and changes the fixed lr.
  Averaging 33 adjacent endpoint weights reached 0.087943 px and isolated the
  issue as constant-lr L1 endpoint oscillation.  Full evidence and all
  non-gate labels are in
  `/home/disk1/MHINet/artifacts/p4_tiny_d8_failure_analysis.json`, SHA256
  `ba54bb1cab083cb2e40aadb2eff1fb07725d6393e723152d19b81b027648bc2b`.
- A fresh D8 run kept B1, accumulation4, lr `1e-3`, no scheduler, uniform L1,
  clip 1, BF16 CNN/FP32 geometry, and only-new-layer training.  It predeclared
  a tiny-only equal-weight parameter readout from step 1,536 and crossed the
  threshold at step 1,728 using 193 in-budget snapshots: mean/median/P90 final
  MACE 0.085699/0.074983/0.157325 px, zero failed pairs and zero rejected
  updates.  The simultaneous raw endpoint was 0.459274 px, so this is recorded
  specifically as an **averaged-readout pass; raw endpoint did not pass**.
  The averaging is disabled in formal training and the diagnostic checkpoint
  is marked non-resumable.  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d8_translation_swa.json`, 208,644
  bytes, SHA256
  `719852b5af01aa446cb8850a0014bc5ae82632e0582fcffc0a3c0193aee6034d`.
  Checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d8_one-pair-residuals_translation_bf16_weight-average-from-1536_seed0.pt`,
  14,482,191 bytes, SHA256
  `dc7360186b688c245909c5b0da01abf491f17234313754a0ea422b0c8cc1af98`.
- TINY-S-D4 used the same registered recipe.  It passed at optimizer step
  1,568 with 33 in-budget averaged snapshots: mean/median/P90 final MACE
  0.069876/0.065841/0.091630 px, zero failed pairs and zero rejected updates.
  The simultaneous raw endpoint was 0.126651 px, so this is again an
  **averaged-readout pass; raw endpoint did not pass**.  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d4_translation_swa.json`, 188,466
  bytes, SHA256
  `b5387f294019e9a3700cda147d04305eec93a770e55a69f78e42c3fea303df13`.
  Checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d4_one-pair-residuals_translation_bf16_weight-average-from-1536_seed0.pt`,
  15,112,741 bytes, SHA256
  `9ff3d1f54fa0ef657907658d63baad70218aaa85489db9b9f5003a120eeb7650`.
- Before starting D2, a fresh real-resource one-step D4 integration exercised
  the newer repeated-pair single-GPU-copy path and both raw and averaged
  evaluations.  All forward/backward/checkpoint branches completed with zero
  rejected updates.  It is deliberately `failed` because it contains only two
  diagnostic residuals and one optimizer step, not because of a runtime error.
  Artifact SHA256:
  `997f739826d1f747d3f210d93e2ffb35ecf63c0d0e5e0b4db50a8ecfd409bbc7`.
- TINY-S-D2 passed at optimizer step 416 using the raw parameter readout,
  before the predeclared averaging window: mean/median/P90 final MACE
  0.095326/0.100922/0.124073 px, zero failed pairs and zero rejected updates.
  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d2_translation_swa.json`, 107,374
  bytes, SHA256
  `78cb079f66d69f13719f9a79627fe90aaa86b0c29cf8cde98999066015e82f73`.
  Checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d2_one-pair-residuals_translation_bf16_weight-average-from-1536_seed0.pt`,
  15,644,987 bytes, SHA256
  `7ea700fb910714bd631f72e7908ba7f3ce1f0c866ac3742b272975aaa45f2c9b`.
- A two-residual, one-step D1 integration then exercised the current grouped
  checkpoint implementation at full 784 resolution.  Forward, backward, raw
  evaluation, averaged evaluation and checkpoint save all completed with
  finite gradients, zero failed pairs and zero rejected updates.  Peak
  allocated/reserved training memory was
  6,697,497,088/7,021,264,896 bytes.  Its `failed` status is expected because
  it is not the 32-residual learnability run.  Artifact SHA256
  `2a3f52d3e395e49ba2ebc53dad253c612b8417a42775a52238d81af32f9cb5e3`.
  TINY-8 has not been run.
- A real-feature signal comparison was added before interpreting the slow D1
  curve.  With the registered half-bound D1 residual, the random-adapter
  nearest-vs-centre correlation margin was only `0.002778` on average, versus
  `0.016658` at D2.  Using the full legal D1 decoder bound increased that D1
  margin to `0.008696`; this is diagnosis only and the active run still uses
  the registered half-bound residual.  Half-bound D1, full-bound D1 and
  half-bound D2 artifact SHA256 values are respectively
  `f8f6e3ebaee6537e213ab544c49d0d1dc67310771e1a4db868f99acded7c1751`,
  `5a6d85493dcd28399008da0efc50cd05b4c0ff118e319f96f65799d92ecc1c67`
  and `72c8c74a108e6f1b548d403b76a595ae99b3f5b41739cd9693afe0a239800a9e`.
- The correlation audit now also emulates the decoder-boundary conversion
  `FP32 -> BF16 -> FP32` without changing production forward precision.  For
  the random 32-channel adapter, half-bound D1 had a mean valid-score
  quantization error of `0.000966`, `30.23%` nearest/centre ties after
  conversion, only `73.12%` retention of originally positive margins,
  `69.77%` margin-sign preservation and `59.16%` argmax preservation.  Full-
  bound D1 improved those values to `16.53%`, `87.13%`, `83.47%` and `60.10%`;
  half-bound D2 was materially stronger at `7.91%`, `94.47%`, `92.09%` and
  `83.47%`.  The raw 256-channel descriptor shows the same scale ordering.
  This makes BF16 loss of small D1 contrast a plausible contributor, not a
  proven sole cause; the separately running FP32 training comparison is the
  causal check.  Updated D1-half/D1-full/D2-half artifact SHA256 values are
  `2166fe976a49b801ff4e581d2d498c4d4206eaa65905ac012334832c51aafff3`,
  `9a0dd15088da2b4c82dff3b78ffa73cd5515f631877fc24e3d3f77a1c19b23a2`
  and `8e9812521051d3c79cb8a13721aebcc2ca07471c7ae3b1b676a20ee2635f579e`.
  Two pure-CPU edge/reference tests cover ties, rank loss, invalid shapes and
  empty valid rows; the complete suite passed 69/69 in 7.794 seconds.
- D1 profiling showed that 614,656 queries create 601 fixed 1,024-query
  sampling chunks.  Correlation now samples the 32-channel target and its
  finite mask in one 33-channel `grid_sample`, and constructs the complete
  candidate grid once inside the same non-reentrant checkpoint before slicing
  it into the unchanged row-major chunks.  No precision, ordering, detach or
  gradient rule changed.  The existing output/gradient/checkpoint reference
  suite passed.  A real two-residual D1 one-step smoke reduced elapsed time
  from `18.1092 s` to `14.9944 s` (17.2%) while peak allocated/reserved memory
  rose from 6,697,497,088/7,021,264,896 to
  7,141,153,280/7,409,238,016 bytes; gradients stayed finite and update
  rejection stayed zero.  The final optimization artifact SHA256 is
  `cd47126d9b3faa97542ddef6187981da10eb64a4a97ddeac494b66314e0e914b`.
  An intermediate one-sampler-only smoke is retained with SHA256
  `92c45486e8a9642691b1abff94ac84c2e9a6bfb6a689af1dc9277573ec11d43b`.
- The registered dense D1 5x5 run used the pre-optimization process and reached
  step 864 before its hosting exec session ended without a Python traceback,
  OOM record, final artifact or checkpoint.  It therefore remains an
  interrupted **non-pass**, not a completed 2,000-step result.  Mean final MACE
  fell from `0.927098 px` at step 32 to `0.474308 px` at step 160, then
  plateaued: the last ten 32-step observations averaged `0.4606961 px` in the
  range `[0.457148, 0.466198]`, with zero rejected updates throughout.  The
  raw log SHA256 is
  `92661c88c1d18d5efe45c9749e20afb52b9cf0380ba4ae1ac3bcf4e6a22fc81f`;
  the preserved observation artifact SHA256 is
  `dbfa8521480c0939e4619f1b6ceca7a066dcae28ebb496d30e4f5a24ccf0fa5d`.
  This is an initial dense full-grid 5x5 failure diagnosis, **not** the formal
  baseline for sparse-query D1 and 3x3-window D1 comparisons: it has no final
  checkpoint/artifact and cannot support a same-checkpoint paired comparison.
  The formal `D1-DENSE-5` reference remains unestablished and must be produced
  only after P4, E00/E01, E02--E05 and main-configuration selection.  Neither
  variant may replace the mainline until matched-budget validation shows its
  accuracy, failure-rate, memory and latency trade-off.
- A dense D1 5x5 **diagnostic**, started at Git `d77b487` after the correlation
  optimization, expanded the controlled residual to the full legal D1 decoder
  bound and completed its declared 256-step budget.  Exact command:
  `CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.tiny_overfit --runtime configs/runtime_paths.server.json --experiments D1 --sample-protocol one_pair_residuals --residual-profile translation --precision bf16 --sample-count 32 --seed 0 --max-steps 256 --eval-interval 32 --threshold-mace-px 0.1 --residual-bound-fraction 1.0 --output artifacts/p4_tiny_s_d1_full_bound_diagnostic_256.json --overwrite`.
  Mean final MACE moved from `1.858603 px` at step 0 to `1.853464`,
  `0.989137`, `0.955050`, `0.929901`, `0.946010`, `0.921074`,
  `0.920057`, and `0.930858 px` at steps 32--256.  The final
  median/P90/max were `0.822484/1.731238/1.863900 px`; H1 and H2 means were
  `0.931836` and `0.930858 px`.  All 32 conditions remained finite with zero
  rejected updates, but the best observed mean was still far above `0.1 px`.
  Peak allocated/reserved memory was
  `7,138,167,296/7,411,335,168` bytes and measured training time was
  `3,241.151 s`.  Artifact SHA256:
  `920a5386fad222c5a648cfadca49f7d107d388109ae468eeac78d3f4d8164353`;
  raw stderr SHA256:
  `9ccb831427b6c37dd5a0b838755770eeee762213350b0b2764d1fad953765cab`;
  16,250,961-byte checkpoint
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d1_one-pair-residuals_translation_bf16_raw_seed0.pt`
  SHA256 `b33f3b148b58e96f7d5af4bfde1681f56c6a2c1be0dda803a51bd6e97b8dadd4`.
  This is a completed short diagnostic failure, not the registered 2,000-step
  averaged D1 gate.  Increasing residual amplitude alone therefore does not
  resolve D1 learnability and does not justify a blind long continuation.
- The independent-run merger was exercised with only D8.  It failed closed
  and listed D4/D2/D1/TINY-8 as missing; formal E01 then rejected that artifact
  before model construction or any optimizer step.  Partial artifact SHA256:
  `0680e3cb6015fc17807aede502ed5b285617a45f514e70f31e8b9fa108a257e4`.
- The gate validator now checks each experiment's two-updates-per-scale
  schedule, B1/accumulation-4 AdamW recipe, lr/weight decay/clip/scheduler,
  BF16 mode, translation residual protocol, finite gradients, controlled-H0
  provenance and seed.  It no longer trusts a reported checkpoint hash: it
  verifies the on-disk path/bytes/SHA256, restricted-loads the payload, binds
  progress and metadata to the experiment, and checks raw/averaged readout
  ownership.  It also recomputes mean/median/P90/max, failures and rejections
  from all 32 per-condition rows, checks history consistency and requires the
  five independent artifacts to share data/resource evidence.  D8/D4 predate
  two readout flags; only their two independently reloaded immutable checkpoint
  hashes receive an explicit legacy migration exception.  A fresh strict
  partial merge accepts D8/D4/D2 and fails only for the legitimately missing
  D1 and TINY-8 experiments.  Partial artifact SHA256:
  `99ff441f1f00b005b0c301dbf899161c9ccdaf9dcd3769d86fb058bcc8901f3b`.
- Long TINY-S/TINY-8 runs now have an independent atomic progress checkpoint
  path and strict `--resume-progress` entrypoint.  The progress role always
  stores the raw iterator and matching AdamW state at a completed optimizer
  boundary; it also stores the exact data cursor, Python/NumPy/Torch/CUDA RNG,
  compact metric history, original step-0 endpoint, accumulated timing/peak
  memory and the FP32 equal-weight parameter averager.  Averager tensors live
  only in the binary `auxiliary_state` and are deliberately omitted from the
  JSON-facing save report.  A restricted `weights_only=True` preflight runs
  before live state or an existing partial JSON is changed.  Resume fails on a
  mismatch in experiment/scales, seed, budget, evaluation interval, threshold,
  sample protocol/order, residual profile/bound, precision, averaging window,
  optimizer recipe, target size, architecture, manifest, runtime config or
  DINO/GHIM/pyramid checkpoint path/size/SHA256.  Heartbeats default to every
  eight optimizer steps and perform no extra forward pass; an evaluation step
  and heartbeat are merged into one save.  The same visible-CUDA-device
  topology is required because all visible CUDA RNG streams are restored.
- The final evidence checkpoint filename now includes residual-bound fraction
  and optimizer budget, preventing a short diagnostic from overwriting a
  registered 2,000-step result.  A resumed checkpoint that had already passed
  at its current evaluated step finalizes without another optimizer update.
  The redundant per-parameter CUDA finite-gradient synchronization was removed;
  `clip_grad_norm_(error_if_nonfinite=True)` remains the single fail-closed
  finite-norm check before every optimizer step.
- A CPU interruption-equivalence regression stops immediately after the
  step-1 heartbeat, reloads the restricted progress payload, and reaches step 3
  with bit-exact model and AdamW tensor states relative to an uninterrupted
  run, including the parameter-average window.  Name mismatch and non-finite
  averager state are rejected.  Command:
  `CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -v`.
  Result: all 67 tests passed in 2.741 seconds.  This validates checkpoint
  mechanics only; it is not a D1/TINY-8 learnability or model-validation pass.
- A real CUDA smoke at Git `1ccedfae8574cf01eb398ad00b12af8e4f7ef610`
  then ran D8 for one optimizer step on two controlled conditions, wrote a
  14,488,399-byte raw progress checkpoint, and resumed it with zero additional
  optimizer steps.  The restored report confirms model, AdamW, Torch CPU RNG
  and CUDA RNG restoration; history remained `[0, 1]`, MACE remained exactly
  `11.89400863647461 -> 2.285935878753662 px`, and failures/rejections remained
  zero.  Both processes correctly returned status `failed` because two
  conditions and one step are not the registered gate.  The progress checkpoint
  is `/home/disk1/MHINet/outputs/tiny_overfit/resume_smoke_1ccedfa_progress.pt`,
  SHA256 `c7419803b9a89baa48cb258960de2c57594965677240bc87067f5afcef9c6be2`;
  the post-resume final checkpoint SHA256 is
  `672709c64d459f2e177a75f500fbc6f64f8ca3c130637cd9de03fe0b6d7ad92e`.
  Exact commands, environment versions, output paths, resource hashes and
  expected-failure scope are in
  `/home/disk1/MHINet/artifacts/p4_tiny_resume_cuda_smoke.json`, SHA256
  `f588599ffaef020a8f6a4f0ab21d5932bc35612baf0619d3af49b7ecdd371058`.
- Added the read-only `tiny-checkpoint-audit` entrypoint for diagnosing a
  serialized raw/averaged endpoint without resuming training.  It snapshots a
  possibly live atomic progress file before restricted preflight, rejects
  formal/unknown checkpoint roles and protocol conflicts, requires explicit
  supplements for missing legacy seed/bound fields, reconstructs the exact
  train-only controlled conditions, and strictly loads only the MHIR iterator
  with `restore_rng=False` and no optimizer.  It records H0, every H, both
  per-scale decoder deltas, support/condition/solve/saturation fields and
  canonical serialized/load/post-forward state hashes.  Its status is
  deliberately `diagnostic_complete`, never `passed`.
- Tiny endpoint serialization now restores the iterator's original train/eval
  mode even when projection diagnosis raises, records per-update corner
  residuals and emits strict JSON (`null` for non-finite diagnostics rather
  than non-standard `Infinity`/`NaN`).  Command:
  `/root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -v`.
  Result: all 77 tests passed in 1.576 seconds, including direct rejected-update,
  invalid-H0 restoration, protocol-conflict, legacy-supplement and exact-state-
  digest regressions.  This is checkpoint/diagnostic engineering only; no new
  D1 learnability or validation result is claimed.
- Checkpoint delta summaries now keep x/y coordinates separate and report the
  fraction of controlled conditions improved by each update, preventing an
  easy horizontal correction from hiding a vertical failure in one aggregate.
  The real-correlation audit accepts an explicit `--condition-index` so the
  exact horizontal- or vertical-dominant H0 conditions from tiny-overfit can be
  reproduced.  Full-suite command unchanged; all 79 tests passed in 1.824
  seconds.  No production forward, loss, precision or training rule changed.
- Added the diagnostic-only `--condition-index-offset` to tiny-overfit so a
  vertical-dominant condition can be trained independently instead of silently
  reusing condition 0.  A nonzero offset is bound into progress/final metadata
  and the final checkpoint filename; the registered default zero deliberately
  omits the new signature key, preserving exact resume compatibility with the
  already-running jobs.  Checkpoint audit reconstructs the recorded offset.
  The cache description now reports its actual sample count instead of always
  saying 32.  Full suite: 80/80 passed in 1.796 seconds.  This option is failure
  isolation only and cannot satisfy the 32-condition P4 gate.
- The completed dense-D1 BF16 step-256 checkpoint was reloaded read-only at Git
  `018fb0a` using:
  `CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli tiny-checkpoint-audit --runtime configs/runtime_paths.server.json --checkpoint outputs/tiny_overfit/tiny-s-d1_one-pair-residuals_translation_bf16_raw_seed0.pt --experiment D1 --sample-count 32 --sample-protocol one_pair_residuals --residual-profile translation --precision bf16 --seed 0 --residual-bound-fraction 1.0 --output artifacts/p4_tiny_s_d1_full_bound_bf16_checkpoint_audit.json --overwrite`.
  The 16,250,961-byte snapshot SHA256 is the expected
  `b33f3b148b58e96f7d5af4bfde1681f56c6a2c1be0dda803a51bd6e97b8dadd4`;
  canonical serialized, loaded and post-forward iterator-state hashes all equal
  `b0ae7b605211a74ebeaf64eb73d4f0f316a0d92c69173f2f878b0fa889c5a419`.
  No optimizer/RNG/backward/step was used, the source inode did not change while
  snapshotting, and DINO/MVT/VGG/CGMDP were each evaluated once for the one
  exact pair.  Artifact: 178,948 bytes, SHA256
  `3c9fde2cc1141dcb68afbb64ac7211c24b4ddafa6137d877622b7e6f97b8569a`;
  stderr was empty.  This legacy checkpoint lacks embedded resource hashes and
  seed/bound, so the artifact explicitly marks those two CLI supplements and
  does not claim strict historical resource binding.
- That audit reproduces H0/H1/H2 mean MACE
  `1.858605 -> 0.931836 -> 0.930858 px`, with no failure/rejection or saturation
  and well-conditioned DLT (`condition_number` mean about `3.438`).  It exposes
  a directional failure: update 1 reduces mean absolute x residual from
  `1.432011` to `0.083152 px`, but y changes from `0.915114` to
  `0.920478 px`; update 2 leaves x/y at `0.120899/0.914302 px` and improves only
  10/32 conditions.  Thus guards, saturation and a hidden state mutation are
  ruled out for this checkpoint; the decoder learned mostly horizontal
  correction and its shared second application added no aggregate benefit.
- Real D1 correlation was then audited on the exact x-dominant condition 0 and
  y-dominant conditions 2/14 with full legal residual bound.  Commands used
  `python -m mhinet.cli real-correlation-audit --runtime
  configs/runtime_paths.server.json --pair-index 0 --scale 1 --seed 0
  --condition-index {0,2,14} --residual-bound-fraction 1.0` on GPU2.  Artifact
  SHA256 values are respectively
  `7d4cdc3ee38e4ba2732d5fe63bf63a63b1a1703eec7e35f1c4fcc7fba530950c`,
  `9a2ad84fd78a764ce95bfa8a6cf4e8c5342fb0f3a78b27ca4c3f868e0270a69d`
  and `08d29c365b8160896303533ce90a35c4b6f5111b869a39875bfe23876d26c600`;
  all stderr files were empty.  Random-adapter nearest-candidate top-1 rates
  were `0.2248/0.2946/0.2392`, while raw-descriptor rates were
  `0.4071/0.5013/0.4521`; condition 2's y-positive correlation margin
  (`0.01221`) was stronger than condition 0's x-dominant margin (`0.00870`).
  Vertical evidence is therefore present before the decoder; this narrows, but
  does not yet prove, the failure to learned representation/optimization or
  multi-condition interference.
- Two condition-0 one-sample D1 diagnostics launched at Git `e39d187` completed
  128 raw steps in BF16 and FP32.  Exact commands matched the full-bound D1
  recipe with `--sample-count 1 --max-steps 128 --eval-interval 16`, precision
  `bf16` or `fp32`, and their separately recorded progress/output paths.  BF16
  moved `1.418984 -> 0.065078 px` (best sampled endpoint `0.012468`); FP32 moved
  `1.418984 -> 0.047024 px` (best `0.045429`).  Both had zero failure/rejection.
  Their top-level status correctly remains `failed`, because one condition can
  never meet the registered 32-condition criterion.  Artifact SHA256 values:
  `0126f91ca908226dd34df7dcea86fb08223dd28102ee1079566a98cc9c0fe8fa`
  and `9a0c5ab3934f9ae96abc6854ab2cb55b2a0826fddf7a03c3c4b9b0998383e00a`.
  Final checkpoint SHA256 values:
  `e2770fbc012d7d186940385f8952bc7b14e3ae3033dd3cfe4b651947e54cca2d`
  and `a01ab71d2745710beaa5e7c2875ca3d855f60a6b38558e74427a434d4210b845`.
  BF16/FP32 training elapsed was `1553.745/1498.777 s`; peak allocated/reserved
  memory was `7,138,167,296/7,411,335,168` and
  `7,921,545,728/8,204,058,624` bytes.  This proves only that the x-dominant
  condition is individually learnable in either precision, not that D1 or P4
  passes.  Separate y-positive/y-negative one-condition jobs are in progress.
- The full 32-condition D1 FP32 diagnostic launched from Git `e39d187`
  completed all 256 optimizer steps.  Exact command:
  `CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m
  mhinet.tiny_overfit --runtime configs/runtime_paths.server.json
  --experiments D1 --sample-protocol one_pair_residuals --residual-profile
  translation --precision fp32 --sample-count 32 --seed 0 --max-steps 256
  --eval-interval 32 --threshold-mace-px 0.1 --residual-bound-fraction 1.0
  --heartbeat-interval 8 --progress-checkpoint
  outputs/tiny_overfit/d1_full_bound_fp32_256_progress.pt --output
  artifacts/p4_tiny_s_d1_full_bound_fp32_diagnostic_256.json --overwrite`.
  Its H0/H1/H2 mean MACE was `1.858605 -> 0.948720 -> 0.929889 px`;
  the best sampled endpoint was `0.918012 px` at step 160.  It had zero
  failures/rejections but missed the registered `<0.1 px` threshold.  Training
  took `3310.370 s`, with peak allocated/reserved CUDA memory
  `7,921,545,728/8,204,058,624` bytes.  Artifact: 101,431 bytes, SHA256
  `8345c778c5a8df086b62131dd1f2d2e6b6db49a7a16214347359457a787fb405`;
  final checkpoint: 16,252,945 bytes, SHA256
  `6b5820e86c370cac9c8320ce1232230186a41a8bfd8d8b6c422ac66d10be729d`.
  FP32 therefore does not explain or fix the 32-condition D1 plateau.
- The condition-2 y-positive and condition-14 y-negative one-sample BF16
  diagnostics launched from Git `b50ef2e` also completed 128 optimizer steps.
  They used the same D1 full-bound command as the earlier one-condition runs,
  with `--condition-index-offset 2` on GPU2 or
  `--condition-index-offset 14` on GPU3 and their separately recorded
  progress/output paths.  Condition 2 moved
  `1.602059 -> 0.810867 -> 0.023410 px`; condition 14 moved
  `1.404833 -> 0.699856 -> 0.011812 px`.  Both had zero failure/rejection.
  The top-level statuses correctly remain `failed` because each has only one
  sample and cannot satisfy the 32-condition gate.  Artifact bytes/SHA256 are
  `52,267`/`b43c45918818130b08db47253751a2d18fdd12c39b81033fd844bc53564e3d23`
  and
  `52,112`/`7bafe83bce9f426d04c0ba59ad2abf8239e08543447d507173dc277db8d7a152`.
  Final-checkpoint SHA256 values are
  `5bf6ea1726f60340784165379225a971d857e40eb796aaf949dfc32f0b501424`
  and
  `aad9fea00bee55000049d696f0694995e6cad9081512d3e99d3f3e54e8e81fdb`.
  Training elapsed was `1486.101/1480.555 s`; each run used
  `7,138,167,296/7,411,335,168` peak allocated/reserved bytes.  Together with
  condition 0, these results show that both horizontal and vertical residuals
  are individually learnable; the unresolved failure is joint coverage or
  multi-condition optimization, not a demonstrated missing vertical signal.
- A registered 32-condition BF16/SWA D1 run with a 2,000-step budget started
  afterward but did **not** finish.  The process is no longer present and its
  stderr ends after a normal heartbeat without a Python exception.  The last
  atomic progress checkpoint is a restricted-loadable `tiny_progress` at step
  184 (19,389,293 bytes, SHA256
  `b046f763ea844087ec3803ecce24a347e11de1056cdcc27648b65b179f17f38e`);
  the last completed evaluation was step 160 at `0.925703 px`.  SWA had not
  begun because its configured start is step 1536.  The partial running JSON is
  deliberately not committed as final evidence.  The checkpoint records the
  exact 2,000-step signature and resource identities, so it can be considered
  for strict resume after the unexplained external termination is accounted
  for; it is not a D1 pass.

CUDA warns that `grid_sampler_2d_backward_cuda` and
`adaptive_avg_pool2d_backward_cuda` have no deterministic implementation.
Seeds, sample IDs and RNG states are recorded, but current CUDA results should
not be described as bitwise deterministic.

## P5/P6 status

Not started.  E00/E01 and later experiments remain blocked by the incomplete
TINY-S/TINY-8 learnability gate.  No test item has been evaluated and no model
validation result has been claimed.

The D1 efficiency work is now explicitly ordered after E00/E01, the equal-start
E02--E05 feature-group comparison and validation-only main-configuration
selection.  The earlier step-864 interrupted dense observation was corrected
from “baseline” to an initial failure diagnosis because it has neither a final
artifact nor a checkpoint for paired evaluation.  The machine-readable sparse-
query/3x3-window registration is a deliberately non-executable draft at
`/home/disk1/MHINet/configs/d1_efficiency_ablation_v1.2.json`, 96 lines, SHA256
`d6cce6aa4ce41caeceab6ee61297c77f9af2ccc5fd6a205862aa017b33d7b2f9`.
Known facts (dense 5x5, D2-guided eligibility, 3x3's theoretical 64% candidate
reduction, grouped validation and seed-0 then seed-0/1/2 policy) are fixed;
unknown selector thresholds, exact validation IDs, runtime repetitions and
accuracy/speed/memory decision margins remain `null` blockers rather than
invented defaults.  No D1 efficiency implementation or result is claimed.

P5 entrypoint preparation has begun without executing E00/E01.  The evaluator
now emits both explicitly named all-finite-geometry diagnostics and
conditional-valid H0/every-update/final summaries, so a finite fallback from a
failed initializer cannot masquerade as a valid estimate.  A rejected proposal
keeps its guarded retained H in the trajectory denominator while rejection is
reported separately.  Per-pair JSONL and flattened CSV now retain every state,
accepted/reason code and name, support count, condition number, solve status,
corner-update magnitude and saturation.  Window recall remains explicitly
unavailable rather than fabricated because the current forward contract does
not retain GT-to-window membership.  Five CPU evaluation regression tests
passed; this is output-contract engineering, not P5 validation evidence.

## 2026-09-09 — MCNet-style MHIR rewrite and D2 mainline cutoff

This section supersedes the **current applicability** of the earlier P3/P4,
D1, TINY-8, parameter-count, and eight-update records; it does not erase their
historical provenance. No model-validation claim is made here.

### User decision and active architecture

- Public module names remain GHIM, CGMDP, and MHIR. The active CGMDP path is
  `F_MVT -> D16 -> D8 -> D4 -> D2`; it stops the cumulative decoder before
  scale `1`. The active MHIR schedule is `(8,8,4,4,2,2)`, producing
  `H0 -> H1 -> ... -> H6 = H_final`.
- D1 was not physically deleted. Its adapter, 25-channel local correlation,
  nine-block decoder, and explicit diagnostic entry remain registered. In all
  mainline training profiles its parameters are frozen and absent from the
  optimizer; a normal forward neither constructs `D1` nor calls D1 adapter,
  correlation, or decoder.
- Active architecture config:
  `/home/disk1/MHINet/configs/mhinet_mcnet_v1.2.json`, 2026-09-09 SHA256
  `92095ffe16a5e650bab641df3fd17e16a0bf1f34ab47c82c59f37faa43ae092a`.
  The original handoff package was deliberately preserved instead of edited.
  Verification command:

  ```bash
  /root/miniconda3/envs/loma-repro/bin/python \
    MHINet_server_handoff_v1.2/verify_package.py \
    MHINet_server_handoff_v1.2
  ```

  Result: 31 files checked, zero errors; original architecture-config SHA256
  remains `4a859f413ee4be59acb1dc6c0cf189e80b8f4e7b3f54dc9f117acd1ac5cfe128`.

### MCNet source pin and dimension adaptation

- Structural reference: `https://github.com/zjuzhk/MCNet.git`, fixed commit
  `cc03479689b3cf40f0c384954f338b434765c155`. On 2026-09-09,
  `git ls-remote` resolved both remote `HEAD` and `refs/heads/master` to this
  commit. Audited reference files were `update.py`, `network.py`, and the
  four-point geometry utilities.
- Each decoder now consumes correlation only. It applies
  `Conv1x1(K,64,bias=True)`, repeated
  `Conv3x3(s1,p1,bias=True) -> GroupNorm(8) -> ReLU -> MaxPool2(s2)`, then
  `Conv1x1(64,2,bias=True)` directly on a `2x2` spatial corner grid. BCHW is
  permuted to BHWC and flattened row-major as TL/TR/BL/BR, with x/y last.
- MCNet's power-of-two inputs do not directly cover the current grids. Pooling
  therefore uses `ceil_mode=True`: D8 uses 6 blocks
  (`98->49->25->13->7->4->2`), D4 uses 7, D2 uses 8, and retained D1 uses 9.
  There is no adaptive pool or MLP in the production path.
- Search windows remain the previously registered 9x9, 9x9, 7x7, and dormant
  5x5 windows (`K=81/81/49/25`). This avoids changing decoder topology and D1
  window policy in the same experiment. Candidate-valid masks remain outside
  the decoder and are used only for support/geometry guards.
- Intentional MHINet differences from official MCNet are explicit: H0 rather
  than identity initialization, normalized FP32 cumulative corner/H state,
  zero-initialized output projections, per-scale tanh bounds
  `32/16/6/(2 dormant) px`, direct H-guided sampling with invalid-zero masks,
  and pre-screened differentiable DLT. A rejected update retains the preceding
  H/T; no update detaches H or T.
- Registered new parameters, including inactive D1, are `1,176,712`. The
  mainline D8/D4/D2 trainable-new-parameter count is `833,222`. Counts include
  the four retained adapters and scale-specific decoders; they do not imply
  D1 activation/FLOP execution.

### Runtime, checkpoint, and gate changes

- `SharedFeatureProvider` defaults to `(8,4,2)` and breaks cumulative decoding
  after scale `"2"`. It records `dedode_steps=4` and `dedode_scale1=0` for the
  mainline; a unit test separately proves that explicit `(8,4,2,1)` still
  reaches scale `"1"`.
- Formal training configs reject D1-containing active schedules. E01 now uses
  `(8,4,2)` twice. Evaluation and profiling default to the same six updates.
- The required tiny gate is now TINY-S-D8, TINY-S-D4, TINY-S-D2, and TINY-6;
  its identifier is `P4_TINY_S_TINY_6`. D1/TINY-8 remain optional diagnostics
  and cannot satisfy or be merged into the current gate.
- MHINet-owned checkpoint format was bumped from version 1 to version 2.
  Version-1 decoder/optimizer/SWA state is rejected. Formal resume/evaluation
  validates architecture SHA256 and `mhir_revision` before mutating model or
  optimizer state. New tiny filenames contain an architecture-hash prefix and
  progress signatures bind architecture, data, runtime, weights, and protocol.
- In particular, the old step-184 D1 progress checkpoint with SHA256
  `b046f763ea844087ec3803ecce24a347e11de1056cdcc27648b65b179f17f38e`
  is not resumable under this architecture. The concurrently written
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d1_translation_swa.json` remains an
  unstaged legacy heartbeat and is not current evidence.

### Verification completed so far

Targeted command:

```bash
/root/miniconda3/envs/loma-repro/bin/python -m unittest \
  tests.test_model tests.test_feature_provider tests.test_modules \
  tests.test_iterator tests.test_tiny_overfit tests.test_tiny_gate \
  tests.test_train tests.test_checkpointing tests.test_config \
  tests.test_losses_metrics -v
```

Result: 51/51 tests passed. This covers exact MCNet block topology and
parameter counts, 784-derived meta shapes, TL/TR/BL/BR mapping, output
zero-initialization, second-step upstream gradient, six-round no-op/update
state, correlation recomputation, no-detach final gradients, default D1
non-execution, explicit D1 availability, CGMDP scale-1 early stop, checkpoint
metadata fail-closed behavior, TINY-6 protocol, and six-update loss/metrics.

Full CPU regression command:

```bash
/root/miniconda3/envs/loma-repro/bin/python -m compileall -q mhinet tests
/root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -v
```

Result: compileall succeeded and 88/88 tests passed in 1.440 seconds.

Real-resource P3 command (physical GPU 1 exposed as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli zero-init-smoke \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p3_zero_init_smoke_mcnet_d2.json --overwrite
```

Result: `passed`. Six of six updates were accepted; schedule was
`8,8,4,4,2,2`; all registered output-projection weights/biases and all deltas
were exactly zero; DINO/MVT/VGG ran once, CGMDP executed scales 16/8/4/2 only
(`dedode_steps=4`, `dedode_scale1=0`). H0-to-H6 maximum/mean corner difference
was `6.103515625e-05/2.986308027175255e-05 px`. Single unwarmed forward was
`742.833 ms` and peak allocated memory was `2,061,168,128` bytes; neither is a
production benchmark. Artifact: 4,486 bytes, SHA256
`29e90988578e1a1cb9ad32a75a0c0146a0282605c50da06c9806807bb53416be`.

Real-resource P4 gradient command (physical GPU 2 exposed as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli gradient-audit \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p4_gradient_audit_mcnet_d2.json --overwrite
```

Result: `passed`. At step 0, all D8/D4/D2 output projections had nonzero finite
gradient and all upstream decoder/adapter gradients were exactly zero as
expected. At step 1, every active upstream decoder and adapter gradient was
finite and nonzero. A final-H6-only loss reached all three earlier decoders.
The MVT-to-GHIM and MVT-to-CGMDP routes both had nonzero finite MVT gradients;
the CGMDP route also reached VGG and the active DeDoDe decoder. DINO and GHIM
head parameter gradients remained absent, and registered D1 parameters stayed
frozen with no gradients. Peak allocated memory was `3,103,597,056` bytes for
the cached six-round heads check and `5,821,080,576` bytes for the separate
joint-branch diagnostics. The artifact also embeds the active architecture,
provider, and checkpoint provenance. Artifact: 10,308 bytes, SHA256
`cca64b4f3573f9c40a70b980f4221b874fa6fe34c3dad06445ef5bf9c3be06fd`.

Real-resource P4 two-step heads-only profile command (physical GPU 1 exposed
as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli profile \
  --runtime configs/runtime_paths.server.json --profile heads \
  --optimizer-steps 2 \
  --output artifacts/p4_profile_heads_mcnet_d2.json --overwrite
```

Result: `passed` on an RTX 4090 at the real `784x784` input size. Peak
allocated/reserved memory was `3,999,755,264/4,882,169,856` bytes. Unwarmed
step-0 forward/backward time was `0.558349/0.558144 s`; step 1 was
`0.257731/0.434172 s`. Both steps accepted all six updates and recorded one
DINO, MVT, VGG, and cumulative DeDoDe call, four decoder steps through D2,
and zero scale-1 steps. At zero initialization, step 0 had 372 nonzero new-head
gradient elements; after the first update, step 1 had 798,968, consistent with
the expected delayed upstream-gradient activation. DINO and the frozen GHIM
head had no gradients. Artifact: 8,582 bytes, SHA256
`c4dcb825c5d01057d3c9a116eee837ff411e353ebdd25727f73856c1832f34a1`.

Real-resource P4 two-step joint profile command (physical GPU 2 exposed as
`cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli profile \
  --runtime configs/runtime_paths.server.json --profile joint \
  --optimizer-steps 2 \
  --output artifacts/p4_profile_joint_mcnet_d2.json --overwrite
```

Result: `passed` on an RTX 4090. Peak allocated/reserved memory was
`10,934,943,232/11,836,325,888` bytes. Unwarmed step-0 forward/backward time
was `0.569247/0.920599 s`; step 1 was `0.265485/0.889338 s`. Both steps used
the six-update `8,8,4,4,2,2` schedule, kept H/proposal tensors differentiable,
ran shared DINO/MVT/VGG once, stopped the cumulative decoder after four steps,
and did not execute D1. Total active trainable parameters were `107,061,671`.
The expected zero-head boundary was visible at step 0: DeDoDe and VGG gradient
tensors existed but had zero norm, while MVT already received gradient through
H0. At step 1, DeDoDe/VGG/MVT gradient norms were respectively
`0.361962/0.489198/34.208104`, all finite and nonzero; DINO and the frozen
GHIM head still had no gradients. Artifact: 8,707 bytes, SHA256
`92a67d98bee23f1de71e8c3d468c6f00442737b65e8357d76b8ff92b1a391fe4`.

Still pending at that checkpoint for the new architecture: checkpoint-v2 exact-boundary resume
smoke, D8/D4/D2/TINY-6 tiny-overfit, E00, and E01. Earlier artifacts do not
fill these gaps. E00/E01 remain blocked until the new P3/P4 and TINY-6 gate
pass.

## 2026-09-10：执行正式训练前完整检查（进行中）

停止边界：完成 P0–P4 审查及 tiny gate 后停止，不执行 E00/E01，也不执行解冻或结构消融。

发现并修复两个入口问题：alignment 的样本来源由 test 改为 train 并显式记录 split/test_used；CGMDP 的 scale1 decoder 虽然已被 forward 截断，但联合 profile 仍将其参数标记为可训练，现强制冻结并保持 eval，同时保留 state_dict。新增回归测试确认 scale1 参数不进入 optimizer，重复 train/eval 不会重新激活。89/89测试通过，原始设计包31项校验通过。

| 检查 | 本次结果 | 证据 |
| --- | --- | --- |
| P0真实资源与环境 | passed | `artifacts/p0_preflight_mcnet_d2.json` |
| P1 train样本旧输出对齐 | H0/D8/D2逐元素一致；D1调用0 | `artifacts/p1_alignment_mcnet_d2_train.json` |
| P2几何和相关性reference | 20/20 passed | `artifacts/p2_reference_mcnet_d2.json` |
| joint双步复测 | passed；活跃参数107,039,206；峰值10,934,943,232 bytes | `artifacts/p4_profile_joint_mcnet_d2_frozen_d1.json` |

上述四项SHA256依次为：

- `b867ee4753f890ee138fdef037e4908d722877e3cb7ff03b88b9d2978ec47659`
- `633faf7a7ffa98053999578f2a27567a9accf87a6c5e48bb6d76c52ca8a50977`
- `32e4861895df398a31951eece97eb1e382de4a7e238b041c26e03ebaa9c787be`
- `773f79164c3fa95469adbb585f82d5523d7f6649e83a947a57e0ba868e4ded67`

真实Python为 `/root/miniconda3/envs/loma-repro/bin/python`，3.10.20；PyTorch2.11.0+cu128、CUDA12.8、cuDNN91900、torchvision0.26.0、NumPy2.2.6、GPU驱动565.57.01。父shell继承的CONDA_PREFIX为base，preflight新增 sys.executable/sys.prefix 以消除歧义。可选pytest/openpyxl不可导入，timm导入触发NumPy旧API错误；当前已执行入口使用unittest和现有LoMa加载路径正常运行，不据此声称全部可选依赖可用。

复测命令（工作目录 `/home/disk1/MHINet`）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli preflight --runtime configs/runtime_paths.server.json --output artifacts/p0_preflight_mcnet_d2.json --overwrite
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli alignment --runtime configs/runtime_paths.server.json --output artifacts/p1_alignment_mcnet_d2_train.json
/root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli geometry-corr-check --output artifacts/p2_reference_mcnet_d2.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli profile --runtime configs/runtime_paths.server.json --profile joint --optimizer-steps 2 --output artifacts/p4_profile_joint_mcnet_d2_frozen_d1.json
```

四项tiny共用参数：`--sample-protocol one_pair_residuals --residual-profile translation --residual-bound-fraction 0.5 --precision bf16 --sample-count 32 --seed 0 --max-steps 2000 --eval-interval 32 --threshold-mace-px 0.1 --weight-average-start-step 1536 --heartbeat-interval 64`。均为 `python -u -m mhinet.cli tiny-overfit --runtime configs/runtime_paths.server.json`，不改变L1或加入额外损失。

| experiments | 物理GPU | progress路径（outputs/tiny_overfit下） | artifact（artifacts下） |
| --- | ---: | --- | --- |
| D8 | 2 | mcnet_d8_full_progress.pt | p4_tiny_s_d8_mcnet.json |
| D4 | 3 | mcnet_d4_full_progress.pt | p4_tiny_s_d4_mcnet.json |
| D2 | 3 | mcnet_d2_full_progress.pt | p4_tiny_s_d2_mcnet.json |
| TINY-6 | 2 | mcnet_6_full_progress.pt | p4_tiny_6_mcnet.json |

D8/D4初次启动未限制线程，观察到各162线程；随后SIGINT结束这两个已确认身份的进程，从各自原始边界以 `--resume-progress` 恢复，设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。D2/TINY-6自启动即使用相同线程限制。此调整为资源配置，未变更预算/优化器/样本顺序，不宣称跨线程设置逐位相同。源码冻结修复不改变heads tiny的执行图或权重。

`scripts/summarize_pretraining.py` 从现有artifact生成 `docs/pretraining_mcnet_d2_summary.md` 与JSON索引，包含每轮误差、原始/平均读出、显存、缓存前向时间和checkpoint身份；缺失或运行中项不会被推定通过。运行时序及最终状态以原始artifact和progress为准。

### D8完整预算与checkpoint重载（已完成）

1664步达到登记门槛并提前停止。1536–1664共129次参数平均后的H0/H1/H2为14.868890/0.291660/0.098226 px，失败0/32、拒绝0/64；原始终点H2为0.290066 px，单独保留。训练循环累计617.205 s，峰值571,810,816 bytes；共享GPU下缓存前向56.555 ms/样本，不能视作独占端到端基准。结果表见 `docs/pretraining_mcnet_d2_summary.md`。

原始artifact SHA256：`7c2f1fae0cf576c7b753fce0c2875c88746101b8b78d3723b85127adc386f294`。平均checkpoint为 `outputs/tiny_overfit/tiny-s-d8_one-pair-residuals_translation_bf16_weight-average-from-1536_arch-92095ffe16a5_bound-0p5_budget-2000_seed0.pt`，SHA256 `0da998f748cb901d0674e2767931603bbff2987e38aea08cc9591a460c39a15b`，不能用于优化器续训。

独立重载命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli tiny-checkpoint-audit --runtime configs/runtime_paths.server.json --checkpoint outputs/tiny_overfit/tiny-s-d8_one-pair-residuals_translation_bf16_weight-average-from-1536_arch-92095ffe16a5_bound-0p5_budget-2000_seed0.pt --experiment D8 --sample-count 32 --sample-protocol one_pair_residuals --residual-profile translation --precision bf16 --seed 0 --residual-bound-fraction 0.5 --output artifacts/p4_tiny_s_d8_mcnet_checkpoint_audit.json
```

重载资源签名严格匹配，序列化/载入/前向后权重哈希一致，重新计算的轨迹与原始artifact一致。重载artifact SHA256：`3a96402d0273aa835d8ebcd1299428d10ae18d0ea08846640589dbda5873cccf`。

`scripts/analyze_tiny_conditions.py` 按H0平均残差的主轴与符号分组，生成表格及JSON。D8平均读出 x+/x-/y+/y- 的最终误差分别为0.090299/0.087680/0.129368/0.120216 px。总体均值门槛通过并不表示每方向或每样本均低于0.1 px。饱和比例和拒绝更新均为0。32步probe与完整结果均保存对应分方向证据；其余三个完整实验仍在运行，合并gate尚未通过。

## 2026-09-10：补齐续训对照与 D8 短程结果

新增 `docs/results_mcnet_d2.md`，用表格与文字同时记录六轮资源、checkpoint-v2 对照和 D8 32步诊断。复核此前保存的两条运行记录及最终权重：恢复边界、RNG、数据游标正确，但 adapter 的119个权重元素存在最大1.9595e-6的差异，不能声明逐位一致。详情、原始目录和 checkpoint SHA256 均见结果表。

此前续训命令（工作目录 `/home/disk1/MHINet`，物理 GPU1）：

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli train --runtime configs/runtime_paths.server.json --config configs/train_minimal_smoke.json --output-dir outputs/mhinet_resume_smoke_mcnet_d2_v2 --stop-after-optimizer-step 1 --overwrite
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli resume --runtime configs/runtime_paths.server.json --config configs/train_minimal_smoke.json --output-dir outputs/mhinet_resume_smoke_mcnet_d2_v2 --resume outputs/mhinet_resume_smoke_mcnet_d2_v2/checkpoints/step_0000001.pt
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli train --runtime configs/runtime_paths.server.json --config configs/train_minimal_smoke.json --output-dir outputs/mhinet_uninterrupted_smoke_mcnet_d2_v2 --overwrite
```

本次 D8 命令（物理 GPU2，独立新输出路径）：

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli tiny-overfit --runtime configs/runtime_paths.server.json --experiments D8 --sample-protocol one_pair_residuals --residual-profile translation --residual-bound-fraction 0.5 --precision bf16 --sample-count 32 --seed 0 --max-steps 32 --eval-interval 32 --threshold-mace-px 0.1 --heartbeat-interval 16 --progress-checkpoint outputs/tiny_overfit/mcnet_d8_probe32_progress.pt --output artifacts/p4_tiny_s_d8_mcnet_probe32.json
```

运行完成32步，梯度有限，H0/H1/H2均值为14.868890/9.773121/7.634817 px，零失败和零拒绝。未达到0.1 px，按失败结果保存，不填补正式 tiny gate。新记录仅执行 D8，未启动 D1。原有 `artifacts/p4_tiny_s_d1_translation_swa.json` 用户工作区改动继续保留且不提交。

## 2026-09-10：正式训练前检查完成，按要求停止

截至07:18 UTC，D8/D4/D2/TINY-6均正常结束并通过登记门槛，四份checkpoint独立重载完成，合并审计与训练入口的只读门槛校验通过。P0–P4当前检查已完成；未执行E00/E01、解冻比较、结构消融或test评估，不启动后续训练进程。

| 实验 | 步数 | H0→每轮H均值 MACE px | 原始终点 px | 主读出来源 | 失败/拒绝 | 峰值allocated GB | 训练循环 s |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: |
| D8 | 1664 | 14.868890→0.291660→0.098226 | 0.290066 | 129次参数平均 | 0/0 | 0.572 | 617.205 |
| D4 | 1568 | 7.434444→0.191940→0.099413 | 0.272937 | 33次参数平均 | 0/0 | 1.982 | 1459.628 |
| D2 | 1152 | 2.787924→0.212890→0.099895 | 0.099895 | 原始参数 | 0/0 | 2.821 | 2639.150 |
| TINY-6 | 1568 | 14.868889→1.304389→0.208130→0.184050→0.141688→0.081053→0.081194 | 0.209412 | 33次参数平均 | 0/0 | 3.099 | 4864.561 |

每项失败率分母为32；单尺度拒绝率分母64，TINY-6为192。梯度均有限。缓存描述子后的主读出前向时间分别为56.555/64.281/109.516/163.817 ms，详情见 `docs/pretraining_mcnet_d2_summary.md`。这些时间在共享GPU运行，且不包含GHIM/CGMDP，不能用于独占端到端速度结论。原始/平均读出和方向诊断均保留。

平均从预登记1536步开始。D2在1152步已由原始参数通过，所以虽checkpoint文件名包含 `weight-average-from-1536`，其metadata明确 `checkpoint_contains_weight_average=false`；文件名不作为读出判断依据。其余三项原始终点均高于0.1 px，不能声称原始优化器状态也通过。D4/D2接近门槛，未来跨种子/跨图像稳健性仍待正式实验。TINY-6的平均H6比H5回退约0.000141 px，原始读出的中尺度也有回退，后续必须继续报告完整轨迹。

| 实验 | artifact SHA256 | 最终checkpoint SHA256 |
| --- | --- | --- |
| D8 | `7c2f1fae0cf576c7b753fce0c2875c88746101b8b78d3723b85127adc386f294` | `0da998f748cb901d0674e2767931603bbff2987e38aea08cc9591a460c39a15b` |
| D4 | `962b3874ea296074e34b2306ebee967a1715921670a7efb466868d69e656243e` | `10838c6a8ae15dbf4ac44d24588c37f577c3113a598e0490074491729d5372d7` |
| D2 | `e35ed757acc633b30dad9af5a791cb17a846352f57f366d3e8911a3a72ef305c` | `00ec15456650d17869beb2a48c45919b7365b307b14336ba6f80da920098ef2c` |
| TINY-6 | `0f60415c4f3c03f235249f2b0cf6eb23a4fdd6ff4c3ec43bc09e9ef2a1546ed2` | `61a1b7c8e131762887084c4dec14d741fc7d7a51a9515829c2374814a647afc1` |

四个checkpoint的绝对路径、完整协议和原始progress路径都保存在 `artifacts/pretraining_mcnet_d2_summary.json`。权重文件位于服务器 `outputs/tiny_overfit`，不将大权重加入Git。

D4/D2/TINY-6的重载均使用物理GPU3、`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，执行与D8相同的 `tiny-checkpoint-audit` 参数，分别替换 `--experiment D4/D2/TINY-6`、对应checkpoint路径及 `--output artifacts/p4_tiny_s_d4_mcnet_checkpoint_audit.json` / `p4_tiny_s_d2_mcnet_checkpoint_audit.json` / `p4_tiny_6_mcnet_checkpoint_audit.json`。三份重载artifact SHA256依次为：

- `b50471457d6835d34afea21ad8c3069232cf6201ea1256d578713b45bc4783ad`
- `a94cbb25dc3fbdbb52e81003da7c3aa746132b43895c7345e5b757520fddb229`
- `2a796335daf71d12953d0051f3b58a750fa2e9bfeeb132ad302eb254846f1b28`

四次重载均 `strict_signature_match`，序列化/加载/前向后权重SHA一致，轨迹最大绝对差0。重载仅做forward，未构造optimizer、未恢复optimizer/RNG、未backward、未访问test。

合并及只读正式入口检查：

```bash
/root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli tiny-gate-merge --inputs artifacts/p4_tiny_s_d8_mcnet.json artifacts/p4_tiny_s_d4_mcnet.json artifacts/p4_tiny_s_d2_mcnet.json artifacts/p4_tiny_6_mcnet.json --output artifacts/p4_tiny_gate_mcnet_d2.json
/root/miniconda3/envs/loma-repro/bin/python -c 'from pathlib import Path; from mhinet.train import _validate_tiny_gate; print(_validate_tiny_gate(Path("artifacts/p4_tiny_gate_mcnet_d2.json"), True))'
```

合并status=`passed`、errors为空，SHA256 `13719169eaed0d1d887a18d846f4a824525231f427dc8ec3d3456edad6a60ea3`；训练入口接受该gate。89/89 unittest通过（1.571 s），此前设计包31项验证通过。架构配置内容继续保持原SHA，不修改配置中的历史status字段来伪造新身份；完成状态记录在本日志与gate中。

D2平台期附加诊断命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli real-correlation-audit --runtime configs/runtime_paths.server.json --scale 2 --seed 0 --condition-index 2 --residual-bound-fraction 0.5 --output artifacts/p2_real_correlation_mcnet_d2_condition2.json
```

该纵向条件下256维原描述子/随机32维adapter的最近候选top1比例为0.755147/0.498455，存在可用局部匹配信号。artifact SHA256 `164d6b80cb6b62cf217b73d765b650ca19c9cd33aa19405e39cfeafd91559ac8`。D2随后在原定预算和损失下突破平台，无需更换损失或放宽门槛。全部方向表由 `scripts/analyze_tiny_conditions.py` 生成至 `artifacts/p4_mcnet_d2_all_condition_analysis.json/.md`。

交付边界：本轮证明工程正确性和登记受控条件的可学习性，不代表真实GHIM初始化上的泛化验证或联合训练成功。下一阶段建议从登记E00/E01开始，使用新的正式初始化；保持test封存、D1不执行、DINO冻结、GHIM head输入梯度保留。用户明确要求在正式训练前停止，本轮到此结束。

## 2026-09-10：单卡batch短测、进度条、H0+六轮验证叠加图

用户要求真实batch尝试、训练进度条、每次验证输出绿框GT/红框预测叠加图，并追加H0形成7张；提供物理GPU1训练命令，但不代启动正式训练。代码、结果表、超参、冻结范围、三模块解释和正式启动/恢复命令见 `docs/batch_size_and_visualization.md`。本轮只执行工程短测/两步冒烟；E00/E01未运行，test未访问。

修改实际文件：`mhinet/feature_provider.py`支持真实B对共享DINO/MVT及累计decoder形状；`mhinet/train.py`按有效pair加权累积并补足无效样本，接入tqdm和每次验证图；`mhinet/evaluate.py`接入验证图及CLI进度；新增 `mhinet/visualization.py`、`mhinet/batch_probe.py`、`tests/test_batch_visualization.py`、三个batch检查/汇总脚本。D1不计算、不删除；原架构hash不变。较大batch尚未通过串行H0对齐，正式配置明确保留BS=1×累积4，BS>1需显式实验性opt-in，不放宽阈值宣称通过。

环境：`/root/miniconda3/envs/loma-repro/bin/python`，Python3.10.20、PyTorch2.11.0+cu128、tqdm4.67.3，RTX4090；所有GPU命令设 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。项目 `/home/disk1/MHINet`，复用源码 `/home/disk1/LoMa`；数据 `/home/disk1/Data/datasets/GoogleEarth_scale_pairs`。实际资源路径、逐个checkpoint SHA256、源文件hash、可视化PNG hash、恢复状态均归档到 `artifacts/batch_visualization_summary.json`；原始probe分别是 `artifacts/batch_probe_heads_bs1.json` / `bs2.json` / `bs4.json`（完整文件名前缀均为batch_probe_heads_）。

| BS | 累积 | GPU | allocated GB | reserved GB | 中位pair/s | 解释 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 3 | 4.001 | 5.098 | 0.779 | 2次预热+5次计时完成 |
| 2 | 2 | 3 | 7.112 | 9.028 | 1.111 | 2次预热+5次计时完成；对齐未通过 |
| 4 | 1 | 0 | 13.361 | 16.624 | 0.934 | 2次预热+5次计时完成；实验路径 |

这些是heads profile工程吞吐，不是精度结果；其他GPU作业负载明显，BS4不同卡，不能据此确认稳定加速比。命令为 `python -u -m mhinet.batch_probe --runtime configs/runtime_paths.server.json --batch-size B --output artifacts/batch_probe_heads_bsB.json`，B依次1/2/4，对应CUDA_VISIBLE_DEVICES依表；日志在 `outputs/batch_probe_heads_bsB.log`。

批内反序H0差0；BS2对比逐对运行H0最大角坐标差0.36767578125px，超过检查脚本预设0.1px，保留 `artifacts/batch_forward_isolation.json` 的failed状态。命令 `CUDA_VISIBLE_DEVICES=0 python -m scripts.check_batched_forward --runtime configs/runtime_paths.server.json --output artifacts/batch_forward_isolation.json`。随后GPU3执行 `python -m scripts.diagnose_batch_precision`，日志 `outputs/batch_precision_diagnosis_fp32_position.log`：DINO最大特征差0.927975，MVT最大差1.25；固定串行DINO/MVT再批量head，四角差0.004883px。仅拆分head不能修复上游差异，尚未定位到第一个差异算子。初版诊断脚本忘记显式传旧头的FP32位置dtype而报错，修正诊断参数后重跑；未为此更改GHIM算法或权重。

第一次BS2真实训练入口冒烟在GPU3外部显存占用增长后OOM（进程allocated6.47GiB、剩余58MiB），日志 `outputs/engineering_batch_visualization_smoke_start.log`。失败输出目录保留不覆盖；最终冒烟配置改为正式推荐的BS1，在GPU0重新执行：

```bash
cd /home/disk1/MHINet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli train --runtime configs/runtime_paths.server.json --config configs/batch_visualization_smoke.json --output-dir outputs/engineering_batch_visualization_smoke_bs1 --tiny-gate-artifact artifacts/p4_tiny_gate_mcnet_d2.json --stop-after-optimizer-step 1
/root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli train --runtime configs/runtime_paths.server.json --config configs/batch_visualization_smoke.json --output-dir outputs/engineering_batch_visualization_smoke_bs1 --tiny-gate-artifact artifacts/p4_tiny_gate_mcnet_d2.json --resume outputs/engineering_batch_visualization_smoke_bs1/checkpoints/step_0000001.pt
```

2步完成，数据流位置4→8，无invalid、无共享调用违规，日志确认训练和验证进度条。该次最终checkpoint SHA256=`15a7feaf142bf72a25a54c057a7909661ddcec9fd0a10316b6c20a295b02469a`。随后按用户追加要求加入H0，用同一step1 checkpoint在新目录重放step2，命令第二行仅将output-dir改为 `outputs/engineering_visualization_seven_smoke`；不覆盖先前6张图的原始证据。

7张图实际输出于 `/home/disk1/MHINet/outputs/engineering_visualization_seven_smoke/visualizations/step_0000002/pair_0000`，H0及D8/D4/D2各2张；manifest记录同一验证pair和每轮H。已目视检查真实叠加图，绿色GT/红色预测一致；不是用GT warp冒充预测。新增非法投影隔离、七张计数、缺失迭代拒绝、batch预算及实验opt-in测试，95/95 unittest通过（1.919s），`git diff --check`通过。

| 7图重放（2对val，仅工程检查） | H0 | H1 | H2 | H3 | H4 | H5 | H6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 平均MACE px | 6.611870 | 6.601425 | 6.599621 | 6.601795 | 6.606786 | 6.609659 | 6.612917 |

失败率0，最终验证单对含指标准备的同步单次延迟1714.687ms（绘图不计入该值，其他GPU作业影响较大）；训练step2峰值allocated3.999976GB、reserved4.888461GB。最终H略差于H0，不能将两步冒烟称为精度提升或模型验证成功。7图重放checkpoint：`outputs/engineering_visualization_seven_smoke/checkpoints/step_0000002.pt`，SHA256=`20ee388f5914850287c8237f45e0ad273c73a6e7d7ce539baa5d6a0c72b003c8`。重复恢复的细小数值差符合此前记录的CUDA grid_sample反传非逐位确定性，不声称bitwise一致。

归档命令 `python -m scripts.summarize_batch_visualization`。正式建议仍为 `configs/e01_heads_v1.2.json`：BS1×4、heads-only、lr1e-4、AdamW wd1e-4、clip1、10000步、500步warmup/cosine到1e-5、每500步验证2500对并输出固定1对的7张图。用户在GPU1自行启动的完整命令见上述文档；本轮未占用GPU1启动正式任务。后续较大batch先修复BF16批量对齐、同卡复测，再升级正式配置。

## 2026-09-10：代码按职责分包，新增两个Shell入口

按用户要求整理 `/home/disk1/MHINet/mhinet`，一级目录只保留 `__init__.py`、`config.py`、`cli.py`，25个原实现文件迁入 `models/`、`ops/`、`engine/`、`dataio/`、`diagnostics/`、`visualization/`；各目录含独立包声明。模型仍为GHIM→CGMDP→MHIR，D2截止、D1保留不执行；没有拆改算法或重新初始化权重。所有源码/测试/辅助脚本导入与mock目标更新；统一 `python -m mhinet.cli <command>` 路径和子命令不变。直接Python导入改用子包路径。历史日志、artifact内的源文件路径和hash保持原样，不伪造迁移后的旧证据。

新增且仅新增两个 `.sh`：

- `/home/disk1/MHINet/scripts/train_e01.sh`：默认物理GPU1，使用 `/root/miniconda3/envs/loma-repro/bin/python`；固定现有runtime/E01/gate/output默认值，支持 `GPU_ID`、`MHINET_PYTHON`、`DRY_RUN=1` 和额外CLI参数（包括 `--resume`）。不自动执行E00，不自动覆盖已存在输出。脚本从自身位置解析项目根，可从其他cwd调用。
- `/home/disk1/MHINet/scripts/test.sh`：同一conda Python，显式隐藏CUDA设备，运行CPU unittest；不访问真实train/val/test影像、不启动正式训练。支持附加unittest参数，例如 `-p test_geometry.py`。

验证：`bash -n scripts/train_e01.sh scripts/test.sh`通过；`bash scripts/test.sh`的99项测试全部通过，含原95项和新增4项入口/导入测试；完整复跑日志为 `/home/disk1/MHINet/outputs/code_reorganization_tests.log`。新增检查覆盖所有14个CLI子命令目标可导入、从 `/tmp` 调用默认GPU1、GPU2覆盖与含空格resume路径、拒绝多GPU编号。`DRY_RUN=1 bash scripts/train_e01.sh`输出预期命令而不构建模型；`bash scripts/train_e01.sh --help`安全退出。

额外只读AST审计：逐一用 `git show HEAD:mhinet/<原文件名>` 与25个迁移后的文件比较，移除import节点后AST完全一致，结果 `PASS: all 25 moved modules have identical non-import AST`。因此本次没有修改模型属性/state_dict键、优化器逻辑或checkpoint格式；单元测试仍覆盖checkpoint round-trip。未执行真实GPU重载或训练，不将此记录成新的模型精度验证。原架构SHA256仍为 `92095ffe16a5e650bab641df3fd17e16a0bf1f34ab47c82c59f37faa43ae092a`，tiny gate仍为 `13719169eaed0d1d887a18d846f4a824525231f427dc8ec3d3456edad6a60ea3`。未生成或修改checkpoint，权重实际路径/hash沿用上一条归档。环境仍为Python3.10.20、PyTorch2.11.0+cu128、conda loma-repro；本次不使用GPU。

README新增目录树与启动/测试/切卡/续训示例，更新当前batch说明文档中的源码路径；保留原来无关的 `artifacts/p4_tiny_s_d1_translation_swa.json` 工作区修改，不纳入本次提交。

## 2026-09-10：真实影像精度评估入口与16对val实测

用户明确需要真实影像精度评估，而非仅工程测试。将原单元测试入口移动到 `scripts/unit_tests.sh`，`scripts/test.sh`改为必须指定checkpoint的真实影像评估：默认物理GPU1、完整val、七阶段指标、首个验证pair的7张图；支持 `--max-pairs`、`--output-dir`、`--visualization-pairs`、显式 `--split test`。test只用于锁定配置后的最终评估，脚本不默认读取test、不默认选择checkpoint、不缺权重退回随机初始化。

修改 `mhinet/engine/evaluate.py`，新增 `mhinet/engine/reporting.py`。独立CLI输出改为 `OUTPUT/metrics/{summary.json,pair_metrics.csv,pair_metrics.jsonl}`、`OUTPUT/report.md`、`OUTPUT/visualizations/pair_0000/`；训练内部原有validation输出位置不变。默认检查非空输出目录，防止跑完后发现不能保存。汇总绑定checkpoint SHA256、manifest SHA256与checkpoint角色，不将两步工程权重误称正式训练权重。

真实指标新增每阶段success/AUC@1/3/5输入像素，AUC精确定义为 `sum(valid*max(0,1-MACE/t))/N`；失败保留分母、贡献零。这是归一化经验CDF积分，不冒称其他论文的梯形插值口径。修正偶数样本中位数：使用0.5分位数而非torch.median的较小中间项。修正原 `latency_ms_per_pair_single_pass` 实际混入I/O/绘图的问题，改为已记录的单对同步计时均值，并另存 `wall_time_ms_per_pair_including_io_visualization`。原历史文件不改写；新指标版本 `real_image_v2_ecdf_auc_quantile_median`。正式warmup/重复性能基准仍另做，window recall仍未实现、不造值。

环境沿用 `/root/miniconda3/envs/loma-repro/bin/python`，Python3.10.20、PyTorch2.11.0+cu128、RTX4090。实际数据为 `/home/disk1/Data/datasets/GoogleEarth_scale_pairs/val/pairs.jsonl` 前16对；未访问test。GPU1有其他负载，因此本次短评估指定GPU0。完整命令：

```bash
cd /home/disk1/MHINet
GPU_ID=0 bash scripts/test.sh outputs/engineering_visualization_seven_smoke/checkpoints/step_0000002.pt --max-pairs 16 --output-dir outputs/real_image_accuracy_val16_verified
bash scripts/unit_tests.sh
```

checkpoint SHA256=`20ee388f5914850287c8237f45e0ad273c73a6e7d7ce539baa5d6a0c72b003c8`；它仅训练过两步，本次真实影像实测是入口验证和该权重的实际测量，绝不是完整正式训练效果或最终test结果。首次输出 `outputs/real_image_accuracy_val16` 保留旧中位数口径；补齐hash和中位数修正后复跑到新的 `outputs/real_image_accuracy_val16_verified`，没有覆盖旧证据。

| 状态 | H0 | H1 | H2 | H3 | H4 | H5 | H6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16对平均MACE px | 4.157159 | 4.100831 | 4.057889 | 4.038963 | 4.023873 | 4.020474 | 4.017675 |

最终success@1/3/5px=`0/0.375/0.75`，AUC@1/3/5px=`0/约0.0828/约0.2691`；失败率和拒绝率0。H6平均MACE相对H0下降约0.1395px，但P90从6.154736增至6.292036，不能声称所有样本改善。峰值allocated/reserved=2060819968/2675965952 bytes；单次延迟367.961ms、含I/O/绘图墙钟均摊591.850ms（共享服务器短测，不是正式性能基准）。七张PNG已生成并在artifact计数核验。

完整表格与文字解读 `docs/results_real_image_val16.md`，机器可读归档 `artifacts/real_image_accuracy_val16.json` 含所有七阶段、manifest/checkpoint/report/summary hash与实际输出目录；操作说明 `docs/real_image_evaluation.md`。102项单元测试通过（1.804s），新增AUC失败分母、标准中位数、真实评估Shell必须checkpoint且调用evaluate的检查；日志 `outputs/real_image_evaluation_unit_tests.log`。真实评估日志 `outputs/real_image_accuracy_val16_verified.log`。本轮无反向传播/权重更新，未启动正式训练；当前未发现E01正式训练checkpoint，后续应由用户显式传入实际训练权重路径进行完整val/test评估。

## 2026-09-10：独立E00 H0基线Shell入口

新增 `/home/disk1/MHINet/scripts/test_e00.sh`，默认物理GPU1、`loma-repro`的实际Python、runtime登记的预训练权重，执行 `mhinet.cli evaluate --split val --max-pairs 2500 --h0-only --visualization-pairs 0`。这里选择与E01相同的val前2500对，而不是不同规模的完整val，以便公平对照。默认结果为 `outputs/E00_h0_seed0/report.md` 与 `metrics/`，不生成精修七轮图；只有GHIM前向，不训练、不执行CGMDP/MHIR。拒绝传入训练checkpoint，已训练模型请用原 `scripts/test.sh`。支持切卡、`DRY_RUN=1`、样本数与输出目录参数；不自动覆盖目录、不读取test。

`bash -n scripts/test_e00.sh`、dry-run、从/tmp调用的参数契约检查均通过；`bash scripts/unit_tests.sh`为103/103通过（1.594s），日志 `outputs/e00_entrypoint_unit_tests.log`。README和真实评估说明同步更新。本轮只创建并检查入口，未启动E00长评估或训练，未生成/修改checkpoint；环境、真实资源路径和checkpoint hash沿用现有登记。

## 2026-09-10：按用户要求更正独立评估划分，检查误跑val结果

用户明确规定train用于训练、val用于训练中验证、test用于独立评估（包含E00）。服务器逐条JSON清单统计：train=36000对、val=2500对、test=1000对。先前未统计就把2500解释为截取上限不准确，此处纠正；现行独立评估不得再默认val。

修改 `scripts/test_e00.sh`、`scripts/test.sh` 和 `mhinet/engine/evaluate.py` 的CLI默认值为test；两个脚本默认均不设置max-pairs，评估完整1000对。保留用户此前对E00脚本删除max-pairs的修改。训练实现与配置不改，仍读取train与val。帮助、README、当前操作文档和实验计划同步说明，test结果不得参与模型选择。历史val16诊断是历史证据，不冒充test结果、不改写记录。

用户要求如这次误跑E00的val有结果则删除。两次只读检查均未发现 `/home/disk1/MHINet/outputs/E00_h0_seed0`；扫描outputs内summary.json也没有mode=H0_only的结果；进程列表中未发现E00/evaluate进程。因此没有可确认属于本次误跑的输出可删，未删除任何文件、未终止其他作业，未触碰历史val16、训练验证或无关D1结果。

`DRY_RUN=1 bash scripts/test_e00.sh`确认实际命令含 `--split test` 且没有 `--max-pairs`；Shell语法检查通过；103项单元测试通过（1.592s），日志 `outputs/test_split_entrypoint_unit_tests.log`。未启动重新评估，由用户执行脚本；无新增模型结果或checkpoint。环境和资源路径/hash沿用前述登记。

## 2026-09-10：训练同目录自动续训

用户要求：输出目录相同则接着训练，新的输出目录则开始新训练。`mhinet/engine/train.py`现在在未传`--resume`时检查输出目录：空/新目录初始化新run；非空且存在`checkpoints/step_*.pt`自动选择最新编号checkpoint并恢复模型、optimizer、scheduler、RNG和数据流；非空无checkpoint安全报错，避免覆盖未知结果。显式`--resume`优先；恢复仍校验training config SHA、architecture SHA和profile。训练元数据写入`resume_mode=auto_latest|explicit|new_run`。`--overwrite`仍是用户明确的例外，不会被自动启用。

此次用户本地尝试的脚本已选择 `e01_heads_v1.2 bs_2.json` 和 `outputs/E01_heads_bs_2`；补入`allow_experimental_batch=true`后可通过实验性batch保护。该配置为BS2、累积4、有效batch8，保持与文件内参数一致；其H0批量数值对齐尚未通过，不能与正式BS1主线混比。该配置随代码一并保存，确保远程clone的默认脚本可运行；规范BS1配置`configs/e01_heads_v1.2.json`仍保留，脚本支持`MHINET_CONFIG`和`MHINET_OUTPUT_DIR`显式切换配置/实验目录。未启动或终止用户训练。

验证：`bash -n scripts/train_e01.sh scripts/test.sh scripts/test_e00.sh scripts/unit_tests.sh`通过；103项原有单元测试加最新checkpoint选择测试共104项通过。`DRY_RUN=1 bash scripts/train_e01.sh`确认默认路径、GPU1和参数转发；无新增checkpoint、无删除输出。当前工作区中的`artifacts/p4_tiny_s_d1_translation_swa.json`仍为用户无关修改，不纳入提交。

评估总表同步：扫描服务器现有outputs中的10份真实评估summary（含刚完成的E00完整test/1000对），生成 `docs/evaluation_summary.md`、`artifacts/evaluation_registry.json` 和 `.csv`。每条记录绑定源summary SHA、split/对数、模式、H0/最终轨迹、成功率/AUC、失败率、显存、延迟、manifest与checkpoint信息；工程val冒烟和独立test分开标注。评估入口已自动登记新summary，`python -m mhinet.engine.evaluation_registry --scan outputs`可重扫。表不删旧记录，不把tiny/性能探测视为精度评估。

## 2026-09-16：独立共享描述子预训练与DINOv3 LoRA分支

用户批准建立分支实施，最终提供卡1冻结DINO的共享网络、卡2 LoRA对照训练命令；不自动启动正式长训练。由`d312c0a`建立`codex/shared-descriptor-lora`，首个中文提交`377e4a8`已推送origin。既有未提交的旧trainer/random-init/评估表修改保留，不纳入本任务提交。

新增实际代码：`mhinet/pretraining/{model,loss,data,evaluation,train}.py`，两份`configs/shared_descriptor_{frozen,lora}.json`，`scripts/train_shared_descriptor{,_lora}.sh`，三个审计入口`audit_shared_pretraining.py/check_shared_bundle.py/check_shared_resume.py`，测试`tests/test_shared_pretraining.py`。复用现有feature_provider、GHIM四项损失、checkpointing；未修改相邻LoMa源码。只运行GHIM/CGMDP至D2，不构建MHIR、不执行D1。LoRA包装保留原QKV masked-bias，blocks8–17/r8/alpha16；梯度路径绕过旧DINO no_grad，冻结主干本体。

正式参数：同预训练初始化、第一档train51978/val5772、母图/地理交集0；BS1累积4、单epoch12995 optimizer steps、warmup5%+cosine到10%、AdamW wd1e-4/clip1；每5000步及末尾全量val，每1000步保存。MVT/GHIM head lr1e-6，VGG5e-6，累计解码器/LoRA1e-5。损失为现有GHIM四项+三尺度双向InfoNCE（每方向最多1024查询、temperature0.1、全局采样及局部hard negatives、双侧mask）；test未读取用于选参。

资源/环境：Python `/root/miniconda3/envs/loma-repro/bin/python` 3.10.20，torch2.11.0+cu128，RTX4090；LoMa `/home/disk1/LoMa`，数据 `/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1`。实际权重路径完整登记在`configs/runtime_paths.quadrant.server.json`及每次`run.json`：LoRetta位于LoMa outputs/loretta_stage1_h/cache/huggingface/hub/models--BRoss123--LoRetta/snapshots/95bf3670465cc80ffca8b1edf94b5eb651211194/loretta.pth，SHA256 `09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e`；金字塔`/root/.cache/torch/hub/checkpoints/loma_B.pt`，SHA256 `3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758`。train manifest SHA `6587d3be58dfc8b1601244bcf7fa05e3cb5dba1b14b7a49871dcb1fe3ef3276c`，val SHA `c4763636e2ca9914c62aec45f5f6f238e0cfa9269343f27571e6c0acdb3d01ee`。

验证：`bash scripts/unit_tests.sh`127项通过（含新增7项），1.991秒；新脚本bash语法检查通过。真实权重LoRA零初始化H0/D8/D4/D2最大差0；GHIM/CGMDP两条分支对MVT及LoRA均非零有限梯度；第二步A梯度非零。严格模式真实单对20步Ldesc1.167799→0.315688、LGHIM0.040362→0.006571，DINO base参数hash始终`611ab0607761c72dc93bcf7379829109f14da32fd43f9a8ed1e5a07f69d1317f`；报告`artifacts/shared_pretraining_real_tiny20_strict.json`。

初次独立连续/恢复比较未通过，最大state差基线0.030666/LoRA0.022022；定位CUDA非确定性反向，warn_only不足以开启确定性Flash Attention。新增与grid_sample前向/梯度对齐的gather双线性采样，开启strict deterministic、cuDNN deterministic、CUBLAS_WORKSPACE_CONFIG。不放宽误差阈值：重跑两组8对/2步连续与中断恢复，模型/optimizer最大差均0，checkpoint SHA各自完全一致。报告`artifacts/shared_{frozen,lora}_resume_audit.json`；基线checkpoint SHA `02125544071e25e9fe5c310b3dfe42dfcf14949175c373f69bdeb52af97ddc6a`，LoRA `6112cd0c1404a1425522b08b3638ed3486756d4ec2cefb67aa83bf09678e6308`。

真实小跑输出位于`outputs/diagnostics/shared_{frozen,lora}_strict_split`与`*_strict_continuous`。基线第二步2.347秒/峰值9.949GB，LoRA2.617秒/10.104GB（不含保存/val，仅诊断，非正式速度承诺）。每次val实际输出H0叠加图和D8/D4/D2 full-gallery匹配图；32查询/尺度/方向的全目标网格检索误差与采样InfoNCE命中率分开记录。2对val的H0 MACE约201/202px，拟合成功不代表精度成功，这些数字不得作为完整数据集结果。

恢复机制：同目录latest.pt恢复模型/optimizer/scheduler/RNG/数据游标并核对配置、数据、权重、代码hash；无checkpoint将本入口旧记录移入previous_no_checkpoint_*再从头训练，不删除未知文件；目录锁防并发覆盖。导出shared_descriptor.pt含MVT/GHIM/VGG/累计解码器/可选LoRA，不含DINO原权重及任务头；load_shared校验DINO hash，供MHINet和LoMa DINOv3共享分支使用，未宣称原版DINOv2 LoMa可直接load_state_dict。

诊断期间卡1被外部作业占约21GB，初次基线bundle/新短跑因此OOM；没有中断外部作业，基线诊断迁到卡3，LoRA主要用卡2。最终脚本默认卡1/卡2，启动前需确认空闲。工程检查、具体命令、结果表与失败修复详情汇总在`docs/shared_descriptor_results.md`和`docs/shared_descriptor_pretraining.md`。未启动正式训练，未宣称模型验证成功。

最终bundle审计：冻结版806项、LoRA版826项state全部与训练checkpoint一致，H0/D8/D4/D2推理差均0。bundle SHA分别`75a23020cbbeac693f310056a295357448b46883add1588be218db568f4b5ce6`和`bd15a3d8e3dc39ed85b9bcd70f9a70002add0aa40c28435b94588e0b967b5bca`，见`artifacts/shared_{frozen,lora}_bundle_audit.json`。本任务所有短程GPU进程已退出。

## 2026-09-17：H0有效重叠投影误差与第二档训练

用户明确不继续LoRA，要求补充有效重叠区域H0误差，再实际启动第二档训练并提供tail命令。已补`mhinet/pretraining/overlap_metrics.py`和旧checkpoint评估入口`scripts/evaluate_shared_overlap.py`；新训练验证保留四角MACE，并额外保存H0_overlap。支持集使用GT和A/B有效mask确定，枚举全部源图像素中心，单位为784重采样目标图像像素，方向A→B；预测出界不剔除，非法预测保留在Recall分母。记录pair均值/中位数/P90、像素加权均值、Recall@1/3/5、无支持对数、非法投影和拟合失败，不删极端标签。

第二档配置`configs/shared_descriptor_tier2.json`、脚本`scripts/train_shared_descriptor_tier2.sh`。初始化第一档冻结版最终`/home/disk1/MHINet/outputs/shared_descriptor_frozen_seed0/latest.pt`，SHA256 `df3ec90b9bc53fd983cc1945a3c028ccdf9bf688325a60055101232c509cc25e`，原step12995；严格加载模型、校验DINO provenance，不继承已结束的optimizer/scheduler。无LoRA，DINO冻结，其余共享层训练、VGG BN统计固定；仍不运行MHIR/D1。峰值lr：MVT/head5e-7、VGG2.5e-6、CGMDP5e-6（第一档峰值一半）。BS1×累积4，新AdamW、wd1e-4、clip1，warmup5%后cosine至峰值10%。同新目录支持自动恢复。

实际数据仍为`/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1`，仅第二档几何+辐射处理；train34652/val3848，母图/地理交集均0，一轮8663步；第5000步及最后全量val，每1000步保存。test未用于选择。环境沿用loma-repro Python3.10.20/torch2.11.0+cu128。

检查：131项单元测试通过，耗时2.097s；新增4项覆盖恒等、已知平移、双侧mask、非法投影及fit失败分母。实际8对train/2对val冒烟两步通过，loss0.190485→0.167080、峰值9.96GB；目录`outputs/diagnostics/shared_tier2_overlap_smoke`，新增overlap指标覆盖788167像素，无非法预测，pair均值1.8584px。这里只是小样本工程检查，不是第二档完整训练精度。最终调整中位数使用quantile(0.5)，并补初始化DINO hash校验。

已提交并推送`bbd95d7`（补充H0有效重叠投影评估并配置冻结DINO第二档续训）。启动时GPU0–3均空闲；以独立session后台启动：卡1训练PID3788458，输出`outputs/shared_descriptor_frozen_tier2_seed0`，日志`outputs/shared_descriptor_frozen_tier2_seed0.console.log`；卡0补评估PID3788465，在第一档最终冻结模型上计算tier1/tier2完整val的H0指标，输出`outputs/shared_descriptor_frozen_seed0/h0_overlap_v1`，日志`outputs/shared_descriptor_frozen_h0_overlap.log`。最初shell nohup后台启动未存活，已核对退出且无GPU占用后改用Popen(start_new_session=True)，没有重复训练实例。

后台训练命令等价于`GPU_ID=1 bash scripts/train_shared_descriptor_tier2.sh`；用户查看`tail -n 30 -f /home/disk1/MHINet/outputs/shared_descriptor_frozen_tier2_seed0.console.log`。每50步额外输出TRAIN行。H0补评估单独运行，不等待补评估完成才使用另一张卡训练；新训练自身每次val也计算同指标。设置与指标定义见`docs/shared_descriptor_tier2.md`。训练和完整补评估结果尚未完成，不能提前宣称成功。

启动确认：独立session脱离启动器后两进程仍存活；交付前卡1实际达到22/8663步（最近loss0.1657，显存约10GB），卡0补评估达到第一档647/5772对。未见异常；这只是启动状态，不是最终结果。

## 2026-09-18：第一档完整回测、第三档训练、极端外推审计

用户要求检查第二档是否损害第一档几何能力、继续第三档并分析外推。新增`scripts/evaluate_shared_full.py`对第二档最终checkpoint运行第一档完整5772对val（GHIM/CGMDP全指标、H0重叠误差），卡0后台PID3792188；输出`outputs/shared_tier2_backtest_tier1`，日志同名`.log`。新增`scripts/compare_shared_backtest.py`在完整report产生后，与第一档最终模型原验证及补算的H0 overlap基线比较，严格校验manifest、对数、pair ID及GT支持像素数。

第三档新配置`configs/shared_descriptor_tier3.json`、脚本`scripts/train_shared_descriptor_tier3.sh`：从第二档最终`outputs/shared_descriptor_frozen_tier2_seed0/latest.pt`初始化，SHA256 `4af7846f8111a64efae182005ba7fa9d1e90fecf9553815d5b88395d80280d7d`。冻结DINO且不开LoRA；其余共享层训练、VGG BN统计冻结；不运行MHIR/D1。峰值LR延续第二档MVT/head5e-7、VGG2.5e-6、CGMDP5e-6，BS1×累积4，warmup5%+cosine至10%，新AdamW状态。一轮34652对train/8663步，3848对完整val，第5000步及最后验证，每1000步保存；不改现有数据、GT或损失。先8对train/2对val两步冒烟，loss0.172034/0.211256、显存9.96GB，反传/验证通过；不同batch的loss不作收敛证据。卡1正式后台PID3792478，输出`outputs/shared_descriptor_frozen_tier3_seed0`，日志`outputs/shared_descriptor_frozen_tier3_seed0.console.log`；已实际进入数百步。第一档回测和第三档独立运行，保留第二档模型，回测完成前不宣称无遗忘。

另在卡2用第二档checkpoint补第三档完整val的训练前H0基线，PID3793109；输出`outputs/shared_descriptor_frozen_tier2_seed0/h0_overlap_tier3_baseline`，日志`outputs/shared_tier2_h0_tier3_baseline.log`，便于第三档结束后同数据比较。

极端外推只读审计：`scripts/analyze_homography_extrapolation.py`，报告`artifacts/homography_extrapolation_audit.json`及`homography_extrapolation_train_audit.json`。val第一/二/三档有78/32/21对GT角点超过10000输入px；train有553/384/249对。所有重新组合T_B@inv(T_A)的H与存储标签一致；角点分母无变号。但现生成器只检查同号/有限，不限制离分母零线距离、投影范围或局部放大率，合法H仍可在非重叠角点极端放大。

第二档最严重样本`past_tier2_1__36.035100129.396223`：GT角点约(-1039396,2348110)px，零线离源角点0.1916px，四角MACE619540.56px、重叠误差1.2503px。单对贡献总四角误差68.1%，32对极端GT贡献94.4%。分组：GT范围<2000的3513对角点/重叠均值6.191/1.518px；2000–10000的303对97.716/2.342px；>=10000的32对26818.398/4.377px。未删除困难样本；仍有真实重叠误差25/59px的坏例，不能把全部模型错误都归因外推。

控制敏感性实验（脚本--sensitivity，OpenCV4.13.0，seed0，128个GT对应点，目标端sigma0.1px噪声，100次无RANSAC拟合）：重叠误差中位数0.02054px、四角误差中位数178721.60px；不是网络预测。CPU float32/float64真值投影检查中最严重样本四角均值差13.5768px，远小于该样本预测619540px，不能单纯归因浮点精度。结论及新数据几何安全约束建议在`docs/homography_extrapolation_analysis.md`，未自动改变生成器。启动MHIR全图四角目标训练前需解决目标外推不稳定，不能只加复杂损失掩盖。

工程验证：133项单元测试通过（1.907秒），新增仿射/近零线但不变号测试；Shell语法通过。初版已中文提交推送`d87b20c`。本节记录启动及审计事实，完整回测比较将在结束后补记；第三档正式训练尚未完成。

完整回测现已完成：5772/5772对第一档val，GT支持2533713851像素保持一致。第二档权重回测相较第一档最终权重，H0 overlap每对均值2.113301→1.686334px、中位数1.178909→1.028497px、P90 2.240741→1.944974px，74.7228%影像对改善；D8/D4/D2双向平均匹配误差3.254289/1.601194/0.829732→3.234129/1.595642/0.824359px，D2 Recall@1px70.1314%→70.6444%，desc loss0.159341→0.145992，无非法投影/拟合失败。H0四角均值74.711518→69.928910px，仍需结合前述外推诊断。该单种子同域val上没有观察到总体能力退化；不泛化为所有域/下游均无遗忘。机器记录`outputs/shared_tier2_backtest_tier1/comparison.json`及`artifacts/shared_tier2_backtest_tier1.json`，完整表`docs/shared_descriptor_tier3.md`。

第三档训练前H0基线也已完成（3848对、1284531292支持像素）：每对均值2.305215px、中位数1.593579px、P90 3.689485px，point Recall@1/3/5为35.2183%/84.4293%/93.8839%，非法投影和拟合失败均0。仅基线评估完成，第三档训练仍在卡1后台运行，不宣称训练完成。卡0/卡2评估任务结束后正常退出。
