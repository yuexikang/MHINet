# 源码复用与审计来源

本包基于已读取的本地 Git 快照，不宣称是服务器或远程最新提交。2-step：899fdb99a4076e312196bbbf99a3c329739b5d7a；MCNet：cc03479689b3cf40f0c384954f338b434765c155。服务器 P0 核对当前版本差异后决定兼容接入。

| 原资源 | 复用方式 | 必要改造 |
| --- | --- | --- |
| stage1_loma_shared_v1/stage1_head.py | 概率匹配、coarse warp/matchability、Stage1 head | 保留参数加载；允许输入 autograd |
| loma_dinov3_l17_v1/descriptor.py | DINO blocks11/17 + MVT，共享pair context | DINO冻结；MVT no_grad与冻结开关按训练阶段拆开 |
| stage1_dedode_pyramid_hroi_v1/pyramid_descriptor.py | VGG19-BN、DeDoDe cumulative decoder | 增加D4快照、实际执行scale1输出D1 |
| stage1_dedode_pyramid_hroi_v1/model.py | 仅参考原共享入口和归一化 | 新训练provider避免继承整网 inference_mode |
| loretta_stage1_h/fitter.py | 保留原Stage1概率对应几何拟合 | 前向对齐和奇异分支反传安全需同时验证 |
| 原数据构造与geometry工具 | 可复用可信标签和坐标约定 | 用明确像素/normalized A→B转换做单元测试 |

上述路径均在仓库 experiments/ 下。原 LoFTR fine、RRU、DaD、LoMa assignment 不纳入新 forward。LoRetta 第三方源码在所审 Git 快照中未完整跟踪，须另定位并固定版本；权重须选择实际现用 Stage1 的 MVT+head 成套 checkpoint，不能自动用随机权重或不匹配的官方权重替代后称为保留当前性能。

MCNet 提供多尺度 correlation refinement、直接四角迭代和序列监督的参考。其默认三尺度/每级两轮、先warp后相关，与 MHINet 四尺度及目标原坐标 H(p)+delta 的算子不同。其 FGO 代码含batch均值和序列内持续flag，见 docs/04。MCNet 对动态前景的隐式 outlier rejection 观察不等于 MHINet 已有地面/屋顶区分能力。

来源：[2-step固定提交](https://github.com/yuexikang/2-step/tree/899fdb99a4076e312196bbbf99a3c329739b5d7a)、[MCNet固定提交](https://github.com/zjuzhk/MCNet/tree/cc03479689b3cf40f0c384954f338b434765c155)、[CVPR 2024论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_MCNet_Rethinking_the_Core_Ingredients_for_Accurate_and_Efficient_Homography_CVPR_2024_paper.pdf)。用户提供的本地PDF已经用于原审视，包中未复制论文。
