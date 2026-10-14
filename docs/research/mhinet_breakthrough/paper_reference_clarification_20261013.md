# 用户指定参考结果与指标对齐

2026-10-13，用户补充了上限研究的具体参考：LoRetta的PCK@1/2/3/5为38.7/67.6/81.3/91.4%，AUC83.3%；SwinMatcher可见光图像对precision63.3%、SR100%、RMSE2.831。后续同时研究严格定位精度和全域配准成功率，不再仅把“八九十”当作未具名SR数字。

## 1. 已核实与待核实

| 方法/数据 | 指标 | 数值 | 当前证据与解释 |
| --- | --- | ---: | --- |
| LoRetta / LEVIR-GM | PCK@1/2/3/5 | 38.7/67.6/81.3/91.4% | 用户数字与v1表III一致；512坐标，在完全可匹配GT像素上评分 |
| LoRetta / LEVIR-GM | AUC | 83.3% | 是其PCK曲线的归一化梯形积分，阈值集0.5…10px，不能当SR@3 |
| SwinMatcher / 用户所指可见光图像对 | precision / SR / RMSE | 63.3% / 100% / 2.831 | 用户提供；遥感TGRS2025论文及作者仓库已定位，具体表格、阈值、成功条件、RMSE集合和样本范围仍需逐项核实 |
| MHINet H0joint / GE 498对 | P@1/3/5，SR@3 | 8.25/33.24/53.42%，47.59% | 所有输出匹配为P分母、逐对宏平均；SR为预测H在GT可见20×20网格RMSE≤3px |
| MHINet H0joint / JL 1755对 | P@1/3/5，SR@3 | 7.10/37.92/59.51%，43.65% | 与GE同协议，数据与训练/适配预算仍不同于参考论文 |

LoRetta公开正文§V-A说明各matcher对应先做RANSAC、再以TPS拟合并栅格化成密集warp，之后在完全可匹配像素Ω上评分。因此这组PCK评价的是整套配准链，而非本项目全部返回匹配点的P。 learned baselines也在LEVIR-GM上适配；需要一并对齐训练来源、checkpoint选择、候选筛选和非刚性GT。[LoRetta v1](https://arxiv.org/html/2608.04106v1)

正确遥感SwinMatcher为*Universal Cross-Modal Remote Sensing Image Matching with Interactive Swin Transformer*，IEEE TGRS 2025，DOI10.1109/TGRS.2025.3592418。[出版方](https://ieeexplore.ieee.org/document/11095750/)、[作者仓库](https://github.com/LotrL/SwinMatcher)。不能混用另篇Scientific Reports的*Swin-transformer for weak feature matching*。目前不将SR100%自行解释成严格RMSE≤3px，也不假定它是完整数据集而非子集/示例。

## 2. 当前模型的另一成功口径已接近100%

CPU只读汇总已有`h0joint_{GoogleEarth,JL1Flight}/pairs.jsonl`的`metrics.match_success20_at5`：

| 数据 | 至少20个5px正确匹配 | 当前严格SR@3 |
| --- | ---: | ---: |
| GE 498对 | 497/498 = 99.80% | 237/498 = 47.59% |
| JL 1755对 | 1711/1755 = 97.49% | 766/1755 = 43.65% |

这里只读取已保存字段，没有重推理、筛点、拟合或调参。指标实现见`scripts/mrsi_gt_metrics.py:50`附近的`match_success20_at5`与网格RMSE/SR分支。原结果目录为`outputs/semidense_rgb512_h0joint_test512_20261012`。

这说明“有足量正确对应”和“全域H达到3px精度”之间有明显差距，可用于定位系统性偏差、覆盖、相关错误与估计问题。它不证明SwinMatcher采用相同宽松成功规则，也不消除当前定位精度低的问题。下一轮必须把不同成功条件明确列出，不通过改指标获得表面提升。

## 3. 下一轮的双主线

1. **定位误差分布：**以LoRetta完整PCK曲线为参照，核对本项目P@1/2/3/5与可匹配区域、原始对应/TPS后warp的差别；缺P@2或共同稠密评价时标记缺项，不能插值冒造。严格1px与宽5px差距用于研究偏差/亚像素细化、真实起点训练与matchability，不直接把相差百分点归因架构。
2. **正确点如何变成可靠的全域配准：**以SwinMatcher precision/SR/RMSE为参照，核其正确匹配阈值、SR是正确点门槛还是变换误差、RMSE是正确点子集/成功对/控制点/全域网格及失败处理。结合已有高NCM、密集点、旧confidence、RANSAC、错误几何自洽和域遗忘证据，推导为何当前严格SR仍低。
3. **同协议真正提升：**提出最多三个大幅提升机制和最小单因素验证，并区分标签/数据、定位/选择、变换模型与估计。若对齐后仍有明显差距，明确理论可改善空间及缺失证据。TPS与单应的对照必须各自配适合的独立GT，不在当前协议下直接换拟合器后混称算法提升。

仍只读研究、轻量CPU诊断，不训练或改模型。旧点数与QRRU研究继续作为子证据；累计记忆记录这次具体参考和方法范围，避免下轮再把已确定论文写成未知。

## 4. SwinMatcher官方实现已核验的机制（原表定义仍未知）

固定作者仓库提交`9dc1de0cddafa4f23eb6cbec7fe3fe3da0fda6d3`，避免把同名ScientificReports模型混入。

- [coarse_matching.py:103](https://github.com/LotrL/SwinMatcher/blob/9dc1de0cddafa4f23eb6cbec7fe3fe3da0fda6d3/src/swinmatcher/utils/coarse_matching.py#L103)：双向softmax后将任一方向达阈值且为该方向最大值的候选作OR合并，不是严格双向互选。适合研究候选召回与歧义代价。
- [fine_matching.py:58](https://github.com/LotrL/SwinMatcher/blob/9dc1de0cddafa4f23eb6cbec7fe3fe3da0fda6d3/src/swinmatcher/utils/fine_matching.py#L58)：局部dual softmax及fine阈值之后，每个coarse窗口保留最高置信fine匹配；两端各采样3×3特征，卷积预测两端偏移，tanh×0.5限幅再映射到影像。下一轮将“扩大粗候选覆盖→窗口内质量选择→两端定位”与MHINet的严格mutual、16子点展开、单侧QRRU及旧置信度比较，每个机制单独验证。
- [demo.py:35](https://github.com/LotrL/SwinMatcher/blob/9dc1de0cddafa4f23eb6cbec7fe3fe3da0fda6d3/demo.py#L35)：demo用coarse0.3/fine0.1、512 resize与原图映射；第86行USAC_MAGSAC、5px并仅展示内点。这是demo，不能用它给论文precision/SR/RMSE补定义，也不能把5px解释为论文正确阈值。

未找到公开仓库的论文指标评测实现，正文访问受限。用户原表及相邻指标定义/实验设置是最小待核材料；仍不猜SR是否NCM>=20，也不把用户数值写成已独立核验。代码机制可先服务理论研究，不等待完整正文、更不以demo筛选流程替代公平基准。
