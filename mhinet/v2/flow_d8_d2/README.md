# 原 Flow → D8 → D2：冻结能力测试

分支从 `codex/mhi-v2` 的 `f6b8051` 建立。保留原 Flow0 + Matchability，不接实验拒配头，不执行 QRRU，不训练参数。

模型：原 Flow 模型第30轮；D8/D2 特征、温度来自 `outputs/hangzhou_cleaned_restart/v2_20261022/best.pt` 第7轮。加载旧下游权重后重新覆盖原 Flow 的完整 MVT/定位头。这个组合未联合重训，因此本轮衡量现有组合的能力，不代表局部匹配训练后的上限。

固定三组，不根据评价结果搜索参数：

- global：现有完整 D8 全局 dual-softmax + Flow 高斯加分。
- local：将 Flow 位移插值至 D8，围绕预测目标的最近 D8 中心搜索 ±4 格（9×9，半径约32原图像素）。只保留数值/边界有效初值，无额外 Matchability 阈值。
- local_mat01：同上，D16 Matchability <0.1 的查询不参与；D8 插值要求贡献的 D16 单元均有效，避免低分单元污染邻居。0.1 是本轮固定的保守试验值，不是已校准部署阈值。

局部匹配沿用原温度、Flow 高斯加分与 mutual 选择，但 dual-softmax 的归一化支持域变成稀疏候选图，因此分数不能直接当作与全局同标度的概率。每个 D8 父对应再展开为两端4×4 D2子点，在16×16局部相关矩阵中选互选对应；最终保留最多2048对。不引入局部几何过滤。

评价：杭州旧开发集64对（G0–G3）作为近似H参考；另用 PCF 留存的杭州同源干净合成35对（不含纹理替换）做精确H诊断。这些都是已使用数据，本轮属于开发性实验，不声称新的独立测试。阈值和窗口不据此调参。

分别报告 Flow0、D8、D2 的 P@1/3/5/10、有效区域EPE、匹配数和每对正确点数；另外在同一批最终D2源点上比较插值Flow初值与D2，避免不同采样密度造成虚假提升。统计 D8 搜索窗真值覆盖、D8选对父单元的比例（即D2父窗口覆盖）、特征和匹配耗时。所有误差单位为512原图像素，杭州结果不是人工真值精度。

运行：`python -m mhinet.v2.flow_d8_d2.run --domain hangzhou --device cuda:0`，另一个任务用 `--domain synthetic --device cuda:1`。报告：`python -m mhinet.v2.flow_d8_d2.report`。

## D8/D2 适配训练

`python -m mhinet.v2.flow_d8_d2.train --device cuda:0`

沿用清理后杭州254训练/64验证对，从上述下游第7轮权重继续训练。冻结DINO、MVT和完整原Flow/Matchability；只训练VGG、D8/D2解码器及匹配温度。QRRU不执行、无损失、无梯度、无优化器参数。沿用现有D8正对应损失与D2近似H不确定区域损失，不增加新的置信度标签。训练监督窗口由参考H提供，验证使用实际Flow局部搜索，二者的窗口来源明确区分。

学习率8e-5，AdamW，累积4对，最多30轮，余弦退火至1e-6；验证局部D2 P@3连续6轮未提高0.1个百分点时早停（至少5轮）。初始未训练权重也可被选为best。验证指标依然只是杭州近似H的一致性。

训练结果：`outputs/flow_d8_d2/training/`。最佳权重复评示例：

```bash
python -m mhinet.v2.flow_d8_d2.run --domain hangzhou --device cuda:1 --checkpoint outputs/flow_d8_d2/training/best.pt --output outputs/flow_d8_d2/trained
```
