# GoogleEarth八组人工刺点独立测试结果

| 匹配集合 | 影像对数 | 人工点数 | 平均匹配数 NM | Ground RMSE 按对平均(px) | 全部人工点合并 RMSE(px) | Ground SR@1px | Ground SR@3px | Ground SR@5px | 拟合失败数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 8 | 78 | 11911.88 | 4.129 | 4.330 | 0.0% | 25.0% | 62.5% | 0 |
| Top-4096 | 8 | 78 | 4096.00 | 4.708 | 4.934 | 0.0% | 12.5% | 50.0% | 0 |
| Top-2048 | 8 | 78 | 2048.00 | 5.474 | 5.524 | 0.0% | 0.0% | 37.5% | 0 |
| Top-1024 | 8 | 78 | 1024.00 | 5.643 | 5.647 | 0.0% | 0.0% | 50.0% | 0 |

## Full逐对结果

| 样本编号 | 人工点数 | NM | Ground RMSE(px) | 点误差中位数(px) | 点误差P90(px) | 最大点误差(px) | Ground SR@3px |
|---|---:|---:|---:|---:|---:|---:|---|
| 043 | 12 | 11995 | 6.201 | 1.363 | 3.543 | 20.599 | 失败 |
| 045 | 6 | 11998 | 6.218 | 4.695 | 9.161 | 11.563 | 失败 |
| 078 | 7 | 11991 | 3.181 | 3.573 | 4.144 | 4.217 | 失败 |
| 092 | 6 | 11416 | 3.163 | 2.923 | 3.828 | 4.015 | 失败 |
| 093 | 12 | 11989 | 5.020 | 3.911 | 5.441 | 12.405 | 失败 |
| 388 | 12 | 11992 | 2.670 | 2.090 | 4.101 | 4.896 | 成功 |
| 468 | 11 | 11918 | 3.679 | 2.518 | 5.842 | 8.139 | 失败 |
| 478 | 12 | 11996 | 2.903 | 2.029 | 4.313 | 6.389 | 成功 |

## 权重与实验配置

- 测试时间：2026-10-19 15:51:55（北京时间）。
- 执行分支：`codex/mhi-v2`；启动时提交 `b99fe162a7435460eaebc9dd9aec63f1817263da`。当前分支的 MHI v2 尚在准备，本次评价的是冻结的 MHI v1 原始 A 组预测。
- 权重：`/home/disk1/MHINet/weight/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt`。
- 权重 SHA256：`43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4`，本次重新读取实际权重文件核验。
- 训练来源：E 组第三档权重为起点，在 MHINet_RGB_512 上完成 5 epoch（25,535/25,535 步）；训练集 40,849 对、验证集 5,106 对，只冻结 DINOv3，其余活跃模块可训练；统一初始学习率 8e-5，累积 8，有效 batch=8，seed=42，Lc+Lf+Lq+Lh0 联合监督；本权重未执行后续 JL 微调。
- 推理配置：512×512，H0 D8 mask 开启，D2 H0 warp 与对应 mask 开启，QRRU 4 轮，inference_window_chunk=1024，最多输出 12,000 对。
- 预测来源：`/home/disk1/MHINet/outputs/h0_position_ablation_512_20261013/A_GoogleEarth`，复用该目录已保存的固定模型匹配坐标和置信度。本次没有重新运行神经网络，没有训练或改权重。
- 评价：CPU；Full保留原顺序，Top-K按置信度稳定降序，不足K保留全部。仅由模型预测点使用cv2.findHomography拟合A→B H；RANSAC阈值3px、maxIters=10000、confidence=0.999、seed=0。人工点不参与筛点、拟合或调参。
- 已逐对核对Full的新H与历史A组H，8/8矩阵逐元素完全一致（最大绝对差0），确认只改变评价真值与评价位置。
- NumPy/OpenCV版本与原预测实验一致；输入图像路径、尺寸、SHA256和manifest SHA256均由评价器核验。
- 人工数据：仅纳入8个completed标注，78点，无草稿、无needs_review点；8组review.status均为unreviewed，如实作为已完成、未经独立审核的人工标注使用。另6个未完成文件未纳入。
- 八组JSON按原字节复制到annotation_snapshot，固定版本与哈希，后续继续标注不影响本次结果。

## 指标定义与解释

对第i对影像的人工点，e_ij=||project(H_est,a_ij)-b_ij||₂，Ground RMSE_i=sqrt(mean_j(e_ij²))。坐标单位为存储的512目标B图像像素。Ground SR@t=mean_i[Ground RMSE_i≤t]；拟合失败保留在分母并计失败。

平均RMSE是8对各自RMSE的算术平均；合并RMSE是78个人工点的平方误差合并后开方。本次全部32次拟合有效，误差统计无排除项。这里不使用此前论文式“失败RMSE=10”的指标。

Full在388、478两对达到3px，SR@3=2/8=25%；在078、092、388、468、478五对达到5px，SR@5=5/8=62.5%。本小样本中Full优于各Top-K的平均RMSE与SR@3，不改变正式配置，也不将8对结论外推到全部498对。

043的点误差中位数为1.363px，但最大误差20.599px，RMSE为6.201px；说明RMSE对少数大误差点敏感。此次保留全部人工点，没有因误差大而删点。

稀疏人工点只能检验这些位置的配准误差，不能给全部神经网络输出点逐一判对错，因此不补造原定义P@1/3/5或NCM。NM为模型输出数。092的A侧人工点集中在较窄区域（包围框约占全图10.63%），它尤其不能代表全图。

## 与名义合成几何的诊断对照

| 样本编号 | 预测H在人工点上的RMSE(px) | 原名义合成H在同一人工点上的RMSE(px) | 历史全图网格RMSE(px) |
|---|---:|---:|---:|
| 043 | 6.201 | 6.311 | 3.964 |
| 045 | 6.218 | 4.515 | 2.533 |
| 078 | 3.181 | 3.443 | 1.618 |
| 092 | 3.163 | 5.081 | 2.254 |
| 093 | 5.020 | 7.130 | 4.934 |
| 388 | 2.670 | 4.037 | 2.785 |
| 468 | 3.679 | 3.044 | 2.617 |
| 478 | 2.903 | 3.882 | 2.141 |

原名义合成H在同一人工点上也有残差，这些残差混合原始跨时相对齐差异和人工定位误差，不能单凭它们判定某一来源错误。历史网格SR与本次Ground SR的真值、评价位置不同，不能把两者差值全部归因于更正标签。

## 可复现文件

- 完整运行目录：`/home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/evaluations/manual8_h0joint_A_20261019_154944`。
- `run_registration.json`：权重、分支、源预测、人工快照版本与SHA256。
- `metrics/summary.json`：按变体汇总及输入文件哈希。
- `metrics/pairs.jsonl`：32条逐对逐变体结果，含逐个人工点误差、ID、预测H和拟合内点数。
- `metrics/pairs.csv`：便于表格软件查看的逐对结果。
- `nominal_geometry_comparison.json`：与原名义合成H的独立诊断对照。
- `independent_validation.json`：独立重算核验。
- `build_report.py`：本报告生成脚本。

执行命令：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
/root/miniconda3/envs/loma-repro/bin/python -m tools.gcp_annotator.evaluate \
  --annotations /home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/evaluations/manual8_h0joint_A_20261019_154944/annotation_snapshot \
  --matches-root /home/disk1/MHINet/outputs/h0_position_ablation_512_20261013/A_GoogleEarth \
  --output /home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/evaluations/manual8_h0joint_A_20261019_154944/metrics \
  --variants Full Top-4096 Top-2048 Top-1024
```

复现时指定新的output目录，避免覆盖本次结果。
