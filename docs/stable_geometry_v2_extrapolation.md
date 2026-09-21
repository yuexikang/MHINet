# 稳定几何 v2 全量外推检查（2026-09-20）

范围：GoogleEarth_quadrant_tiers_stable_v2 全部train/val，合计134750对。方向A→B和B→A，按实际半像素resize约定转换到784输入像素坐标。先检查四角齐次分母同号且非零，再除法；矩形内分母为线性函数，因此无内部过零。数据真值检查，不是网络预测评估。原版合成test未纳入，也没有使用test选择阈值。

| 集合 | 对数 | 任一方向角坐标绝对值>10000px | 无穷远线距离<10px | 最大角坐标绝对值 | 最小无穷远线距离 | 最大角点局部放大率 |
|---|---:|---:|---:|---:|---:|---:|
| train | 121282 | 0 | 0 | 2342.551px | 199.450px | 8.944 |
| val | 13468 | 0 | 0 | 2323.659px | 215.059px | 9.044 |

所有样本投影分母同号。仍允许正常图外投影：角坐标绝对值超过2000px的train690对、val69对。投影坐标大小不是预测误差。新协议避免旧版百万像素角投影和接近无穷远线，但并不要求四角落在另一图像内部，也不保证所有外推误差与重叠区误差一致。

控制敏感性实验：每个split选双向最大角投影最远的3对，共6对；每对正反两个方向，各100次。128个均匀网格中采样的GT几何重叠对应点，目标端高斯噪声sigma0.1px、seed0、cv2.findHomography(method=0)，不使用网络预测，也不包含遮挡mask/真实误匹配。12组实验中，重叠误差中位数0.02021–0.02254px，四角误差中位数0.04721–0.20497px，四角误差P90为0.07438–0.40258px。没有旧例中0.1px扰动导致十万像素四角误差的爆炸。只代表选定6对在该噪声条件下，不是全量扰动鲁棒性证明。

可复现命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/loma-repro/bin/python -u -m scripts.audit_stable_extrapolation --root /home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2 --output artifacts/stable_geometry_v2_extrapolation.json
```

机器报告包含分档统计、manifest SHA256、被选pair ID及每项控制实验。新增单测覆盖恒等、反向尺度及分母变号先于除法拒绝，3项通过。未训练模型、未改变损失或训练配置。结论：本版train/val未发现此前定义的极端真值外推，下一步仍需当前模型在新版val的预测基线，不能称模型已经解决所有外推错误。
