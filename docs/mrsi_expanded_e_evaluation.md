# E组 expanded MRSI 全量真值评估

原始MRSIDatasets的46对恒等映射评估按用户要求放弃。其输出保留ABANDONED.json标记及代码快照，不作为本报告结果。

## 固定输入与执行

- 数据：`/home/disk1/Data/datasets/MRSI/expanded_MRSI_dataset`，600对，6个模态目录各100对。每场景00.PNG为源，01至10.PNG为目标；00_.PNG仅作辅助参考。
- 真值：目标影像同名.npy中的原图像素3×3单应矩阵，方向source→target。与remote_archive归档0to1矩阵核对，600个矩阵最大元素绝对差约5e-10（文本精度）。不使用原始46对的恒等映射假设。
- 权重：`outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt`，完整E最终权重；DINO依赖SHA核验、模型strict加载；不在MRSI上训练或选权重。
- 实现：正式推理1024版本`7cddb2a`，独立detached worktree加载；当前主工作区保留main，不合并实验训练分支。main本身没有半密集QRRU模型代码。
- 推理：D8→H残差D2→4轮QRRU，inference_window_chunk=1024，固定输入784，坐标映射回各原图像素；预测不读取GT。
- GPU1，固定权重、seed0；首对额外预热，随后模块CUDA event计时，排除读图、传输前等待、GT评分、RANSAC和绘图IO。

## 指标口径

1. **P@1/3/5px**：GT投影在有效目标范围、预测坐标有效且误差≤阈值的匹配数/全部输出匹配数。空输出为0。总体按600对宏平均，分模态按各100对宏平均，另存点数加权micro结果。
2. **匹配RMSE**：每对全部有限、可进行GT投影的匹配的sqrt(mean(||H_GT(a)-b||²))，再逐对宏平均。包含误匹配，不只算正确子集或RANSAC内点。AEPE也报告。无匹配的对不赋0，报告有效对数及失败率。
3. **配准SR@1/3/5px**：仅由预测匹配用OpenCV RANSAC估计H（阈值3px、最多10000次、confidence0.999、seed0）；在20×20均匀源网格中保留GT投影可见的点，与GT比较网格配准RMSE。RMSE≤阈值即成功；拟合失败作为失败计入全体pair分母。另报配准RMSE及有效对数。这不是RANSAC内点自洽误差。
4. **NCM@1/3/5px**：正确匹配数量的逐对均值；补充至少20个5px正确匹配的匹配成功率。
5. 补充平均输出匹配数、推理失败率、配准拟合失败率、推理均值/P95。coarse/fine/final分别保存，主表使用final。

## 复现

```bash
# 仅在该目录尚不存在时执行；不改变当前main检出。
git worktree add --detach /tmp/mhinet-mrsi-e-eval 7cddb2a
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /root/miniconda3/envs/loma-repro/bin/python -u scripts/evaluate_e_mrsi_gt.py \
 --implementation-root /tmp/mhinet-mrsi-e-eval \
 --checkpoint /home/disk1/MHINet/outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt \
 --manifest artifacts/mrsi_expanded_e_manifest.jsonl \
 --output outputs/mrsi_expanded_e_gt_20260929
```

输出目录拒绝覆盖；每对坐标/confidence保存为NPZ，预测与指标保存pairs.jsonl；每张影像、每个GT、checkpoint、代码和manifest均记录SHA。report.json保存总体/分模态/分阶段统计；index.html展示结果和全部匹配图。

数学检查：非恒等投影真值方向、固定2px误差阈值、RMSE保留离群点、失败进入SR分母、不等尺寸不自动缩放等5项测试通过。
