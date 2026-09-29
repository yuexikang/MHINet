# JL1Flight：E 权重的 512 / 784 对照

同一 E 权重、1755 对测试影像（原图 640×640）、1024 推理分块，未微调。只改变模型输入尺寸；全部精度按原图像素计算，所有影像对计入宏平均。

| 指标 | 512 | 784 |
|---|---:|---:|
| P@1px | 3.46% | 4.91% |
| P@3px | 21.07% | 29.05% |
| P@5px | 37.77% | 49.63% |
| RMSE(px) | 7.648 | 7.431 |
| 平均推理(ms/对) | 167.513 | 239.001 |
| P95推理(ms/对) | 284.000 | 286.411 |
| SR@3px | 32.99% | 35.56% |
| 平均NCM@5px | 4517.3 | 5954.2 |

RMSE：配准 SR@3px 成功对统计 GT 误差≤5px 正确匹配的 RMSE，失败记10。推理时间包括共享网络和匹配头，不含读图、指标计算及结果保存；不同时间运行的单次耗时，仅代表这两次实测。

## 784 变换分组

| 分组 | 对数 | P@1px | P@3px | P@5px | RMSE(px) | SR@3px |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 351 | 4.82% | 28.69% | 49.10% | 7.584 | 33.33% |
| t1 | 351 | 4.88% | 28.98% | 49.65% | 7.313 | 37.32% |
| t2 | 351 | 4.94% | 28.98% | 49.40% | 7.399 | 35.90% |
| t3 | 351 | 4.88% | 29.09% | 50.03% | 7.656 | 32.48% |
| t4 | 351 | 5.03% | 29.49% | 49.99% | 7.201 | 38.75% |

t0–t4 表示变换编号，不是模态类别。

784 成功配准 624 对；推理状态计数：{'none': 1714, 'invalid_h0': 41}。配对 SR@3px 转移：{'512_False_784_False': 973, '512_False_784_True': 203, '512_True_784_True': 421, '512_True_784_False': 158}。

## 模块平均耗时（ms/对）

| 模块 | 512 | 784 |
|---|---:|---:|
| dino | 19.407 | 32.347 |
| mvt | 9.079 | 13.395 |
| h0 | 2.430 | 2.723 |
| vgg | 4.825 | 13.004 |
| cgmdp | 5.437 | 12.939 |
| shared_other | 0.103 | 0.116 |
| matcher_setup | 0.661 | 1.312 |
| coarse | 10.291 | 24.752 |
| fine | 13.675 | 29.207 |
| qrru | 101.537 | 109.117 |
| pipeline_other | 0.024 | 0.026 |

## 复现与结果

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/root/miniconda3/envs/loma-repro/bin/python scripts/evaluate_e_jl1flight.py \
  --checkpoint outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt \
  --manifest artifacts/jl1flight_test_manifest.jsonl --input-size 784 \
  --output outputs/jl1flight_e_784_rerun
```

原始结果：`outputs/jl1flight_e_784_20260929/`；完整统计：`artifacts/jl1flight_e_784_results.json`。HTML、1755 对匹配坐标及匹配图均已保存。核对两组权重、清单、GT、影像校验值及原图尺寸完全一致。
