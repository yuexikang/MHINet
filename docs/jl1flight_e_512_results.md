# JL1Flight test：E 权重，512 输入

完整测试 1755 对（351 个源影像，每个对应 t0–t4 五个变换）。原图 640×640，模型输入 512×512，指标在原图坐标计算。无本数据集微调；推理分块 1024。

| 变换组 | 对数 | P@1px | P@3px | P@5px | RMSE(px) | SR@3px | NCM@5px |
|---|---:|---:|---:|---:|---:|---:|---:|
| 总体 | 1755 | 3.46% | 21.07% | 37.77% | 7.648 | 32.99% | 4517.3 |
| t0 | 351 | 3.37% | 20.77% | 37.28% | 7.544 | 34.47% | 4459.8 |
| t1 | 351 | 3.44% | 20.90% | 37.56% | 7.724 | 31.91% | 4491.0 |
| t2 | 351 | 3.47% | 21.05% | 37.67% | 7.523 | 34.76% | 4502.8 |
| t3 | 351 | 3.42% | 21.13% | 38.09% | 7.925 | 29.06% | 4554.3 |
| t4 | 351 | 3.58% | 21.51% | 38.26% | 7.523 | 34.76% | 4578.8 |

RMSE 沿用指定定义：配准 SR@3px 成功时，统计 GT 误差≤5px 正确匹配的 RMSE；失败记 10；全体影像对宏平均。579 对成功，1176 对失败，其中 48 对因 H0 无效而没有输出匹配，均计入分母。t0–t4 是文件中的变换编号，不是模态类别。

平均单对模型推理 167.51 ms，中位数 162.80 ms，P95 284.00 ms。不含读图、指标拟合、可视化与文件保存。

粗匹配→精匹配→QRRU 的 P@5px：34.14%→38.00%→37.77%；对应 SR@3px：17.32%→32.54%→32.99%。本次没有同集 784 对照，无法将误差归因于 512 分辨率。

## 复现

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/root/miniconda3/envs/loma-repro/bin/python scripts/evaluate_e_jl1flight.py \
  --checkpoint outputs/semidense_stable_v2_tier3_lr_e_seed0/latest.pt \
  --manifest artifacts/jl1flight_test_manifest.jsonl --input-size 512 \
  --output outputs/jl1flight_e_512_rerun
```

结果目录：`outputs/jl1flight_e_512_20260929/`，包含 HTML、逐对指标、全部匹配 NPZ 和 1755 张匹配图。精简完整统计与运行溯源：`artifacts/jl1flight_e_512_results.json`。评估入口和配对清单保存于提交 `9ff6192`；运行注册的模型实现提交为 `4b62ea7`，脚本哈希另行记录。
