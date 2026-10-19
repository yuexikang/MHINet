# GoogleEarth 地面同名点标注工具交付结论

日期：2026-10-19。

本次软件代码已单独保存到 `main`；本分支仅补充本次交付结论，原有 Top-K 实验记录保留。

- 软件提交：[`9c283750c7fd0907c5a924ef6507bc568bafa529`](https://github.com/yuexikang/MHINet/commit/9c283750c7fd0907c5a924ef6507bc568bafa529)。
- 代码和完整使用说明：[main 上的 tools/gcp_annotator](https://github.com/yuexikang/MHINet/tree/9c283750c7fd0907c5a924ef6507bc568bafa529/tools/gcp_annotator)。共新增 17 个文件，使用 FastAPI 与原生 HTML/CSS/JavaScript，无桌面和 GPU 依赖。
- 范围：只支持 GoogleEarth 的 498 对有效测试清单，已保存 seed=42、20 对的固定分层试标注清单，后续支持扩展并保留原有样本。
- 预览：默认显示自制 H 变形前的原始影像，标出各侧测试图有效范围；A/B 地面点分别由人工选择，保存时独立换算到测试图浮点像素坐标。
- 操作：支持双图独立缩放平移、成对点击、拖点修正、坐标编辑、撤销重做、自动保存恢复、跳过原因、逐点复核及独立审核。
- 数据：统一保存到 `/home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/`；提供逐对 JSON、原子写入与历史备份，以及正式 JSONL/CSV 导出。
- 评价：人工 Ground GCP 只用于独立误差计算，不参与拟合 H；复用已保存模型 NPZ、既有 RANSAC 配置及 Full / Top-4096 / Top-2048 / Top-1024 策略，输出 Ground RMSE、Median、P90 和 SR@1/3/5。
- 验证：17 项自动化测试及 32 项真实图像浏览器检查通过；已验证草稿和服务重启恢复、坐标精度及真实预测 NPZ 的四档评价。迁移到 `main` 后再次通过全部 17 项自动化测试。
- 科研数据状态：QA 模拟点击已清理，正式标注目录为空，待人工建立地面真值；未修改模型、训练代码、原始测试数据和既有实验结果。

验证报告和软件自测截图位于：

```text
/home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/verification/20261019/
```

在仓库根目录启动：

```bash
git switch main
python -m tools.gcp_annotator.app --port 8765
```

SSH 端口转发、完整文件清单和独立评价命令见 `main:tools/gcp_annotator/README.md`。
