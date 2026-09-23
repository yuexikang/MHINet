# 半密集可视化中文界面

入口仍为 `outputs/semidense_lr_comparison.html`。中文分组报告位于各实验的 `visualizations/zh/index.html`，包含中文导航、说明、样本标题以及从原始日志重绘的中文训练/级联曲线。D8、D2、QRRU、GT、PCA、EPE、coarse、fine、loss等技术关键词保留。

历史细节PNG中的文字已经嵌入像素，且历史checkpoint没有逐步保留，因此不伪造或覆盖原始图像。中文页面为每幅图提供中文标题、图注以及英文状态的对应解释；原始PNG、NPZ、JSON、模型和训练日志保持原样。

独立生成命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/loma-repro/bin/python scripts/localize_semidense_reports.py
```

正在训练时使用 `--watch --interval 120`，每120秒按数据变化更新中文报告；A/B/C最终诊断均完成后自动退出。它不修改训练源码，不改变checkpoint的实现哈希或恢复协议。

Matplotlib需要中文字体：系统 `fonts-noto-cjk`，或用户目录 `~/.local/share/fonts/NotoSansCJKsc-Regular.otf`。本机使用Noto CJK官方字体：
`https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf`。

浏览器刷新原入口即可。中文曲线读取原始数值，不重新计算训练/匹配指标；细节图中的概率色阶、选点与坐标不变。
