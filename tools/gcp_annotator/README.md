# GoogleEarth 地面同名点标注工具

独立的 FastAPI + 原生 HTML/CSS/JavaScript 工具，代码全部位于本目录。只适配 GoogleEarth，不加载模型、不训练、不自动匹配、不修改原始影像、测试清单、GT 或实验结果。

默认标注数据目录为：

```text
/home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/
```

遵循本次要求，不使用任务模板中的仓库 `outputs/` 作为默认保存目录。

## 仓库读取结论

- GoogleEarth：`/home/disk1/Data/datasets/Test_Data/GoogleEarth/val/pairs.jsonl`，498 对。清单 SHA256 为 `55088ef26b24170cad2da2feb7917bcc3e2051e94b576a48a9018ecd12872b51`。
- 读取 `image_A/image_B` 时相对于 **manifest 所在目录** `val/`，不是凭文件名配对。A 来自 `target_path`，B 来自 `source_path`。
- 生成实现为 `mhinet/dataio/generate_googleearth_temporal_val.py`：A 是完整 target，B 是 source 的随机凸四边形到 512×512 的变换；`T_source_to_A/B` 是生成阶段的原像素中心到测试像素中心的变换。A 的矩阵虽然名为 source，实际上应用于同尺寸 target。
- 原始 1080×1080 target/source 均存在，因此默认直接显示变形前的清晰原始影像，不对裁剪后的图像反复插值。原貌预览没有被 H0、预测 H 或配准模型对齐。
- JL1Flight 已核对：实际清单是 `.../JL1Flight/test/pairs.jsonl`，1755 对，含 `transform_group=t0..t4`、`native_image_A/B` 和各侧 native-to-output 矩阵；本次按要求不提供 JL1Flight 标注适配。
- 原有 `scripts/mrsi_gt_metrics.py::score_matches` 使用预测点进行 RANSAC（3 px、10000 iterations、confidence=.999、seed=0），随后用合成 H 比较网格。**本工具的评价只保留相同拟合步骤，以独立人工地面点计算误差，不沿用合成 H 网格真值。**
- `scripts/evaluate_topk_correspondence_ablation.py` 的 Full 保留原顺序，Top-K 对置信度降序稳定排序；本工具沿用该策略。

## 环境和启动

Python ≥3.10。服务器不需要 X11、桌面或 GPU。推荐在已有包含 NumPy、Pillow、OpenCV 的环境中，仅补充缺少的 Web 依赖：

```bash
cd /home/disk1/MHINet
python -m pip install 'fastapi>=0.100' 'uvicorn>=0.23'
python -m tools.gcp_annotator.app --port 8765
```

需要独立环境时：

```bash
cd /home/disk1/MHINet
python -m venv tools/gcp_annotator/.venv
source tools/gcp_annotator/.venv/bin/activate
python -m pip install -r tools/gcp_annotator/requirements.txt
python -m tools.gcp_annotator.app --port 8765
```

已安装 `opencv-python` 时不需要再安装 `opencv-python-headless`；requirements 供空环境使用，两者择一即可。依赖不包含 Torch、前端打包器或数据库。

本地电脑建立 SSH 转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 用户名@服务器地址
```

随后浏览器访问 [标注工具](http://127.0.0.1:8765)。服务默认监听 `127.0.0.1`，可用 `--host/--port` 调整。停止服务使用 Ctrl+C，自动保存的数据可在下次继续。

所有可选参数：

```bash
python -m tools.gcp_annotator.app --help
```

## 标注操作

1. 默认打开 seed=42 的固定 20 对试标注清单。左侧可切换全部 498 对、搜索 pair_id、按状态筛选；显示当前范围内的完成/标注中/跳过/未开始数量。
2. 默认“原貌预览”。黄色边界是**此侧**测试图的有效覆盖范围，阴影部分不接受标注；两侧边界由各自的生成矩阵独立得到，不表示地面真值已经对齐。请在两侧都能看到的稳定地面位置标注。
3. 先左键点 A，再点 B，生成一组 `GCP-001` 等。A 草稿显示为“待配对”，会自动保存；重新点 A 可以更换草稿。已有点附近点击用于选中，放大后更容易区分相邻点。
4. 滚轮以光标为中心缩放；空白处按住左键拖动、中键拖动或 Space+拖动平移。两幅图独立。F 恢复两幅完整视图。支持局部放大镜及十字准星。
5. 左键拖已有点可修正单侧位置。点击表格编号定位两图；数值编辑使用**测试影像坐标**，显示四位小数，内部和 JSON 保存完整浮点精度。
6. 标注者、备注、跳过原因和点的“待复核”状态都可记录。推荐 12–16 组但不强制；标注中允许任意数量及 A 草稿。完成要求至少一个完整点对、无草稿、无待复核点。跳过必须记录原因。
7. “独立审核模式”锁定坐标，逐点定位检查 A/B 并勾选“审核已看”，填写审核者后通过审核。修改坐标、点集合、复核标记或影像对状态会使已有审核失效。
8. “导出 JSONL / CSV”在服务器的 `exports/` 保存汇总并下载；仅导出**已完成**的影像对，不导出未配对草稿。允许完成但尚未审核；评价可加 `--reviewed-only` 限制。

快捷键：

| 操作 | 快捷键 |
| --- | --- |
| 保存 | Ctrl/Cmd+S（含坐标输入框内） |
| 撤销 / 重做 | Ctrl/Cmd+Z / Ctrl/Cmd+Y 或 Ctrl/Cmd+Shift+Z |
| 删除选中整组点 | Delete / Backspace |
| 取消 A 草稿 | Esc |
| 恢复完整视图 | F |
| 上一对 / 下一对 | `[` / `]` |
| 拖动平移 | Space+拖动或中键拖动 |

应选道路交叉点、地面标线、广场铺装角点等，避开楼顶、树冠、车辆、遮挡和变化区域。界面的点集包围框面积只帮助观察分散度，不是地面质量评分。

## 坐标与原貌预览的约定

保存 `points[].a/b` 是原始**测试输入影像**的像素坐标，通常为 512×512。整数是像素中心，左上中心 `(0,0)`，x 右/y 下，合法范围 `[0,W-1]×[0,H-1]`，不取整。

若在原貌图中人工点击 `p_native`，则：

```text
p_test_A = project(T_source_to_A, p_native_A)
p_test_B = project(T_source_to_B, p_native_B)
```

这只是各侧图像的已知生成坐标转换。**不会把 A 点通过合成 H 生成 B 点**；A/B 两侧始终人工分别选择。JSON 同时记录逆变换得到的 `native_a/native_b`、原貌路径/尺寸、各侧矩阵和文件哈希。

浏览器以 CSS 像素做缩放/平移，独立处理 canvas 的设备像素比。`screen=(image*scale)+pan`，逆变换为 `(screen-pan)/scale`。绘图图像边缘位于 `-0.5`，使屏幕上的像素中心与数据一致。仅对矩阵运算导致的边界浮点误差使用 `1e-7` 容差，不裁剪真实越界标注。

## 抽样和扩展

首次启动生成 `selection.json`；之后始终读取原清单，重新指定 seed/sample-size 不会重抽。GoogleEarth 只有 single-quadrilateral 变换类别，按 manifest 中已有 `area_ratio_A_to_B` 分成 `[2,3)` 和 `[3,4]` 两层，seed=42 各选 10 对。抽样从不读取模型结果。

扩至 50/100 对（原有清单是新清单的完整前缀，不替换已选样本）：

```bash
python -m tools.gcp_annotator.app --extend-sample 50
# 后续可执行 --extend-sample 100
```

扩展后重启服务以刷新侧栏。原始 manifest 哈希变化时拒绝复用旧选样目录，避免配对清单悄悄变化。

## 文件保存与恢复

```text
ground_gcp_annotations/
├── selection.json                   # 固定抽样及 manifest 哈希
├── .store.lock                      # 跨进程写入锁
├── annotations/
│   ├── googleearth_eval__043.json   # 每对当前版本（访问未标注影像不会创建空文件）
│   └── googleearth_eval__043.json.bak
├── history/<pair_id>/000001.json    # 每个成功提交版本的完整快照
├── exports/ground_gcps.jsonl
├── exports/ground_gcps.csv
├── evaluations/<run>/              # 建议的独立 Ground GCP 评价输出
└── verification/<run>/             # 可选真实图像软件 QA 证据
```

写入使用同目录临时文件、文件 fsync、原子替换、父目录 fsync；保存前验证有限值、范围、完整点对、唯一 ID 和状态。版本号比较配合文件锁，避免多个标签页静默覆盖。

每次编辑约 500 ms 后自动保存，切换影像对前等待保存成功。浏览器还保留未同步修改；重开时可恢复同版本的本地草稿。如服务器已更新，会提示下载本地恢复 JSON 人工核对，不覆盖服务器。

文件损坏时停止服务，从 `.json.bak` 或 `history/<pair_id>/` 最新的完整版本复制到对应 `annotations/<pair_id>.json` 再启动。浏览器有未同步版本时先下载恢复 JSON。

服务会自动恢复服务器端草稿；撤销/重做历史保留在本次打开的影像对中，服务重启后仍有磁盘版本快照可人工恢复。

## 独立 Ground GCP 评价

先有人工作业完成的 GCP，再执行；目前交付不包含伪造的人工真值。推荐与原实验相同的 `loma-repro` 环境（已核对 NumPy 2.2.6、OpenCV 4.13.0）：

```bash
cd /home/disk1/MHINet
/root/miniconda3/envs/loma-repro/bin/python -m tools.gcp_annotator.evaluate \
  --annotations /home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/annotations \
  --matches-root /home/disk1/MHINet/outputs/h0_position_ablation_512_20261013/A_GoogleEarth \
  --output /home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/evaluations/run_001 \
  --variants Full Top-4096 Top-2048 Top-1024
```

`--annotations` 也接受导出的 JSONL 或单个 JSON。`--matches-root` 为包含 `match_index.json` 的**数据集实验根目录**，不是其 `matches/` 子目录。支持 `--match-index` 指定其他索引；无索引时扫描 NPZ 内嵌 `pair_id`，不根据文件名猜测。NPZ 必须含标量 `pair_id`、`points_a`、`points_b`、`confidence`，坐标须已在测试像素系，不能直接拿 784 输入的结果评价 512 标注。

可加 `--reviewed-only` 仅使用已审核影像对。检测到原实验 `plan.json` 时检查 NumPy/OpenCV 版本；若只是探索可显式 `--allow-version-mismatch`，报告会记录偏差，不应作为严格复现结果。

评价过程：

1. 验证人工数据类型、manifest 哈希、影像路径/尺寸/哈希、点对完整性及复核标志。
2. Full 按保存顺序使用全部预测点；Top-K 稳定降序置信度排序，同分按原顺序，N<K 使用所有可用点。
3. 用**模型匹配点**按原实验配置拟合 H，不向拟合函数传入 GCP，不使用合成 H 筛选。
4. 逐个人工点计算 `e_i=||project(H,a_i)-b_i||₂`（B 测试图像素），`Ground RMSE=sqrt(mean(e_i²))`、Median、P90。
5. 每对 `Ground SR@t = (Ground RMSE <= t)`，t=1/3/5；总体 SR 为影像对平均。
6. 缺 NPZ、匹配不足、退化 H、投影无效均记录失败，在总体 SR 分母中保留并计为 false。误差均值仅对有效对计算，报告有效/失败数量；同时提供按对平均和按点合并统计，不用任意常数替代失败误差。

输出 `summary.json`、逐对逐变体的 `pairs.jsonl` 和 `pairs.csv`。JSONL 保留逐点误差、点 ID、H、内点数量和失败原因；summary 保留所有输入文件哈希、版本、RANSAC 参数和纳入/排除范围。输出目录必须新建，不覆盖已有结果。

人工 Ground GCP 与原来的合成 H 评价是两条独立评价：前者测量人工地面位置误差，后者测量相对名义配准的生成几何误差。两者都可以报告，但不可混称同一种 GT，也不可把软件 QA 的模拟点击用于科研结果。

## 自动化测试与真实图像验证

不需要 GPU，也不需要加载 MHINet：

```bash
cd /home/disk1/MHINet
python -m pip install 'httpx>=0.24'   # 仅测试需要；已安装则跳过
python -m unittest discover -s tools/gcp_annotator/tests -v
```

测试覆盖亚像素矩阵往返、坐标校验、完整关联/草稿恢复、原子写入失败、双写者冲突、固定抽样、审核状态、RANSAC 与旧实现一致、Top-K 同分稳定顺序、Ground 公式和失败分母，并验证改变 GCP 不会改变拟合 H。

可选真实浏览器验证（Playwright 只用于开发 QA，不是工具运行依赖）：

```bash
python -m pip install playwright
python -m playwright install chromium --only-shell
python -m tools.gcp_annotator.tests.browser_smoke \
  --report-dir /home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/verification/manual_run \
  --matches-root /home/disk1/MHINet/outputs/h0_position_ablation_512_20261013/A_GoogleEarth \
  --eval-python /root/miniconda3/envs/loma-repro/bin/python
```

验证使用实际 GoogleEarth 图像、DPR=2 浏览器、真实鼠标事件、服务进程重启和实际预测 NPZ，输出截图及 `verification.json`。所有模拟点击的 JSON/历史/导出写入 `qa/browser_*` 临时目录，验证结束自动删除，不混入正式 annotations；保留的 QA 报告不是标注数据或精度结论。

极简服务器若缺浏览器共享库，可执行 `python -m playwright install-deps chromium` 补齐，仅开发 QA 需要；服务本身不依赖 Chromium、桌面库或服务器中文字体。

本次已执行 17 项自动化测试及 32 项真实图像浏览器检查，并使用 `googleearth_eval__043` 的真实保存预测验证全部四个评价变体。证据位于：

```text
/home/disk1/Data/datasets/Test_Data/GoogleEarth/ground_gcp_annotations/verification/20261019/
├── verification.json   # 检查项、源代码/原始文件哈希和环境版本
├── browser.png         # 软件自测模拟点击界面，非人工真值
└── server.log
```

## 本次新增文件

```text
tools/gcp_annotator/
├── __init__.py
├── app.py
├── core.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── static/
│   ├── index.html
│   ├── style.css
│   ├── geometry.js
│   └── app.js
└── tests/
    ├── __init__.py
    ├── helpers.py
    ├── test_core.py
    ├── test_api.py
    ├── test_evaluate.py
    └── browser_smoke.py
```

仅新增本目录下文件；原有模型、训练/测试脚本、GT、manifest 和已有实验结果均不修改。运行产生的选样、人工标注、导出、审核历史和独立评价结果保存在 GoogleEarth 数据目录下。
