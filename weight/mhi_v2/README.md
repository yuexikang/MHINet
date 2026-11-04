# MHI-v2 权重

正式入口：`mhinet.v2.load_mhi_v2()`；配置与权重清单位于 `configs/mhi_v2/releases/`。

- `flow_pair_d2_deep_20261030.pt`：VGG/CGMDP、D2加深、局部Pair与匹配温度，epoch10。
- `qrru_local_20261104.pt`：QRRU局部精修，epoch10，定位损失＋改坏惩罚训练。
- Flow权重路径由清单指定，沿用已验收epoch4。

权重文件不提交Git。换机器时复制清单指向的文件，使用 `assets_root` 指定根目录，并提供原DINO权重及外部源码路径。
部署权重不包含优化器状态；继续训练时应使用原实验检查点。详情见[正式模型说明](../../docs/mhi_v2/flow_pair_release.md)。
