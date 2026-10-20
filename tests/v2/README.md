# MHI v2 验证目录

当前57项 CPU 工程测试已通过。覆盖 [逐条清单](../../versions/v2/ACCEPTANCE.md) 中的迁移与坐标、B/C 全局匹配、零 adapter 等价、DGFO mask/梯度、训练组/损失/评价协议和 RGB_512 数据分组。运行方式：

```bash
python -m unittest discover -s tests/v2 -t . -v
```

v1 原测试保留原路径。v2 测试检验旧输出等价、异常回退、分块一致性及阶段所需的不变量；[真实资源/GPU 验收](../../versions/v2/evidence/real_feature_acceptance_20261020_final.json) 单独登记为 `passed`，范围为两对 val、三步 QRRU 梯度诊断，另有 Flow Head 两步训练/断点恢复烟测。它们不能替代完整训练和清单级精度评价。

正式 Flow Head 已在 GPU0 启动，C/D/E 排队，状态以 `campaign.json` 为准。首轮使用原 RGB_512 train/val、单seed42；数据检查证明源图身份交叉为0，未证明地理邻近或真正空间隔离。准确率与多seed稳定性均须有实际独立证据后再签收。

独立P6损失实验新增 objective、严格续训、点级评价与自动验收测试，目前合计85项CPU测试通过。CPU测试含合成fixture；P6真实GPU校准、断点验收与精度结论须以其自动队列产出的独立证据为准。
