# 实现路线：先正确、再可学、再联合训练

以下文件是建议新增路径，尚未实现。目标目录为仓库 `experiments/mhinet_v1/`；不改写旧实验接口。每完成一阶段在 implementation_log.md 留实际命令、提交、配置 hash、结果及失败原因。

| 阶段 | 建议文件 / 工作 | 必须交付的验收证据 | 后续门槛 |
| --- | --- | --- | --- |
| P0 资源与复用核验 | preflight.py、configs.py；定位仓库/依赖/权重/数据，固定 split | 环境版本与资源 hash；旧共享路径可 forward，记录 H0/context/D8/D2；审计源提交差异 | 无真实资源可先实现独立几何测试，不能编造 H0 baseline |
| P1 可训练共享特征 | feature_provider.py、adapters.py；导出真正 D8/D4/D2/D1 | 一份 DINO/MVT；四级 shape；旧输出对齐；冻结/解冻参数组正确 | 真 D1 未导出不能宣布完整金字塔完成 |
| P2 几何与相关性 | geometry.py、correlation.py；四角 DLT、坐标与候选采样、mask、分块 | identity/平移/透视、角点顺序、chunk reference、finite difference、非法分支安全 | geometry/corr 不过先修复，不进训练 |
| P3 更新器与完整 forward | update.py、model.py；各尺度 CNN、两轮重采样、八轮串联 | 2,544,160 新参数；zero-init投影差<1e-3px；H0到Hfinal不 detach；每轮 diagnostics | 全尺度 FP32 和 AMP forward/backward finite |
| P4 loss/训练与小样本 | losses.py、train.py、evaluate.py、tests/；训练组、记录与 resume | TINY-S/TINY-8、两条 MVT 梯度、联合VGG反传；默认L1正确；全无效batch规则 | 精确小样本 MACE<0.1px是建议诊断门槛，未达标解释并排查 |
| P5 主线训练 | 固定数据/seed/config；E00→E01，profile后再E02–E05 | 固定val的 H0/八轮/Hfinal、失败率和速度显存；保存逐pair CSV | baseline确实改善且非no-move，才扩大消融 |
| P6 机制与定稿 | 尺度/轮数/重采样/损失对照；三种子与封存test | 主表、消融表、误差轨迹、难例分组、实际运行命令与checkpoint | 不以训练集收益或单一均值替代泛化证据 |

## 必须实现的接口

结构 JSON → dataclass/schema 验证 → model 构造；runtime JSON/CLI 与结构配置分开。核心方法包括 model.set_training_phase(profile)、forward(images)、loss(outputs, H_gt, status)、evaluate(checkpoint, split_manifest)。返回字典按 docs/03，不把 failure reason 只写成日志字符串而遗漏逐样本字段。

需要真正提供并在 README 记录的 CLI：preflight、geometry/corr check、gradient audit、tiny-overfit、train、evaluate、profile、resume。当前包没有这些模型运行入口，因此不提供伪造的可运行训练命令；后续会话实现后填写。

## 训练规模推进

先固定 B=1、单卡与梯度累积，完成实际784训练显存 profile。若 D1 超显存，先使用分块采样、activation checkpoint 和更合理的张量生命周期，并实测梯度等价性。若仍无法满足硬件，应记录阻塞并提出 D8/D4/D2 运行变体；不能删除 D1 后仍命名为完整四尺度结果。MVT/VGG 联合训练显存独立评估，冻结训练可行不代表 joint 可行。

先一个种子跑通主线，再平行比较解冻策略；不依赖多卡完成第一批验证。DDP 后续需处理各 rank 有效样本不一致、同步 skip 和全局 loss 分母，不能直接复制单卡 skip 逻辑。

## 阶段完成的定义

代码存在、测试通过、可学习性通过、真实数据泛化改善是四件不同的事。交付日志逐项写证据；本计划的 NumPy 数值检查不能代替 PyTorch 梯度测试，更不能代替模型训练结果。
