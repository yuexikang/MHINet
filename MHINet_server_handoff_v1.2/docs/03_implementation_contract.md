# MHINet v1 实现接口与执行顺序

本文是后续编码的明确规范，模块名和伪代码尚未对应到已实现文件。建议在原2-step新增experiments/mhinet_v1，复用固定源提交；当前设计文档保存在PRHNet历史目录的MHINet_v1子目录。

## 1. 类与职责

| 类/模块 | 职责 | 参数归属 |
| --- | --- | --- |
| SharedFeatureProvider | 一次DINO/MVT、Stage1、VGG、DeDoDe并返回四尺度 | 复用原权重；一个owner |
| PyramidAdapter | 256→64/64/32/32，A/B共享，L2 normalize | 每尺度独立 |
| HGuidedLocalCorrelation | 完整源网格、局部候选、bilinear采样与mask | 无参数 |
| ResidualGeometryDecoder | 2K+4输入→8维四角残差 | 每尺度独立、同尺度两轮共享 |
| FourCornerUpdater | T累加、4point DLT、数值guard | 无参数 |
| MultiScaleHIterator | 按8/4/2/1每级两轮推进H/T | 持有4个decoder |
| MHINet | 封装输入、共享资源、输出与冻结策略 | 总模型 |

## 2. 张量契约

| 名称 | dtype/shape | 单位/含义 |
| --- | --- | --- |
| images | float, B×2×3×784×784 | RGB [0,1]，首版B=1 |
| H0_norm | FP32, B×3×3 | Stage1原normalized A→B |
| context | BF16/FP32, B×2×49×49×1024 | 共享MVT输出 |
| pyramid[s] | BF16/FP32, B×2×256×h_s×w_s | 累计描述子 |
| features[s] | BF16/FP32, B×2×c_s×h_s×w_s | adapter后描述子 |
| correlation | BF16/FP32, B×K_s×h_s×w_s | 每候选cosine，无softmax |
| candidate_valid | bool, B×K_s×h_s×w_s | 确定性边界/范数合法性 |
| decoder_input | float, B×(2K_s+4)×h_s×w_s | C、M、P、U依次拼接 |
| delta_px | FP32, B×4×2 | 输入目标像素位移，不是尺度像素 |
| T_norm | FP32, B×4×2 | target normalized角位移 |
| H_updates_norm | FP32, B×8×3×3 | 每次guard之后H |
| H_scales_norm | FP32, B×4×3×3 | 取updates的索引1/3/5/7 |
| proposal_Q_norm | FP32, B×8×4×2 | 训练loss所需guard前角点 |
| update_accepted | bool, B×8 | 数值/支持合法性 |

H_final_norm=H_updates_norm[:,7]。同时返回stage1_valid、overall_valid、failure_reasons和runtime diagnostics。精修被拒时保留上次合法H，因此只要初始化合法且后续保持合法，overall_valid仍可为true；另用refinement_any_accepted和更新范数区分“有H输出”与“确实有精修”。

## 3. forward伪代码

以下是算法伪代码，尚不可作为运行命令。

```text
verify RGB input and paired dimensions
context, H0, stage1_status, pyramid = feature_provider(images)
initialize normalized source control corners c
H = H0
T = project(H0, c) - c
for scale in [8, 4, 2, 1]:
    FA, FB = adapter[scale](pyramid[scale])
    for iteration in [0, 1]:
        corr, candidate_valid = local_corr(FA, FB, H, scale)
        source_positions, H_flow = geometry_channels(H, scale)
        X = concatenate(corr, candidate_valid, source_positions, H_flow)
        delta_px = decoder[scale](X)
        proposal_T = T + pixel_delta_to_target_normalized(delta_px)
        system = build_four_point_system(c, c + proposal_T)
        eligible = screen_finite_support_condition_without_grad(system, candidate_valid)
        proposal_H = differentiable_solve_only_safe_subset(system, eligible)
        accepted, reason = postcheck_projection(proposal_H, eligible)
        H = select(accepted, proposal_H, H)
        T = select(accepted, proposal_T, T)
        append proposal corners, H, accepted, reason
return H0, eight updates, four scale outputs, final H, status
```

Stage1失败样本在迭代前分流，其输出H为原占位值且overall_valid=False；上述循环针对有效样本。batch混合失败时必须按样本处理，不能因单个失败丢弃整个batch。

## 4. 冻结策略接口

model.set_training_phase(profile) 读取 configs/training_profiles.json。heads 仅训练新增层；decoder_finetune 额外解冻DeDoDe；vgg_finetune/mvt_finetune分别再解冻VGG/MVT；joint两者均解冻。DINO和Stage1 head参数始终冻结，joint下head正常参与autograd且H0不detach。VGG BN运行统计eval，affine随VGG参数组。model.eval()只改变推理行为；不能替代requires_grad或禁梯度上下文管理。

必须审查旧train()覆盖与no_grad的范围。检查同一共享MVT是否真的只forward一次、原Stage1头是否没有第二个encoder。当前descriptor.extract_pair_descriptors限制两张图，因此外层B=1先显式断言，不能靠reshape假装支持多对图像。

## 5. 数值验收

1. 与旧共享入口比较H0/context和旧D8/D2；相同checkpoint、resize、dtype与设备条件下对齐。新增D4/D1不改变原有快照。
2. adapter与decoder参数实数合计必须为2,544,160；空间尺寸与layer_shapes.csv匹配。
3. 全零FC2时，角更新严格0，H0与最终H的固定点投影差<1e-3输入px（非病态合成样本）。不使用参数矩阵逐项误差作为唯一几何判断。
4. 给已知平移/旋转/透视H，检查像素-normalized-scale往返；identity应将像素中心精确对应自身。
5. 四角TL/TR/BL/BR顺序贯穿预测、标签和DLT；若可视化多边形连线，改用TL/TR/BR/BL绘制，不能混成DLT控制点顺序。
6. 在小网格上与直接逐候选reference correlation比较，含亚像素、零范数和越界；分块与非分块一致。
7. 检查两个不同H导致不同sampling grids；第2轮实际使用第1轮输出，且保持FA/FB不变。
8. DLT梯度与小扰动有限差分检查，冻结DINO/Stage1 head参数梯度为None；joint下MVT/VGG梯度有限且路径有效；新head在第2个optimizer step后前层梯度非零。
9. 病态角点、非法分母、全无候选、NaN检查必须给出显式reason；不能把NaN零填后报告成功。
10. 完整8轮FP32/AMP forward及backward smoke、checkpoint/resume与显存profiling；未运行前不写“测试已通过”。

## 6. 实现优先级

第一批：SharedFeatureProvider、四尺度adapter和无参数几何/采样层；第二批：单尺度decoder和四角updater，验证tiny-overfit；第三批：4尺度8轮串联、完整loss；第四批：训练/验证/统计、activation checkpoint和DeDoDe/VGG/MVT分阶段联合训练。详细验收按docs/06。

暂不增加Planar、confidence head、GRU、光流输出、密集对应solver、semantic branch或额外backbone。未来扩展必须以完整v1作为固定baseline，不在实现中临时混入。

## 7. 环境输入

运行时从CLI显式提供repo/LoRetta source/checkpoint/data/output路径；脚本不写死中文路径。结构配置与运行配置分开：结构配置无悬而未决的层定义，运行配置仍需真实权重、数据和硬件信息。当前未启动训练，因此这些路径不伪造。
