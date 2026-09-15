# 四象限三档单母图数据

每张母图7对：几何3对（比例.8/.6/.4各一），辐射2对，干扰2对。
past/current各自独立生成，区域划分沿用9:1。后两档比例按母图索引循环分配。
train计划121282对，val计划13468对。原始test500对只引用、不生成H。

四边形：四象限各一点，凸，面积>=30%，边长>=短边15%，内角25–155度。
B四边形绕母图中心随机旋转[-30,30]度，重新检查界内及象限；一次透视采样。
这是额外旋转参数，不等于总相对视角；约束拒绝采样会影响最终角度分布。
双侧几何重叠>=50%；第三档母图重叠门槛65%，遮挡后双侧共同可见>=30%。
输出A保持母图宽高，B按比例取整；不将输出分辨率比例解释为固定地物尺度比。

第二、三档两侧独立亮度/对比度/gamma [0.8,1.2]。
第三档随机A侧、B侧或双侧，每个被遮挡图一个矩形，占其面积10–30%；
黑/灰/随机颜色/随机噪声。几何H不变。
mask_A/B_overlap现在表示双方共同可见位置，兼容现有GHIM matchability/geo/cls监督。
另存mask_A/B_geometry、mask_A/B_visible，区分几何重叠与遮挡。
每对metadata记录档位、比例、采样角、四点、H、辐射、遮挡及重叠率。

生成：`bash scripts/generate_three_tiers.sh --generate`，不用GPU。
输出 `/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1`，拒绝覆盖。
进度日志 `/home/disk1/MHINet/outputs/quadrant_tiers_generation.log`。
完成时dataset_summary.json的status才为completed；中断或采样耗尽会报错，不静默缺对。
当前训练默认runtime仍为single_parent_v2；本次仅生成新数据，不启动训练。
正式训练前须切换runtime和独立输出，并按tier分别统计验证指标。
