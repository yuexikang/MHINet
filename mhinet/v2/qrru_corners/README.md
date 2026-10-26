# QRRU中心／四角监督对照

在 `codex/flow-d8-d2` 上独立实验，不改变主线QRRU损失默认行为。

四组：杭州中心、杭州中心+四角、RGB_512中心、RGB_512中心+四角。四角权重0或0.1，中心权重1；四轮迭代均监督，迭代权重0.8^(3-t)。杭州中心和四角均使用数据已有的不确定范围（原图像素转换到D2）；精确合成无死区。四角都是H推算的参考位置，没有新增人工真值。

冻结原Flow与刚完成杭州D8/D2训练的第11轮权重，QRRU全新初始化seed42、末层为零。只训练QRRU，包括其特征投影；不更新任何上游或匹配温度。四角监督通过独立实验标签函数启用，主线近似H默认屏蔽四角的行为保持不变。

每域254训练、64验证、64测试。杭州沿用清理后的训练/验证，测试从既有hz_test按G0—G3各16对抽取；RGB_512从既有train/dev/test按source_dataset均衡抽取（flair_rgb、googleearth_current、whu均为该数据集原有来源，未新增数据源）。随机种子42。各域本轮三划分无同源图／空间块交叉，但这些数据属于既有研究选择池，不声称此前从未使用。最终两个测试集合不用于本轮选权重。

每对训练最多128输入：最多64个实际局部D8→D2输出（前512高分中，中心误差≤12原图px、可见四角所需修正每坐标≤12px）；其余使用参考位置+高斯扰动（D2 std0.8、每坐标截断±2.4）。同域两组的过滤规则、随机查询、数据顺序完全相同；只有四角损失权重变化。

四组统一30轮，AdamW 8e-5、weight_decay0.01、累积4对、梯度裁剪1、余弦退火至1e-6。为保证预算相同不分别早停，按各自训练域验证集最终迭代P@3选best，初始不修正权重可被选中，同时保存last。不能拿不同数据域间的变化全部归因于监督差异，应重点比较同域中心／四角两组。

评估直接使用全部最多2048个实际D2候选，不按真值剔除难点，不进行QRRU自适应提前停止，也不在修正后删点。记录输入及四轮的P@1/3/5、参考可见EPE、原本≤3px但改错的比例、原本不正确但救回的比例；记录四角EPE。出界点计为不正确。QRRU不预测拒配置信度。

GPU0杭州中心，GPU1杭州四角，GPU2合成中心后接合成四角。各组命令：

```bash
python -m mhinet.v2.qrru_corners.experiment --domain hangzhou --supervision center --device cuda:0
python -m mhinet.v2.qrru_corners.experiment --domain hangzhou --supervision corners --device cuda:1
python -m mhinet.v2.qrru_corners.experiment --domain synthetic --supervision center --device cuda:2
python -m mhinet.v2.qrru_corners.experiment --domain synthetic --supervision corners --device cuda:2
```

每组保存逐轮验证、训练日志、best/last权重及最佳权重在两个域测试集上的四轮轨迹统计。训练中不使用QRRU输出重新筛选查询。
