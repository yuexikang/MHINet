**本轮结论：WHU原始val/test共3452张影像具有继续核查的价值。现存训练和评估清单未引用它们的完整路径，作者也提供了空间区域划分证据；但本地编号到母图的映射尚未确认，不能将其宣布为合格的地理隔离开发集。** 官方边界文件请求超时，故status=limited；没有独立验证改进，breakthrough_verified=false。

**1．范围与变化检查。** 本轮为2026-10-13北京时间08时研究，已读state、首轮综合、最近06时报告及必要专题，延续WHU资格审计。当前训练记录中的12项downstream实现、4项loader/runtime代码及train/val清单，共18项SHA256全部匹配；额外核验GHIM loss亦一致。联合配置一致，tracked diff为空，训练run顶层文本文件无前轮以来更新。诊断设置OMP/MKL/OpenBLAS线程为1、禁字节码、120秒封顶；完整TIFF头检查102.15秒，清单审计24.66秒。未写文件、读取影像像素或权重、启动推理、训练和H拟合。

**2．新增文件身份与使用证据。** 令W=`/home/disk1/Data/datasets/WHU-Building`，R=`/home/disk1/Data/datasets/MHINet_RGB_512`。统计全部文件名：

| 原始集合 | 张数 | 本地编号系列 |
| --- | ---: | --- |
| train | 4736 | `0.tif…4735.tif` |
| val | 1036 | `val_0.tif…val_1035.tif` |
| test第一组 | 714 | `0.tif…713.tif` |
| test第二组 | 1702 | `2_0.tif…2_1701.tif` |

各系列编号连续、无缺号。RGB512的train/val/test分别使用3788/474/474张WHU来源，全部来自W/train/image；三者完整源路径并集恰好等于原始train的4736张。[生成器第33行](/home/disk1/MHINet/mhinet/dataio/generate_rgb512.py:33)明确限定这一入口。R/source_split_manifest与三个pairs清单的六个身份字段逐行差异0。

全读16份现存主清单，共384863行，涵盖RGB512、两代GE quadrant、合成2000对测试、已观察GE498/JL1755及JL微调数据；检查12类源图、父图、原生图和影像路径字段。WHU原始val/test完整路径交集均为0。这是记录中的路径分离，不能排除改名、未记录调用或不存在的历史材料。

发现一个容易误判的替代解释：原始test有714个basename与原始train相同，在RGB512 train/val/test分别碰撞574/62/78个。例如[R/train第37062行](/home/disk1/Data/datasets/MHINet_RGB_512/train/pairs.jsonl:37062)使用的是train/image/186.tif，而非test/image/186.tif。714组同名文件的inode及文件大小全部不同，排除了同一路径对象和字节完全相同文件，但不能证明像素内容或地理范围不同。来源身份不能只用basename或数字尾号。

**3．初始化历史的覆盖与缺口。** 当前run从E初始化，E从C初始化；继续追溯得到八个本地祖先训练阶段：前三个shared frozen阶段使用GE quadrant v1，之后三个shared stable阶段及C、E使用stable_v2。两套现存train/val清单哈希与祖先记录一致，没有WHU路径交集。[当前run第66行](/home/disk1/MHINet/outputs/semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep/run.json:66)、[E/run第55行](/home/disk1/MHINet/outputs/semidense_stable_v2_tier3_lr_e_seed0/run.json:55)可追溯接续关系。

另审阅32个现存run.json，含归档和chunk子run。旧GoogleEarth_scale_pairs、temporal4及single_parent_v2目录已不存在，只能核runtime声明，不能逐样本复核。部分祖先源码与当前版本不同，完整历史源码副本未恢复；最早底座的官方LoRetta、LoMa-B外部预训练来源也没有逐样本记录。因此只能表述为“现存证据未发现原始val/test路径被使用”，不能表述为“历次训练绝对未见”。

**4．定位元数据与实时一手补证。** 仅读8188个TIFF的首个IFD：全部512×512、3通道，均没有描述标签270、地面scale/tiepoint/transform标签33550/33922/34264、GeoKey标签34735/34736/34737或GDAL标签42112/42113。W根下未发现非TIFF文件。这扩大了此前抽查证据，但检查范围是首个IFD，没有由此恢复CRS或footprint。

检索日期2026-10-13。Shunping Ji、Shiqing Wei、Meng Lu，*Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set*，IEEE TGRS 57(1):574–586，2019，DOI 10.1109/TGRS.2018.2858817；2018年在线发表，核验版本为最终发表版。[作者机构记录](https://research-portal.uu.nl/en/publications/fully-convolutional-networks-for-multisource-building-extraction-/)

论文印刷页576说明无重叠切片、顺序编号及整体影像重建，页577图3展示空间分区。[发表版PDF](https://research-portal.uu.nl/ws/portalfiles/portal/235991427/Fully_Convolutional_Networks_for_Multisource_Building_Extraction_From_an_Open_Aerial_and_Satellite_Imagery_Data_Set.pdf) 作者官网列train为64×74、val为14×74、test为17×42与23×74，与本地四系列数量吻合。这支持作者发布时按空间区域划分，仍未证明本地扫描方向、起点和重命名映射保持原样；网页总数8189与分组之和8188亦存在一张差异。[官方数据页](https://gpcv.whu.edu.cn/data/building_dataset.html)

官网提供[split边界文件](https://gpcv.whu.edu.cn/data/shapefiles.zip)，本轮请求超时，未核CRS、坐标、区域间距或缓冲带。不能用官网split名称、0.3米名义分辨率和相符数量代替本地定位证据，更不能据此证明与已观察GE/JL零地理交叠。

**5．接回点数、质量判断与跨域问题。** 点数专题已由18时及后续报告覆盖，本轮不重复选择性轨迹统计。当前[semidense.py第189行](/home/disk1/MHINet/mhinet/downstream/semidense.py:189)组合fine概率与粗置信度，第195行截断，第211行QRRU后沿用旧分数；来源核查没有验证这一分数能否排序最终质量，也没有验证多点有害。WHU若获得资格，可提供标签明确的航空合成几何对照，用来排除部分标签残差解释；它仍不能替代真实跨时相和JL的matchability、拒绝及跨域保留验证。现行来源路径分组也没有检验WHU母图区域隔离。

**6．按证据收益与成本排序的最小后续方案。**

1. 优先仅核小型定位材料：作者边界、切片目录或重建说明，以及现有下载/解压记录。确认四编号系列对应的母图区域、编号方向、CRS和完整源footprint，再比较本地训练祖先及已观察来源。缺映射则保持资格未定，不凭数量吻合补造坐标，不下载大型母图或整包。
2. 资格成立后再提出最小完整缓存方案，本轮不执行。固定检查点、候选池和512像素约定，保存全部最终坐标、原分数、lineage、合法mask与来源哈希；诊断QRRU时加入完整fine起点。按母图/空间区域分组，合成航空与真实跨域panel分别报告。
3. 单因素比较原输出、置信top-K、同K固定多种子随机、置信分层随机及空间均衡；K示例沿用256/512/1000/2000/4000/全部，只在隔离开发集锁定。固定RANSAC假设尝试预算、阈值、配对种子与评分规则，报告P@1/3/5、NCM、NM、双侧覆盖、SR、失败率及网格RMSE分位数。P升而NCM/覆盖/SR降仅称权衡；随机也改善才支持冗余或拟合负担，只有置信筛选改善更支持质量筛选。拟合收益与推理剪枝收益分开，锁定后使用新的封存测试。

**供findings收录：WHU现存清单的路径分离与作者空间划分已有证据，但8188个裁片首个IFD均缺定位字段；714组跨split同名是明确的身份审计陷阱。开发资格仍未闭合，本轮没有新增匹配提升证据。**
