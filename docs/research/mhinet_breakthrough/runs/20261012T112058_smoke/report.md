本地读取：成功，只读打开指定报告，未修改任何文件。H0joint 最终指标如下（阈值单位为 px）：

| 数据集 | P@1 | P@5 | SR@3 |
|---|---:|---:|---:|
| GE（GoogleEarth） | 8.25% | 53.42% | 47.59% |
| JL（JL1Flight） | 7.10% | 59.51% | 43.65% |

实时网络：成功，已实时搜索并打开 LoRetta 官方代码页面：https://github.com/yusiwei77-star/LoRetta/ 。可核验事实：该仓库 README 明确说明，LoRetta 先以仿射先验定位可匹配的重叠区域，再在对齐坐标框架内细化稠密对应关系。([github.com](https://github.com/yusiwei77-star/LoRetta/))
