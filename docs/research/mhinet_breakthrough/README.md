# MHINet 静默研究记录

## 当前主路线（2026-10-14 用户决策）

当前唯一主路线为 **D8→D2 密集匹配下游：共享 GHIM/CGMDP → D8 粗匹配 → D2 细匹配 → 4 轮 QRRU**。现有实现和配置保留 `semidense` 命名，描述半密集返回匹配；旧六轮 MHIR、零样本 P3/P6 作为历史对照保存。

路线、基线与权威文档以[当前 D8→D2 主路线](../../mainline_d8_d2.md)为准；过时内容及保留方式见[文档状态清单](../../document_status_20261014.md)。以下研究主题围绕这条路线推进，历史报告中的优先级和“下一轮”描述按其记录时间解读。

第一次阅读请先看[研究成果阅读指南](/home/disk1/MHINet/docs/research/mhinet_breakthrough/reading_guide.md)，日常看[累计研究记忆](/home/disk1/MHINet/docs/research/mhinet_breakthrough/research_memory.md)和[最新报告](/home/disk1/MHINet/docs/research/mhinet_breakthrough/latest.md)。

任务通过服务器cron运行，2026-10-12按用户要求改为**每两小时一次，北京时间偶数小时整点（00、02、04…22点）**。当时变更后的首个计划时点为2026-10-12北京时间16:00；后续运行读取已有结论和状态，推进未解问题。最新尝试、交付和研究轮数以`state.json`为准，历史计划时间不表示当前下一次运行。

最初配置时没有可调用的`automation_update`接口，因此采用服务器任务。实现使用已安装、已登录的Codex CLI，复用默认模型和推理设置，启用实时搜索。服务器、cron、登录和网络代理须可用；关机期间不执行，恢复后继续下一个计划时点。配置时已启动并核验cron进程，以下保存其设置与验证记录。

## 阅读入口

- `round_0001_synthesis.md`：2026-10-12首轮综合结论、候选突破和当时验证顺序，作为历史起点。
- `round_0001_data_results.md`、`round_0001_model_training.md`、`round_0001_literature.md`：首轮专题证据，假设及优先级须结合后续修正。
- `latest.md`：最近一次有效交付的研究报告（ok或limited，failed不替换）。
- `state.json`：最后尝试/成功时间、状态、报告路径及下一项研究。
- `runs/<北京时间戳>/`：每轮`run.json`、结构化`response.json`和`report.md`。
- `smoke_status.json`：真实本地读取+实时检索的链路验证；它不是精度实验。
- `findings.jsonl`：仅当报告标记已有独立验证支持的突破时追加索引；尚未出现时不存在。

## 执行范围与静默

模型子进程使用只读sandbox、隐藏GPU、CPU线程数1。外层脚本仅保存研究文件，不改现有实现/数据/权重；网络仅用于研究。每轮最多45分钟，进程锁防止并发，同一小时不重复尝试。模型退出、超时或响应错误会记录failed，保留上一份成功报告，下一个计划时点重新推进。终止launcher时会先停止其Codex进程组，再释放锁。

无聊天进度、无通知、无邮件。重要发现和需要用户决策的内容也只落盘，用户主动查看时再汇报。不得因周期任务自动启动训练或大型评估。

私有运行日志/代理环境保存于`/home/disk1/.mhinet-breakthrough-research`，目录权限700，代理配置权限600，未复制账户令牌。cron的标准输出和错误均重定向到该目录，避免cron邮件。每轮原始事件和错误日志也保存在该目录。

实际cron条目：

```cron
0 */2 * * * /usr/bin/nice -n 10 /usr/bin/python3 /home/disk1/MHINet/scripts/run_mhinet_breakthrough_research.py >> /home/disk1/.mhinet-breakthrough-research/launcher.log 2>&1 # mhinet-breakthrough-every-2-hours
```

保留了原有crontab其他内容；修改前备份存放私有运行目录。源脚本：`/home/disk1/MHINet/scripts/run_mhinet_breakthrough_research.py`；任务要求：`brief.md`；结构化输出：`response.schema.json`。

## 停止与恢复

在本目录创建`PAUSED`文件，下一个计划时点立即退出，不启动研究；这不打断已运行的轮次。移除该文件后，从下一个计划时点恢复。需要立即取消或彻底移除调度时，向Codex明确提出；取消只移除带`# mhinet-breakthrough-every-2-hours`的条目，保留其他任务。

## 验证记录

两次真实CLI链路验证成功：本地只读报告、实时搜索并打开作者来源、JSON结构输出和报告保存；第二次使用最小cron环境，退出码0、耗时84.27秒，验证了持久代理环境和登录。cron进程与已安装条目均已核验。

调度可靠性复核后，修复了资源读取失败留下running状态和终止launcher遗留子进程的问题。临时故障注入验证缺失brief、损坏代理JSON均记录failed；信号终止验证子进程已退出且状态记录failed。这些检查未调用模型训练或GPU。

## 2026-10-12运行核查与频率变更

2026-10-12的12、13、14点均实际触发，但最终因Codex使用额度限制未完成；15点运行于15:07:56成功保存第二轮报告。截至该次核查，共完成首轮及一轮后续研究，未证实新精度突破。第二轮核验了合成验证同点精修收益与逐轮探针的高分偏置，并将fine窗口与lineage关联、可重建覆盖率核算列为当时的下一项任务。详细尝试记录见`progress_check_20261012.json`；后续进展见累计记忆及运行状态。

该次变更将后续时点设为当天16:00、18:00、20:00等北京时间偶数小时整点，保持静默；当时未改变研究内容、既有成功报告或其他cron任务。

## 2026-10-12：新增输出点数量主题

2026-10-12按用户要求加入“输出点过多对精度的影响”，当时将候选/输出点来源、置信度、空间冗余、固定K对照与RANSAC稳定性列为下一成功轮次的重点；同时看P、NCM、覆盖、SR和误差长尾，避免将删点后的P提升误报为整体改进。后续延续累计记忆中的未解子问题，原fine/QRRU缓存诊断保留。定时频率仍为每两小时，保持静默。

## 2026-10-13：累计记忆与新旧迭代

按用户要求新增`research_memory.md`，汇总首轮与截至08时报告的稳定事实、支持/反证、已完成诊断与未解问题。每轮将记忆注入研究提示，报告明确“旧判断→新证据→保留/修正/否定→下一验证”，并返回完整更新记忆。有效交付后原子保存最新记忆，同时保留该轮快照；failed轮不覆盖。初始记忆快照在`evidence/research_memory_seed_20261013.md`。已完成的fine缓存诊断保留历史，未解决的全量坐标/真实域尾部另记，不重复从头检查。

点数问题持续与QRRU、最终质量、空间覆盖、RANSAC、数据分布、GT和跨域保留一起推敲；地理前置核查需说明通往哪个实验，连续两轮同一缺件不可得时暂存并推进其他相关问题。频率仍为每两小时，保持静默。

## 2026-10-13：下一轮精度上限与大幅提升主线

用户追加要求解释论文八九十与当前不到五十的差距。具体任务见`accuracy_ceiling_request_20261013.md`：先对齐指标，再用旧证据和一手论文推导可达性/瓶颈，输出最多三个高收益机制及最小证伪实验。参考已明确为LoRetta PCK/AUC与SwinMatcher precision/SR/RMSE，具体核验范围见下一节；不把理论/oracle或不同指标当已验证提升。此前点数与旧分数诊断保留作为子问题，频率仍每两小时，保持静默。

## 2026-10-13：参考论文已明确

用户指定LoRetta PCK/AUC和遥感SwinMatcher可见光precision/SR/RMSE。详情见`paper_reference_clarification_20261013.md`，已纳入下一轮理论上限与大幅提升任务；不再将论文名视为缺件。LoRetta定义/数值已核，SwinMatcher具体表和成功/误差集合待核。当前足量正确点成功率与严格全域配准之间的差距已由保存字段复算，证据在`evidence/paper_reference_alignment_20261013.json`。频率仍两小时静默，旧机制与反证继续保留。
