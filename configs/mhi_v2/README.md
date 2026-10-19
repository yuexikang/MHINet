# MHI v2 配置目录

当前没有可执行训练配置。研究参数初稿登记于 [experiment_plan.json](../../versions/v2/experiment_plan.json)，不能传给旧训练入口。

配置实现时默认 `input_size=512`，只允许512×512模型输入；新增 `max_output_matches=2048`，允许显式正整数覆盖。该字段只限制 QRRU 与有效性过滤后的最终输出；原 `max_matches=12000` 作为 QRRU 前候选预算单独保留，不从旧 checkpoint 把12000误继承成最终输出默认值。

逐项完成配置解析器、协议和训练组验收后，在此保存有 schema/task、数据/权重来源、训练预算、冻结策略和输出目录的独立配置。旧协议缺省值、正式 v1 配置快照均在 v1 档案中保留，不覆盖历史配置。
