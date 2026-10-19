# MHI v2 专属算法目录

此处仅预留实现边界，算法尚未实现。顺序与验收见 [ACCEPTANCE.md](../../versions/v2/ACCEPTANCE.md)。

Flow0 接口、D8 soft prior、Direct-D8 fine matching、Discrepancy/DGFO、版本化加载及训练评价的新代码均放在此目录。可复用旧模块的稳定功能；使用 wrapper、子类或显式适配保留旧 checkpoint 参数键名，不往原 `models/`、`downstream/` 注入 v2 专属条件分支，不改写 v1 来源核或默认 CLI。
