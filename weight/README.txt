常用权重目录

每个权重均附同名 TXT：训练数据、训练方法、超参数、冻结策略、初始化链、进度和 SHA256。
基础权重：loretta_base.pth 与 loma_B_base.pt。
常用 E 组：semidense_stable_v2_tier3_lr_e_seed0.pt。
当前 512 训练快照：semidense_rgb512_synth_seed42_5ep.pt；具体步数见同名 TXT，不能视为自动跟随 latest.pt 的最终结果。
原 outputs/ 与运行配置不变，以保证正在运行的训练和历史来源校验可继续使用。新配置可显式引用这里的基础权重副本。
二进制权重被 Git 忽略，仅存本机；说明与索引可提交 Git。

更新副本与说明：
/root/miniconda3/envs/loma-repro/bin/python /home/disk1/MHINet/scripts/collect_common_weights.py

loretta_base.pth -> loretta_base.txt
loma_B_base.pt -> loma_B_base.txt
shared_stable_v2_full_tier1_seed0.pt -> shared_stable_v2_full_tier1_seed0.txt
shared_stable_v2_full_tier2_seed0.pt -> shared_stable_v2_full_tier2_seed0.txt
shared_stable_v2_full_tier3_seed0.pt -> shared_stable_v2_full_tier3_seed0.txt
semidense_stable_v2_tier1_lr_a_seed0.pt -> semidense_stable_v2_tier1_lr_a_seed0.txt
semidense_stable_v2_tier1_lr_b_seed0.pt -> semidense_stable_v2_tier1_lr_b_seed0.txt
semidense_stable_v2_tier1_lr_c_seed0.pt -> semidense_stable_v2_tier1_lr_c_seed0.txt
semidense_stable_v2_tier2_lr_a_seed0.pt -> semidense_stable_v2_tier2_lr_a_seed0.txt
semidense_stable_v2_tier3_lr_c_seed0.pt -> semidense_stable_v2_tier3_lr_c_seed0.txt
semidense_stable_v2_tier3_lr_d_seed0.pt -> semidense_stable_v2_tier3_lr_d_seed0.txt
semidense_stable_v2_tier3_lr_e_seed0.pt -> semidense_stable_v2_tier3_lr_e_seed0.txt
semidense_rgb512_synth_seed42_5ep.pt -> semidense_rgb512_synth_seed42_5ep.txt
