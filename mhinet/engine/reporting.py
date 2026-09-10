"""Readable real-image accuracy table derived only from measured evaluation metrics."""
from pathlib import Path


def write_accuracy_report(path, summary):
    trajectory = summary['trajectory']
    lines = ['# 真实影像精度评估', '',
             f"影像对数：{summary['pairs']}；manifest：`{summary['manifest']}`。",
             f"checkpoint角色：`{summary.get('checkpoint_role')}`；仅完成评估不等于模型验证成功。", '',
             'MACE/网格误差使用784输入像素；误差均值仅统计有效输出，成功率和AUC分母包含全部样本。', '',
             '| H | MACE均值 | 中位数 | P90 | 网格均值 | 无效率 | 成功@1/3/5 | AUC@1/3/5 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for i, label in enumerate(trajectory['labels']):
        m = trajectory['mace_input_px_conditional_valid'][i]
        g = trajectory['grid5_input_px_conditional_valid'][i]
        t = trajectory['all_pair_threshold_scores_input_px'][i]
        success = '/'.join(f"{t['success'][str(k)]:.4f}" for k in (1,3,5))
        auc = '/'.join(f"{t['auc'][str(k)]:.4f}" for k in (1,3,5))
        lines.append(f"| {label} | {m['mean']:.6f} | {m['median']:.6f} | {m['p90']:.6f} | {g['mean']:.6f} | {m['conditional_failure_rate']:.4f} | {success} | {auc} |")
    lines += ['', trajectory['auc_definition'], '',
              f"更新拒绝率：{summary['rejected_update_rate']:.6f}；最终失败率：{summary['failure_rate']:.6f}。",
              f"平均单次延迟：{summary['latency_ms_per_pair_single_pass']:.3f} ms；含I/O与绘图墙钟均摊：{summary['wall_time_ms_per_pair_including_io_visualization']:.3f} ms。",
              f"峰值allocated/reserved：{summary['peak_allocated_bytes']}/{summary['peak_reserved_bytes']} bytes。",
              '此处延迟不是统一预热和重复计时的正式性能基准。',
              '原始目标像素指标、每对影像及每轮诊断详见metrics中的JSON/CSV；test不得用于选配置。']
    path = Path(path)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return path
