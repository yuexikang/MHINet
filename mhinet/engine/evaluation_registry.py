"""Append-preserving evaluation snapshots and one generated summary table."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

from mhinet.config import PROJECT_ROOT


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    os.replace(temporary, path)


def _record(path):
    payload = path.read_bytes()
    summary = json.loads(payload)
    if 'trajectory' not in summary or 'pairs' not in summary:
        raise ValueError(f'Not an evaluation summary: {path}')
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = summary.get('checkpoint') or {}
    metadata = checkpoint.get('metadata', {})
    provider = summary.get('build', {}).get('provider', {})
    manifest = summary.get('manifest')
    split = Path(manifest).parent.name if manifest else summary.get('split', 'unknown')
    is_training = 'validation' in path.parts
    experiment = metadata.get('experiment_id') or summary.get('checkpoint_role')
    scope = '训练中验证' if is_training else '独立评估'
    if experiment and ('smoke' in experiment or 'engineering' in experiment):
        scope += ' / 工程权重'
    mode = summary.get('mode', 'unknown')
    if mode == 'H0_only':
        scope += ' / 预训练H0'
    t = summary['trajectory']
    errors = t.get('mace_input_px_conditional_valid', [])
    final = errors[-1] if errors else {}
    scores = t.get('all_pair_threshold_scores_input_px', [])
    success = scores[-1]['success'] if scores else summary.get('all_pair_success_at_input_px', {})
    auc = scores[-1]['auc'] if scores else {}
    new_timing = 'wall_time_ms_per_pair_including_io_visualization' in summary
    return _clean({
        'id': hashlib.sha256((str(path)+':'+digest).encode()).hexdigest()[:16],
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        'source_summary': str(path), 'source_summary_sha256': digest,
        'run': path.parents[2].name if is_training else path.parents[1].name,
        'scope': scope, 'split': split, 'pairs': summary['pairs'], 'mode': mode,
        'labels': t.get('labels', []), 'trajectory_mace_px': [e.get('mean') for e in errors],
        'H0_mace_px': errors[0].get('mean') if errors else None,
        'final_mace_px': final.get('mean'), 'final_median_px': final.get('median'),
        'final_p90_px': final.get('p90'), 'success': success, 'auc': auc,
        'failure_rate': summary.get('failure_rate'), 'rejected_update_rate': summary.get('rejected_update_rate'),
        'peak_allocated_bytes': summary.get('peak_allocated_bytes'),
        'peak_reserved_bytes': summary.get('peak_reserved_bytes'),
        'latency_ms': summary.get('latency_ms_per_pair_single_pass') if new_timing else None,
        'wall_ms': summary.get('wall_time_ms_per_pair_including_io_visualization', summary.get('latency_ms_per_pair_single_pass')),
        'metrics_revision': summary.get('metrics_revision', 'legacy_lower_median'),
        'checkpoint_path': checkpoint.get('path') or provider.get('selected_stage1_checkpoint'),
        'checkpoint_sha256': summary.get('checkpoint_sha256') or checkpoint.get('sha256'),
        'weight_note': '运行摘要未记录权重hash' if not (summary.get('checkpoint_sha256') or checkpoint.get('sha256')) else '运行摘要记录的hash',
        'architecture_sha256': summary.get('build', {}).get('architecture_sha256') or metadata.get('architecture_sha256'),
        'manifest_sha256': summary.get('manifest_sha256'),
        'snapshot': summary,
    })


def _number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def _render(records, root):
    lines = ['# 评估总表', '',
        '自动登记独立评估及训练中验证；每次以原始summary路径+内容SHA去重，同一路径的新结果保留为新记录。',
        '只汇总已找到的真实影像评估summary；tiny-overfit、性能探测及单元测试不冒充精度评估。',
        '不同split、影像子集、架构及指标版本不可直接排名。误差为有效输出上的统计，必须同时看失败率。',
        '旧记录没有AUC或独立推理计时时显示“—”，不补造；旧中位数口径与新版本不同。', '',
        '| 记录/原始结果 | 类型 | split/对数 | H0均值px | 最终均值px | 中位/P90 px | 成功@1/3/5 | AUC@1/3/5 | 失败率 | 显存GB | 推理/墙钟ms | 指标版本 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in records:
        triple = lambda key: '/'.join(_number(r[key].get(str(t))) for t in (1,3,5))
        memory = None if r['peak_allocated_bytes'] is None else r['peak_allocated_bytes']/1e9
        lines.append(f"| [{r['run']} · {r['id'][:6]}]({r['source_summary']}) | {r['scope']} | {r['split']}/{r['pairs']} | {_number(r['H0_mace_px'])} | {_number(r['final_mace_px'])} | {_number(r['final_median_px'])}/{_number(r['final_p90_px'])} | {triple('success')} | {triple('auc')} | {_number(r['failure_rate'])} | {_number(memory)} | {_number(r['latency_ms'])}/{_number(r['wall_ms'])} | {r['metrics_revision']} |")
    lines += ['', '成功率/AUC/失败率以[0,1]比例表示。显存列为peak allocated，reserved和H0～各轮完整轨迹见机器可读记录。',
              '权重路径/hash、数据清单hash、原始summary快照保存在 `artifacts/evaluation_registry.json`；缺失的运行时hash保持缺失，不把当前磁盘文件hash冒充运行时证据。',
              '登记时间是入表时间，source_mtime是原文件修改时间，不冒充未知的实验开始时间。']
    _atomic_write(root/'docs/evaluation_summary.md', '\n'.join(lines)+'\n')
    table = io.StringIO()
    fields = [k for k in records[0] if k != 'snapshot'] if records else ['id']
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    for r in records:
        writer.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], (dict,list)) else r[k] for k in fields})
    _atomic_write(root/'artifacts/evaluation_registry.csv', table.getvalue())


def register_evaluation(summary_path, *, root=PROJECT_ROOT):
    root, path = Path(root), Path(summary_path).resolve()
    record = _record(path)
    database = root/'artifacts/evaluation_registry.json'
    database.parent.mkdir(parents=True, exist_ok=True)
    with (database.parent/'evaluation_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads(database.read_text())['records'] if database.exists() else []
        if not any(r['id'] == record['id'] for r in records):
            records.append(record)
        _atomic_write(database, json.dumps({'version':1, 'records':records}, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        _render(records, root)
    return record['id']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, default=PROJECT_ROOT/'outputs')
    args = parser.parse_args(argv)
    count = 0
    for path in sorted(args.scan.rglob('summary.json')):
        summary = json.loads(path.read_text())
        if 'trajectory' in summary and 'pairs' in summary:
            register_evaluation(path)
            count += 1
    print(f'Registered {count} existing summaries; table: {PROJECT_ROOT / "docs/evaluation_summary.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
