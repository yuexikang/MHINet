#!/usr/bin/env python3
"""Validate and summarize the fixed-weight H0 inference-position experiment.

This CPU-only report never changes a checkpoint, prediction or test manifest.
--verify-matches independently checks correspondence-based NCM/RMSE and the
historical A correspondence files; it does not rerun a model or RANSAC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VARIANTS = {
    'A': dict(use_h0_d8_mask=True, use_h0_d2_warp=True),
    'B': dict(use_h0_d8_mask=True, use_h0_d2_warp=False),
    'C': dict(use_h0_d8_mask=False, use_h0_d2_warp=True),
    'D': dict(use_h0_d8_mask=False, use_h0_d2_warp=False),
}
COUNTS = {'GoogleEarth': 498, 'JL1Flight': 1755}
NAMES = {'A': '原始模型', 'B': '仅 D8 几何约束',
         'C': '仅 D2 几何引导', 'D': '无 H0 几何引导'}
THRESHOLDS = ('1', '3', '5')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    if actual is None or expected is None:
        require(actual is None and expected is None, label + ': undefined value differs')
        return
    require(np.isclose(actual, expected, rtol=1e-9, atol=1e-9),
            f'{label}: {actual!r} != {expected!r}')


def summarize(records):
    from scripts.mrsi_gt_metrics import aggregate
    if not records:
        return None
    result = aggregate(records)
    rmse = [r['metrics']['rmse_correct5_or_failure10'] for r in records
            if r['metrics']['rmse_correct5_or_failure10'] is not None]
    result.update(rmse_correct5_or_failure10=float(np.mean(rmse)) if rmse else None,
                  rmse_defined_pairs=len(rmse), rmse_undefined_pairs=len(records) - len(rmse))
    return result


def compare_aggregate(actual, expected, label):
    require(actual['pairs'] == expected['pairs'], label + ': pair count differs')
    for group in ('precision', 'precision_micro', 'mean_ncm', 'sr'):
        for threshold in THRESHOLDS:
            close(actual[group][threshold], expected[group][threshold], label + '/' + group + '/' + threshold)
    for key in ('mean_matches', 'matching_rmse', 'matching_rmse_pairs', 'registration_rmse',
                'registration_rmse_pairs', 'rmse_correct5_or_failure10',
                'rmse_defined_pairs', 'rmse_undefined_pairs'):
        close(actual[key], expected[key], label + '/' + key)


def verify_correspondences(path, row, record):
    """Check metrics directly from stored points, using the registered SR decision."""
    from scripts.mrsi_gt_metrics import project, conditional_rmse
    with np.load(path, allow_pickle=False) as stored:
        require(str(stored['pair_id'].item()) == row['pair_id'], f'{path}: wrong pair ID')
        a, b, confidence = [stored[key] for key in ('points_a', 'points_b', 'confidence')]
    require(a.shape == b.shape == (len(confidence), 2), f'{path}: invalid array shapes')
    metric = record['metrics']
    require(metric['matches'] == len(a), f'{path}: NM differs')
    af, bf = a.astype(np.float64), b.astype(np.float64)
    truth, projectable = project(af, row['H_A_to_B'])
    finite = np.isfinite(af).all(1) & np.isfinite(bf).all(1)
    scorable = finite & projectable
    inside = scorable & (af >= 0).all(1) & (af <= np.asarray(row['size_A']) - 1).all(1)
    inside &= (truth >= 0).all(1) & (truth <= np.asarray(row['size_B']) - 1).all(1)
    inside &= (bf >= 0).all(1) & (bf <= np.asarray(row['size_B']) - 1).all(1)
    errors = np.linalg.norm(truth - bf, axis=1)
    for threshold in THRESHOLDS:
        ncm = int(np.count_nonzero(inside & (errors <= float(threshold))))
        require(metric['ncm'][threshold] == ncm, f'{path}: NCM@{threshold} differs')
        close(metric['precision'][threshold], ncm / max(1, len(a)), f'{path}: P@{threshold}')
    matching_rmse = float(np.sqrt(np.mean(errors[scorable] ** 2))) if scorable.any() else None
    close(metric['matching_rmse'], matching_rmse, f'{path}: matching RMSE')
    if metric['sr']['3'] and metric['ncm']['5'] == 0:
        expected_rmse = None
    else:
        expected_rmse = conditional_rmse(af, bf, row['size_A'], row['size_B'], row['H_A_to_B'],
                                         success_override=metric['sr']['3'])['rmse_correct5_or_failure10']
    close(metric['rmse_correct5_or_failure10'], expected_rmse, f'{path}: paper RMSE')
    # Check the serialized fitted-H grid error without re-fitting or choosing matches.
    if metric['registration_rmse'] is not None:
        require(metric['H_estimated'] is not None, f'{path}: grid RMSE lacks fitted H')
        x, y = np.meshgrid(np.linspace(0, row['size_A'][0] - 1, 20),
                           np.linspace(0, row['size_A'][1] - 1, 20))
        grid = np.stack((x.ravel(), y.ravel()), axis=1)
        target, visible = project(grid, row['H_A_to_B'])
        visible &= (target >= 0).all(1) & (target <= np.asarray(row['size_B']) - 1).all(1)
        prediction, valid = project(grid[visible], metric['H_estimated'])
        require(valid.all() and visible.any(), f'{path}: invalid fitted-H grid')
        grid_rmse = float(np.sqrt(np.mean(np.sum((prediction - target[visible]) ** 2, axis=1))))
        close(metric['registration_rmse'], grid_rmse, f'{path}: grid RMSE')
        require(metric['sr'] == {t: grid_rmse <= float(t) for t in THRESHOLDS}, f'{path}: SR differs')
    return a, b, confidence


def delta(current, baseline):
    return dict(precision_change_pp={t: 100 * (current['precision'][t] - baseline['precision'][t]) for t in THRESHOLDS},
                sr_change_pp={t: 100 * (current['sr'][t] - baseline['sr'][t]) for t in THRESHOLDS},
                ncm_change={t: current['mean_ncm'][t] - baseline['mean_ncm'][t] for t in THRESHOLDS},
                nm_change=current['mean_matches'] - baseline['mean_matches'],
                paper_rmse_change=(current['rmse_correct5_or_failure10'] - baseline['rmse_correct5_or_failure10']
                                   if current['rmse_correct5_or_failure10'] is not None and baseline['rmse_correct5_or_failure10'] is not None else None))


def validate(root, verify_matches):
    plan = read_json(root / 'plan.json')
    evidence = read_json(root / 'validation.json')
    require(plan['scope'] == 'full_test_cohorts' and plan['limit'] == 0, 'Only full test cohorts can produce the final summary')
    require(plan['input_size'] == plan['matcher']['input_size'] == 512, 'Wrong input size')
    require(plan['inference_window_chunk'] == plan['matcher']['inference_window_chunk'] == 1024, 'Inference chunk changed')
    require(plan['variants'] == VARIANTS, 'A/B/C/D inference switches changed')
    require(set(plan['datasets']) == set(COUNTS), 'Expected both full test datasets')
    require(evidence.get('complete') and evidence.get('model_unchanged'), 'Model validation is incomplete')
    state_sha = evidence['model_state_sha256_before']
    require(state_sha == evidence['model_state_sha256_after'] == evidence['checkpoint_model_state_sha256'],
            'Parameters or buffers changed')
    require(evidence['plan_fingerprint'] == plan['fingerprint'], 'Validation plan differs')
    require(evidence['feature_hash_checked_pairs'] == sum(COUNTS.values()), 'Feature audit is not full-cohort')
    require(sha_file(plan['checkpoint']['path']) == plan['checkpoint']['sha256'], 'Checkpoint file changed')
    require(sha_file(REPO / 'scripts/mrsi_gt_metrics.py') == plan['implementation']['scripts/mrsi_gt_metrics.py'],
            'Metric implementation changed; use the registered source')
    summaries, audit = {}, dict(model_state_sha256=state_sha, checkpoint_sha256=plan['checkpoint']['sha256'],
                                feature_pairs=0, a_baseline_exact_pairs=0, verified_npz=0,
                                verified_historical_a_npz=0, datasets={})
    for dataset, count in COUNTS.items():
        entry = plan['datasets'][dataset]
        require(entry['pairs'] == entry['evaluated_pairs'] == count, 'Wrong cohort size: ' + dataset)
        manifest = Path(entry['path'])
        require(sha_file(manifest) == entry['sha256'], 'Manifest changed: ' + dataset)
        rows = read_rows(manifest)
        ids = [row['pair_id'] for row in rows]
        require(len(ids) == len(set(ids)) == count, 'Duplicate or missing IDs: ' + dataset)
        baseline = Path(entry['baseline_root'])
        require(sha_file(baseline / 'report.json') == entry['baseline_report_sha256'], 'Historical report changed: ' + dataset)
        require(sha_file(baseline / 'pairs.jsonl') == entry['baseline_pairs_sha256'], 'Historical pair stream changed: ' + dataset)
        historical_report = read_json(baseline / 'report.json')
        old_registration = historical_report['registration']
        require(historical_report.get('complete') and old_registration['checkpoint']['sha256'] == plan['checkpoint']['sha256'],
                'Historical A checkpoint differs: ' + dataset)
        require(old_registration['manifest']['sha256'] == entry['sha256'] and old_registration['manifest']['pairs'] == count,
                'Historical A cohort differs: ' + dataset)
        require(old_registration['matcher'] == plan['matcher'], 'Historical A matcher settings differ: ' + dataset)
        require([r['pair_id'] for r in read_rows(baseline / 'pairs.jsonl')] == ids, 'Historical pair order differs: ' + dataset)
        bv = evidence['baseline_validation'][dataset]
        require(bv['a_baseline_exact_pairs'] == bv['feature_hash_checked_pairs'] == count and bv['first_pair_legacy_exact'],
                'A baseline / feature equality audit is incomplete: ' + dataset)
        group_records = {}
        summaries[dataset] = {}
        for variant, flags in VARIANTS.items():
            directory = root / f'{variant}_{dataset}'
            report = read_json(directory / 'report.json')
            registration = read_json(directory / 'registration.json')
            require(report.get('complete'), 'Incomplete report: ' + str(directory))
            require(registration == report['registration'], 'Report registration mismatch: ' + str(directory))
            require(registration['variant'] == variant and registration['dataset'] == dataset and registration['flags'] == flags,
                    'Wrong experiment group: ' + str(directory))
            require(registration['checkpoint'] == plan['checkpoint'] and registration['manifest'] == entry and
                    registration['matcher'] == plan['matcher'] and registration['protocol'] == plan,
                    'Group provenance differs: ' + str(directory))
            require(registration['model_state_sha256_before'] == registration['model_state_sha256_after'] ==
                    registration['checkpoint_model_state_sha256'] == state_sha, 'Group model state changed: ' + str(directory))
            records = read_rows(directory / 'pairs.jsonl')
            require([r['pair_id'] for r in records] == ids, 'Pair stream count/order differs: ' + str(directory))
            require(len(list((directory / 'matches').glob('*.npz'))) == count, 'Correspondence inventory differs: ' + str(directory))
            pair_map = read_json(directory / 'match_index.json')
            require([r['pair_id'] for r in pair_map] == ids and [r['index'] for r in pair_map] == list(range(count)),
                    'Correspondence index differs: ' + str(directory))
            for index, (row, record) in enumerate(zip(rows, records)):
                require(record['index'] == index and record['variant'] == variant and record['flags'] == flags,
                        'Pair variant or index differs: ' + row['pair_id'])
                require(record.get('shared_features_unchanged') is True, 'Shared feature mutation: ' + row['pair_id'])
                require(set(record['feature_sha256']) == {'D8', 'D2', 'H0_norm', 'stage1_valid'}, 'Missing feature hashes')
                require(record['a_baseline_exact']['checked'] == ['points_a', 'points_b', 'confidence'], 'Missing A equality evidence')
                require(Path(record['a_baseline_exact']['old_npz']) == baseline / 'matches' / f'{index:05d}.npz',
                        'Wrong original A correspondence path')
                require(type(record['h0_valid_predicted']) is bool, 'Missing independent H0 validity')
                require(record['h0_valid_predicted'] == (record['h0_fit_valid'] and record['h0_geometry_valid']),
                        'H0 validity components differ: ' + row['pair_id'])
                if verify_matches:
                    arrays = verify_correspondences(directory / 'matches' / f'{index:05d}.npz', row, record)
                    audit['verified_npz'] += 1
                    if variant == 'A':
                        with np.load(baseline / 'matches' / f'{index:05d}.npz', allow_pickle=False) as old:
                            for key, actual in zip(('points_a', 'points_b', 'confidence'), arrays):
                                require(actual.dtype == old[key].dtype and actual.shape == old[key].shape and
                                        np.array_equal(actual, old[key]), 'A differs from original NPZ: ' + row['pair_id'] + '/' + key)
                        audit['verified_historical_a_npz'] += 1
            calculated = summarize(records)
            compare_aggregate(report['overall'], calculated, dataset + '/' + variant)
            stages = {}
            for stage in ('coarse', 'fine'):
                stage_records = [dict(record, metrics=record['stages'][stage]) for record in records]
                stages[stage] = summarize(stage_records)
                compare_aggregate(report['stages'][stage], stages[stage], dataset + '/' + variant + '/' + stage)
            invalid = [r for r in records if not r['h0_valid_predicted']]
            require(report['h0_validity']['invalid_pairs'] == len(invalid) and
                    report['h0_validity']['valid_pairs'] == count - len(invalid), 'H0 subset count differs')
            summaries[dataset][variant] = dict(name=NAMES[variant], flags=flags, overall=calculated, stages=stages,
                                               invalid_h0_pairs=len(invalid), invalid_h0_overall=summarize(invalid),
                                               report=str((directory / 'report.json').resolve()))
            group_records[variant] = records
        for index in range(count):
            reference = group_records['A'][index]
            for variant in ('B', 'C', 'D'):
                current = group_records[variant][index]
                require(current['feature_sha256'] == reference['feature_sha256'] and
                        current['image_sha256'] == reference['image_sha256'] and
                        current['h0_valid_predicted'] == reference['h0_valid_predicted'],
                        'Variants received different shared features/input/H0 validity: ' + ids[index])
        original = summaries[dataset]['A']['overall']
        compare_aggregate(historical_report['overall'], original, dataset + '/historical A')
        for variant in VARIANTS:
            summaries[dataset][variant]['delta_from_a'] = delta(summaries[dataset][variant]['overall'], original)
            changed = [(a['metrics']['sr']['3'], b['metrics']['sr']['3'])
                       for a, b in zip(group_records['A'], group_records[variant])]
            summaries[dataset][variant]['sr3_pair_transitions_from_a'] = dict(
                improved_pairs=sum(not a and b for a, b in changed), worsened_pairs=sum(a and not b for a, b in changed))
        if verify_matches:
            digest = hashlib.sha256()
            for index, row in enumerate(rows):
                digest.update((row['pair_id'] + ':' + sha_file(baseline / 'matches' / f'{index:05d}.npz') + '\n').encode())
            require(digest.hexdigest() == entry['baseline_npz_inventory_sha256'], 'Historical NPZ inventory changed: ' + dataset)
        audit['feature_pairs'] += count
        audit['a_baseline_exact_pairs'] += count
        audit['datasets'][dataset] = dict(pairs=count, manifest_sha256=entry['sha256'],
                                         all_variants_model_state_identical=True,
                                         all_pairs_features_identical=True, a_historical_baseline_exact=True,
                                         invalid_h0_pairs=summaries[dataset]['A']['invalid_h0_pairs'])
    audit['verify_matches_requested'] = verify_matches
    return dict(plan=plan, audit=audit, results=summaries)


def number(value, digits=3):
    return '无定义' if value is None else f'{value:.{digits}f}'


def percent(value):
    return f'{100 * value:.2f}%'


def table(headers, rows):
    return '<div class="table"><table><thead><tr>' + ''.join('<th>' + html.escape(str(v)) + '</th>' for v in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + \
        '</tbody></table></div>'


def export(root, result):
    plan, audit, results = result['plan'], result['audit'], result['results']
    output_rows, headline_rows, stage_rows, invalid_rows, delta_rows = [], [], [], [], []
    for dataset in COUNTS:
        for variant in VARIANTS:
            value = results[dataset][variant]
            m, d = value['overall'], value['delta_from_a']
            row = dict(dataset=dataset, variant=variant, name=NAMES[variant], pairs=m['pairs'],
                       d8_h0_mask=VARIANTS[variant]['use_h0_d8_mask'],
                       d2_h0_warp_and_mask=VARIANTS[variant]['use_h0_d2_warp'])
            for threshold in THRESHOLDS:
                row.update({f'p{threshold}': m['precision'][threshold], f'ncm{threshold}': m['mean_ncm'][threshold],
                            f'sr{threshold}': m['sr'][threshold], f'delta_p{threshold}_pp': d['precision_change_pp'][threshold],
                            f'delta_sr{threshold}_pp': d['sr_change_pp'][threshold]})
            row.update(nm=m['mean_matches'], paper_rmse=m['rmse_correct5_or_failure10'],
                       matching_rmse=m['matching_rmse'], matching_rmse_defined_pairs=m['matching_rmse_pairs'],
                       grid_rmse=m['registration_rmse'], grid_rmse_defined_pairs=m['registration_rmse_pairs'],
                       paper_rmse_undefined_pairs=m['rmse_undefined_pairs'],
                       estimated_inference_ms=m['inference_ms_mean'], invalid_h0_pairs=value['invalid_h0_pairs'])
            for stage in ('coarse', 'fine'):
                sm = value['stages'][stage]
                for threshold in THRESHOLDS:
                    row[f'{stage}_p{threshold}'] = sm['precision'][threshold]
                row[f'{stage}_sr3'] = sm['sr']['3']
                stage_rows.append([dataset, variant, 'D8 粗匹配' if stage == 'coarse' else 'D2 精匹配',
                                   *[percent(sm['precision'][t]) for t in THRESHOLDS], percent(sm['sr']['3']),
                                   number(sm['mean_ncm']['5'], 1), number(sm['mean_matches'], 1)])
            output_rows.append(row)
            headline_rows.append([dataset, variant, m['pairs'], *[percent(m['precision'][t]) for t in THRESHOLDS],
                                  *[number(m['mean_ncm'][t], 1) for t in THRESHOLDS], number(m['mean_matches'], 1),
                                  *[percent(m['sr'][t]) for t in THRESHOLDS], number(m['rmse_correct5_or_failure10'])])
            changes = value['sr3_pair_transitions_from_a']
            delta_rows.append([dataset, variant, *[f'{d["precision_change_pp"][t]:+.2f}' for t in THRESHOLDS],
                               f'{d["sr_change_pp"]["3"]:+.2f}', number(d['paper_rmse_change']),
                               changes['improved_pairs'], changes['worsened_pairs']])
            im = value['invalid_h0_overall']
            invalid_rows.append([dataset, variant, value['invalid_h0_pairs'],
                                 percent(im['precision']['5']) if im else '无样本',
                                 percent(im['sr']['3']) if im else '无样本',
                                 number(im['rmse_correct5_or_failure10']) if im else '无样本',
                                 number(im['mean_matches'], 1) if im else '无样本'])
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    descriptions = [
        '四组使用同一组完成 H0 联合训练的权重，所有参数及 BatchNorm 等缓冲区保持不变；仅改变推理中的 H0 使用位置，没有重新训练或调参。',
        '每对影像只计算一次共享特征，再交给 A/B/C/D。DINOv3、MVT、CGMDP、QRRU、匹配温度、打分公式、阈值、匹配数量上限及 1024 分块一致。候选集合和采样位置改变后，实际相似度与归一化分数数值可能改变。',
        'D2 开关同时控制 H0 的重叠支持区 mask 和 H 残差 warp。关闭后，使用预测目标 D8 格的规则 4×4 D2 子格，仅检查影像边界。因此这里消融的是 D2 的 H0 几何引导整体。',
        'A 保持原始数值路径，并逐对与之前同权重、同测试清单的结果检查 points_a、points_b、confidence 完全相同。D 完全忽略 H0 及其有效性；A/B/C 保留原有无效 H0 拒绝逻辑，另列该子集结果。',
        '所有阈值采用存储影像的 512 目标像素。P@1/3/5px、NCM@1/3/5px、NM 按全部影像对平均，无匹配计 0。SR 由预测匹配独立拟合 H，再计算 GT 可见 20×20 网格配准误差。',
        '论文 RMSE：SR@3px 成功对统计 GT 误差 ≤5px 的正确匹配，失败对记 10；成功但无正确匹配时无定义并单独计数。未封顶匹配 RMSE 与网格 RMSE 仅作参考，分别在其有定义的影像对上平均。',
        '运行时间包含共享特征耗时加各组匹配耗时，不含读取、评价、哈希和预热。由于与其他 GPU 任务并行，时间仅作记录，不用于公平速度结论。GoogleEarth 真值是在名义原始配准上的合成变换，已删除确认错误的 431、432。',
    ]
    headers = ['数据集', '组', '对数', 'P@1px', 'P@3px', 'P@5px', 'NCM@1px', 'NCM@3px', 'NCM@5px', 'NM',
               'SR@1px', 'SR@3px', 'SR@5px', '论文 RMSE']
    definition_rows = [[v + '：' + NAMES[v], '使用' if f['use_h0_d8_mask'] else '不使用',
                        '使用' if f['use_h0_d2_warp'] else '不使用',
                        '使用' if f['use_h0_d2_warp'] else '不使用'] for v, f in VARIANTS.items()]
    reference_rows = [[r['dataset'], r['variant'], number(r['matching_rmse']), r['matching_rmse_defined_pairs'],
                       number(r['grid_rmse']), r['grid_rmse_defined_pairs'], r['paper_rmse_undefined_pairs'],
                       number(r['estimated_inference_ms'], 1)] for r in output_rows]
    sections = [
        ('实验定义', ['组', 'D8 H0 Mask', 'D2 H0 Warp', 'D2 H0 支持区 Mask'], definition_rows),
        ('总体结果', headers, headline_rows),
        ('相对 A 的变化', ['数据集', '组', 'ΔP@1（百分点）', 'ΔP@3（百分点）', 'ΔP@5（百分点）',
                          'ΔSR@3（百分点）', 'Δ论文 RMSE', 'SR@3 改善对数', 'SR@3 退步对数'], delta_rows),
        ('粗细阶段', ['数据集', '组', '阶段', 'P@1px', 'P@3px', 'P@5px', 'SR@3px', 'NCM@5px', 'NM'], stage_rows),
        ('无效 H0 子集', ['数据集', '组', '对数', 'P@5px', 'SR@3px', '论文 RMSE', 'NM'], invalid_rows),
        ('参考误差与时间', ['数据集', '组', '匹配 RMSE', '有效对数', '网格 RMSE', '有效对数', '论文 RMSE 无定义对数',
                            '推理估算 ms/对'], reference_rows),
    ]
    audit_text = (f'权重 SHA256：{audit["checkpoint_sha256"]}；参数及缓冲区 SHA256：{audit["model_state_sha256"]}；'
                  f'全部 {audit["feature_pairs"]} 对输入特征审计一致；A 的 {audit["a_baseline_exact_pairs"]} 对旧结果完全相同。'
                  f'独立核验新 NPZ：{audit["verified_npz"]} 份；重新检查历史 A NPZ：{audit["verified_historical_a_npz"]} 份。')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>H0 推理位置消融</title>' \
           '<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;line-height:1.6;color:#182238}' \
           'h1,h2{line-height:1.3}.table{overflow-x:auto;margin-bottom:24px}table{border-collapse:collapse;white-space:nowrap}' \
           'td,th{padding:8px 12px;border-bottom:1px solid #d9dfe8;text-align:right}th{background:#eef2f7}' \
           'td:first-child,th:first-child{text-align:left}img{width:min(100%,1100px)}code{overflow-wrap:anywhere}.audit{background:#edf7ef;padding:14px}</style>' \
           '<h1>固定权重：H0 推理位置消融</h1>' + ''.join('<p>' + html.escape(p) + '</p>' for p in descriptions)
    page += '<p class="audit">' + html.escape(audit_text) + '</p><p><a href="summary.csv">结果 CSV</a> · <a href="summary.json">完整结果及审计</a></p>'
    page += '<img src="overview.png" alt="四组 P@5px 与 SR@3px 对比">'
    markdown = '# 固定权重：H0 推理位置消融\n\n' + '\n\n'.join(descriptions) + '\n\n' + audit_text + '\n\n'
    for title, columns, cells in sections:
        page += '<h2>' + html.escape(title) + '</h2>' + table(columns, cells)
        markdown += '## ' + title + '\n\n| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
        markdown += ''.join('| ' + ' | '.join(map(str, values)) + ' |\n' for values in cells) + '\n'
    page += '<h2>原始结果</h2><ul>' + ''.join(
        f'<li><a href="{variant}_{dataset}/report.json">{variant} / {dataset}</a></li>'
        for dataset in COUNTS for variant in VARIANTS) + '</ul></html>'
    (root / 'summary.html').write_text(page)
    (root / 'summary.md').write_text(markdown)
    plot(root, results)


def plot(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(4)
    for ax, dataset in zip(axes, COUNTS):
        for offset, (key, threshold, label, color) in zip((-.19, .19),
                (('precision', '5', 'P@5px', '#3d77b9'), ('sr', '3', 'SR@3px', '#c67c35'))):
            values = [100 * results[dataset][v]['overall'][key][threshold] for v in VARIANTS]
            bars = ax.bar(x + offset, values, .36, label=label, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set(title=dataset, ylabel='Percent (%)', xticks=x, xticklabels=list(VARIANTS), ylim=(0, 110))
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle('Fixed weights: H0 inference-position ablation')
    fig.savefig(root / 'overview.png', dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'outputs/h0_position_ablation_512_20261013')
    parser.add_argument('--verify-matches', action='store_true', help='CPU verification of all new NCM/RMSE and original A NPZ arrays')
    args = parser.parse_args(argv)
    result = validate(args.root.resolve(), args.verify_matches)
    export(args.root.resolve(), result)
    print(json.dumps(dict(summary=str(args.root.resolve() / 'summary.html'), audit=result['audit']), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
