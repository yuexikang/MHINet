#!/usr/bin/env python3
"""CPU-only report and independent audit of fixed-correspondence Top-K selection.

No neural modules are imported and no model or RANSAC fit is executed. Source
coordinates and confidence are immutable; only stored original-index selections
are inspected. --verify-selections recomputes correspondence and fitted-H metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
COUNTS = {"GoogleEarth": 498, "JL1Flight": 1755}
VARIANTS = {"Full": None, "Top-4096": 4096, "Top-2048": 2048, "Top-1024": 1024}
SELECTION_KEYS = {"Full": "full_indices", "Top-4096": "top4096_indices",
                  "Top-2048": "top2048_indices", "Top-1024": "top1024_indices"}
THRESHOLDS = ("1", "3", "5")
TRANSITIONS = ("both_success", "both_failure", "gained_success", "lost_success")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    if actual is None or expected is None:
        require(actual is None and expected is None, label + ": undefined value differs")
    else:
        require(np.isclose(actual, expected, rtol=1e-9, atol=1e-9),
                f"{label}: {actual!r} != {expected!r}")


def compare_metrics(actual, expected, label):
    """Match the runner's absolute 1e-9 Full-reproduction tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ": metric keys differ")
        for key in expected:
            compare_metrics(actual[key], expected[key], label + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ": metric shape differs")
        for index, value in enumerate(expected):
            compare_metrics(actual[index], value, label + "/" + str(index))
    elif isinstance(expected, float):
        require(np.isclose(actual, expected, rtol=0, atol=1e-9), label + ": metric value differs")
    else:
        require(type(actual) is type(expected) and actual == expected, label + ": metric value differs")


def project(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    value = np.c_[points, np.ones(len(points))] @ np.asarray(homography, dtype=np.float64).T
    valid = np.isfinite(value).all(1) & (np.abs(value[:, 2]) > 1e-10)
    output = np.full_like(points, np.nan)
    output[valid] = value[valid, :2] / value[valid, 2:]
    return output, valid


def verify_metrics(a, b, row, metric, label):
    """Recompute metrics using stored predictions/H; never select with GT or refit H."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 2,
            label + ": point shapes differ")
    require(metric["matches"] == len(a), label + ": NM differs")
    truth, projectable = project(a, row["H_A_to_B"])
    finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    scorable = finite & projectable
    inside = scorable & (a >= 0).all(1) & (a <= np.asarray(row["size_A"]) - 1).all(1)
    inside &= (truth >= 0).all(1) & (truth <= np.asarray(row["size_B"]) - 1).all(1)
    inside &= (b >= 0).all(1) & (b <= np.asarray(row["size_B"]) - 1).all(1)
    errors = np.linalg.norm(truth - b, axis=1)
    for threshold in THRESHOLDS:
        count = int(np.count_nonzero(inside & (errors <= float(threshold))))
        require(count == metric["ncm"][threshold], label + "/NCM@" + threshold)
        close(metric["precision"][threshold], count / max(1, len(a)), label + "/P@" + threshold)
    for key, expected in (("finite_matches", int(finite.sum())),
                          ("scorable_matches", int(scorable.sum())),
                          ("common_support_matches", int(inside.sum()))):
        if key in metric:
            require(metric[key] == expected, label + "/" + key)
    matching_rmse = float(np.sqrt(np.mean(errors[scorable] ** 2))) if scorable.any() else None
    close(metric["matching_rmse"], matching_rmse, label + "/matching RMSE")
    correct = inside & (errors <= 5.)
    paper_rmse = (float(np.sqrt(np.mean(errors[correct] ** 2))) if correct.any() else None) \
        if metric["sr"]["3"] else 10.
    close(metric["rmse_correct5_or_failure10"], paper_rmse, label + "/paper RMSE")
    grid_rmse = metric["registration_rmse"]
    if grid_rmse is None:
        require(not any(metric["sr"].values()), label + ": undefined grid error has successful SR")
        require(metric.get("fit_failure") is not None, label + ": missing fit-failure reason")
        return
    require(metric.get("H_estimated") is not None, label + ": grid error lacks fitted H")
    x, y = np.meshgrid(np.linspace(0, row["size_A"][0] - 1, 20),
                       np.linspace(0, row["size_A"][1] - 1, 20))
    grid = np.stack((x.ravel(), y.ravel()), axis=1)
    target, visible = project(grid, row["H_A_to_B"])
    visible &= (target >= 0).all(1) & (target <= np.asarray(row["size_B"]) - 1).all(1)
    prediction, valid = project(grid[visible], metric["H_estimated"])
    require(visible.any() and valid.all(), label + ": invalid fitted-H evaluation grid")
    expected = float(np.sqrt(np.mean(np.sum((prediction - target[visible]) ** 2, axis=1))))
    close(grid_rmse, expected, label + "/grid RMSE")
    require(metric["sr"] == {t: expected <= float(t) for t in THRESHOLDS}, label + "/SR")
    if "registration_grid_points" in metric:
        require(metric["registration_grid_points"] == int(visible.sum()), label + "/grid count")


def summarize(records):
    metrics = [record["metrics"] for record in records]

    def average(key):
        values = [m[key] for m in metrics if m.get(key) is not None]
        return float(np.mean(values)) if values else None

    result = dict(
        pairs=len(records),
        precision={t: float(np.mean([m["precision"][t] for m in metrics])) for t in THRESHOLDS},
        precision_micro={t: sum(m["ncm"][t] for m in metrics) /
                            max(1, sum(m["matches"] for m in metrics)) for t in THRESHOLDS},
        sr={t: float(np.mean([m["sr"][t] for m in metrics])) for t in THRESHOLDS},
        mean_ncm={t: float(np.mean([m["ncm"][t] for m in metrics])) for t in THRESHOLDS},
        mean_matches=average("matches"),
        rmse_correct5_or_failure10=average("rmse_correct5_or_failure10"),
        paper_rmse_defined_pairs=sum(m.get("rmse_correct5_or_failure10") is not None for m in metrics),
        paper_rmse_undefined_pairs=sum(m.get("rmse_correct5_or_failure10") is None for m in metrics),
        registration_rmse=average("registration_rmse"),
        registration_rmse_pairs=sum(m.get("registration_rmse") is not None for m in metrics),
        matching_rmse=average("matching_rmse"),
        matching_rmse_pairs=sum(m.get("matching_rmse") is not None for m in metrics))
    return result


def compare_summary(actual, expected, label):
    require(actual["pairs"] == expected["pairs"], label + ": pair count differs")
    for group in ("precision", "mean_ncm", "sr"):
        for threshold in THRESHOLDS:
            close(actual[group][threshold], expected[group][threshold], label + "/" + group + "/" + threshold)
    for key in ("mean_matches", "rmse_correct5_or_failure10", "registration_rmse"):
        close(actual[key], expected[key], label + "/" + key)
    for key in ("registration_rmse_pairs", "paper_rmse_defined_pairs", "paper_rmse_undefined_pairs"):
        if key in actual:
            close(actual[key], expected[key], label + "/" + key)


def source_directory(entry, dataset):
    root = Path(entry["source_root"]).resolve()
    return root if (root / "matches").is_dir() else root / ("A_" + dataset)


def transition(baseline, current):
    if baseline and current:
        return "both_success"
    if not baseline and not current:
        return "both_failure"
    return "lost_success" if baseline else "gained_success"


def verify_inventory(path, expected=None):
    inventory = read_json(path)
    files = inventory["files"]
    digest = canonical_sha(files)
    require(inventory["sha256"] == digest, str(path) + ": canonical inventory digest differs")
    if expected is not None:
        require(digest == expected, str(path) + ": registered inventory differs")
    for filename, expected_sha in files.items():
        require(sha_file(filename) == expected_sha, "Source file changed: " + filename)
    return files, digest


def validate(root, verify_selections):
    started = time.perf_counter()
    plan = read_json(root / "plan.json")
    validation = read_json(root / "validation.json")
    require(validation.get("complete") and validation.get("source_unchanged"),
            "Runner is incomplete or source inventory changed")
    require(set(plan["datasets"]) == set(COUNTS), "Expected both full test cohorts")
    require(plan["variants"] == VARIANTS and plan["selection_keys"] == SELECTION_KEYS,
            "Expected registered Full/4096/2048/1024 selections")
    require(plan["scope"] == "full_test_cohorts" and plan["limit"] == 0,
            "Final report requires full test cohorts")
    require(validation["plan_fingerprint"] == plan["fingerprint"], "Completion plan differs")
    require(validation["neural_inference_run"] is False and validation["model_weights_loaded"] is False and
            validation["cuda_used"] is False, "Expected cached CPU-only selection evaluation")
    require(sha_file(REPO / "scripts/mrsi_gt_metrics.py") == plan["metrics_sha256"],
            "Registered metric implementation changed")
    require(plan.get("input_size", 512) == 512, "Expected 512-coordinate evaluation")
    require(validation["total_pairs"] == sum(COUNTS.values()), "Wrong number of source pairs")
    require(validation["total_records"] == 4 * sum(COUNTS.values()), "Wrong number of group records")
    require(validation["full_baseline_reproduced_pairs"] == sum(COUNTS.values()),
            "Full baseline reproduction is incomplete")
    inventory, inventory_sha = verify_inventory(
        root / "source_inventory.json", plan["source_inventory_sha256"])
    require(validation["source_inventory_sha256_before"] ==
            validation["source_inventory_sha256_after"] == inventory_sha,
            "Source inventory differs before/after evaluation")
    selection_files, results = {}, {}
    audit = dict(complete=False, verify_selections_requested=verify_selections,
                 source_inventory_sha256=inventory_sha, verified_source_files=len(inventory),
                 source_pairs=0, group_records=0, verified_selection_npz=0,
                 verified_metric_records=0, verified_source_npz=0,
                 stable_ranking_pairs=0, nested_selection_pairs=0, datasets={})
    for dataset, count in COUNTS.items():
        entry = plan["datasets"][dataset]
        require(entry["manifest"]["pairs"] == count and entry["source_npz_count"] == count,
                "Wrong full cohort: " + dataset)
        manifest = Path(entry["manifest"]["path"]).resolve()
        require(sha_file(manifest) == entry["manifest"]["sha256"], "Manifest changed: " + dataset)
        rows = read_rows(manifest)
        ids = [row["pair_id"] for row in rows]
        require(len(ids) == len(set(ids)) == count, "Duplicate or missing pair IDs: " + dataset)
        require(all(row["size_A"] == row["size_B"] == [512, 512] for row in rows),
                "Metric coordinates are not 512: " + dataset)
        source = source_directory(entry, dataset)
        for filename, key in (("report.json", "source_report_sha256"),
                              ("pairs.jsonl", "source_pairs_sha256"),
                              ("match_index.json", "source_index_sha256"),
                              ("registration.json", "source_registration_sha256")):
            require(sha_file(source / filename) == entry[key], "Source provenance changed: " + str(source / filename))
        source_records = read_rows(source / "pairs.jsonl")
        require([record["pair_id"] for record in source_records] == ids, "Source order differs: " + dataset)
        require(len(list((source / "matches").glob("*.npz"))) == count, "Source correspondence count differs")
        group_records = {}
        results[dataset] = {}
        for variant, k in VARIANTS.items():
            directory = root / f"{variant}_{dataset}"
            report, registration = read_json(directory / "report.json"), read_json(directory / "registration.json")
            require(report.get("complete"), "Incomplete group: " + str(directory))
            require(registration["variant"] == variant and registration["k"] == k and
                    registration["dataset"] == dataset, "Wrong selection group: " + str(directory))
            require(registration["source"] == entry and registration["manifest"] == entry["manifest"],
                    "Group source provenance differs: " + str(directory))
            require(registration["source_inventory_sha256"] == inventory_sha, "Group inventory differs")
            require(registration["protocol"] == plan and registration["checkpoint"] == plan["checkpoint"] and
                    registration["metrics_sha256"] == plan["metrics_sha256"], "Group protocol differs")
            if "registration" in report:
                require(report["registration"] == registration, "Report registration differs: " + str(directory))
            records = read_rows(directory / "pairs.jsonl")
            require([record["pair_id"] for record in records] == ids, "Group pair stream/order differs: " + str(directory))
            group_records[variant] = records
            calculated = summarize(records)
            compare_summary(report["overall"], calculated, dataset + "/" + variant)
            under_k = None if k is None else sum(record["source_nm"] < k for record in records)
            changes = {t: {name: 0 for name in TRANSITIONS} for t in THRESHOLDS}
            for index, (row, record, old) in enumerate(zip(rows, records, source_records)):
                label = dataset + "/" + variant + "/" + row["pair_id"]
                require(record["index"] == index and record["variant"] == variant and record["k"] == k,
                        label + ": group/index differs")
                source_nm = old["metrics"]["matches"]
                expected_nm = source_nm if k is None else min(source_nm, k)
                require(record["source_nm"] == source_nm and record["selected_nm"] ==
                        record["metrics"]["matches"] == expected_nm, label + ": selected NM differs")
                require(record["under_k"] is (None if k is None else source_nm < k), label + ": under-K differs")
                require(record["baseline_metrics"] == old["metrics"], label + ": Full reference metrics changed")
                require(record["selection_key"] == SELECTION_KEYS[variant], label + ": selection key differs")
                require(record.get("nested_prefixes_verified") and record.get("full_baseline_reproduced"),
                        label + ": missing nested/Full evidence")
                expected_source = source / "matches" / f"{index:05d}.npz"
                require(Path(record["source_npz"]).resolve() == expected_source.resolve(), label + ": source NPZ differs")
                require(record["source_npz_sha256"] == inventory[str(expected_source.resolve())],
                        label + ": source NPZ inventory differs")
                selection_path = Path(record["selection_npz"]).resolve()
                require(selection_path == root / "selections" / dataset / f"{index:05d}.npz",
                        label + ": selection NPZ path differs")
                previous = selection_files.setdefault(str(selection_path), record["selection_sha256"])
                require(previous == record["selection_sha256"], label + ": variants reference different selections")
                for threshold in THRESHOLDS:
                    status = transition(old["metrics"]["sr"][threshold], record["metrics"]["sr"][threshold])
                    require(record["sr_transition_vs_full"][threshold] == status, label + ": SR transition differs")
                    changes[threshold][status] += 1
                    require(record["metrics"]["ncm"][threshold] <= old["metrics"]["ncm"][threshold],
                            label + ": subset has more correct matches than source")
                if variant == "Full":
                    compare_metrics(record["metrics"], old["metrics"], label + ": Full baseline")
            require(report["selection_counts"]["under_k_pairs"] == under_k,
                    "Report under-K differs: " + dataset + "/" + variant)
            require(report["sr_transitions_vs_full"] == changes,
                    "Report SR transitions differ: " + dataset + "/" + variant)
            result = dict(k=k, overall=calculated, under_k_pairs=under_k,
                          under_k_rate=None if under_k is None else under_k / count,
                          under_k_examples=[] if k is None else [
                              dict(index=record["index"], pair_id=record["pair_id"],
                                   source_nm=record["source_nm"], selected_nm=record["selected_nm"])
                              for record in records if record["under_k"]],
                          sr_transitions_from_full=changes,
                          report=str((directory / "report.json").resolve()))
            results[dataset][variant] = result
            audit["group_records"] += len(records)
        source_npz_inventory = {}
        cutoff_ties = {variant: 0 for variant in VARIANTS if variant != "Full"}
        for index, row in enumerate(rows):
            source_path = source / "matches" / f"{index:05d}.npz"
            source_sha = inventory[str(source_path.resolve())]
            source_npz_inventory[str(source_path.resolve())] = source_sha
            selection_path = root / "selections" / dataset / f"{index:05d}.npz"
            require(sha_file(selection_path) == selection_files[str(selection_path)],
                    "Selection NPZ changed: " + str(selection_path))
            if not verify_selections:
                continue
            with np.load(source_path, allow_pickle=False) as stored:
                require(str(stored["pair_id"].item()) == row["pair_id"], "Wrong source NPZ pair ID")
                a, b, confidence = [stored[key] for key in ("points_a", "points_b", "confidence")]
            require(a.shape == b.shape == (len(confidence), 2), "Invalid source point shapes")
            require(np.isfinite(confidence).all(), "Stable confidence ranking requires finite scores")
            order = np.argsort(-confidence, kind="stable").astype(np.int64)
            with np.load(selection_path, allow_pickle=False) as stored:
                require(str(stored["pair_id"].item()) == row["pair_id"], "Wrong selection pair ID")
                original = np.arange(len(confidence), dtype=np.int64)
                for key, expected in (("source_original_indices", original),
                                      ("ranked_original_indices", order),
                                      ("full_indices", original)):
                    require(stored[key].dtype == np.int64 and np.array_equal(stored[key], expected),
                            str(selection_path) + ": " + key + " differs")
                for variant, k in VARIANTS.items():
                    indices = stored[SELECTION_KEYS[variant]]
                    expected = original if k is None else order[:k]
                    require(indices.dtype == np.int64 and np.array_equal(indices, expected),
                            str(selection_path) + ": wrong stable selection " + variant)
                    if k is not None and len(order) > k and confidence[order[k - 1]] == confidence[order[k]]:
                        cutoff_ties[variant] += 1
                    verify_metrics(a[indices], b[indices], row,
                                   group_records[variant][index]["metrics"],
                                   dataset + "/" + variant + "/" + row["pair_id"])
                    audit["verified_metric_records"] += 1
            audit["verified_source_npz"] += 1
            audit["verified_selection_npz"] += 1
            audit["stable_ranking_pairs"] += 1
            audit["nested_selection_pairs"] += 1
        require(canonical_sha(source_npz_inventory) == entry["source_npz_inventory_sha256"],
                "Source NPZ inventory checksum differs: " + dataset)
        actual_selection_count = len(list((root / "selections" / dataset).glob("*.npz")))
        require(actual_selection_count == count, "Selection inventory count differs: " + dataset)
        ds_evidence = validation["datasets"][dataset]
        require(ds_evidence["pairs"] == ds_evidence["full_baseline_reproduced_pairs"] ==
                ds_evidence["nested_pairs"] == count, "Dataset completion evidence differs")
        require(ds_evidence["group_records"] == 4 * count, "Dataset group record count differs")
        for variant in VARIANTS:
            require(ds_evidence["under_k_counts"][variant] == results[dataset][variant]["under_k_pairs"],
                    "Dataset under-K evidence differs")
            results[dataset][variant]["cutoff_tie_pairs"] = None if variant == "Full" or not verify_selections \
                else cutoff_ties[variant]
            baseline = results[dataset]["Full"]["overall"]
            current = results[dataset][variant]["overall"]
            results[dataset][variant]["delta_from_full"] = dict(
                precision_pp={t: 100 * (current["precision"][t] - baseline["precision"][t]) for t in THRESHOLDS},
                sr_pp={t: 100 * (current["sr"][t] - baseline["sr"][t]) for t in THRESHOLDS},
                ncm={t: current["mean_ncm"][t] - baseline["mean_ncm"][t] for t in THRESHOLDS},
                nm=current["mean_matches"] - baseline["mean_matches"])
        # Every pair must have nondecreasing NCM as the retained prefix grows.
        for index in range(count):
            for threshold in THRESHOLDS:
                counts = [group_records[v][index]["metrics"]["ncm"][threshold]
                          for v in ("Top-1024", "Top-2048", "Top-4096", "Full")]
                require(counts == sorted(counts), "Nested prefixes violate NCM monotonicity")
        audit["source_pairs"] += count
        audit["datasets"][dataset] = dict(pairs=count, source_root=str(source),
                                          manifest_sha256=entry["manifest"]["sha256"],
                                          source_npz_inventory_sha256=canonical_sha(source_npz_inventory),
                                          selection_files=actual_selection_count,
                                          cutoff_tie_pairs=cutoff_ties if verify_selections else None)
    selections_sha = canonical_sha(selection_files)
    require(selections_sha == validation["selections_inventory_sha256"], "Selection inventory digest differs")
    declared_selections = read_json(root / "selection_inventory.json")
    require(declared_selections == dict(sha256=selections_sha, files=selection_files),
            "Declared selection-file inventory differs")
    require(len(selection_files) == sum(COUNTS.values()), "Wrong total number of selection files")
    audit.update(complete=True, selections_inventory_sha256=selections_sha,
                 validation_seconds=time.perf_counter() - started,
                 coordinates_and_confidence="Original arrays indexed without modification; source file hashes unchanged",
                 gt_selection="GT is used only for metric recomputation, never ranking or prefix selection")
    return dict(plan=plan, audit=audit, results=results)


def number(value, digits=3):
    return "无定义" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "不适用" if value is None else f"{100 * value:.2f}%"


def markdown_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + \
        "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows) + "\n"


def html_table(headers, rows):
    return '<div class="table"><table><thead><tr>' + \
        "".join("<th>" + html.escape(str(value)) + "</th>" for value in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(
            "<td>" + html.escape(str(value)) + "</td>" for value in row) + "</tr>" for row in rows) + \
        "</tbody></table></div>"


def conclusions(results):
    paragraphs = []
    for dataset in COUNTS:
        full = results[dataset]["Full"]["overall"]
        best_p = max(VARIANTS, key=lambda v: results[dataset][v]["overall"]["precision"]["3"])
        best_sr = max(VARIANTS, key=lambda v: results[dataset][v]["overall"]["sr"]["3"])
        p = results[dataset][best_p]["overall"]
        s = results[dataset][best_sr]["overall"]
        paragraphs.append(
            f"{dataset}：Full 的 P@3px 为 {percent(full['precision']['3'])}、SR@3px 为 {percent(full['sr']['3'])}。"
            f"P@3px 最高的是 {best_p}（{percent(p['precision']['3'])}），"
            f"SR@3px 最高的是 {best_sr}（{percent(s['sr']['3'])}）。")
        for variant in VARIANTS:
            if variant == "Full":
                continue
            changes = results[dataset][variant]["sr_transitions_from_full"]["3"]
            delta = results[dataset][variant]["delta_from_full"]
            paragraphs.append(
                f"{dataset} / {variant}：相对 Full，P@3px 变化 {delta['precision_pp']['3']:+.2f} 个百分点，"
                f"SR@3px 变化 {delta['sr_pp']['3']:+.2f} 个百分点；"
                f"原本失败后成功 {changes['gained_success']} 对，原本成功后失败 {changes['lost_success']} 对。")
    paragraphs.extend([
        "P 的分母随 K 缩小，P 提高不等于正确匹配数量提高；NCM 是相同预测集合子集中的绝对正确点对数，"
        "本报告同时保留 NCM、NM、SR 和未封顶网格误差，判断匹配质量与配准稳定性。",
        "这里的 confidence 来自原始粗细匹配打分，QRRU 细化后未重新计算置信度；"
        "因此排序分数与最终正确性可能存在偏差，筛去低分点也可能改变 RANSAC 的空间覆盖和拟合结果。",
        "本实验从已保存的 QRRU 后最终输出中选点，仅评价选择策略对精度及配准的影响。"
        "没有重新运行模型，不表示减少了 DINOv3、MVT、CGMDP 或 QRRU 的计算，也不支持端到端提速结论。"])
    return paragraphs


def plot(root, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = {"Full": "#b45b1d", "Top-4096": "#155e91",
              "Top-2048": "#378db6", "Top-1024": "#6fb7c5"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 10, "axes.titlesize": 12,
                         "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
        order = ("Top-1024", "Top-2048", "Top-4096", "Full")
        for row_index, dataset in enumerate(COUNTS):
            ax, bx = axes[row_index]
            nm = [results[dataset][v]["overall"]["mean_matches"] for v in order]
            p3 = [100 * results[dataset][v]["overall"]["precision"]["3"] for v in order]
            sr3 = [100 * results[dataset][v]["overall"]["sr"]["3"] for v in order]
            ax.plot(nm, p3, color="#a9b5bf", linewidth=1.2, zorder=1)
            for index, variant in enumerate(order):
                ax.scatter(nm[index], p3[index], color=colors[variant], s=95 if variant == "Full" else 58,
                           marker="*" if variant == "Full" else "o", zorder=3)
                ax.annotate(f"{variant}\n{p3[index]:.2f}%", (nm[index], p3[index]),
                            xytext=(0, 12 if index in (0, 3) else -31), textcoords="offset points",
                            ha="center", fontsize=9)
            ax.set(title=f"{dataset}: precision vs retained matches",
                   xlabel="Mean returned matches (NM)", ylabel="P@3px (%)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.margins(x=.13, y=.28)
            ax.set_ylim(max(0, min(p3) - 12), min(100, max(p3) + 12))
            bx.plot(range(4), sr3, color="#155e91", linewidth=1.8, zorder=1)
            for index, variant in enumerate(order):
                bx.scatter(index, sr3[index], color=colors[variant],
                           s=95 if variant == "Full" else 58,
                           marker="*" if variant == "Full" else "o", zorder=3)
                bx.annotate(f"{sr3[index]:.2f}%", (index, sr3[index]),
                            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
            bx.set(title=f"{dataset}: registration vs match budget",
                   xlabel="Retained-match budget K (Full = all original matches)",
                   ylabel="SR@3px (%)", xticks=range(4),
                   xticklabels=("1024", "2048", "4096", "Full"))
            bx.set_ylim(max(0, min(sr3) - 8), min(100, max(sr3) + 10))
            bx.margins(x=.13)
            for axis in (ax, bx):
                axis.set_axisbelow(True)
                axis.grid(alpha=.22, linewidth=.6)
        fig.suptitle("Confidence-ranked subsets of fixed predicted correspondences", fontsize=14)
        fig.savefig(root / "overview.png", dpi=220)
        plt.close(fig)


def export(root, result, markdown_output=None):
    plan, audit, results = result["plan"], result["audit"], result["results"]
    csv_rows, headline, selection_rows, delta_rows = [], [], [], []
    for dataset in COUNTS:
        for variant, k in VARIANTS.items():
            value, base = results[dataset][variant], results[dataset]["Full"]["overall"]
            m, delta = value["overall"], value["delta_from_full"]
            transitions = value["sr_transitions_from_full"]["3"]
            row = dict(dataset=dataset, variant=variant, k=k, pairs=m["pairs"])
            for threshold in THRESHOLDS:
                row.update({f"p{threshold}": m["precision"][threshold],
                            f"ncm{threshold}": m["mean_ncm"][threshold],
                            f"sr{threshold}": m["sr"][threshold],
                            f"delta_p{threshold}_pp": delta["precision_pp"][threshold],
                            f"delta_sr{threshold}_pp": delta["sr_pp"][threshold]})
            row.update(nm=m["mean_matches"], paper_rmse=m["rmse_correct5_or_failure10"],
                       paper_rmse_defined_pairs=m["paper_rmse_defined_pairs"],
                       paper_rmse_undefined_pairs=m["paper_rmse_undefined_pairs"],
                       grid_rmse=m["registration_rmse"],
                       grid_rmse_defined_pairs=m["registration_rmse_pairs"],
                       matching_rmse=m["matching_rmse"],
                       matching_rmse_defined_pairs=m["matching_rmse_pairs"],
                       under_k_pairs=value["under_k_pairs"], under_k_rate=value["under_k_rate"],
                       cutoff_tie_pairs=value["cutoff_tie_pairs"],
                       **{f"sr3_{name}": transitions[name] for name in TRANSITIONS})
            csv_rows.append(row)
            headline.append([dataset, variant, m["pairs"],
                             *[percent(m["precision"][t]) for t in THRESHOLDS],
                             *[number(m["mean_ncm"][t], 1) for t in THRESHOLDS],
                             number(m["mean_matches"], 1),
                             *[percent(m["sr"][t]) for t in THRESHOLDS],
                             number(m["rmse_correct5_or_failure10"]),
                             number(m["registration_rmse"]), m["registration_rmse_pairs"]])
            selection_rows.append([dataset, variant,
                                   "不适用" if value["under_k_pairs"] is None else value["under_k_pairs"],
                                   percent(value["under_k_rate"]),
                                   "不适用" if k is None else value["cutoff_tie_pairs"]
                                   if value["cutoff_tie_pairs"] is not None else "未独立核验",
                                   m["paper_rmse_defined_pairs"], m["paper_rmse_undefined_pairs"],
                                   transitions["both_success"], transitions["both_failure"],
                                   transitions["gained_success"], transitions["lost_success"]])
            delta_rows.append([dataset, variant, *[f"{delta['precision_pp'][t]:+.2f}" for t in THRESHOLDS],
                               *[f"{delta['sr_pp'][t]:+.2f}" for t in THRESHOLDS],
                               number(m["mean_ncm"]["5"] - base["mean_ncm"]["5"], 1),
                               number(m["mean_matches"] - base["mean_matches"], 1)])
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    headers = ["数据集", "组", "对数", "P@1px", "P@3px", "P@5px", "NCM@1px", "NCM@3px", "NCM@5px", "NM",
               "SR@1px", "SR@3px", "SR@5px", "论文RMSE", "未封顶网格RMSE", "网格误差有效对数"]
    sections = [
        ("总体结果", headers, headline),
        ("选点与逐对成功变化",
         ["数据集", "组", "不足K对数", "不足K比例", "K边界同分对数", "论文RMSE有效对数", "论文RMSE无定义对数",
          "Full成功仍成功", "Full失败仍失败", "Full失败后成功", "Full成功后失败"], selection_rows),
        ("相对Full的变化",
         ["数据集", "组", "ΔP@1百分点", "ΔP@3百分点", "ΔP@5百分点",
          "ΔSR@1百分点", "ΔSR@3百分点", "ΔSR@5百分点", "ΔNCM@5", "ΔNM"], delta_rows)]
    methods = [
        "固定同一组已完成 H0 联合训练的权重，使用原 A 组最终 QRRU 后保存的匹配坐标及 confidence；"
        "所有 DINOv3、MVT、CGMDP、匹配温度与 QRRU 参数不变，模型没有重跑。",
        "源 A 的推理设置：D8 H0 Mask 开启、D2 H0 Warp 及 D2 H0 支持区 Mask 开启，保留完整 QRRU 细化；"
        "原输出上限为12000点。Full 保留该原输出全部点，其余预算固定为4096、2048、1024。",
        "Full 保留原输出的全部点和原顺序。Top-4096、Top-2048、Top-1024 分别按 confidence 稳定降序取前 K 个；"
        "同分按原始索引保持顺序，原输出不足 K 时保留全部点，不补点、不改变坐标或 confidence。",
        "各 Top-K 是同一置信度排序的嵌套前缀。GT 不参与排序、选点或 H 拟合，"
        "仅在选点完成后用于评价。原预测坐标按原样保留，未额外裁剪边缘点。",
        "GoogleEarth 为 498 对（已剔除确认错误的 431、432），JL1Flight 为 1755 对；"
        "输入和全部误差阈值采用存储影像的 512 坐标。共 8 组、2253 个源影像对、9012 条评价记录。",
        "P@t=NCM@t/NM，无输出计0；NCM 为 GT 可见支持区内误差≤t的正确匹配数量。"
        "P、NCM、NM、SR 按全部影像对宏平均，不足 K 比例以全部影像对为分母，Full 不适用。",
        "每个选点集合单独使用 OpenCV RANSAC 拟合 H（阈值3px，最多10000次，confidence=0.999，seed=0），"
        "再对 GT 可见20×20源网格计算配准误差。SR@1/3/5px 为该误差≤阈值的全对成功比例，拟合失败计失败。",
        "论文RMSE：沿用 SR@3px 判断成功，成功对统计≤5px正确匹配的点误差RMSE，失败记10；"
        "成功但无正确点时无定义并单独计数。未封顶网格RMSE在拟合和网格投影均有效的影像对上平均，"
        "另报有效对数，不将拟合失败当作0。",
        "仅从缓存结果重算匹配与配准指标，未测网络推理耗时。CPU评价耗时不代表模型推理耗时。"]
    checkpoint = plan["checkpoint"]
    methods.extend([
        f"权重：{checkpoint['path']}；SHA256：{checkpoint['sha256']}；"
        f"训练进度：{checkpoint['step']}/{checkpoint['total_steps']}步。",
        "指标代码：" + str((REPO / "scripts/mrsi_gt_metrics.py").resolve()) +
        "；SHA256：" + plan["metrics_sha256"] + "；评价环境：" +
        json.dumps(plan["versions"], ensure_ascii=False, sort_keys=True)])
    for dataset in COUNTS:
        entry = plan["datasets"][dataset]
        manifest = entry["manifest"]
        methods.append(f"{dataset} 清单：{manifest['path']}；SHA256：{manifest['sha256']}；"
                       f"影像对数：{manifest['pairs']}；原 A 结果目录：{entry['source_root']}。")
        for variant in VARIANTS:
            for example in results[dataset][variant]["under_k_examples"][:10]:
                methods.append(f"不足K示例：{dataset} / {variant} / index {example['index']} / "
                               f"{example['pair_id']}，原始{example['source_nm']}点，"
                               f"实际保留{example['selected_nm']}点，未补点。")
    source_text = [
        "源文件清单 SHA256：" + audit["source_inventory_sha256"],
        "选点清单 SHA256：" + audit["selections_inventory_sha256"],
        f"完整性核验：{audit['source_pairs']} 个源影像对、{audit['group_records']} 条分组记录；"
        f"逐数组独立核验 {audit['verified_source_npz']} 份源匹配、{audit['verified_selection_npz']} 份选点，"
        f"独立重算 {audit['verified_metric_records']} 条指标（NCM/P、论文RMSE、保存H的网格RMSE/SR；不重新拟合）。",
        "实验原始配置：" + str((root / "plan.json").resolve())]
    findings = conclusions(results)
    markdown = "# 固定匹配点的 Top-K 选择消融\n\n"
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定匹配点的Top-K选择消融</title>' \
        '<style>body{font-family:system-ui,sans-serif;max-width:1600px;margin:30px auto;padding:0 20px;' \
        'line-height:1.65;color:#162438}.table{overflow-x:auto;margin:18px 0 26px}table{border-collapse:collapse;' \
        'white-space:nowrap}td,th{padding:7px 10px;border-bottom:1px solid #dce3eb;text-align:right}' \
        'td:first-child,th:first-child{text-align:left}th{background:#edf2f7}h1,h2{line-height:1.3}' \
        'img{max-width:100%;width:1200px}p{overflow-wrap:anywhere}</style>' \
        '<h1>固定匹配点的 Top-K 选择消融</h1>'
    for title, columns, rows in sections:
        markdown += "## " + title + "\n\n" + markdown_table(columns, rows)
        page += "<h2>" + html.escape(title) + "</h2>" + html_table(columns, rows)
    markdown += "![Top-K对精度和配准成功率的影响](" + str((root / "overview.png").resolve()) + ")\n\n"
    page += '<img src="overview.png" alt="P@3px与NM，以及SR@3px与K">'
    for title, paragraphs in (("实验配置与指标", methods), ("结论", findings), ("可复现证据", source_text)):
        markdown += "## " + title + "\n\n" + "\n\n".join(paragraphs) + "\n\n"
        page += "<h2>" + html.escape(title) + "</h2>" + "".join(
            "<p>" + html.escape(paragraph) + "</p>" for paragraph in paragraphs)
    markdown += "[结果CSV](" + str((root / "summary.csv").resolve()) + ") · [完整JSON](" + \
        str((root / "summary.json").resolve()) + ")\n"
    page += '<p><a href="summary.csv">结果CSV</a> · <a href="summary.json">完整JSON和审计</a> · ' \
        '<a href="summary.md">Markdown报告</a></p></html>'
    (root / "summary.html").write_text(page)
    (root / "summary.md").write_text(markdown)
    if markdown_output is not None:
        markdown_output = Path(markdown_output).resolve()
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown)
    plot(root, results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=REPO / "outputs/topk_correspondence_ablation_512_20261015")
    parser.add_argument("--verify-selections", action="store_true",
                        help="Independently verify all stable nested indices and point/grid metrics, without RANSAC refit")
    parser.add_argument("--markdown-output", type=Path, default=None,
                        help="Explicit optional destination for the Chinese report; defaults to outputs only")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    result = validate(root, args.verify_selections)
    export(root, result, args.markdown_output)
    print(json.dumps(dict(summary=str(root / "summary.html"), audit=result["audit"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
