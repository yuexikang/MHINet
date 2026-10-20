"""Fail-closed engineering replay and preregistered pilot acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mhinet.config import sha256_file
from mhinet.downstream.supervision import coarse_labels
from mhinet.ops.geometry import pixel_homography_to_normalized
from mhinet.v2.train import timestamp, write_json
from .common import SOURCE_SHA, SOURCE_STEP, VAL_SHA, digest_tree, setup
from .train import assert_tree_equal


def payload(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def engineering(output):
    root = output/"engineering"
    continuous = payload(root/"L0_continuous/latest.pt")
    resumed = payload(root/"L0_resumed/latest.pt")
    candidate = payload(root/"L1/latest.pt")
    for key in ("model", "optimizer", "scheduler", "progress", "rng", "auxiliary_state"):
        assert_tree_equal(resumed[key], continuous[key], key)
    clones = [p["metadata"]["flow_loss_pilot"]["source_state_clone"]
              for p in (continuous, resumed, candidate)]
    assert_tree_equal(clones[1], clones[0], "L0 resume source clone")
    assert_tree_equal(clones[2], clones[0], "L1 source clone")
    source_path = Path(continuous["metadata"]["source_checkpoint"])
    if sha256_file(source_path) != SOURCE_SHA:
        raise RuntimeError("Source changed before acceptance")
    source = payload(source_path)
    trainable = set(continuous["metadata"]["trainable_parameters"])
    if len(trainable) != 10:
        raise RuntimeError("Only ten Flow Head tensors may train")
    frozen_count = 0
    changed = {}
    for name, reference in source["model"].items():
        if name not in trainable:
            for arm in (continuous, candidate):
                assert_tree_equal(arm["model"][name], reference, "frozen."+name)
            frozen_count += 1
    for label, arm, steps in (("L0", continuous, 2), ("L1", candidate, 1)):
        if arm["progress"]["optimizer_step"] != SOURCE_STEP+steps:
            raise RuntimeError("Smoke budget mismatch")
        changed[label] = [name for name in sorted(trainable)
                          if not torch.equal(arm["model"][name], source["model"][name])]
        if not changed[label]:
            raise RuntimeError("Smoke did not update any trainable weights")
    # Exact first update/epoch/cursor and common shuffle are also registered by
    # the runner. Dataset decoding has no random augmentations.
    runs = {name: json.loads((root/directory/"run.json").read_text())
            for name, directory in (("L0", "L0_continuous"), ("L1", "L1"))}
    for key in ("source_sha256", "train_sha256", "val_sha256", "trainable_parameters"):
        assert_tree_equal(runs["L0"][key], runs["L1"][key], key)
    left, right = [runs[name]["flow_loss_pilot"]["config"] for name in ("L0", "L1")]
    assert_tree_equal({k:v for k,v in left.items() if k not in ("arm", "experiment_id", "notes")},
                      {k:v for k,v in right.items() if k not in ("arm", "experiment_id", "notes")}, "common pilot configuration")
    for key in ("sixth_pass_order_sha256", "sixth_pass_first16_pair_ids"):
        if key not in runs["L0"]["flow_loss_pilot"] or key not in runs["L1"]["flow_loss_pilot"]:
            raise RuntimeError("Missing registered training order evidence")
        assert_tree_equal(runs["L0"]["flow_loss_pilot"][key], runs["L1"]["flow_loss_pilot"][key], key)
    logs = {name:[json.loads(line) for line in (root/directory/"train.jsonl").read_text().splitlines()]
            for name,directory in (("L0", "L0_continuous"), ("L0_resumed", "L0_resumed"), ("L1", "L1"))}
    assert_tree_equal([r["pair_ids"] for r in logs["L0_resumed"]],
                      [r["pair_ids"] for r in logs["L0"]], "resumed pair order")
    assert_tree_equal(logs["L1"][0]["pair_ids"], logs["L0"][0]["pair_ids"], "paired first update")
    evidence = dict(schema="mhi.flow-loss-engineering-acceptance.v1", status="passed",
        checked_at=timestamp(), real_data=True, source_checkpoint_sha256=SOURCE_SHA,
        uninterrupted_vs_resumed_exact=["model", "optimizer", "scheduler", "progress", "rng", "auxiliary_state"],
        source_clone=clones[0], frozen_tensors_bitwise_verified=frozen_count, changed_head_tensors=changed,
        replay_state_sha256={k:digest_tree(continuous[k]) for k in ("model", "optimizer", "rng")},
        budgets={"L0_continuous": 2, "L0_interrupted_then_resumed": [1,2], "L1": 1},
        calibration_sha256=sha256_file(output/"calibration/calibration.json"))
    write_json(root/"acceptance.json", evidence)
    print(json.dumps(evidence)); return 0


def metric_gates(old, new, flow_old, flow_new, negative_control_exact):
    a, b = old["stages"]["final"], new["stages"]["final"]
    return dict(
        flow_epe_improved=flow_new < flow_old,
        final_pre3_gain_at_least_point2pp=(b["macro_raster_mask_precision"]["3.0"]
                                         -a["macro_raster_mask_precision"]["3.0"]) >= .002,
        coverage_a_preserved=b["mean_coverage_a_16x16"] >= .98*a["mean_coverage_a_16x16"],
        coverage_b_preserved=b["mean_coverage_b_16x16"] >= .98*a["mean_coverage_b_16x16"],
        match_count_preserved=b["mean_matches"] >= .95*a["mean_matches"],
        failure_increase_at_most_point5pp=(b["failure_below20_correct5_rate"]
                                         -a["failure_below20_correct5_rate"]) <= .005,
        prior_off_exact=negative_control_exact)


def selected_parent_ids(array):
    """Convert native pixel D8 centers (3.5+8*i) back to integer cell IDs."""
    uv = (array+.5)/8-.5
    rounded = np.rint(uv)
    if (not np.all(np.isfinite(uv)) or not np.allclose(uv, rounded, atol=1e-4, rtol=0)
            or not np.all((rounded >= 0)&(rounded < 64))):
        raise ValueError("Expected exact native D8 cell centers")
    return rounded[:,1].astype(np.int64)*64+rounded[:,0].astype(np.int64)


def parent_retention(row, manifest_root, predictions):
    masks = []
    for side in ("A", "B"):
        with Image.open(manifest_root/row[f"mask_{side}_overlap"]) as image:
            masks.append(torch.from_numpy(np.asarray(image.convert("L")).copy()).float()[None]/255)
    H = pixel_homography_to_normalized(torch.tensor(row["H_A_to_B"], dtype=torch.float64),
                                     (512,512), (512,512)).float()
    source, target = coarse_labels(H, *masks, (64,64))
    truth = set((source*4096+target).tolist())
    counts = {}
    for label, arrays in predictions.items():
        if arrays["d8_a"] is None or arrays["d8_b"] is None:
            counts[label] = dict(gt_parents=len(truth), retained=None, recall=None, available=False,
                                 reason="D8 confidence/coordinates unavailable in evaluation artifact")
            continue
        a, b = selected_parent_ids(arrays["d8_a"]), selected_parent_ids(arrays["d8_b"])
        selected = set((a*4096+b).tolist())
        counts[label] = dict(gt_parents=len(truth), retained=len(truth & selected), available=True,
                             recall=len(truth & selected)/len(truth) if truth else None)
    return counts


def decision(output):
    setup()
    evaluations = output/"evaluations"
    labels = ("L0_B", "L0_C", "L1_B", "L1_C")
    summaries = {label:json.loads((evaluations/label/"summary.json").read_text()) for label in labels}
    expected_code = summaries["L0_B"]["code_sha256"]
    for label, summary in summaries.items():
        if (summary["pairs"] != 5106 or summary["manifest_sha256"] != VAL_SHA
                or summary["shared_output_budget"] != 2048 or summary["code_sha256"] != expected_code
                or summary["prior_toggle"] != label[-1]):
            raise RuntimeError("Matching result identity/protocol mismatch")
        arm_path = output/label[:2]/"last.pt"
        if summary["checkpoint_sha256"] != sha256_file(arm_path):
            raise RuntimeError("Fixed-budget checkpoint identity mismatch")
    manifest = Path(summaries["L0_B"]["manifest"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    records = {label:[json.loads(line) for line in (evaluations/label/"pairs.jsonl").read_text().splitlines() if line.strip()]
               for label in labels}
    retention_rows = []
    for index, row in enumerate(rows):
        arrays = {}
        for label in labels:
            record = records[label][index]
            if record["pair_id"] != str(row["pair_id"]):
                raise RuntimeError("Matching pair identity/order mismatch")
            with np.load(evaluations/label/record["npz"], allow_pickle=False) as blob:
                arrays[label] = {key:blob[key] for key in ("final_a", "final_b", "final_confidence")}
                arrays[label].update({key:blob[key] if key in blob else None for key in ("d8_a", "d8_b")})
        for key in ("final_a", "final_b", "final_confidence"):
            if not np.array_equal(arrays["L0_B"][key], arrays["L1_B"][key]):
                raise RuntimeError(f"Full-val prior-off negative control differs: {row['pair_id']} {key}")
        retention_rows.append(dict(pair_id=str(row["pair_id"]), values=parent_retention(row, manifest.parent, arrays)))
        if index%100 == 0:
            write_json(output/"decision_progress.json", dict(status="checking_full_negative_control_and_parent_retention", pairs=index+1, total=5106))
    with (output/"parent_retention.jsonl").open("w") as stream:
        for row in retention_rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    flow = {arm:json.loads((evaluations/arm/"summary.json").read_text()) for arm in ("L0", "L1")}
    gates = metric_gates(summaries["L0_C"], summaries["L1_C"],
                         flow["L0"]["macro_flow_epe_px"], flow["L1"]["macro_flow_epe_px"], True)
    retention_summary = {}
    for label in labels:
        values = [r["values"][label] for r in retention_rows]
        usable = [v for v in values if v["available"]]
        recalls = [v["recall"] for v in usable if v["recall"] is not None]
        denominator = sum(v["gt_parents"] for v in usable)
        retention_summary[label] = dict(available_pairs=len(usable), unavailable_pairs=len(values)-len(usable),
            gt_parents_total=sum(v["gt_parents"] for v in values), gt_parents_in_available_pairs=denominator,
            macro_recall=float(np.mean(recalls)) if recalls else None,
            micro_recall=sum(v["retained"] for v in usable)/denominator if denominator else None)
    report = dict(schema="mhi.flow-loss-pilot-decision.v1", status="completed", completed_at=timestamp(),
        gates=gates, pilot_worth_confirming=all(gates.values()),
        recommendation="register_equal_budget_from_v1_confirmation" if all(gates.values()) else "retain_locked_flow_source",
        adoption="No pilot replaces the current C/D/E source; matched confirmation and further seeds remain separate work",
        flow_epe_px={arm:report["macro_flow_epe_px"] for arm,report in flow.items()},
        final_pre3={label:summary["stages"]["final"]["macro_raster_mask_precision"]["3.0"] for label,summary in summaries.items()},
        prior_off_negative_control=dict(pairs=5106, final_points_and_confidence_exact=True),
        parent_retention=dict(protocol="Exact nearest-cell inverse-roundtrip GT parents from existing coarse_labels; selected D8 top2048 native centers. Macro over available pairs with GT parents; micro over GT parents in available pairs. Missing D8 artifacts are explicitly unavailable, never inferred as zero retention.",
            per_arm=retention_summary,
            artifact=str(output/"parent_retention.jsonl"), sha256=sha256_file(output/"parent_retention.jsonl")),
        scope="Single-seed exploratory RGB512 val; raster diagnostics, no sealed sets")
    write_json(output/"decision.json", report)
    print(json.dumps(report)); return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", action="store_true")
    args = parser.parse_args(argv)
    return decision(args.output) if args.decision else engineering(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
