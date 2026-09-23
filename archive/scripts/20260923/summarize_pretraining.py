"""Render existing pretraining evidence; never infer a pass from missing runs."""

from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "D8": "p4_tiny_s_d8_mcnet.json",
    "D4": "p4_tiny_s_d4_mcnet.json",
    "D2": "p4_tiny_s_d2_mcnet.json",
    "TINY-6": "p4_tiny_6_mcnet.json",
}


def main() -> None:
    rows = []
    for name, filename in FILES.items():
        path = ROOT / "artifacts" / filename
        if not path.exists():
            rows.append({"name": name, "status": "missing"})
            continue
        data = json.loads(path.read_text())
        experiments = data.get("experiments", [])
        if not experiments:
            rows.append({"name": name, "status": "running", "path": str(path)})
            continue
        result = experiments[0]
        rows.append({
            "name": name,
            "status": result["status"],
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "architecture_sha256": data["build"]["architecture_sha256"],
            "optimizer_steps": result["optimizer"]["optimizer_steps"],
            "primary_readout": result["primary_readout"],
            "trajectory_mean_mace_px": result["final"]["trajectory_mean_mace_px"],
            "raw_final_mean_mace_px": result["raw_final"]["H_final_mace_px"]["mean"],
            "final_mean_mace_px": result["final"]["H_final_mace_px"]["mean"],
            "failed_pairs": result["final"]["failed_pairs"],
            "sample_count": result["sample_count"],
            "rejected_updates": result["final"]["rejected_updates"],
            "gradient_finite": result["gradient_finite"],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "cached_forward_ms": result["final"]["latency_ms_per_pair_single_pass"],
            "training_elapsed_seconds": result["training_elapsed_seconds"],
            "progress": result["progress"],
            "resume": result["resume"],
            "checkpoint": result["checkpoint"],
        })
    report = {
        "scope": "pretraining controlled-H0 learnability only; no E00/E01",
        "all_reported_passed": all(row["status"] == "passed" for row in rows),
        "gate_note": "This table does not replace tiny-gate-merge validation.",
        "experiments": rows,
    }
    gate_path = ROOT / "artifacts/p4_tiny_gate_mcnet_d2.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        source_hashes = {s["path"]: s["sha256"] for s in gate.get("sources", [])}
        bound = len(source_hashes) == len(rows) and all(
            r.get("sha256") is not None and source_hashes.get(r.get("path")) == r["sha256"]
            for r in rows
        )
        report["merged_gate"] = {
            "path": str(gate_path),
            "sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            "status": gate.get("status"),
            "current_sources_match": bound,
            "passed_with_current_sources": bool(bound and gate.get("status") == "passed"
                                                 and gate.get("errors") == []),
        }
    audits = []
    for row in rows:
        if "path" not in row or "checkpoint" not in row:
            continue
        audit_path = Path(row["path"]).with_name(Path(row["path"]).stem + "_checkpoint_audit.json")
        if not audit_path.exists():
            continue
        audit = json.loads(audit_path.read_text())
        checkpoint = audit["checkpoint"]
        original = row["trajectory_mean_mace_px"]
        replay = audit["endpoint"]["trajectory_mean_mace_px"]
        matching = (
            audit["resource_binding"] == "strict_signature_match"
            and checkpoint["audited_snapshot_sha256"] == row["checkpoint"]["sha256"]
            and checkpoint["serialized_iterator_state_sha256"]
            == checkpoint["loaded_iterator_state_sha256"]
            == checkpoint["post_forward_iterator_state_sha256"]
            and len(original) == len(replay)
            and all(abs(a - b) <= 1e-6 for a, b in zip(original, replay))
        )
        audits.append({"name": row["name"], "path": str(audit_path),
                       "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                       "source_and_trajectory_match": matching})
    report["checkpoint_replay_audits"] = audits
    report["all_checkpoint_replays_match"] = len(audits) == len(rows) and all(
        a["source_and_trajectory_match"] for a in audits)
    (ROOT / "artifacts/pretraining_mcnet_d2_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# 正式训练前 tiny 检查汇总", "",
        "由 `scripts/summarize_pretraining.py` 根据原始 artifact 生成。",
        "32个受控 H0 样本，seed=0；主读出可能包含预先登记的参数平均，原始终点始终单列。",
        "延迟仅为缓存描述子后的精修，不含 GHIM/CGMDP。未使用 test；未启动 E00/E01。", "",
        "| 实验 | 状态 | 步数 | H0→各轮H MACE px | 原始终点 px | 主读出 px | 失败/样本 | 拒绝更新 | 显存 GB | 缓存前向 ms |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "optimizer_steps" not in row:
            lines.append(f"| {row['name']} | {row['status']} | — | — | — | — | — | — | — | — |")
            continue
        trajectory = " → ".join(f"{value:.6f}" for value in row["trajectory_mean_mace_px"])
        lines.append(
            f"| {row['name']} | {row['status']} | {row['optimizer_steps']} | {trajectory} | "
            f"{row['raw_final_mean_mace_px']:.6f} | {row['final_mean_mace_px']:.6f} | "
            f"{row['failed_pairs']}/{row['sample_count']} | {row['rejected_updates']} | "
            f"{row['peak_allocated_bytes']/1e9:.3f} | {row['cached_forward_ms']:.3f} |"
        )
    lines += ["", "通过要求为平均最终 MACE < 0.1 px、零失败、零拒绝更新，并满足梯度和协议审计。"]
    merged = report.get("merged_gate", {})
    if merged.get("passed_with_current_sources"):
        lines.append("四项与合并审计均已通过，合并结果绑定当前四份artifact。按用户要求停止在正式训练前，未执行 E00/E01。")
        lines.append(f"合并gate SHA256：`{merged['sha256']}`。")
        if report["all_checkpoint_replays_match"]:
            lines.append("四个checkpoint独立重载均匹配资源签名及保存的轨迹；序列化、加载和前向后权重身份一致。")
    else:
        lines.append("四项均报告通过，仍须执行合并器核验资源及协议一致性。" if report["all_reported_passed"]
                     else "尚有缺失、进行中或未通过项；正式实验门槛保持关闭。")
    lines += ["", "## 证据", ""]
    for row in rows:
        if "sha256" in row:
            lines.append(f"- {row['name']}：`{row['path']}`；SHA256 `{row['sha256']}`；主读出 `{row['primary_readout']}`。")
    (ROOT / "docs/pretraining_mcnet_d2_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"all_reported_passed": report["all_reported_passed"],
                      "rows": [{"name": r["name"], "status": r["status"]} for r in rows]}))


if __name__ == "__main__":
    main()
