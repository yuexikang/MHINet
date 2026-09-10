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
            "checkpoint": result["checkpoint"],
        })
    report = {
        "scope": "pretraining controlled-H0 learnability only; no E00/E01",
        "all_reported_passed": all(row["status"] == "passed" for row in rows),
        "gate_note": "This table does not replace tiny-gate-merge validation.",
        "experiments": rows,
    }
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
