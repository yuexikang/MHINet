"""Summarize completed tiny conditions by dominant H0 residual direction."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(block):
    groups = {}
    for row in block["per_pair"]:
        residual = row["H0_corner_residual_px"]
        dx = sum(p[0] for p in residual) / 4
        dy = sum(p[1] for p in residual) / 4
        axis, value = ("x", dx) if abs(dx) >= abs(dy) else ("y", dy)
        label = axis + ("+" if value >= 0 else "-")
        groups.setdefault(label, []).append(row)
    report = {}
    for label, rows in sorted(groups.items()):
        count = len(rows)
        rounds = len(rows[0]["trajectory_mace_px"])
        trajectory = [sum(r["trajectory_mace_px"][i] for r in rows) / count
                      for i in range(rounds)]
        report[label] = {
            "count": count,
            "trajectory_mean_mace_px": trajectory,
            "worst_final_mace_px": max(r["H_final_mace_px"] for r in rows),
            "rejected_updates": sum(sum(not accepted for accepted in r["update_accepted"])
                                    for r in rows),
            "mean_saturation_fraction": sum(sum(r["tanh_saturation_fraction"]) for r in rows)
                                        / (count * (rounds - 1)),
            "minimum_supported_queries": min(min(r["supported_query_count"]) for r in rows),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = []
    lines = ["# Tiny 平移方向诊断", "",
             "按 H0 四角平均残差的主轴及符号分组。只重算已有训练诊断，不使用验证或测试选择配置。", "",
             "| 实验 | 读出 | 方向 | 样本数 | H0→各轮H均值 px | 最差最终 px | 拒绝数 | 饱和比例 | 最少支持query |",
             "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for path in args.inputs:
        data = json.loads(path.read_text())
        if not data.get("experiments"):
            raise ValueError(f"No completed experiment in {path}")
        for experiment in data["experiments"]:
            groups = {key: summarize(experiment[key]) for key in ("raw_final", "final")}
            evidence.append({"name": experiment["name"], "status": experiment["status"],
                             "source": str(path.resolve()),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "primary_readout": experiment["primary_readout"], "groups": groups})
            for readout, directions in groups.items():
                for label, row in directions.items():
                    trajectory = " → ".join(f"{v:.6f}" for v in row["trajectory_mean_mace_px"])
                    lines.append(f"| {experiment['name']} | {readout} | {label} | {row['count']} | "
                                 f"{trajectory} | {row['worst_final_mace_px']:.6f} | "
                                 f"{row['rejected_updates']} | {row['mean_saturation_fraction']:.6f} | "
                                 f"{row['minimum_supported_queries']} |")
    lines += ["", "方向均值、最差样本、支持数量和饱和比例用于定位问题，不能单独证明原因。",
              "如主读出未通过门槛，继续关闭正式训练入口；不以某个方向或某个样本的通过替代完整门槛。"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"experiments": len(evidence), "output": str(args.output)}))


if __name__ == "__main__":
    main()
