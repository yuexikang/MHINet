"""Launch the registered three-stage, full-epoch experiment sequentially."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path("/home/disk1/MHINet")
STEPS = {1: 12995, 2: 8663, 3: 8663}
VAL_PAIRS = {1: 5772, 2: 3848, 3: 3848}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / f"configs/shared_stable_v2_full_tier{args.tier}.json"
    config = json.loads(config_path.read_text())
    source = Path(config["initialize_checkpoint"])
    if not source.is_file():
        raise SystemExit(f"初始化权重尚不存在，请先完成上一阶段：{source}")
    if args.tier > 1:
        import torch
        previous = args.tier - 1
        state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        expected_config = json.loads(
            (ROOT / f"configs/shared_stable_v2_full_tier{previous}.json").read_text()
        )
        if state["metadata"]["config"] != expected_config:
            raise SystemExit("上一阶段 checkpoint 配置不匹配，拒绝串接其他实验。")
        if (state["metadata"]["total_steps"] != STEPS[previous]
                or state["progress"]["optimizer_step"] != STEPS[previous]):
            raise SystemExit("上一阶段尚未训练完完整一轮，请完成后再启动。")
        summary_path = source.parent / "validation" / f"step_{STEPS[previous]:06d}" / "summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"上一阶段最终验证尚未完成：{summary_path}")
        summary = json.loads(summary_path.read_text())
        if summary["pairs"] != VAL_PAIRS[previous] or summary["step"] != STEPS[previous]:
            raise SystemExit("上一阶段最终验证不是预期的完整 val。")
        del state
    output = ROOT / f"outputs/shared_stable_v2_full_tier{args.tier}_seed0"
    command = [sys.executable, "-u", "-m", "mhinet.pretraining.train",
               "--config", str(config_path), "--output", str(output)]
    print(f"第 {args.tier} 档完整一轮；初始化：{source}；输出：{output}", flush=True)
    if not args.check_only:
        os.chdir(ROOT)
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
