#!/usr/bin/env python3
"""Run the preregistered Flow Head -> equal-budget C/D/E campaign serially."""
import argparse
import fcntl
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write(path, payload):
    temporary = path.with_suffix(".temporary.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--engineering-report", type=Path, required=True)
    args = p.parse_args()
    engineering = json.loads(args.engineering_report.read_text())
    if engineering.get("status") not in ("pass", "passed"):
        raise ValueError("Real-resource engineering acceptance is required before long training")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output/".campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    phases = [("Flow0", "configs/mhi_v2/flow_head_rgb512_seed42.json",
               "outputs/mhi_v2/Flow0/P5/flow_head_rgb512_seed42_5ep_20261020")]
    phases.extend((f"MHIv2-{variant}", f"configs/mhi_v2/qrru_{variant.lower()}_rgb512_seed42.json",
                   f"outputs/mhi_v2/MHIv2-{variant}/P5/qrru_rgb512_seed42_5ep_20261020") for variant in "CDE")
    state = dict(schema="mhi.v2-long-campaign.v1", status="running", started_at=now(), gpu=args.gpu,
                 engineering_report=str(args.engineering_report.resolve()), phases=[],
                 protocol="RGB512 train only; source-image-disjoint val; 512 input; final cap 2048; fresh optimizers; sequential single GPU")
    write(output/"campaign.json", state)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    active = {"process": None, "stop": False}
    def stop(signum, frame):
        active["stop"] = True
        process = active["process"]
        if process is not None and process.poll() is None:
            process.terminate()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    for name, configuration, directory in phases:
        if active["stop"]:
            break
        cfg = json.loads((ROOT/configuration).read_text())
        if not Path(cfg["checkpoint"]).is_file():
            raise FileNotFoundError(f"Phase {name} requires its registered completed source: {cfg['checkpoint']}")
        command = [sys.executable, "-u", "-m", "mhinet.v2.train", "--config", str(ROOT/configuration),
                   "--output", str(ROOT/directory), "--device", "cuda:0"]
        phase = dict(name=name, config=configuration, output=directory, actual_command=command, status="running", started_at=now())
        state["phases"].append(phase)
        write(output/"campaign.json", state)
        with (output/f"{name}.console.log").open("a") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            active["process"] = child
            phase["pid"] = child.pid
            while child.poll() is None:
                progress = ROOT/directory/"status.json"
                if progress.exists():
                    try:
                        phase["progress"] = json.loads(progress.read_text())
                    except (OSError, ValueError):
                        pass
                phase["checked_at"] = now()
                write(output/"campaign.json", state)
                time.sleep(10)
            phase["exit_code"] = child.returncode
        active["process"] = None
        progress = ROOT/directory/"status.json"
        result = json.loads(progress.read_text()) if progress.exists() else {}
        phase.update(status=result.get("status", "failed"), completed_at=now(), progress=result)
        if child.returncode != 0 or result.get("status") != "completed":
            state.update(status="interrupted" if active["stop"] else "failed", checked_at=now())
            write(output/"campaign.json", state)
            return child.returncode or 1
        write(output/"campaign.json", state)
    state.update(status="interrupted" if active["stop"] else "completed", checked_at=now())
    write(output/"campaign.json", state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
