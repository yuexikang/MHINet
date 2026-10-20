#!/usr/bin/env python3
"""Durable, bounded loss-pilot queue. GPU memory is checked even for hidden PIDs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/root/miniconda3/envs/loma-repro/bin/python"
DEFAULT_REGISTRATION = ROOT/"versions/v2/evidence/flow_loss_pilot_20261020/registration.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path, value):
    temporary = path.with_suffix(path.suffix+".temporary")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def gpu_status(gpu):
    line = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    memory, utilization = [int(v.strip()) for v in line.strip().split(",")]
    return dict(memory_used_mib=memory, utilization_percent=utilization,
                available=memory < 512 and utilization < 5)


def verify_pins(registration):
    for path, expected in registration["pinned_files"].items():
        if sha(ROOT/path) != expected:
            raise RuntimeError(f"Registered input/code changed: {path}")


def build_commands(reg):
    out = Path(reg["output"])
    val, source, runtime = reg["manifest"], reg["source_checkpoint"], reg["runtime"]
    prefix = [PYTHON, "-u", "-m"]
    def train(arm, target, limit=None):
        command = prefix+["mhinet.v2.flow_loss.train", "--config", reg["configs"][arm],
            "--output", str(out/target), "--device", "cuda:0"]
        return command+(["--max-steps", str(limit)] if limit else [])
    def flow(checkpoint, name):
        return prefix+["mhinet.v2.flow_loss.evaluate", "flow", "--checkpoint", str(checkpoint),
            "--manifest", val, "--runtime", runtime, "--hard-cases", reg["hard_cases"],
            "--output", str(out/"evaluations"/name), "--device", "cuda:0"]
    stages = [dict(name="calibration", command=prefix+["mhinet.v2.flow_loss.calibrate",
        "--base-config", reg["base_config"], "--runtime", runtime,
        "--checkpoint", source, "--output", str(out/"calibration"), "--device", "cuda:0"],
        artifact=str(out/"calibration/calibration.json"))]
    if reg.get("calibration_selection"):
        stages[0]["command"] += ["--selection", reg["calibration_selection"]]
    for name, arm, target, limit in (
            ("smoke_L0_continuous", "L0", "engineering/L0_continuous", 2),
            ("smoke_L0_first", "L0", "engineering/L0_resumed", 1),
            ("smoke_L0_resume", "L0", "engineering/L0_resumed", 2),
            ("smoke_L1", "L1", "engineering/L1", 1)):
        stages.append(dict(name=name, command=train(arm, target, limit),
                           artifact=str(out/target/"status.json"), training=True))
    stages.append(dict(name="engineering_acceptance", command=prefix+[
        "mhinet.v2.flow_loss.acceptance", "--output", str(out)],
        artifact=str(out/"engineering/acceptance.json"), cpu=True))
    for arm in ("L0", "L1"):
        stages.append(dict(name="train_"+arm, command=train(arm, arm),
            artifact=str(out/arm/"status.json"), training=True, full=True))
    for name, checkpoint in (("source", source), ("L0", out/"L0/last.pt"), ("L1", out/"L1/last.pt")):
        stages.append(dict(name="flow_"+name, command=flow(checkpoint, name),
            artifact=str(out/"evaluations"/name/"summary.json"), full=True))
    stages.append(dict(name="flow_comparison", command=prefix+["mhinet.v2.flow_loss.evaluate", "compare",
        "--arm-a", str(out/"evaluations/L0"), "--arm-b", str(out/"evaluations/L1"),
        "--source", str(out/"evaluations/source"), "--output", str(out/"evaluations/comparison")],
        artifact=str(out/"evaluations/comparison/comparison.json"), cpu=True))
    stages.append(dict(name="negative_control", command=prefix+[
        "mhinet.v2.flow_loss.evaluate", "negative-control", "--arm-a", str(out/"L0/last.pt"),
        "--arm-b", str(out/"L1/last.pt"), "--manifest", val, "--runtime", runtime,
        "--output", str(out/"evaluations/negative_control"), "--limit", "64", "--device", "cuda:0"],
        artifact=str(out/"evaluations/negative_control/negative_control.json")))
    for arm in ("L0", "L1"):
        for variant in ("B", "C"):
            stages.append(dict(name=f"matching_{arm}_{variant}", command=prefix+[
                "mhinet.v2.evaluate", "--checkpoint", str(out/arm/"last.pt"), "--manifest", val,
                "--output", str(out/"evaluations"/(arm+"_"+variant)), "--device", "cuda:0",
                "--limit", "0", "--variant", variant],
                artifact=str(out/"evaluations"/(arm+"_"+variant)/"summary.json"), full=True))
    stages.append(dict(name="decision", command=prefix+["mhinet.v2.flow_loss.acceptance",
        "--output", str(out), "--decision"], artifact=str(out/"decision.json"), cpu=True))
    return stages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args(argv)
    reg = json.loads(args.registration.read_text())
    verify_pins(reg)
    stages = build_commands(reg)
    if args.print_commands:
        print(json.dumps(stages, indent=2)); return 0
    out = Path(reg["output"])
    if out.exists() and not args.resume:
        raise FileExistsError("Existing queue output requires --resume; artifacts are preserved")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/".queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((out/"status.json").read_text()) if args.resume else {}
    if previous and previous["registration_sha256"] != sha(args.registration):
        raise ValueError("Queue registration changed")
    state = dict(schema="mhi.flow-loss-queue.v1", status="registered", pid=os.getpid(),
        registration_sha256=sha(args.registration), registration=str(args.registration.resolve()),
        started_at=previous.get("started_at", now()), phases=previous.get("phases", []),
        physical_gpu=reg["physical_gpu"], checked_at=now())
    active = dict(child=None, stop=False)
    def stop(signum, frame):
        active["stop"] = True
        if active["child"] is not None and active["child"].poll() is None:
            active["child"].terminate()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    def persist():
        state["checked_at"] = now(); write(out/"status.json", state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(reg["physical_gpu"]),
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONUNBUFFERED="1", TMPDIR=str(out/"temporary"))
    (out/"temporary").mkdir(exist_ok=True)
    try:
        for stage in stages:
            for completed in state["phases"]:
                if completed["status"] == "completed" and completed["name"] != "smoke_L0_first":
                    artifact = Path(completed["artifact"])
                    if not artifact.is_file() or sha(artifact) != completed["artifact_sha256"]:
                        raise RuntimeError("Completed stage artifact changed: "+completed["name"])
            old = next((p for p in state["phases"] if p["name"] == stage["name"] and p["status"] == "completed"), None)
            if old:
                if not Path(stage["artifact"]).is_file() or sha(stage["artifact"]) != old["artifact_sha256"]:
                    # The first smoke status is intentionally superseded by its resume stage.
                    if stage["name"] != "smoke_L0_first":
                        raise RuntimeError("Completed artifact changed: "+stage["name"])
                continue
            if active["stop"]:
                break
            if not stage.get("cpu"):
                consecutive = 0
                while consecutive < 3 and not active["stop"]:
                    availability = gpu_status(reg["physical_gpu"])
                    state.update(status="waiting_gpu_idle", next_stage=stage["name"], gpu=availability)
                    consecutive = consecutive+1 if availability["available"] else 0
                    persist(); time.sleep(10)
                if active["stop"]:
                    break
            verify_pins(reg)
            # A failed evaluation is restarted into a new directory, preserving its partial attempt.
            target = Path(stage["artifact"]).parent
            if not stage.get("training") and target.exists() and stage["name"] not in ("engineering_acceptance", "decision"):
                backup = out/"partial_attempts"/(stage["name"]+"_"+str(time.time_ns()))
                backup.parent.mkdir(exist_ok=True)
                target.rename(backup)
            phase = dict(name=stage["name"], status="running", command=stage["command"],
                started_at=now(), artifact=stage["artifact"])
            state["phases"].append(phase)
            state.update(status="running", current_stage=stage["name"])
            logpath = out/(stage["name"]+".console.log")
            with logpath.open("a") as stream:
                child = subprocess.Popen(stage["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active["child"] = child; phase["pid"] = child.pid
                while child.poll() is None:
                    if stage.get("training") and Path(stage["artifact"]).is_file():
                        phase["progress"] = json.loads(Path(stage["artifact"]).read_text())
                    persist(); time.sleep(5)
            active["child"] = None
            phase["exit_code"] = child.returncode
            if child.returncode or active["stop"]:
                raise RuntimeError(f"Stage {stage['name']} stopped with code {child.returncode}; see {logpath}")
            artifact = json.loads(Path(stage["artifact"]).read_text())
            if stage.get("training"):
                if artifact["status"] != ("completed" if stage.get("full") else "diagnostic_stop"):
                    raise RuntimeError("Training did not finish its registered budget/validation")
                if stage.get("full") and artifact.get("extra_steps") != 5107:
                    raise RuntimeError("Training budget mismatch")
            if stage["name"].startswith(("flow_", "matching_")) and stage.get("full"):
                if artifact.get("pairs") != 5106:
                    raise RuntimeError("Full validation count mismatch")
            if stage["name"].startswith("flow_") and stage.get("full"):
                if not artifact.get("complete") or artifact.get("hard_cases_evaluated") != 52 or artifact.get("scope") != "full_val":
                    raise RuntimeError("Incomplete Flow/hard-case diagnostics")
            phase.update(status="completed", completed_at=now(), artifact_sha256=sha(stage["artifact"]))
            persist()
        state["status"] = "interrupted" if active["stop"] else "completed"
        if not active["stop"]:
            state["decision"] = json.loads((out/"decision.json").read_text())
            state["completed_at"] = now()
        persist()
        return 130 if active["stop"] else 0
    except BaseException as error:
        if active["child"] is not None and active["child"].poll() is None:
            active["child"].terminate(); active["child"].wait()
        state.update(status="interrupted" if active["stop"] else "failed", error=f"{type(error).__name__}: {error}")
        persist()
        return 130 if active["stop"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
