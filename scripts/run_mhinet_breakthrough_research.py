#!/usr/bin/env python3
"""Run a silent, read-only Codex research round; publish only schema-valid output."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "docs/research/mhinet_breakthrough"
RUNTIME = Path("/home/disk1/.mhinet-breakthrough-research")
CODEX = Path("/root/.local/bin/codex")
CHINA = timezone(timedelta(hours=8))


def timestamp():
    return datetime.now(CHINA).isoformat(timespec="seconds")


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n")
    temporary.replace(path)


def load_state():
    path = REPORTS / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def validate_output(value):
    keys = {"status", "title", "report_markdown", "research_memory_markdown", "sources", "breakthrough_verified", "next_focus"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Unexpected response fields")
    if value["status"] not in ("completed", "limited"):
        raise ValueError("Invalid research status")
    for key in ("title", "report_markdown", "research_memory_markdown", "next_focus"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError("Missing " + key)
    if not isinstance(value["sources"], list) or not all(isinstance(x, str) for x in value["sources"]):
        raise ValueError("Invalid sources")
    if not isinstance(value["breakthrough_verified"], bool):
        raise ValueError("Invalid breakthrough flag")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Test auth, local reads, live search and structured output")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (REPORTS / "PAUSED").exists():
        return 0
    with (RUNTIME / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        state = load_state()
        hour = datetime.now(CHINA).strftime("%Y-%m-%dT%H")
        if not args.smoke and state.get("last_attempt_hour") == hour:
            return 0
        name = datetime.now(CHINA).strftime("%Y%m%dT%H%M%S") + ("_smoke" if args.smoke else "")
        run_dir = REPORTS / "runs" / name
        run_dir.mkdir(parents=True)
        metadata = {"started_at": timestamp(), "smoke": args.smoke, "status": "running",
                    "sandbox": "read-only", "live_search": True}
        save_json(run_dir / "run.json", metadata)
        if not args.smoke:
            state.update(last_attempt_at=timestamp(), last_attempt_hour=hour, last_attempt_dir=str(run_dir), last_status="running")
            save_json(REPORTS / "state.json", state)
        began = time.monotonic()
        try:
            brief = (REPORTS / "brief.md").read_text()
            if args.smoke:
                task = (
                    "这是调度链路的轻量验证，限三分钟，不展开完整研究。只读打开 "
                    + str(REPO / "docs/semidense_rgb512_h0_joint_test512_results.md")
                    + "，提取GE和JL的H0joint P@1/P@5/SR@3。再用实时web搜索并打开一个"
                    "LoMa或LoRetta作者论文/官方代码页面，提供准确链接和可核验的一句事实。"
                    "按schema返回JSON：title说明链路验证；report_markdown简短中文，明确本地读取与实时网络"
                    "分别是否成功；网络不可用则status=limited。breakthrough_verified=false；"
                    "next_focus为核查QRRU训练噪声与真实预测误差的差异。"
                    "research_memory_markdown填入读取到的research_memory.md原文；"
                    "若该文件不存在，说明此测试不产生科学记忆。不要修改任何文件。"
                )
            else:
                memory_path = REPORTS / "research_memory.md"
                memory = memory_path.read_text() if memory_path.exists() else "累计记忆尚未建立；先从已有研究报告恢复关键结论与反证。"
                task = (brief + "\n本轮开始北京时间：" + timestamp()
                        + "\n前轮状态：" + json.dumps(state, ensure_ascii=False)
                        + "\n已保存的累计研究记忆：\n" + memory
                        + "\n执行一轮实质研究，报告约1500–3000中文字，按JSON schema交付。"
                        "结合累计记忆、相关历史报告及最近报告推进next_focus，"
                        "明确历史判断的保留/修正/否定，并返回整份更新后的research_memory_markdown。"
                        "若存在更新的用户约束，优先遵守。")
            command = [str(CODEX), "--no-daemon", "--search", "-a", "never", "exec", "--ephemeral",
                       "--sandbox", "read-only", "-C", str(REPO), "--json", "--color", "never",
                       "--output-schema", str(REPORTS / "response.schema.json"),
                       "-o", str(run_dir / "response.json"), "-"]
            env = dict(os.environ)
            network_config = RUNTIME / "network_env.json"
            if network_config.exists():
                proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                              "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
                saved_network = json.loads(network_config.read_text())
                env.update({k: v for k, v in saved_network.items() if k in proxy_keys and isinstance(v, str)})
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
            env["PATH"] = "/root/.local/bin:/root/miniconda3/envs/loma-repro/bin:/usr/local/bin:/usr/bin:/bin"
            with (RUNTIME / (name + ".events.jsonl")).open("w") as events, (RUNTIME / (name + ".stderr.log")).open("w") as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                           text=True, env=env, start_new_session=True)
                def interrupted(signum, frame):
                    raise InterruptedError("Research round stopped by signal " + str(signum))
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                for sig in handlers:
                    signal.signal(sig, interrupted)
                try:
                    process.communicate(task, timeout=240 if args.smoke else 2700)
                except BaseException as failure:
                    # Keep the lock until the independent Codex process group has stopped.
                    for sig in handlers:
                        signal.signal(sig, signal.SIG_IGN)
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.communicate(timeout=20)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.communicate()
                    if isinstance(failure, subprocess.TimeoutExpired):
                        raise RuntimeError("Research round exceeded its time limit") from failure
                    raise
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
            if process.returncode:
                raise RuntimeError("Codex exit code " + str(process.returncode) + "; inspect private runtime logs")
            value = json.loads((run_dir / "response.json").read_text())
            validate_output(value)
            save_text(run_dir / "report.md", value["report_markdown"])
            metadata.update(status=value["status"], title=value["title"], finished_at=timestamp(),
                            duration_seconds=round(time.monotonic() - began, 2), breakthrough_verified=value["breakthrough_verified"])
            if args.smoke:
                save_json(REPORTS / "smoke_status.json", metadata | {"report": str(run_dir / "report.md")})
            else:
                save_text(run_dir / "research_memory.md", value["research_memory_markdown"])
                save_text(REPORTS / "research_memory.md", value["research_memory_markdown"])
                state.update(last_status=value["status"], last_success_at=timestamp(), last_report=str(run_dir / "report.md"),
                             next_focus=value["next_focus"], completed_rounds=state.get("completed_rounds", 0) + 1,
                             research_memory_path=str(REPORTS / "research_memory.md"), research_memory_updated_at=timestamp())
                state.pop("last_error", None)
                save_json(REPORTS / "state.json", state)
                save_text(REPORTS / "latest.md", value["report_markdown"])
                if value["breakthrough_verified"]:
                    with (REPORTS / "findings.jsonl").open("a") as findings:
                        findings.write(json.dumps({"at": timestamp(), "title": value["title"], "report": str(run_dir / "report.md")}, ensure_ascii=False) + "\n")
            save_json(run_dir / "run.json", metadata)
            return 0
        except Exception as error:
            metadata.update(status="failed", finished_at=timestamp(), error=str(error), duration_seconds=round(time.monotonic() - began, 2))
            save_json(run_dir / "run.json", metadata)
            if args.smoke:
                save_json(REPORTS / "smoke_status.json", metadata)
            else:
                state.update(last_status="failed", last_error=str(error))
                save_json(REPORTS / "state.json", state)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
