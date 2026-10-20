"""An isolated, fixed-budget sixth-epoch Flow-loss continuation experiment."""
import argparse
import copy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mhinet.config import sha256_file
from mhinet.engine.checkpointing import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from mhinet.v2 import SCHEMA, TASK
from mhinet.v2.train import buffer_digest, means, timestamp, validate, write_json
from mhinet.v2.training import assert_frozen_gradients
from .common import (EXTRA_STEPS, SOURCE_SHA, SOURCE_STEP, TRAIN_SHA, VAL_SHA,
                     code_hashes, data, digest_tree, load_source, read_calibration, setup)
from .objective import flow_objective


PILOT_SCHEMA = "mhi.flow-loss-pilot.v1"


def read_pilot_config(path):
    config = json.loads(Path(path).read_text())
    allowed = {"schema", "arm", "base_config", "runtime", "source_checkpoint", "source_sha256",
               "calibration_report", "calibration_sha256", "source_validation", "source_validation_sha256",
               "seed", "order_seed", "additional_steps", "learning_rate", "accumulation", "workers",
               "save_every", "validate_every", "validation_limit", "epsilon_px", "experiment_id", "notes"}
    if not isinstance(config, dict) or set(config)-allowed:
        raise ValueError("Unknown pilot configuration fields")
    if config.get("schema") != PILOT_SCHEMA or config.get("arm") not in ("L0", "L1"):
        raise ValueError("Expected isolated pilot schema and arm L0/L1")
    for name in ("base_config", "runtime", "source_checkpoint", "calibration_report"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"Missing pilot input: {name}")
    fixed = dict(seed=42, order_seed=47, additional_steps=EXTRA_STEPS, learning_rate=1e-6,
                 accumulation=8, save_every=250, validate_every=1000, validation_limit=64, epsilon_px=.5)
    for name, expected in fixed.items():
        value = config.setdefault(name, expected)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
            raise ValueError(f"Pilot fixes {name}={expected}")
        if type(expected) is int and type(value) is not int:
            raise ValueError(f"{name} must be integer")
    if type(config.setdefault("workers", 4)) is not int or config["workers"] not in (2, 4):
        raise ValueError("Pilot workers must be 2 or 4, common to both arms")
    if config.setdefault("source_sha256", SOURCE_SHA) != SOURCE_SHA:
        raise ValueError("Source checkpoint identity mismatch")
    return config


def assert_tree_equal(actual, expected, path="state"):
    """Check every value/tensor, independent of storage device."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise RuntimeError(f"Tensor clone mismatch: {path}")
        return 1
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError(f"Mapping clone mismatch: {path}")
        return sum(assert_tree_equal(actual[k], v, f"{path}.{k}") for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise RuntimeError(f"Sequence clone mismatch: {path}")
        return sum(assert_tree_equal(a, b, f"{path}.{i}") for i, (a, b) in enumerate(zip(actual, expected)))
    if actual != expected:
        raise RuntimeError(f"Value clone mismatch: {path}")
    return 0


class FixedExtensionScheduler:
    """An explicitly new constant-LR extension, never a restored cosine."""
    def __init__(self, optimizer, learning_rate=1e-6):
        self.optimizer, self.learning_rate, self.completed = optimizer, learning_rate, 0
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            if "initial_lr" in group:
                group["initial_lr"] = learning_rate

    def step(self):
        self.completed += 1
        for group in self.optimizer.param_groups:
            if group["lr"] != self.learning_rate:
                raise RuntimeError("Fixed extension learning rate changed")

    def state_dict(self):
        return dict(kind="fixed_lr_extension_v1", learning_rate=self.learning_rate, completed=self.completed)

    def load_state_dict(self, state):
        if state.get("kind") != "fixed_lr_extension_v1" or state.get("learning_rate") != self.learning_rate:
            raise ValueError("Cannot restore a different extension schedule")
        self.completed = int(state["completed"])
        if not 0 <= self.completed <= EXTRA_STEPS or any(g["lr"] != self.learning_rate for g in self.optimizer.param_groups):
            raise ValueError("Invalid extension scheduler progress/LR")


def clone_source_state(system, optimizer, payload, *, restore_rng=True):
    system.load_state_dict(payload["model"], strict=True)
    model_tensors = assert_tree_equal(system.state_dict(), payload["model"], "model")
    # CPU load_state_dict may otherwise alias checkpoint moment tensors. The
    # source payload must remain immutable for independent arms and CPU replay.
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    optimizer_tensors = assert_tree_equal(optimizer.state_dict(), payload["optimizer"], "optimizer")
    if not payload["optimizer"]["state"]:
        raise ValueError("Source optimizer has no moments")
    for state in optimizer.state_dict()["state"].values():
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError("Expected complete AdamW source moments/steps")
    evidence = dict(model_tensors_verified=model_tensors, optimizer_tensors_verified=optimizer_tensors,
                    model_sha256=digest_tree(payload["model"]), optimizer_sha256=digest_tree(payload["optimizer"]),
                    rng_sha256=digest_tree(payload["rng"]), every_source_tensor_exact=True)
    if restore_rng:
        evidence["rng_restored"] = restore_rng_state(payload["rng"])
        assert_tree_equal(capture_rng_state(), payload["rng"], "rng")
    return evidence


def progress_valid(step, cursor, epoch, scheduler_completed, train_pairs=40849):
    extra = step-SOURCE_STEP
    expected_cursor = min(extra*8, train_pairs)
    if not 0 <= extra <= EXTRA_STEPS or epoch != 5 or cursor != expected_cursor or scheduler_completed != extra:
        raise ValueError("Pilot checkpoint progress/order/scheduler mismatch")


def completion_status(step, full_step, stopped):
    if step == SOURCE_STEP+EXTRA_STEPS and full_step == step:
        return "completed"
    return "interrupted" if stopped or step == SOURCE_STEP+EXTRA_STEPS else "diagnostic_stop"


def pilot_batch_loss(system, batch, device, *, arm, alpha, epsilon):
    tensors = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    shared = system(tensors["images"])
    loss, metrics = flow_objective(shared["stage1"], tensors["H_gt_norm"], tensors["mask_A_overlap"],
                                   tensors["mask_B_overlap"], arm=arm, alpha=alpha, smooth_eps_px=epsilon)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite pilot objective")
    return loss, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Static registration validation; calibration may still be pending")
    parser.add_argument("--max-steps", type=int, help="Stop after N extension updates; registered budget is unchanged")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("Diagnostic max-steps must be positive")
    config = read_pilot_config(args.config)
    if args.dry_run:
        from mhinet.v2.train import read_config
        base, matcher = read_config(config["base_config"])
        if base["stage"] != "flow_head" or matcher.geometry_mode != "flow_d16":
            raise ValueError("Expected Flow-head base configuration")
        print(json.dumps(dict(status="static_registration_passed", config=config,
            source_step=SOURCE_STEP, total_steps=SOURCE_STEP+EXTRA_STEPS,
            calibration_status="exists" if Path(config["calibration_report"]).is_file() else "pending",
            note="No model/data/GPU initialized; execution rechecks source, data, calibration and complete validation identities")))
        return 0
    setup(config["seed"])
    output = args.output.resolve(); latest = output/"latest.pt"
    resume = args.resume or (latest if latest.exists() else None)
    if output.exists() and any(p.name != ".training.lock" for p in output.iterdir()) and resume is None:
        raise FileExistsError("Nonempty pilot output without a resumable checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/".training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calibration = read_calibration(config["calibration_report"], config["epsilon_px"])
    alpha = 1. if config["arm"] == "L0" else float(calibration["alpha"])
    for field, filename in (("calibration_sha256", config["calibration_report"]),):
        if field in config and config[field] != sha256_file(filename):
            raise ValueError(f"{field} differs from registered input")
    runtime, base, matcher, system, payload, provider, migration = load_source(
        config["base_config"], config["runtime"], config["source_checkpoint"], args.device)
    train, val = data(runtime)
    source_validation = Path(config.get("source_validation", str(Path(config["source_checkpoint"]).parent/"validation/validation_0025535.json")))
    baseline = json.loads(source_validation.read_text())
    summary = baseline["summary"]
    if (summary.get("step") != SOURCE_STEP or summary.get("pairs") != 5106 or not summary.get("complete")
            or len(baseline.get("pairs", [])) != 5106 or summary.get("scope") != "full_val"):
        raise ValueError("Source validation must be complete full-val at step25535")
    if "source_validation_sha256" in config and config["source_validation_sha256"] != sha256_file(source_validation):
        raise ValueError("Source validation identity mismatch")
    source_aux = payload.get("auxiliary_state") or {}
    if source_aux.get("last_full_validation_step") != SOURCE_STEP or not math.isclose(
            summary["metrics"]["flow_epe_px"], source_aux.get("best_score", -1), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Source checkpoint/full-validation selection mismatch")
    if math.ceil(len(train)/config["accumulation"]) != EXTRA_STEPS:
        raise ValueError("Extension budget differs from one complete epoch")
    optimizer = torch.optim.AdamW(system.optimizer_groups(config["learning_rate"]), weight_decay=base.get("weight_decay", .0001))
    clone = clone_source_state(system, optimizer, payload)
    # The source cosine has ended. Restore its exact moments first, then start
    # an explicitly declared, common fixed-LR extension for both pilot arms.
    scheduler = FixedExtensionScheduler(optimizer, config["learning_rate"])
    frozen = buffer_digest(system)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config["order_seed"])).tolist()
    metadata = dict(task=TASK, schema=SCHEMA, experiment_kind="flow_loss_pilot_v1", config=base,
        matcher=asdict(matcher), runtime={k:str(v) if isinstance(v, Path) else v for k,v in asdict(runtime).items()},
        source_checkpoint=str(Path(config["source_checkpoint"]).resolve()), source_sha256=SOURCE_SHA,
        dino_sha256=payload["metadata"]["dino_sha256"], train_sha256=TRAIN_SHA, val_sha256=VAL_SHA,
        train_pairs=len(train), val_pairs=len(val), total_steps=SOURCE_STEP+EXTRA_STEPS,
        runtime_sha256=sha256_file(config["runtime"]), implementation=code_hashes(runtime),
        torch_version=str(torch.__version__), numpy_version=str(np.__version__),
        cuda_version=str(torch.version.cuda), deterministic_algorithms=True,
        cpu_threads=torch.get_num_threads(),
        trainable_parameters=system.training_parameter_names(), frozen_buffer_sha256=frozen,
        flow_loss_pilot=dict(config=config, config_sha256=sha256_file(args.config), alpha=alpha,
            base_config_sha256=sha256_file(config["base_config"]), calibration_sha256=sha256_file(config["calibration_report"]),
            source_validation=str(source_validation.resolve()), source_validation_sha256=sha256_file(source_validation),
            source_state_clone=clone, source_scheduler_sha256=digest_tree(payload["scheduler"]),
            extension_scheduler="fixed_lr_extension_v1", primary_checkpoint="last.pt; fixed complete budget",
            sixth_pass_order_sha256=digest_tree(order),
            sixth_pass_first16_pair_ids=[train.index[i].pair_id for i in order[:16]],
            validation_protocol="fixed64 every1000 extension steps; full5106 at terminal5107 only; subset never selects checkpoint"))
    del payload
    step, cursor, epoch, full_step, best_score = SOURCE_STEP, 0, 5, -1, None
    if resume:
        restored = load_checkpoint(resume, model=system, optimizer=optimizer, scheduler=scheduler,
                                   map_location="cpu", expected_metadata=metadata)
        step = restored["optimizer_step"]; cursor = restored["data_progress"]["cursor"]; epoch = restored["data_progress"]["epoch"]
        extra = restored.get("auxiliary_state") or {}
        full_step = extra.get("last_full_validation_step", -1); best_score = extra.get("best_score")
    progress_valid(step, cursor, epoch, scheduler.completed, len(train))
    if buffer_digest(system) != frozen:
        raise RuntimeError("Frozen buffers differ after initialization/resume")
    write_json(output/"run.json", {**metadata, "provider":provider, "migration":migration,
               "source_validation_summary":summary, "registered_at":timestamp()})
    stop = {"requested":False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    def save(path):
        save_checkpoint(path, model=system, optimizer=optimizer, scheduler=scheduler, optimizer_step=step,
            data_progress=dict(epoch=epoch, cursor=cursor), metadata=metadata,
            auxiliary_state=dict(best_score=best_score, last_full_validation_step=full_step))
    def persist_status(status="running"):
        write_json(output/"status.json", dict(status=status, arm=config["arm"], step=step,
            extra_steps=step-SOURCE_STEP, total_steps=SOURCE_STEP+EXTRA_STEPS, additional_steps=EXTRA_STEPS,
            epoch=epoch, cursor=cursor, last_full_validation_step=full_step, best_score=best_score,
            checked_at=timestamp()))
    if not resume:
        save(latest)
    persist_status()
    end = SOURCE_STEP+min(EXTRA_STEPS, args.max_steps or EXTRA_STEPS)
    loader = iter(DataLoader(Subset(train, order[cursor:]), batch_size=1, num_workers=config["workers"],
        generator=torch.Generator().manual_seed(config["order_seed"]), pin_memory=args.device.startswith("cuda"))) if cursor < len(train) else None
    system.train()
    validation_config = {**base, "workers":config["workers"]}
    try:
        while step < end and not stop["requested"]:
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            count = min(config["accumulation"], len(train)-cursor); records = []; pair_ids = []
            for _ in range(count):
                batch = next(loader)
                pair_ids.append(batch["pair_id"][0])
                loss, record = pilot_batch_loss(system, batch, runtime.device, arm=config["arm"], alpha=alpha, epsilon=config["epsilon_px"])
                (loss/count).backward(); cursor += 1
                records.append(dict(loss=float(loss.detach()), **record))
            assert_frozen_gradients(system)
            gradient = torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            if (step-SOURCE_STEP) == 1 or (step-SOURCE_STEP)%250 == 0:
                if buffer_digest(system) != frozen:
                    raise RuntimeError("Frozen buffers changed during pilot optimization")
            row = dict(event="training", arm=config["arm"], step=step, extra_step=step-SOURCE_STEP,
                pairs=count, epoch=epoch, cursor=cursor, learning_rate=optimizer.param_groups[0]["lr"],
                gradient_norm=float(gradient), seconds=time.perf_counter()-started, metrics=means(records), checked_at=timestamp())
            row["pair_ids"] = pair_ids
            with (output/"train.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps(row), flush=True)
            if (step-SOURCE_STEP)%250 == 0 or step == end:
                save(latest)
            if (step-SOURCE_STEP)%1000 == 0 and step < SOURCE_STEP+EXTRA_STEPS and not stop["requested"]:
                validate(system, val, runtime, output, step, validation_config, limit=64,
                         label="fixed64", should_stop=lambda:stop["requested"])
                if buffer_digest(system) != frozen:
                    raise RuntimeError("Frozen buffers changed during subset validation")
            persist_status()
        if step == SOURCE_STEP+EXTRA_STEPS and full_step != step and not stop["requested"]:
            save(output/"last.pt")  # Fixed budget exists even if validation is interrupted.
            final = validate(system, val, runtime, output, step, validation_config,
                label="full_terminal", should_stop=lambda:stop["requested"])
            if buffer_digest(system) != frozen:
                raise RuntimeError("Frozen buffers changed during full validation")
            if final["complete"] and final["scope"] == "full_val" and final["pairs"] == 5106:
                full_step, best_score = step, final["metrics"]["flow_epe_px"]
                save(output/"last.pt"); save(output/"best.pt"); save(output/"best_full.pt")
        save(latest)
        persist_status(completion_status(step, full_step, stop["requested"]))
    except BaseException as error:
        # A partially accumulated update cannot be resumed exactly. Preserve the
        # latest completed-update checkpoint, never publish the current cursor.
        write_json(output/"failure.json", dict(status="failed", error=repr(error), checked_at=timestamp(),
                   resume_checkpoint=str(latest), note="Resume latest completed optimizer boundary; partial microbatch discarded"))
        persist_status("failed")
        raise
    finally:
        if loader is not None and hasattr(loader, "_shutdown_workers"):
            loader._shutdown_workers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
