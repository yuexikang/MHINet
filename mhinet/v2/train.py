"""Resumable, separately supervised RGB512 long experiments for MHI v2."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mhinet.config import RuntimePaths, sha256_file
from mhinet.engine.checkpointing import (capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint)
from mhinet.downstream.metrics import dense_metrics
from . import SCHEMA, TASK
from .checkpointing import migrate_v1_state
from .config import SemidenseConfigV2
from .data import RGB512Dataset
from .feature_provider import build_v2_shared
from .semidense import SemidenseMatcherV2
from .training import SemidenseSystemV2, assert_frozen_gradients, batch_loss


ROOT = Path(__file__).resolve().parents[2]


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def final_run_status(step, total_steps, last_full_validation_step, stop_requested):
    # A campaign may advance only after the final epoch's validation is saved.
    if step == total_steps and last_full_validation_step == step:
        return "completed"
    if stop_requested or step == total_steps:
        return "interrupted"
    return "diagnostic_stop"


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+".temporary")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(target)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def read_config(path):
    config = json.loads(Path(path).read_text())
    allowed = {"schema", "task", "experiment_id", "variant", "stage", "runtime", "checkpoint", "seed",
               "epochs", "accumulation", "learning_rate", "minimum_learning_rate", "warmup_steps", "weight_decay",
               "save_every", "validate_every", "validation_limit", "initial_validation_limit", "workers", "matcher",
               "predicted_fraction", "correction_limit_d2", "d2_cache", "cache_source_identity", "notes"}
    if set(config)-allowed:
        raise ValueError(f"Unknown run fields: {sorted(set(config)-allowed)}")
    if config.get("schema") != SCHEMA or config.get("task") != TASK:
        raise ValueError("Expected executable MHI v2 run schema/task")
    if config.get("stage") not in ("flow_head", "qrru"):
        raise ValueError("Expected separate flow_head or qrru stage")
    for name in ("epochs", "accumulation", "save_every", "validate_every"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seed", "warmup_steps", "validation_limit", "initial_validation_limit", "workers"):
        value = config.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError(name)
    for name in ("learning_rate", "minimum_learning_rate", "weight_decay"):
        value = config.get(name, 0.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name)
    if config["learning_rate"] <= 0 or config.get("minimum_learning_rate", 0) > config["learning_rate"]:
        raise ValueError("Invalid learning rates")
    fraction = config.get("predicted_fraction", .5)
    correction = config.get("correction_limit_d2", 6.)
    if not 0 <= fraction <= 1 or not 0 < correction <= 6:
        raise ValueError("Invalid QRRU input mixture/correction range")
    matcher = SemidenseConfigV2.from_dict(config.get("matcher", {}))
    if config["stage"] == "flow_head" and matcher.geometry_mode != "flow_d16":
        raise ValueError("Flow-head supervision requires flow_d16 geometry")
    if config["stage"] == "qrru" and matcher.geometry_mode == "h0":
        raise ValueError("A is a frozen evaluation control; mixed QRRU training requires a no-H variant")
    return config, matcher


def initialize_system(runtime, config, matcher_config, checkpoint):
    shared, provider_report = build_v2_shared(runtime, matcher_config.geometry_mode)
    system = SemidenseSystemV2(shared, SemidenseMatcherV2(matcher_config), config["stage"]).to(runtime.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    source_metadata = payload.get("metadata", {})
    if source_metadata.get("task") not in ("semidense_qrru_v1", TASK):
        raise ValueError("Source is neither the frozen v1 nor a versioned v2 checkpoint")
    if source_metadata.get("dino_sha256") != sha256_file(runtime.dino_checkpoint):
        raise ValueError("Frozen DINO source identity mismatch")
    migration = migrate_v1_state(system, payload["model"])
    system.configure_stage(config["stage"])
    del payload
    return system, provider_report, migration


def buffer_digest(system):
    digest = hashlib.sha256()
    for name, value in sorted(system.named_buffers()):
        digest.update(name.encode()); digest.update(str((value.dtype, tuple(value.shape))).encode())
        # NumPy cannot represent BF16; hash its raw bytes without a lossy cast.
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def means(records):
    keys = sorted({key for row in records for key, value in row.items() if isinstance(value, (int, float))})
    return {key: float(np.mean([row[key] for row in records if isinstance(row.get(key), (int, float))])) for key in keys}


def implementation_hashes(runtime):
    """Pin local implementation and imported external LoMa/LoRetta sources."""
    paths = set(Path(__file__).parent.glob("*.py"))
    roots = (ROOT, runtime.loma_root, runtime.loretta_source_root)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_file() and any(path.is_relative_to(root.resolve()) for root in roots):
            paths.add(path)
    return {str(path): sha256_file(path) for path in sorted(paths)}


@torch.no_grad()
def validate(system, dataset, runtime, output, step, config, *, limit=0, label="validation", should_stop=None):
    rng = capture_rng_state(); mode = system.training
    system.eval(); seed_all(123)
    indices = range(min(limit, len(dataset))) if limit else range(len(dataset))
    records = []
    started = time.perf_counter()
    try:
        for index, batch in zip(indices, DataLoader(Subset(dataset, indices), batch_size=1, num_workers=config.get("workers", 0))):
            if should_stop is not None and should_stop():
                break
            if system.stage == "qrru" and not limit:
                # Model selection uses cascaded validation precision, not the
                # easier GT-noise refinement loss used for gradient diagnostics.
                shared = system.shared(batch["images"].to(runtime.device))
                prediction = system.matcher.infer(shared)[0]
                row = dataset._read_record(dataset.index[index].offset)
                if row["size_A"] != [512, 512] or row["size_B"] != [512, 512]:
                    raise ValueError("RGB512 validation rows must use stored 512-pixel coordinates")
                measured = dense_metrics(row, dataset.split_root, prediction["points_a"].cpu().numpy(), prediction["points_b"].cpu().numpy())
                metrics = {f"pre_at_{threshold}": measured["thresholds"][f"{threshold}.0"]["precision"] for threshold in (1, 3, 5, 10)}
                records.append(dict(pair_id=batch["pair_id"][0], **metrics, matches=measured["matches"],
                                    failure=int(prediction["failure_reason"] != "none")))
            else:
                loss, metrics = batch_loss(system, batch, runtime.device, config)
                records.append(dict(pair_id=batch["pair_id"][0], loss=float(loss), **metrics))
        summary = dict(step=step, pairs=len(records), scope="full_val" if len(records) == len(dataset) else "fixed_val_subset",
                       complete=len(records) == len(indices),
                       metrics=means(records), seconds=time.perf_counter()-started, label=label, checked_at=timestamp(),
                       selection_protocol="Flow EPE" if system.stage == "flow_head" else "Full-val cascaded raster-mask Pre@3; subset losses are diagnostics only")
        write_json(output/"validation"/f"{label}_{step:07d}.json", dict(summary=summary, pairs=records))
        return summary
    finally:
        restore_rng_state(rng); system.train(mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, help="Diagnostic stop; never changes the preregistered LR schedule")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for value in (args.max_steps, args.limit_train, args.limit_val):
        if value is not None and value < 1:
            raise ValueError("Diagnostic limits must be positive")
    config, mc = read_config(args.config)
    runtime = replace(RuntimePaths.from_json(config["runtime"]), device=args.device)
    source = (args.checkpoint or Path(config["checkpoint"])).resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    seed_all(config["seed"])
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    train = RGB512Dataset(runtime.data_root/"train/pairs.jsonl", max_pairs=args.limit_train, image_size=512)
    val = RGB512Dataset(runtime.data_root/"val/pairs.jsonl", max_pairs=args.limit_val, image_size=512)
    if not len(train) or not len(val):
        raise ValueError("Empty train/validation manifest")
    if ({x.geo_group for x in train.index} & {x.geo_group for x in val.index}
            or {x.parent_group for x in train.index} & {x.parent_group for x in val.index}):
        raise ValueError("Train/validation source-image overlap")
    per_epoch = math.ceil(len(train)/config["accumulation"])
    total_steps = per_epoch*config["epochs"]
    dry = dict(task=TASK, stage=config["stage"], experiment_id=config["experiment_id"],
               train_pairs=len(train), val_pairs=len(val), epochs=config["epochs"], total_steps=total_steps,
               effective_batch=config["accumulation"], input_size=512, max_output_matches=mc.max_output_matches,
               data_root=str(runtime.data_root), checkpoint=str(source), device=runtime.device,
               source_counts={"train":dict(train.source_counts),"val":dict(val.source_counts)},
               grouping_policy=train.grouping_policy,
               scope="diagnostic" if args.limit_train or args.limit_val or args.max_steps else "full_train_long_run",
               train_sha256=sha256_file(train.manifest), val_sha256=sha256_file(val.manifest))
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False)); return 0
    output = args.output.resolve()
    latest = output/"latest.pt"
    resume = args.resume or (latest if latest.exists() else None)
    if output.exists() and any(p.name != ".training.lock" for p in output.iterdir()) and resume is None:
        raise FileExistsError("Nonempty output without a resumable latest.pt; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/".training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    system, provider_report, migration = initialize_system(runtime, config, mc, source)
    implementation = implementation_hashes(runtime)
    frozen_buffers = buffer_digest(system)
    source_sha = sha256_file(source)
    if config["stage"] == "qrru" and config.get("d2_cache"):
        candidate_settings = {key: value for key, value in asdict(mc).items()
                              if key not in ("use_discrepancy", "use_dgfo", "dgfo_radius", "dgfo_dim")}
        cache_identity = hashlib.sha256(json.dumps(dict(source_sha256=source_sha,
            train_sha256=dry["train_sha256"], val_sha256=dry["val_sha256"],
            matching=candidate_settings), sort_keys=True).encode()).hexdigest()
        configured_identity = config.get("cache_source_identity")
        if configured_identity and configured_identity != cache_identity:
            raise ValueError("Configured D2 cache identity differs from its actual fixed model/data/settings")
        config["cache_source_identity"] = cache_identity
    metadata = dict(task=TASK, schema=SCHEMA, config=config, matcher=asdict(mc), runtime=json.loads(json.dumps(asdict(runtime), default=str)),
                    source_checkpoint=str(source), source_sha256=source_sha,
                    dino_sha256=sha256_file(runtime.dino_checkpoint), train_sha256=dry["train_sha256"], val_sha256=dry["val_sha256"],
                    train_pairs=len(train), val_pairs=len(val), total_steps=total_steps, implementation=implementation,
                    torch_version=str(torch.__version__), numpy_version=str(np.__version__), cuda_version=str(torch.version.cuda),
                    freeze_policy="Only Flow Head" if system.stage == "flow_head" else "Only QRRU and its enabled branches",
                    trainable_parameters=system.training_parameter_names(), frozen_buffer_sha256=frozen_buffers)
    optimizer = torch.optim.AdamW(system.optimizer_groups(config["learning_rate"]), weight_decay=config.get("weight_decay", .0001))
    warmup = config.get("warmup_steps", 0); minimum = config.get("minimum_learning_rate", 0)/config["learning_rate"]
    def multiplier(step):
        if warmup and step < warmup:
            return (step+1)/warmup
        phase = min(1., max(0., (step-warmup)/max(1, total_steps-warmup)))
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*phase))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    step = cursor = epoch = 0; best_score = None; best_step = None; last_full_validation_step = -1
    if resume:
        restored = load_checkpoint(resume, model=system, optimizer=optimizer, scheduler=scheduler, map_location="cpu", expected_metadata=metadata)
        step = restored["optimizer_step"]; cursor = restored["data_progress"]["cursor"]; epoch = restored["data_progress"]["epoch"]
        best_score = (restored.get("auxiliary_state") or {}).get("best_score")
        best_step = (restored.get("auxiliary_state") or {}).get("best_step")
        last_full_validation_step = (restored.get("auxiliary_state") or {}).get("last_full_validation_step", -1)
    else:
        # Architecture initialization must not perturb common C/D/E input/noise RNG.
        seed_all(config["seed"])
    write_json(output/"run.json", {**metadata, "registration": dry, "provider": provider_report, "migration": migration,
               "registered_at": timestamp(), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    def save(path):
        save_checkpoint(path, model=system, optimizer=optimizer, scheduler=scheduler, optimizer_step=step,
                        data_progress=dict(epoch=epoch, cursor=cursor), metadata=metadata,
                        auxiliary_state=dict(best_score=best_score, best_step=best_step,
                                             last_full_validation_step=last_full_validation_step))
    if step == 0:
        # A restart remains possible even if initial validation is interrupted.
        save(latest)
        initial = validate(system, val, runtime, output, step, config, limit=config.get("initial_validation_limit", 64), label="initial",
                           should_stop=lambda: stop["requested"])
        if buffer_digest(system) != frozen_buffers:
            raise RuntimeError("Frozen buffers changed during initial validation")
        print(json.dumps(dict(event="initialized", **dry, validation=initial, trainable_tensors=len(metadata["trainable_parameters"])), ensure_ascii=False), flush=True)
    if step > 0 and cursor == len(train) and last_full_validation_step != step:
        recovered = validate(system, val, runtime, output, step, config, label="recovered_epoch_validation",
                             should_stop=lambda: stop["requested"])
        if recovered["complete"]:
            score = recovered["metrics"].get("flow_epe_px" if system.stage == "flow_head" else "pre_at_3")
            last_full_validation_step = step
            better = best_score is None or (score < best_score if system.stage == "flow_head" else score > best_score) if score is not None else False
            if score is not None and better:
                best_score, best_step = score, step
                save(output/"best.pt")
            save(latest)
    end = min(total_steps, args.max_steps or total_steps)
    system.train()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    while step < end and not stop["requested"]:
        if cursor >= len(train):
            cursor = 0; epoch += 1
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config["seed"]+epoch)).tolist()
        loader = iter(DataLoader(Subset(train, order[cursor:]), batch_size=1, num_workers=config.get("workers", 0),
                                 generator=torch.Generator().manual_seed(config["seed"]+epoch), pin_memory=args.device.startswith("cuda")))
        while cursor < len(train) and step < end and not stop["requested"]:
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            count = min(config["accumulation"], len(train)-cursor)
            records = []
            for _ in range(count):
                loss, record = batch_loss(system, next(loader), runtime.device, config)
                (loss/count).backward(); cursor += 1
                records.append(dict(loss=float(loss.detach()), **record))
            assert_frozen_gradients(system)
            trainable = [p for p in system.parameters() if p.requires_grad]
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            if step == 1 and buffer_digest(system) != frozen_buffers:
                raise RuntimeError("Frozen buffers changed after optimization")
            seconds = time.perf_counter()-started
            row = dict(event="training", step=step, total_steps=total_steps, epoch=epoch, cursor=cursor, pairs=count,
                       gradient_norm=float(gradient), learning_rate=optimizer.param_groups[0]["lr"], seconds=seconds,
                       metrics=means(records), checked_at=timestamp(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0)
            with (output/"train.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps(row), flush=True)
            if step % config["save_every"] == 0 or step == end or cursor == len(train):
                save(latest)
            full = cursor == len(train) or step == total_steps
            if full or step % config["validate_every"] == 0 or step == end:
                validation = validate(system, val, runtime, output, step, config,
                                      limit=0 if full else config.get("validation_limit", 64),
                                      should_stop=lambda: stop["requested"])
                if buffer_digest(system) != frozen_buffers:
                    raise RuntimeError("Frozen buffers changed during validation")
                if validation["scope"] == "full_val" and validation["complete"]:
                    score = validation["metrics"].get("flow_epe_px" if system.stage == "flow_head" else "pre_at_3")
                    last_full_validation_step = step
                    better = best_score is None or (score < best_score if system.stage == "flow_head" else score > best_score) if score is not None else False
                    if score is not None and better:
                        best_score, best_step = score, step
                        save(output/"best.pt")
                    save(output/f"epoch_{epoch+1:03d}.pt")
                    save(latest)
                print(json.dumps(dict(event="validation", **validation)), flush=True)
            write_json(output/"status.json", dict(status="running", step=step, total_steps=total_steps, epoch=epoch, cursor=cursor,
                                                  best_score=best_score, best_step=best_step, checked_at=timestamp()))
    save(latest)
    status = final_run_status(step, total_steps, last_full_validation_step, stop["requested"])
    write_json(output/"status.json", dict(status=status, step=step, total_steps=total_steps, epoch=epoch, cursor=cursor,
                                          best_score=best_score, best_step=best_step, checked_at=timestamp()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
