"""Protocol-v1.2 single-GPU MHINet trainer with exact boundary resume."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from .checkpointing import load_checkpoint, save_checkpoint
from .config import RuntimePaths, sha256_file
from .data import HomographyPairDataset, iter_manifest, parse_geo_region
from .evaluate import evaluate_model, write_evaluation
from .losses import sequence_corner_l1
from .metrics import homography_trajectory_metrics
from .model import MHINet, build_model
from .tiny_gate import (
    REGISTERED_PROTOCOL,
    REQUIRED_EXPERIMENTS,
    registered_experiment_errors,
)


@dataclass(frozen=True)
class TrainConfig:
    path: Path
    raw: Mapping[str, Any]
    sha256: str

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Training config root must be an object")
        result = cls(path=source, raw=payload, sha256=sha256_file(source))
        result.validate()
        return result

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment_id"])

    @property
    def profile(self) -> str:
        return str(self.raw["profile"])

    @property
    def active_scales(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["active_scales"])

    def validate(self) -> None:
        errors: list[str] = []
        if self.raw.get("training_revision") != "1.2":
            errors.append("training_revision must be 1.2")
        if self.raw.get("loss_profile") != "sequence_corner_l1":
            errors.append("only sequence_corner_l1 is a mainline training loss")
        if self.raw.get("fgo_enabled") is not False:
            errors.append("FGO must be disabled in the mainline config")
        if self.raw.get("extra_losses_enabled") is not False:
            errors.append("extra losses must be disabled in the mainline config")
        if self.active_scales not in {
            (8,),
            (8, 4),
            (8, 4, 2),
            (8, 4, 2, 1),
        }:
            errors.append("active_scales must be a coarse-to-fine prefix")
        if int(self.raw.get("iterations_per_scale", 0)) not in (1, 2):
            errors.append("iterations_per_scale must be 1 or 2")
        if str(self.raw.get("profile")) not in {
            "heads",
            "decoder_finetune",
            "vgg_finetune",
            "mvt_finetune",
            "joint",
        }:
            errors.append("profile is not a protocol-v1.2 training group")
        if int(self.raw.get("gradient_accumulation", 0)) != 4:
            errors.append("gradient_accumulation must be 4")
        if not math.isclose(
            float(self.raw.get("gradient_clip_norm", 0)), 1.0, rel_tol=0, abs_tol=1e-12
        ):
            errors.append("gradient_clip_norm must be 1")
        if not math.isclose(
            float(self.raw.get("weight_decay", -1)), 1e-4, rel_tol=0, abs_tol=1e-12
        ):
            errors.append("weight_decay must be 1e-4")
        if int(self.raw.get("max_optimizer_steps", 0)) <= 0:
            errors.append("max_optimizer_steps must be positive")
        if not math.isclose(
            float(self.raw.get("warmup_fraction", -1)), 0.05, rel_tol=0, abs_tol=1e-12
        ):
            errors.append("warmup_fraction must be 0.05")
        if not math.isclose(
            float(self.raw.get("minimum_lr_ratio", 0)), 0.1, rel_tol=0, abs_tol=1e-12
        ):
            errors.append("minimum_lr_ratio must be 0.1")
        if int(self.raw.get("validation_interval", 0)) <= 0:
            errors.append("validation_interval must be positive")
        if int(self.raw.get("checkpoint_interval", 0)) <= 0:
            errors.append("checkpoint_interval must be positive")
        if errors:
            raise ValueError("Invalid training config: " + "; ".join(errors))


class DeterministicIndexStream:
    """Epoch-shuffled indices reconstructible from seed/epoch/position."""

    def __init__(
        self,
        length: int,
        seed: int,
        *,
        epoch: int = 0,
        position: int = 0,
    ) -> None:
        if length <= 0:
            raise ValueError("Training dataset is empty")
        if epoch < 0 or position < 0 or position > length:
            raise ValueError("Invalid deterministic data progress")
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.position = int(position)
        self.order = self._make_order(self.epoch)

    def _make_order(self, epoch: int) -> list[int]:
        order = list(range(self.length))
        random.Random(self.seed + int(epoch)).shuffle(order)
        return order

    def next(self) -> int:
        if self.position >= self.length:
            self.epoch += 1
            self.position = 0
            self.order = self._make_order(self.epoch)
        index = self.order[self.position]
        self.position += 1
        return index

    def state_dict(self) -> dict[str, int]:
        return {
            "length": self.length,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }


def warmup_cosine_factor(
    scheduler_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    """5% linear warmup followed by cosine decay to 0.1 of initial LR."""

    scheduler_step = int(scheduler_step)
    if warmup_steps > 0 and scheduler_step < warmup_steps:
        return max((scheduler_step + 1) / warmup_steps, 1.0 / warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((scheduler_step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_train_dataset(
    runtime: RuntimePaths,
    *,
    max_pairs: int | None,
) -> tuple[HomographyPairDataset, list[str]]:
    validation_regions = {
        parse_geo_region(record)
        for record in iter_manifest(runtime.data_root / "val/pairs.jsonl")
    }
    dataset = HomographyPairDataset(
        runtime.data_root / "train/pairs.jsonl",
        max_pairs=max_pairs,
        exclude_geo_groups=validation_regions,
    )
    return dataset, sorted(validation_regions)


def _validate_tiny_gate(path: Path | None, required: bool) -> dict[str, Any] | None:
    if not required:
        return None
    if path is None:
        raise RuntimeError(
            "Formal training requires --tiny-gate-artifact with a passed TINY-S/TINY-8 report"
        )
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("gate") != "P4_TINY_S_TINY_8" or payload.get("status") != "passed":
        raise RuntimeError(f"Tiny gate is not passed: {source}")
    if payload.get("errors"):
        raise RuntimeError(f"Tiny gate contains merge errors: {payload['errors']}")
    protocol = payload.get("protocol", {})
    cache = payload.get("cache", {})
    sample_protocol = protocol.get(
        "sample_protocol", cache.get("sample_protocol")
    )
    diagnostic_count = protocol.get(
        "diagnostic_sample_count", cache.get("diagnostic_sample_count")
    )
    test_used = protocol.get("test_used", cache.get("test_used"))
    observed_protocol = {
        "training_revision": protocol.get("training_revision"),
        "loss": protocol.get("loss"),
        "FGO": protocol.get("FGO"),
        "extra_losses": protocol.get("extra_losses"),
        "planar_head": protocol.get("planar_head"),
        "precision": protocol.get("precision"),
        "tiny_weight_average_start_step": protocol.get(
            "tiny_weight_average_start_step"
        ),
        "sample_protocol": sample_protocol,
        "diagnostic_sample_count": diagnostic_count,
        "test_used": test_used,
    }
    if observed_protocol != REGISTERED_PROTOCOL:
        raise RuntimeError(
            "Tiny gate protocol/sample definition does not match the registered recipe"
        )
    experiments = payload.get("experiments", [])
    names = [item.get("name") for item in experiments]
    if len(names) != len(set(names)):
        raise RuntimeError("Tiny gate contains duplicate experiment names")
    if set(names) != set(REQUIRED_EXPERIMENTS):
        missing = sorted(set(REQUIRED_EXPERIMENTS) - set(names))
        extra = sorted(
            repr(name) for name in set(names) - set(REQUIRED_EXPERIMENTS)
        )
        raise RuntimeError(
            f"Tiny gate experiment set mismatch; missing={missing}, extra={extra}"
        )
    for item in experiments:
        name = str(item["name"])
        item_errors = registered_experiment_errors(item, name, source=str(source))
        if item_errors:
            raise RuntimeError("Tiny gate experiment mismatch: " + "; ".join(item_errors))
    return {"path": str(source), "sha256": sha256_file(source)}


def _compact_parameter_report(model: MHINet) -> dict[str, Any]:
    report = model.training_parameter_report()
    for group in report["groups"].values():
        group.pop("optimizer_parameter_ids", None)
    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_training_state(
    path: Path,
    *,
    model: MHINet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer_step: int,
    data_stream: DeterministicIndexStream,
    invalid_attempts_total: int,
    shared_call_violations: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    data_progress = {
        **data_stream.state_dict(),
        "invalid_attempts_total": int(invalid_attempts_total),
        "shared_call_count_violations": int(shared_call_violations),
    }
    report = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_step=optimizer_step,
        microbatch_progress={"accumulated_valid": 0},
        data_progress=data_progress,
        metadata=metadata,
    )
    report["sha256"] = sha256_file(path)
    return report


def train(
    runtime: RuntimePaths,
    config: TrainConfig,
    *,
    output_dir: Path,
    resume: Path | None,
    tiny_gate_artifact: Path | None,
    overwrite: bool,
    stop_after_optimizer_step: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None and not overwrite:
        raise FileExistsError(
            f"Refusing to start in non-empty {output_dir}; pass --overwrite or --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tiny_gate = _validate_tiny_gate(
        tiny_gate_artifact, bool(config.raw.get("require_passed_tiny_gate", True))
    )
    seed = int(config.raw["seed"])
    _seed_everything(seed)
    device = torch.device(runtime.device)
    model, build_report = build_model(runtime)
    for group in build_report["training_parameters"]["groups"].values():
        group.pop("optimizer_parameter_ids", None)
    model.set_training_phase(config.profile)
    optimizer_groups = model.optimizer_group_spec()
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(config.raw["weight_decay"]),
    )
    total_steps = int(config.raw["max_optimizer_steps"])
    run_limit = (
        total_steps
        if stop_after_optimizer_step is None
        else min(int(stop_after_optimizer_step), total_steps)
    )
    if run_limit <= 0:
        raise ValueError("stop_after_optimizer_step must be positive")
    warmup_steps = round(total_steps * float(config.raw["warmup_fraction"]))
    minimum_ratio = float(config.raw["minimum_lr_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            minimum_ratio=minimum_ratio,
        ),
    )
    max_train_pairs = config.raw.get("max_train_pairs")
    train_dataset, excluded_regions = _safe_train_dataset(
        runtime,
        max_pairs=None if max_train_pairs is None else int(max_train_pairs),
    )
    max_val_pairs = config.raw.get("max_val_pairs")
    validation_dataset = HomographyPairDataset(
        runtime.data_root / "val/pairs.jsonl",
        max_pairs=None if max_val_pairs is None else int(max_val_pairs),
    )

    optimizer_step = 0
    resume_report = None
    data_progress: Mapping[str, Any] = {}
    if resume is not None:
        resume_report = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            restore_rng=True,
        )
        metadata = resume_report.get("metadata", {})
        if metadata.get("training_config_sha256") != config.sha256:
            raise RuntimeError("Resume training config hash does not match checkpoint")
        if metadata.get("profile") != config.profile:
            raise RuntimeError("Resume profile does not match checkpoint")
        optimizer_step = int(resume_report["optimizer_step"])
        data_progress = resume_report.get("data_progress") or {}
    data_stream = DeterministicIndexStream(
        len(train_dataset),
        seed,
        epoch=int(data_progress.get("epoch", 0)),
        position=int(data_progress.get("position", 0)),
    )
    if data_progress and int(data_progress.get("length", len(train_dataset))) != len(
        train_dataset
    ):
        raise RuntimeError("Resume dataset length changed")

    active_scales = config.active_scales
    iterations_per_scale = int(config.raw["iterations_per_scale"])
    accumulation = int(config.raw["gradient_accumulation"])
    clip_norm = float(config.raw["gradient_clip_norm"])
    validation_interval = int(config.raw["validation_interval"])
    checkpoint_interval = int(config.raw["checkpoint_interval"])
    metadata = {
        "experiment_id": config.experiment_id,
        "training_revision": "1.2",
        "loss_revision": "1.1",
        "training_config": str(config.path),
        "training_config_sha256": config.sha256,
        "architecture_sha256": build_report["architecture_sha256"],
        "runtime_config": str(runtime.source_path),
        "profile": config.profile,
        "active_scales": active_scales,
        "iterations_per_scale": iterations_per_scale,
        "tiny_gate": tiny_gate,
        "test_used": False,
    }
    run_record = {
        "status": "running",
        "metadata": metadata,
        "build": build_report,
        "train_pairs": len(train_dataset),
        "validation_pairs": len(validation_dataset),
        "excluded_train_geo_regions": excluded_regions,
        "parameter_report": _compact_parameter_report(model),
        "resume": resume_report,
    }
    _write_json(output_dir / "run.json", run_record)
    log_path = output_dir / "train.jsonl"
    log_mode = "a" if resume is not None else "w"
    last_checkpoint: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None
    invalid_attempts_total = int(data_progress.get("invalid_attempts_total", 0))
    shared_call_violations = int(
        data_progress.get("shared_call_count_violations", 0)
    )
    training_started = time.perf_counter()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    with log_path.open(log_mode, encoding="utf-8") as log_stream:
        while optimizer_step < run_limit:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            trajectories: list[torch.Tensor] = []
            accepted_fractions: list[float] = []
            valid_microbatches = 0
            step_attempts = 0
            step_started = time.perf_counter()
            while valid_microbatches < accumulation:
                if step_attempts >= max(1000, accumulation * 100):
                    raise RuntimeError("Too many invalid Stage1 samples for one optimizer step")
                step_attempts += 1
                index = data_stream.next()
                sample = train_dataset[index]
                images = sample["images"].unsqueeze(0).to(device)
                H_gt = sample["H_gt_norm"].unsqueeze(0).to(device)
                outputs = model(
                    images,
                    active_scales=active_scales,
                    iterations_per_scale=iterations_per_scale,
                )
                shared_call_violations += int(
                    outputs["shared_call_counts"].get("dino") != 1
                    or outputs["shared_call_counts"].get("mvt") != 1
                )
                loss_result = sequence_corner_l1(outputs, H_gt)
                if loss_result["skip_step"]:
                    invalid_attempts_total += 1
                    del images, H_gt, outputs, loss_result
                    continue
                loss = loss_result["loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Training loss became non-finite")
                (loss / accumulation).backward()
                with torch.no_grad():
                    metrics = homography_trajectory_metrics(outputs, H_gt)
                    trajectories.append(metrics["trajectory_mace_px"][0].cpu())
                    accepted_fractions.append(
                        float(outputs["update_accepted"].float().mean().item())
                    )
                losses.append(float(loss.detach().item()))
                valid_microbatches += 1
                del images, H_gt, outputs, loss_result, loss, metrics

            preclip_norm = torch.nn.utils.clip_grad_norm_(
                trainable, clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            trajectory_mean = torch.stack(trajectories).mean(dim=0).tolist()
            log_row = {
                "optimizer_step": optimizer_step,
                "profile": config.profile,
                "loss_sequence_corner_l1_px": sum(losses) / len(losses),
                "microbatch_trajectory_mean_mace_px": trajectory_mean,
                "microbatch_H0_mean_mace_px": trajectory_mean[0],
                "microbatch_Hfinal_mean_mace_px": trajectory_mean[-1],
                "accepted_update_fraction": sum(accepted_fractions)
                / len(accepted_fractions),
                "valid_microbatches": valid_microbatches,
                "invalid_attempts_this_step": step_attempts - valid_microbatches,
                "invalid_attempts_total": invalid_attempts_total,
                "preclip_gradient_norm": float(preclip_norm.detach().item()),
                "learning_rates": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "step_elapsed_seconds": time.perf_counter() - step_started,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0
                ),
                "data_progress": data_stream.state_dict(),
            }
            log_stream.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            log_stream.flush()
            print(
                f"step={optimizer_step}/{total_steps} "
                f"loss={log_row['loss_sequence_corner_l1_px']:.6f}px "
                f"H0={trajectory_mean[0]:.4f}px Hfinal={trajectory_mean[-1]:.4f}px",
                flush=True,
            )

            checkpoint_due = optimizer_step % checkpoint_interval == 0
            validation_due = optimizer_step % validation_interval == 0
            if checkpoint_due or validation_due or optimizer_step == run_limit:
                checkpoint_path = output_dir / "checkpoints" / f"step_{optimizer_step:07d}.pt"
                last_checkpoint = _save_training_state(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_step=optimizer_step,
                    data_stream=data_stream,
                    invalid_attempts_total=invalid_attempts_total,
                    shared_call_violations=shared_call_violations,
                    metadata=metadata,
                )
            if validation_due or optimizer_step == total_steps:
                validation_summary, validation_rows = evaluate_model(
                    model,
                    validation_dataset,
                    device=device,
                    active_scales=active_scales,
                    iterations_per_scale=iterations_per_scale,
                )
                validation_summary.update(
                    {
                        "optimizer_step": optimizer_step,
                        "checkpoint": last_checkpoint,
                        "split": "val",
                        "test_used": False,
                    }
                )
                validation_files = write_evaluation(
                    output_dir / "validation" / f"step_{optimizer_step:07d}",
                    validation_summary,
                    validation_rows,
                    overwrite=overwrite or resume is not None,
                )
                last_validation = {
                    "summary": validation_summary,
                    "files": validation_files,
                }

    completed = {
        **run_record,
        "status": "completed" if optimizer_step >= total_steps else "paused",
        "optimizer_steps": optimizer_step,
        "invalid_attempts_total": invalid_attempts_total,
        "shared_call_count_violations": shared_call_violations,
        "elapsed_seconds": time.perf_counter() - training_started,
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
        "last_checkpoint": last_checkpoint,
        "last_validation": last_validation,
        "note": (
            "Completion means the requested optimizer budget ran; paused means an "
            "explicit optimizer boundary was saved for resume. Model validation "
            "success depends on validation metrics and is never implied by status."
        ),
    }
    _write_json(output_dir / "run.json", completed)
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny-gate-artifact", type=Path)
    parser.add_argument("--stop-after-optimizer-step", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    runtime = RuntimePaths.from_json(args.runtime)
    config = TrainConfig.from_json(args.config)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else runtime.output_root / config.experiment_id
    )
    report = train(
        runtime,
        config,
        output_dir=output_dir,
        resume=args.resume,
        tiny_gate_artifact=args.tiny_gate_artifact,
        overwrite=args.overwrite,
        stop_after_optimizer_step=args.stop_after_optimizer_step,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
