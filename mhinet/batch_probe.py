"""Isolated single-GPU true-batch throughput probe; never saves fitted weights."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from .config import RuntimePaths
from .losses import sequence_corner_l1
from .model import build_model
from .train import _safe_train_dataset, _seed_everything


def run(runtime, batch, steps, warmup, profile):
    _seed_everything(0)
    device = torch.device(runtime.device)
    free, total = torch.cuda.mem_get_info(device)
    dataset, _ = _safe_train_dataset(runtime, max_pairs=max(8, batch))
    # All runs use the same deterministic prefix of real train pairs; no val fitting.
    samples = [dataset[i] for i in range(max(4, batch))]
    effective = max(4, batch)
    accumulation = effective // batch
    model, build = build_model(runtime)
    model.set_training_phase(profile)
    model.train()
    optimizer = torch.optim.AdamW(model.optimizer_group_spec(), weight_decay=1e-4)
    records = []
    for step in range(warmup + steps):
        if step == warmup:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        step_loss = 0.
        accepted = 0
        for micro in range(accumulation):
            selected = samples[micro*batch:(micro+1)*batch]
            images = torch.stack([s["images"] for s in selected]).to(device)
            gt = torch.stack([s["H_gt_norm"] for s in selected]).to(device)
            output = model(images)
            loss = sequence_corner_l1(output, gt)
            if int(loss["valid_pairs"]) != batch:
                raise RuntimeError("Probe prefix contains invalid GHIM outputs; select a documented valid prefix")
            if output["shared_call_counts"]["dino"] != 1 or output["shared_call_counts"]["mvt"] != 1 or output["shared_call_counts"]["dedode_scale1"] != 0:
                raise RuntimeError("Shared forward or D2 cutoff contract violated")
            (loss["loss"] / accumulation).backward()
            step_loss += float(loss["loss"].detach()) / accumulation
            accepted += int(output["update_accepted"].sum())
            del images, gt, output, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        records.append({"step": step, "warmup": step < warmup, "seconds": elapsed,
                        "pairs_per_second": effective / elapsed, "loss": step_loss,
                        "preclip_norm": float(norm), "accepted_updates": accepted})
        print(f"BS={batch} step={step} pairs/s={effective/elapsed:.3f}", flush=True)
    measured = records[warmup:]
    return {"status": "passed", "profile": profile, "batch_size": batch,
            "effective_batch_size": effective, "gradient_accumulation": accumulation,
            "warmup_steps": warmup, "measured_steps": steps,
            "median_pairs_per_second": statistics.median(r["pairs_per_second"] for r in measured),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_before_bytes": free, "total_device_bytes": total,
            "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "architecture_sha256": build["architecture_sha256"],
            "pair_ids": [s["pair_id"] for s in samples], "test_used": False,
            "scope": "timed forward/loss/backward/clip/optimizer with CPU-prefetched real train images; excludes disk I/O, validation, checkpoint writes",
            "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True, choices=(1, 2, 4, 8))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--profile", default="heads", choices=("heads", "joint"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = run(RuntimePaths.from_json(args.runtime), args.batch_size, args.steps, args.warmup, args.profile)
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "batch_size": args.batch_size, "profile": args.profile,
                  "error": str(error), "scope": "isolated diagnostic; process exits to release CUDA allocations"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in result if k not in ("records", "pair_ids", "error")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
