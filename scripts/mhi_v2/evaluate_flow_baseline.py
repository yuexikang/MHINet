"""Full RGB_512 val baseline for the frozen v1 Flow locator, B=1/no training.

Use an idle physical GPU exposed as logical cuda:0. The original runtime is
read unchanged; only data_root points to the fully verified NVMe byte copy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from mhinet.config import RuntimePaths, sha256_file
from mhinet.v2.data import RGB512Dataset
from mhinet.v2.losses import flow_head_loss
from mhinet.v2.train import (implementation_hashes, initialize_system, means,
                             read_config, seed_all, timestamp, write_json)


SOURCE_SHA = "43c020b213306c1ba517460092e4007799c2f445edce09cb00bf88fc8ac69cb4"
MANIFEST_SHA = {
    "train": "11f37bcfcebc61a24310421a5eff2e1921c618d461b8dac8142ff050843f88dc",
    "val": "ad281e31dfd612a239ddfa63f25a42812511e22445f571359477e709c8207f5b",
}


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), ensure_ascii=False, allow_nan=False), flush=True)


def gpu_snapshot(index):
    command = ["nvidia-smi", f"--id={index}", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    fields = [part.strip() for part in subprocess.check_output(command, text=True).strip().split(",")]
    return dict(physical_index=int(fields[0]), uuid=fields[1], name=fields[2],
                memory_used_MiB=int(fields[3]), utilization_percent=int(fields[4]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--runtime", type=Path, default=ROOT/"configs/mhi_v2/runtime.rgb512.server.json")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/mhi_v2/flow_head_rgb512_seed42.json")
    parser.add_argument("--nvme-root", type=Path, default=Path("/home/u2ssd2_TQL/mhi_v2_cache/MHINet_RGB_512"))
    parser.add_argument("--physical-gpu", type=int, default=2)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = dict(schema="mhi.frozen-flow-fullval.v1", status="initializing", pid=os.getpid(),
                 started_at=timestamp(), checked_at=timestamp(), pairs=0, total_pairs=5106,
                 command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                 output=str(output), console_log=str(output.parent/(output.name+".console.log")),
                 protocol="Frozen v1 locator, 512, B=1, eval/no_grad; full fixed RGB512 val; macro mean over valid per-pair metrics; no training/model save")
    write_json(output/"status.json", state)
    fitter_hook = None
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
            raise ValueError("Expose only the requested physical GPU with CUDA_VISIBLE_DEVICES")
        snapshot = gpu_snapshot(args.physical_gpu)
        if snapshot["memory_used_MiB"] > 256 or snapshot["utilization_percent"] > 5:
            raise RuntimeError(f"Requested GPU is no longer idle: {snapshot}")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
        config, matcher = read_config(args.config)
        if config["stage"] != "flow_head" or matcher.geometry_mode != "flow_d16":
            raise ValueError("Use the registered Flow Head configuration")
        original_runtime = RuntimePaths.from_json(args.runtime)
        nvme = args.nvme_root.expanduser().resolve()
        cache_status_file = nvme/"cache_status.json"
        cache_identity_file = nvme/"cache_identity.json"
        cache_status = json.loads(cache_status_file.read_text())
        cache_identity = json.loads(cache_identity_file.read_text())
        if (cache_status.get("status") != "completed" or not cache_status.get("manifests_verified")
                or cache_status.get("files_completed") != cache_status.get("total_files")
                or cache_status.get("identity") != cache_identity
                or Path(cache_identity["source_root"]).resolve() != original_runtime.data_root.resolve()
                or Path(cache_status["destination_root"]).resolve() != nvme):
            raise ValueError("NVMe copy lacks its completed exact-byte verification")
        manifest_identities = {}
        for split, expected in MANIFEST_SHA.items():
            source_manifest = original_runtime.data_root/split/"pairs.jsonl"
            copied_manifest = nvme/split/"pairs.jsonl"
            source_actual = sha256_file(source_manifest)
            copied_actual = sha256_file(copied_manifest)
            if (source_actual != expected or copied_actual != expected
                    or cache_identity["manifests"][split]["sha256"] != expected
                    or cache_status["manifest_sha256"][split] != expected):
                raise ValueError(f"Frozen {split} manifest identity mismatch")
            manifest_identities[split] = dict(expected_sha256=expected, source_actual_sha256=source_actual,
                                               nvme_actual_sha256=copied_actual,
                                               source_path=str(source_manifest), nvme_path=str(copied_manifest))
        # No persisted runtime/config is written. Preserve every runtime field
        # including split_manifest; this evaluation reads val through data_root.
        runtime = replace(original_runtime, data_root=nvme)
        if runtime.device != "cuda:0":
            raise ValueError("Original runtime must address logical cuda:0")
        freeze_path = ROOT/"versions/v1/freeze_manifest.json"
        freeze = json.loads(freeze_path.read_text())
        checkpoint = Path(freeze["checkpoint"]["path"]).resolve()
        actual_source = sha256_file(checkpoint)
        if actual_source != SOURCE_SHA or freeze["checkpoint"]["sha256"] != SOURCE_SHA:
            raise ValueError("Checkpoint differs from the explicitly frozen v1 baseline")
        seed_all(config["seed"])
        system, provider, migration = initialize_system(runtime, config, matcher, checkpoint)
        system.requires_grad_(False)
        system.eval()
        if any(parameter.requires_grad for parameter in system.parameters()):
            raise AssertionError("Pure evaluation has trainable parameters")
        fitter_calls = [0]
        def count_fit(module, inputs, result):
            fitter_calls[0] += 1
        fitter_hook = system.shared.stage1_head.homography_fitter.register_forward_hook(count_fit)
        dataset = RGB512Dataset(runtime.data_root/"val/pairs.jsonl", image_size=512)
        if len(dataset) != 5106:
            raise ValueError(f"Expected full 5106-pair val, got {len(dataset)}")
        source_by_pair = {}
        with dataset.manifest.open() as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    source_by_pair[row["pair_id"]] = row["source_dataset"]
        weights_actual = {}
        for path in (runtime.dino_checkpoint, runtime.selected_stage1_checkpoint, runtime.pyramid_checkpoint):
            resolved = str(path.resolve())
            if resolved not in weights_actual:
                weights_actual[resolved] = sha256_file(path)
        registration = dict(**state, checkpoint=dict(path=str(checkpoint), expected_sha256=SOURCE_SHA,
            actual_sha256=actual_source, task=freeze["checkpoint"]["task"]),
            config=dict(path=str(args.config.resolve()), sha256=sha256_file(args.config), value=config),
            original_runtime=dict(path=str(args.runtime.resolve()), sha256=sha256_file(args.runtime),
                                  value=json.loads(json.dumps(asdict(original_runtime), default=str))),
            effective_runtime=json.loads(json.dumps(asdict(runtime), default=str)),
            runtime_override_fields=["data_root"], manifests=manifest_identities,
            initialization_weights_actual_sha256=weights_actual,
            nvme=dict(status_file=str(cache_status_file), status_actual_sha256=sha256_file(cache_status_file),
                      identity_file=str(cache_identity_file), identity_actual_sha256=sha256_file(cache_identity_file),
                      completed_status=cache_status),
            implementation_actual_sha256=implementation_hashes(runtime), provider=provider, migration=migration,
            torch_version=str(torch.__version__), cuda_version=str(torch.version.cuda),
            physical_gpu_before_start=snapshot, logical_device="cuda:0",
            logical_device_name=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            batch_size=1, workers=config.get("workers", 0), all_parameters_frozen=True,
            model_or_optimizer_saved=False, optimizer_created=False,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        write_json(output/"run.json", registration)
        state.update(status="running", checked_at=timestamp(), initialized_seconds=time.perf_counter()-started)
        write_json(output/"status.json", state)
        emit("initialized", **state)
        records = []
        seed_all(123)  # Same evaluation seed as the completed Flow full-val pass.
        evaluated_at = time.perf_counter()
        loader = DataLoader(dataset, batch_size=1, num_workers=config.get("workers", 0))
        with (output/"pairs.jsonl").open("x") as journal, torch.no_grad():
            for batch in loader:
                tensors = {key: value.to(runtime.device, non_blocking=True)
                           for key, value in batch.items() if isinstance(value, torch.Tensor)}
                shared = system(tensors["images"])
                if shared["pyramid"] or shared["H0_norm"] is not None or fitter_calls[0]:
                    raise AssertionError("Flow baseline unexpectedly ran descriptors or H fitting")
                loss, metrics = flow_head_loss(shared["stage1"], tensors["H_gt_norm"],
                    tensors["mask_A_overlap"], tensors["mask_B_overlap"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite frozen Flow baseline loss")
                pair_id = batch["pair_id"][0]
                record = dict(pair_id=pair_id, loss=float(loss), **metrics)
                # Match the old validation pair fields exactly; source membership
                # is stored separately in registration/summary for aggregation.
                journal.write(json.dumps(record, allow_nan=False)+"\n")
                records.append(record)
                if len(records) % 64 == 0 or len(records) == len(dataset):
                    journal.flush()
                    elapsed = time.perf_counter()-evaluated_at
                    state.update(pairs=len(records), metrics=means(records), checked_at=timestamp(),
                                 evaluation_seconds=elapsed, pairs_per_second=len(records)/max(elapsed, 1e-6),
                                 actual_homography_fitter_calls=fitter_calls[0])
                    write_json(output/"status.json", state)
                    emit("progress", **state)
        summary = dict(step=0, pairs=len(records), scope="full_val", complete=len(records) == len(dataset),
                       metrics=means(records), seconds=time.perf_counter()-evaluated_at,
                       label="frozen_v1_flow_baseline", checked_at=timestamp(),
                       selection_protocol="Pure frozen baseline; Flow EPE, macro per pair, fixed full RGB512 val",
                       actual_homography_fitter_calls=fitter_calls[0], source_checkpoint_actual_sha256=actual_source)
        source_summary = {}
        for source in sorted(set(source_by_pair.values())):
            selected = [row for row in records if source_by_pair[row["pair_id"]] == source]
            source_summary[source] = dict(pairs=len(selected), metrics=means(selected))
        summary["by_source_dataset"] = source_summary
        write_json(output/"validation_0000000.json", dict(summary=summary, pairs=records))
        write_json(output/"summary.json", summary)
        state.update(status="completed", pairs=len(records), completed_at=timestamp(), checked_at=timestamp(),
                     seconds=time.perf_counter()-started, metrics=summary["metrics"],
                     actual_homography_fitter_calls=fitter_calls[0])
        write_json(output/"status.json", state)
        emit("completed", **state)
        return 0
    except BaseException as error:
        state.update(status="failed", checked_at=timestamp(), error=repr(error), traceback=traceback.format_exc())
        write_json(output/"status.json", state)
        emit("failed", **state)
        raise
    finally:
        if fitter_hook is not None:
            fitter_hook.remove()


if __name__ == "__main__":
    raise SystemExit(main())
