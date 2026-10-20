#!/usr/bin/env python3
"""Create a verified train/val-only RGB512 NVMe cache without changing training."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import statistics
import threading
import time
import uuid
from zoneinfo import ZoneInfo


FROZEN = {
    "train": "11f37bcfcebc61a24310421a5eff2e1921c618d461b8dac8142ff050843f88dc",
    "val": "ad281e31dfd612a239ddfa63f25a42812511e22445f571359477e709c8207f5b",
}
FIELDS = ("image_A", "image_B", "mask_A_overlap", "mask_B_overlap")
BLOCK = 1024*1024


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def atomic_json(path, payload):
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def file_hash(path, stop=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(BLOCK):
            if stop is not None and stop.is_set():
                raise InterruptedError("Cache copy stopped")
            digest.update(chunk)
    return digest.hexdigest()


def make_plan(source, expected):
    files = set(); manifests = {}; stats = {}; counts = {}
    for split in ("train", "val"):
        path = source/split/"pairs.jsonl"
        before = signature(path); digest = hashlib.sha256(); count = 0; pair_ids = set()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != split or row.get("size_A") != [512, 512] or row.get("size_B") != [512, 512]:
                    raise ValueError(f"Invalid {split} RGB512 row")
                pair_id = str(row["pair_id"])
                if pair_id in pair_ids:
                    raise ValueError(f"Duplicate {split} pair ID: {pair_id}")
                pair_ids.add(pair_id)
                for field in FIELDS:
                    raw = row[field]
                    if not isinstance(raw, str):
                        raise ValueError("Image reference must be a string")
                    relative = PurePosixPath(raw)
                    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("images", "masks"):
                        raise ValueError(f"Unsafe/non-image reference: {raw}")
                    files.add(f"{split}/{relative.as_posix()}")
                count += 1
        actual = digest.hexdigest()
        if actual != expected[split] or signature(path) != before:
            raise ValueError(f"Frozen {split} manifest identity changed: {actual}")
        manifests[split] = dict(sha256=actual, bytes=before[2], pairs=count)
        stats[split] = before; counts[split] = count
    paths = sorted(files)
    plan_sha = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    identity = dict(schema="mhi.rgb512.verified-cache.v1", source_root=str(source),
                    splits=["train", "val"], fields=list(FIELDS), manifests=manifests,
                    referenced_files=len(paths), referenced_paths_sha256=plan_sha,
                    policy="Only train/val manifest-referenced PNGs; sealed test excluded; source remains read-only")
    return paths, identity, stats


def copy_one(relative, source, destination, stop, previous=None):
    if stop.is_set():
        raise InterruptedError("Cache copy stopped")
    src = source/relative; dst = destination/relative
    if not src.resolve(strict=True).is_relative_to(source):
        raise ValueError(f"Source reference leaves source root: {relative}")
    before = signature(src)
    if not src.is_file() or src.is_symlink() or dst.is_symlink():
        raise ValueError(f"Unexpected nonregular/symlink file: {relative}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.parent.resolve().is_relative_to(destination):
        raise ValueError(f"Destination leaves cache root: {relative}")
    reused = dst.exists()
    if previous is not None and not reused:
        raise FileNotFoundError(f"Previously verified cached file disappeared: {relative}")
    temporary = dst.with_name(dst.name+f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        digest = hashlib.sha256(); copied = 0
        with src.open("rb") as incoming:
            outgoing = None if reused else temporary.open("xb")
            try:
                while chunk := incoming.read(BLOCK):
                    if stop.is_set():
                        raise InterruptedError("Cache copy stopped")
                    digest.update(chunk); copied += len(chunk)
                    if outgoing is not None:
                        outgoing.write(chunk)
                if outgoing is not None:
                    outgoing.flush(); os.fsync(outgoing.fileno())
            finally:
                if outgoing is not None:
                    outgoing.close()
        source_hash = digest.hexdigest()
        target_hash = file_hash(dst if reused else temporary, stop)
        if copied != before[2] or signature(src) != before:
            raise RuntimeError(f"Source file changed while reading: {relative}")
        if source_hash != target_hash:
            raise RuntimeError(f"Source/target SHA256 mismatch: {relative}")
        if previous and (source_hash != previous["source_sha256"] or before[2] != previous["bytes"]):
            raise RuntimeError(f"Previously verified source identity changed: {relative}")
        if not reused:
            temporary.replace(dst)
        return dict(path=relative, bytes=copied, source_sha256=source_hash,
                    destination_sha256=target_hash, source_signature=list(before),
                    reused_and_reverified=reused, verified_at=now())
    except BaseException:
        stop.set()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/disk1/Data/datasets/MHINet_RGB_512"))
    parser.add_argument("--destination", type=Path, default=Path("/home/u2ssd2_TQL/mhi_v2_cache/MHINet_RGB_512"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--copy", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-train-sha256", default=FROZEN["train"])
    parser.add_argument("--expected-val-sha256", default=FROZEN["val"])
    args = parser.parse_args()
    if not 8 <= args.workers <= 16:
        raise ValueError("Use 8-16 copy threads")
    source = args.source.resolve(strict=True); destination = args.destination.absolute()
    if destination.is_symlink() or destination.resolve().is_relative_to(source) or source.is_relative_to(destination.resolve()):
        raise ValueError("Source and destination must be separate, nonsymlink roots")
    expected = {"train":args.expected_train_sha256, "val":args.expected_val_sha256}
    started = time.monotonic()
    paths, identity, manifest_stats = make_plan(source, expected)
    # Sample only the first 32 training pairs' referenced files. Do not perform
    # a full source-tree stat walk, which would compete with active training.
    sample = []
    for section in ("train/images/", "train/masks/"):
        sample.extend([p for p in paths if p.startswith(section)][:64])
    sampled_sizes = [(source/p).stat().st_size for p in sample]
    estimated_bytes = int(statistics.mean(sampled_sizes)*len(paths))+sum(m["bytes"] for m in identity["manifests"].values())
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    summary = dict(identity=identity, destination_root=str(destination), workers=args.workers,
                   estimated_bytes=estimated_bytes, estimated_GiB=estimated_bytes/1024**3,
                   estimate_sample_files=len(sample), destination_free_bytes=free,
                   estimate_policy="Mean compressed size of 64 image and 64 mask train references (32 pairs); estimate only")
    if args.dry_run:
        print(json.dumps(dict(status="dry_run", checked_at=now(), **summary), indent=2)); return 0
    if free < estimated_bytes*1.5:
        raise RuntimeError("Destination lacks 1.5x estimated free capacity")
    if destination.exists() and not args.resume:
        raise FileExistsError("Destination exists; use --resume only for a matching identified cache")
    destination.mkdir(parents=True, exist_ok=args.resume)
    destination = destination.resolve(strict=True)
    lock = (destination/".cache_copy.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity_path = destination/"cache_identity.json"
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("Resume cache identity differs from frozen source/manifests")
    else:
        atomic_json(identity_path, identity)
    journal_path = destination/"cache_files.jsonl"; previous = {}
    if args.resume and journal_path.exists():
        with journal_path.open() as stream:
            for line in stream:
                row = json.loads(line); previous[row["path"]] = row
        if not set(previous).issubset(set(paths)|{"train/pairs.jsonl", "val/pairs.jsonl"}):
            raise ValueError("Journal contains files outside this train/val identity")
    status_path = destination/"cache_status.json"
    state = dict(schema="mhi.rgb512.cache-status.v1", status="copying", pid=os.getpid(),
                 started_at=now(), checked_at=now(), files_completed=0, bytes_verified=0,
                 reused_files=0, total_files=len(paths)+2, progress_interval_seconds=5,
                 journal=str(journal_path), **summary)
    stop = threading.Event()
    def on_signal(signum, frame):
        stop.set()
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)
    atomic_json(status_path, state)
    journal = journal_path.open("a")
    last_progress = time.monotonic()
    def record(row):
        journal.write(json.dumps(row, separators=(",", ":"))+"\n")
        state["files_completed"] += 1; state["bytes_verified"] += row["bytes"]
        state["reused_files"] += int(row["reused_and_reverified"])
    def progress():
        for split, original in manifest_stats.items():
            if signature(source/split/"pairs.jsonl") != original:
                stop.set()
                raise RuntimeError(f"Frozen {split} manifest changed during copy")
        journal.flush(); os.fsync(journal.fileno())
        state.update(checked_at=now(), elapsed_seconds=time.monotonic()-started)
        atomic_json(status_path, state)
        print(json.dumps({k:state[k] for k in ("status", "pid", "files_completed", "total_files", "bytes_verified", "elapsed_seconds", "checked_at")}), flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="rgb512-cache") as pool:
            todo = iter(paths); pending = {}
            def replenish():
                while not stop.is_set() and len(pending) < args.workers*2:
                    try:
                        relative = next(todo)
                    except StopIteration:
                        break
                    future = pool.submit(copy_one, relative, source, destination, stop, previous.get(relative))
                    pending[future] = relative
            replenish()
            while pending:
                if stop.is_set():
                    raise InterruptedError("Cache copy interrupted; verified journal supports --resume")
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future); record(future.result())
                replenish()
                if time.monotonic()-last_progress >= 5:
                    progress(); last_progress = time.monotonic()
        # Publish byte-identical manifests only after all referenced data are
        # verified, so a partial target cannot look like a ready training set.
        for split in ("train", "val"):
            relative = f"{split}/pairs.jsonl"
            row = copy_one(relative, source, destination, stop, previous.get(relative))
            if row["source_sha256"] != expected[split] or file_hash(source/relative, stop) != expected[split]:
                raise RuntimeError(f"Final frozen {split} manifest hash changed")
            record(row)
        state.update(status="completed", completed_at=now(), manifests_verified=True,
                     manifest_sha256={split:expected[split] for split in ("train", "val")})
        progress(); return 0
    except BaseException as error:
        stop.set(); state.update(status="interrupted" if isinstance(error, InterruptedError) else "failed",
                                 error=repr(error), failed_at=now())
        journal.flush(); os.fsync(journal.fileno())
        state.update(checked_at=now(), elapsed_seconds=time.monotonic()-started)
        atomic_json(status_path, state)
        raise
    finally:
        stop.set(); journal.close()


if __name__ == "__main__":
    raise SystemExit(main())
