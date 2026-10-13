"""Transactionally remove one source dataset from an MHINet_RGB_512 dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from scripts.verify_rgb512_dataset import verify_dataset


SPLITS = ("train", "val", "test")
FILE_KEYS = ("image_A", "image_B", "mask_A_overlap", "mask_B_overlap")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pair_counts: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}
    modes: dict[str, dict[str, int]] = {}
    textures: dict[str, dict[str, int]] = {}
    atmospheres: dict[str, dict[str, int]] = {}
    for split, rows in rows_by_split.items():
        pair_counts[split] = len(rows)
        source_counts[split] = dict(sorted(Counter(row["source_dataset"] for row in rows).items()))
        modes[split] = dict(sorted(Counter(row["augmentation_mode"] for row in rows).items()))
        strong = [row for row in rows if row["augmentation_mode"] == "strong"]
        textures[split] = dict(
            sorted(Counter(row["augmentation_parameters"]["texture"]["kind"] for row in strong).items())
        )
        atmospheres[split] = dict(
            sorted(Counter(row["augmentation_parameters"]["atmosphere"]["kind"] for row in strong).items())
        )
    return {
        "pair_counts": pair_counts,
        "source_counts": source_counts,
        "modes": modes,
        "textures": textures,
        "atmospheres": atmospheres,
    }


def remove_source(root: Path, source_dataset: str) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    summary_path = root / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError(f"Dataset status is not completed: {summary.get('status')}")

    rows_by_split = {split: _read_rows(root / split / "pairs.jsonl") for split in SPLITS}
    removed_by_split = {
        split: [row for row in rows if row["source_dataset"] == source_dataset]
        for split, rows in rows_by_split.items()
    }
    removed_count = sum(map(len, removed_by_split.values()))
    if removed_count == 0:
        raise ValueError(f"No rows use source_dataset={source_dataset!r}")

    source_manifest_path = root / "source_split_manifest.jsonl"
    source_manifest = _read_rows(source_manifest_path)
    source_removed = [row for row in source_manifest if row["source_dataset"] == source_dataset]
    if len(source_removed) != removed_count:
        raise AssertionError(
            f"Source manifest mismatch: pairs={removed_count}, source rows={len(source_removed)}"
        )

    quarantine = root / f".{source_dataset}_removal_staging"
    if quarantine.exists():
        raise FileExistsError(f"Removal staging already exists: {quarantine}")
    quarantine.mkdir()
    backups = quarantine / "backups"
    backups.mkdir()
    shutil.copy2(summary_path, backups / "dataset_summary.json")
    shutil.copy2(source_manifest_path, backups / "source_split_manifest.jsonl")
    for split in SPLITS:
        shutil.copy2(root / split / "pairs.jsonl", backups / f"{split}_pairs.jsonl")

    moved: list[tuple[Path, Path]] = []
    try:
        for split in SPLITS:
            split_root = (root / split).resolve()
            for row in removed_by_split[split]:
                for key in FILE_KEYS:
                    source = (split_root / row[key]).resolve()
                    if split_root not in source.parents:
                        raise ValueError(f"Refusing out-of-split file path: {source}")
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    destination = quarantine / "files" / split / row[key]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
                    moved.append((source, destination))

        filtered_by_split = {
            split: [row for row in rows if row["source_dataset"] != source_dataset]
            for split, rows in rows_by_split.items()
        }
        for split in SPLITS:
            _write_rows_atomic(root / split / "pairs.jsonl", filtered_by_split[split])
        _write_rows_atomic(
            source_manifest_path,
            [row for row in source_manifest if row["source_dataset"] != source_dataset],
        )

        realized = _summarize(filtered_by_split)
        summary["status"] = "generated_pending_verification"
        summary["planned_pair_counts"] = realized["pair_counts"]
        summary["realized_pair_counts"] = realized["pair_counts"]
        summary["planned_source_counts"] = realized["source_counts"]
        summary["realized_augmentation_modes"] = realized["modes"]
        summary["realized_strong_texture_choices"] = realized["textures"]
        summary["realized_strong_atmosphere_choices"] = realized["atmospheres"]
        summary.get("discovered_source_counts", {}).pop(source_dataset, None)
        summary.setdefault("configuration", {}).setdefault("excluded_source_datasets", [])
        if source_dataset not in summary["configuration"]["excluded_source_datasets"]:
            summary["configuration"]["excluded_source_datasets"].append(source_dataset)
        removal = {
            "source_dataset": source_dataset,
            "removed_at_utc": datetime.now(timezone.utc).isoformat(),
            "removed_pair_counts": {split: len(rows) for split, rows in removed_by_split.items()},
            "removed_pairs_total": removed_count,
            "removed_derived_files": len(moved),
            "original_source_files_deleted": False,
            "reason": "reserved for real-pair fine-tuning and strengthening",
        }
        summary.setdefault("source_removals", []).append(removal)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        verification = verify_dataset(root, full_decode=False)
        summary["verification"] = verification
        summary["status"] = "completed"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.rmtree(quarantine)
        return {"removal": removal, "remaining_pair_counts": realized["pair_counts"], "verification": verification}
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
        shutil.copy2(backups / "dataset_summary.json", summary_path)
        shutil.copy2(backups / "source_split_manifest.jsonl", source_manifest_path)
        for split in SPLITS:
            shutil.copy2(backups / f"{split}_pairs.jsonl", root / split / "pairs.jsonl")
        shutil.rmtree(quarantine)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True)
    args = parser.parse_args(argv)
    result = remove_source(args.root, args.source_dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
