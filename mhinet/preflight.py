"""P0 resource, provenance, environment, and split-integrity audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

from .config import (
    DEFAULT_ARCHITECTURE_CONFIG,
    ArchitectureConfig,
    RuntimePaths,
    load_architecture_config,
    sha256_file,
)


def _command(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def _git_snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "head": _command(["git", "rev-parse", "HEAD"], cwd=path),
        "branch": _command(["git", "branch", "--show-current"], cwd=path),
        "dirty_paths": _command(["git", "status", "--porcelain=v1"], cwd=path).splitlines(),
        "remote": _command(["git", "remote", "get-url", "origin"], cwd=path),
    }


def _module_version(name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(name)
        return {"status": "available", "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as error:  # environment audit must report broken imports, not hide them
        return {
            "status": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }


def _asset_record(path: Path, hash_cache: dict[Path, str]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in hash_cache:
        hash_cache[resolved] = sha256_file(resolved)
    digest = hash_cache[resolved]
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest,
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield record


def audit_splits(data_root: Path) -> dict[str, Any]:
    """Verify parent/geographic identities and materialized files do not leak."""

    split_records: dict[str, dict[str, Any]] = {}
    parents_by_split: dict[str, set[str]] = {}
    missing_by_split: dict[str, list[str]] = defaultdict(list)
    for split in ("train", "val", "test"):
        split_root = data_root / split
        manifest = split_root / "pairs.jsonl"
        parents: set[str] = set()
        pair_ids: set[str] = set()
        rows = 0
        for row in _iter_jsonl(manifest):
            rows += 1
            pair_id = str(row["pair_id"])
            if pair_id in pair_ids:
                raise ValueError(f"Duplicate pair_id in {manifest}: {pair_id}")
            pair_ids.add(pair_id)
            parents.add(str(row.get("parent_image_A", row.get("source", ""))))
            parents.add(str(row.get("parent_image_B", row.get("target", ""))))
            for field in ("image_A", "image_B", "metadata"):
                candidate = split_root / str(row[field])
                if not candidate.is_file() and len(missing_by_split[split]) < 20:
                    missing_by_split[split].append(str(candidate))
        parents.discard("")
        parents_by_split[split] = parents
        split_records[split] = {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "rows": rows,
            "unique_pair_ids": len(pair_ids),
            "unique_parent_ids": len(parents),
            "missing_required_files": missing_by_split[split],
        }
    overlap: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        values = sorted(parents_by_split[left] & parents_by_split[right])
        overlap[f"{left}_{right}"] = values[:20]
    return {
        "splits": split_records,
        "parent_overlap_examples": overlap,
        "parent_disjoint": all(not values for values in overlap.values()),
        "all_materialized": all(not values for values in missing_by_split.values()),
    }


def collect_preflight(runtime: RuntimePaths, architecture: ArchitectureConfig) -> dict[str, Any]:
    import torch

    runtime.validate()
    hash_cache: dict[Path, str] = {}
    loma_git = _git_snapshot(runtime.loma_root)
    loretta_git = _git_snapshot(runtime.loretta_source_root)
    expected_base = str(architecture.raw["base_commit"])
    gpu_lines: list[str] = []
    try:
        gpu_lines = _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    except Exception as error:
        gpu_lines = [f"unavailable: {type(error).__name__}: {error}"]
    return {
        "gate": "P0",
        "status": "passed" if loma_git["head"] == expected_base else "failed",
        "architecture": {
            "path": str(architecture.path),
            "sha256": architecture.sha256,
            "training_revision": architecture.raw["training_revision"],
            "loss_revision": architecture.raw["loss_revision"],
        },
        "runtime_config": str(runtime.source_path) if runtime.source_path else None,
        "source": {
            "mhinet_repo": str(runtime.repo_root),
            "loma": loma_git,
            "loma_expected_head": expected_base,
            "loma_head_matches_design": loma_git["head"] == expected_base,
            "loretta": loretta_git,
        },
        "assets": {
            "dino_checkpoint": _asset_record(runtime.dino_checkpoint, hash_cache),
            "selected_stage1_checkpoint": _asset_record(
                runtime.selected_stage1_checkpoint, hash_cache
            ),
            "pyramid_checkpoint": _asset_record(runtime.pyramid_checkpoint, hash_cache),
        },
        "data": audit_splits(runtime.data_root),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "platform": platform.platform(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": bool(
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            "modules": {
                name: _module_version(name)
                for name in (
                    "torchvision",
                    "numpy",
                    "scipy",
                    "cv2",
                    "kornia",
                    "einops",
                    "pytest",
                    "openpyxl",
                    "timm",
                )
            },
            "gpus": gpu_lines,
        },
        "known_training_wrapper_audit": {
            "legacy_full_model_train_forces_eval": True,
            "legacy_extract_pyramid_inference_mode": True,
            "legacy_mvt_contextualize_no_grad": True,
            "stage1_head_forward_allows_input_gradient": True,
            "new_provider_must_bypass_legacy_wrappers": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--architecture", type=Path, default=DEFAULT_ARCHITECTURE_CONFIG
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = RuntimePaths.from_json(args.runtime)
    architecture = load_architecture_config(args.architecture)
    report = collect_preflight(runtime, architecture)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
