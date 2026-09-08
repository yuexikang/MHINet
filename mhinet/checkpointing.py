"""Safe, resumable checkpoint I/O for MHINet training.

The checkpoint format deliberately stores metadata and RNG state using only
plain Python containers/scalars and tensors.  This keeps checkpoints produced
here compatible with PyTorch's restricted ``weights_only=True`` loader.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Final

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT: Final = "mhinet.training"
CHECKPOINT_VERSION: Final = 1


def _safe_value(value: Any, *, location: str) -> Any:
    """Convert user progress/metadata to weights-only-safe values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"Object NumPy arrays are unsafe in {location}")
        return torch.from_numpy(value.copy())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            if not (key is None or isinstance(key, (bool, int, float, str))):
                raise TypeError(
                    f"Unsupported mapping key {type(key).__name__} in {location}"
                )
            converted[key] = _safe_value(item, location=f"{location}[{key!r}]")
        return converted
    if isinstance(value, tuple):
        return tuple(
            _safe_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _safe_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Unsupported checkpoint value {type(value).__name__} in {location}; "
        "use tensors or plain containers/scalars"
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and initialized CUDA RNG streams."""

    python_version, python_state, python_gauss = random.getstate()
    numpy_algorithm, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    cuda_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else []
    return {
        "python": {
            "version": int(python_version),
            "state": list(python_state),
            "gauss_next": python_gauss,
        },
        "numpy": {
            "algorithm": str(numpy_algorithm),
            "keys": torch.from_numpy(numpy_keys.copy()),
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.cpu().clone() for state in cuda_states],
        "cuda_initialized": cuda_initialized,
        "cuda_device_count": len(cuda_states),
    }


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, bool]:
    """Restore RNG streams and report which CPU/CUDA portions were restored."""

    try:
        python_state = state["python"]
        numpy_state = state["numpy"]
        torch_cpu_state = state["torch_cpu"]
    except KeyError as error:
        raise ValueError(f"Checkpoint RNG state is incomplete: missing {error.args[0]}") from error

    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(item) for item in python_state["state"]),
            python_state["gauss_next"],
        )
    )
    numpy_keys = numpy_state["keys"]
    if not isinstance(numpy_keys, torch.Tensor):
        raise TypeError("Checkpoint NumPy RNG keys must be a tensor")
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise TypeError("Checkpoint Torch CPU RNG state must be a tensor")
    torch.set_rng_state(torch_cpu_state.detach().cpu())

    cuda_was_initialized = bool(state.get("cuda_initialized", False))
    cuda_restored = False
    if cuda_was_initialized:
        cuda_states = state.get("torch_cuda")
        if not isinstance(cuda_states, list) or not all(
            isinstance(item, torch.Tensor) for item in cuda_states
        ):
            raise TypeError("Checkpoint Torch CUDA RNG states must be a tensor list")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains initialized CUDA RNG state but CUDA is unavailable; "
                "load with restore_rng=False for evaluation-only use"
            )
        current_devices = torch.cuda.device_count()
        if len(cuda_states) != current_devices:
            raise RuntimeError(
                "CUDA RNG device-count mismatch: checkpoint has "
                f"{len(cuda_states)}, current runtime has {current_devices}"
            )
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])
        cuda_restored = True

    return {"cpu": True, "cuda": cuda_restored}


def _atomic_torch_save(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    optimizer_step: int,
    microbatch_progress: Any = None,
    data_progress: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save all state required for exact training continuation."""

    if optimizer_step < 0:
        raise ValueError("optimizer_step cannot be negative")
    target = Path(path).expanduser().resolve()
    safe_microbatch = _safe_value(microbatch_progress, location="microbatch_progress")
    safe_data = _safe_value(data_progress, location="data_progress")
    safe_metadata = _safe_value(dict(metadata or {}), location="metadata")
    rng_state = capture_rng_state()
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "progress": {
            "optimizer_step": int(optimizer_step),
            "microbatch": safe_microbatch,
            "data": safe_data,
        },
        "rng": rng_state,
        "metadata": safe_metadata,
    }
    _atomic_torch_save(payload, target)
    return {
        "path": str(target),
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "bytes": target.stat().st_size,
        "optimizer_step": int(optimizer_step),
        "microbatch_progress": safe_microbatch,
        "data_progress": safe_data,
        "metadata": safe_metadata,
        "cuda_rng_captured": bool(rng_state["cuda_initialized"]),
    }


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: Any,
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load an MHINet-owned checkpoint with PyTorch's restricted loader.

    ``map_location`` is intentionally required so callers cannot accidentally
    restore large training checkpoints onto their original CUDA devices.
    """

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("MHINet checkpoint root must be a dictionary")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')!r}")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')!r}")

    required = ("model", "optimizer", "scheduler", "scaler", "progress", "rng")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("Checkpoint is missing required fields: " + ", ".join(missing))

    incompatible = model.load_state_dict(payload["model"], strict=strict)
    restored = {"model": True, "optimizer": False, "scheduler": False, "scaler": False}
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        restored["optimizer"] = True
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("A scheduler was supplied but the checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
        restored["scheduler"] = True
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("A scaler was supplied but the checkpoint has no scaler state")
        scaler.load_state_dict(payload["scaler"])
        restored["scaler"] = True

    rng_result = {"cpu": False, "cuda": False}
    if restore_rng:
        rng_result = restore_rng_state(payload["rng"])

    progress = payload["progress"]
    if not isinstance(progress, dict) or "optimizer_step" not in progress:
        raise ValueError("Checkpoint progress record is invalid")
    return {
        "path": str(source),
        "format": payload["format"],
        "version": int(payload["version"]),
        "optimizer_step": int(progress["optimizer_step"]),
        "microbatch_progress": progress.get("microbatch"),
        "data_progress": progress.get("data"),
        "metadata": payload.get("metadata", {}),
        "missing_model_keys": list(incompatible.missing_keys),
        "unexpected_model_keys": list(incompatible.unexpected_keys),
        "restored": restored,
        "rng_restored": rng_result,
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "capture_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]
