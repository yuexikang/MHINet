"""Audited v1-to-v2 weight migration, separate from training resumption."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.torch_version import TorchVersion

from mhinet.engine.checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION


# Only these new QRRU branches may retain their explicit v2 initialization.
# In particular, no missing shared/DINO/MVT/CGMDP or original QRRU key is allowed.
_NEW_BRANCH_PREFIXES = tuple(
    f"{root}{branch}."
    for root in ("matcher.qrru.", "qrru.")
    for branch in ("diff_encoder", "diff_adapter", "dgfo", "flow_adapter")
)


def migrate_v1_state(system: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Validate the entire legacy topology before copying any checkpoint tensor.

    Only missing keys in named newly added QRRU branches are permitted. Unknown
    checkpoint keys, shape mismatches and missing original keys fail atomically
    with respect to these validation errors. Optimizer/RNG state is not migrated.
    """
    if not isinstance(state_dict, Mapping) or not all(isinstance(k, str) for k in state_dict):
        raise ValueError("v1 model state must be a string-keyed mapping")
    expected = system.state_dict()
    unexpected = sorted(set(state_dict) - set(expected))
    missing = sorted(set(expected) - set(state_dict))
    forbidden_missing = [k for k in missing if not k.startswith(_NEW_BRANCH_PREFIXES)]
    malformed = [k for k in state_dict if not isinstance(state_dict[k], torch.Tensor)]
    mismatched = [
        f"{k}: checkpoint {tuple(state_dict[k].shape)} != model {tuple(expected[k].shape)}"
        for k in sorted(set(state_dict) & set(expected))
        if isinstance(state_dict[k], torch.Tensor) and state_dict[k].shape != expected[k].shape
    ]
    errors = []
    if unexpected:
        errors.append(f"unexpected keys: {unexpected}")
    if forbidden_missing:
        errors.append(f"missing original keys: {forbidden_missing}")
    if malformed:
        errors.append(f"non-tensor state values: {malformed}")
    if mismatched:
        errors.append(f"shape mismatches: {mismatched}")
    if errors:
        raise ValueError("Rejected v1 migration; " + "; ".join(errors))
    incompatible = system.load_state_dict(state_dict, strict=False)
    if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise RuntimeError("State loading disagreed with the completed migration audit")
    return {
        "source_version": "v1", "target_version": "v2",
        "loaded_tensors": len(state_dict), "expected_tensors": len(expected),
        "missing_keys": missing, "initialized_v2_keys": missing,
        "unexpected_keys": [], "shape_mismatches": [],
        "all_original_keys_verified": True,
        "optimizer_migrated": False,
    }


def load_v1_weights(system: nn.Module, path: str | Path) -> dict[str, Any]:
    """Read an audited v1 second-stage checkpoint using restricted CPU mmap."""
    source = Path(path).expanduser().resolve()
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping):
        raise ValueError("v1 checkpoint must be a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Expected a supported mhinet.training v1 model checkpoint")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("task") != "semidense_qrru_v1":
        raise ValueError("Expected task=semidense_qrru_v1")
    report = migrate_v1_state(system, payload.get("model"))
    return {**report, "path": str(source), "metadata": dict(metadata),
            "source_optimizer_step": payload.get("progress", {}).get("optimizer_step")}


__all__ = ["migrate_v1_state", "load_v1_weights"]
