"""Strict structure/runtime configuration loading for MHINet v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_ROOT = PROJECT_ROOT / "MHINet_server_handoff_v1.2"
DEFAULT_ARCHITECTURE_CONFIG = DESIGN_ROOT / "configs/mhinet_v1.json"
DEFAULT_TRAINING_PROFILES = DESIGN_ROOT / "configs/training_profiles.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class ArchitectureConfig:
    """Validated view of the immutable v1 architecture document."""

    path: Path
    raw: Mapping[str, Any]
    sha256: str

    @property
    def scales(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["iteration"]["scales"])

    @property
    def radii(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["correlation"]["radius"])

    @property
    def adapter_channels(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["adapter"]["channels"])

    @property
    def down_blocks(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["decoder"]["down_blocks"])

    @property
    def max_delta_px(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.raw["decoder"]["max_delta_input_pixels"])

    @property
    def image_size(self) -> tuple[int, int]:
        height, width = self.raw["input"]["size_hw"]
        return int(height), int(width)

    @property
    def expected_new_parameters(self) -> int:
        return int(self.raw["expected_new_parameters"])

    def validate(self) -> None:
        errors: list[str] = []
        if self.raw.get("model_name") != "MHINet":
            errors.append("model_name must be MHINet")
        if self.raw.get("training_revision") != "1.2":
            errors.append("training_revision must be 1.2")
        if self.raw.get("loss_revision") != "1.1":
            errors.append("loss_revision must be 1.1")
        if self.scales != (8, 4, 2, 1):
            errors.append(f"scales must be (8,4,2,1), got {self.scales}")
        if tuple(self.raw["iteration"]["iterations_per_scale"]) != (2, 2, 2, 2):
            errors.append("each scale must have exactly two updates")
        if self.radii != (4, 4, 3, 2):
            errors.append(f"correlation radii mismatch: {self.radii}")
        expected_candidates = tuple((2 * radius + 1) ** 2 for radius in self.radii)
        if tuple(self.raw["correlation"]["candidates"]) != expected_candidates:
            errors.append("candidate counts do not match radii")
        if self.raw["iteration"].get("detach_between_updates") is not False:
            errors.append("H/T detach between updates is forbidden")
        if self.raw["stage1"].get("detach_H0_in_joint") is not False:
            errors.append("H0 detach in joint training is forbidden")
        if self.raw["stage1"].get("dino_frozen") is not True:
            errors.append("DINOv3 must remain frozen")
        if self.raw.get("new_planar_or_confidence_heads") is not False:
            errors.append("Planar/confidence heads are outside v1")
        loss = self.raw["loss"]
        if loss.get("profile") != "sequence_corner_l1" or loss.get("fgo_enabled") is not False:
            errors.append("v1 default must be equal-weight proposal corner L1 with FGO off")
        if errors:
            raise ValueError("Invalid MHINet architecture config: " + "; ".join(errors))


def load_architecture_config(path: str | Path = DEFAULT_ARCHITECTURE_CONFIG) -> ArchitectureConfig:
    resolved = Path(path).expanduser().resolve()
    config = ArchitectureConfig(path=resolved, raw=_read_json(resolved), sha256=sha256_file(resolved))
    config.validate()
    return config


@dataclass(frozen=True)
class RuntimePaths:
    """Server-specific paths, deliberately separate from model structure."""

    repo_root: Path
    loma_root: Path
    loretta_source_root: Path
    dino_checkpoint: Path
    selected_stage1_checkpoint: Path
    pyramid_checkpoint: Path
    data_root: Path
    split_manifest: Path
    output_root: Path
    device: str
    source_path: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path, *, require_files: bool = True) -> "RuntimePaths":
        source = Path(path).expanduser().resolve()
        payload = _read_json(source)
        aliases = {"loma_root": payload.get("loma_root", payload.get("repo_root"))}

        def required(name: str) -> Any:
            value = aliases.get(name, payload.get(name))
            if value is None or value == "":
                raise ValueError(f"Runtime path {name!r} is unresolved in {source}")
            return value

        result = cls(
            repo_root=Path(required("repo_root")).expanduser().resolve(),
            loma_root=Path(required("loma_root")).expanduser().resolve(),
            loretta_source_root=Path(required("loretta_source_root")).expanduser().resolve(),
            dino_checkpoint=Path(required("dino_checkpoint")).expanduser().resolve(),
            selected_stage1_checkpoint=Path(required("selected_stage1_checkpoint")).expanduser().resolve(),
            pyramid_checkpoint=Path(required("pyramid_checkpoint")).expanduser().resolve(),
            data_root=Path(required("data_root")).expanduser().resolve(),
            split_manifest=Path(required("split_manifest")).expanduser().resolve(),
            output_root=Path(required("output_root")).expanduser().resolve(),
            device=str(required("device")),
            source_path=source,
        )
        if require_files:
            result.validate()
        return result

    def validate(self) -> None:
        directories = {
            "repo_root": self.repo_root,
            "loma_root": self.loma_root,
            "loretta_source_root": self.loretta_source_root,
            "data_root": self.data_root,
        }
        files = {
            "dino_checkpoint": self.dino_checkpoint,
            "selected_stage1_checkpoint": self.selected_stage1_checkpoint,
            "pyramid_checkpoint": self.pyramid_checkpoint,
            "split_manifest": self.split_manifest,
        }
        missing = [f"{name}={path}" for name, path in directories.items() if not path.is_dir()]
        missing += [f"{name}={path}" for name, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing runtime resources: " + ", ".join(missing))


def load_training_profiles(path: str | Path = DEFAULT_TRAINING_PROFILES) -> dict[str, tuple[str, ...]]:
    source = Path(path).expanduser().resolve()
    payload = _read_json(source)
    if payload.get("revision") != "1.2":
        raise ValueError("Only training profile revision 1.2 is supported")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("training_profiles.json has no profiles object")
    return {str(name): tuple(str(group) for group in groups) for name, groups in profiles.items()}
