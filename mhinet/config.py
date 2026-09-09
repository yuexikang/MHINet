"""Strict structure/runtime configuration loading for MHINet v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_ROOT = PROJECT_ROOT / "MHINet_server_handoff_v1.2"
DEFAULT_ARCHITECTURE_CONFIG = PROJECT_ROOT / "configs/mhinet_mcnet_v1.2.json"
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
        return tuple(int(value) for value in self.raw["iteration"]["primary_scales"])

    @property
    def registered_scales(self) -> tuple[int, ...]:
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
    def decoder_input_channels(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["decoder"]["input_channels"])

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
        if self.raw.get("mhir_revision") != "mcnet_correlation_decoder_784_v1":
            errors.append("MHIR must use the registered MCNet-faithful decoder revision")
        if self.raw.get("mcnet_reference", {}).get("commit") != (
            "cc03479689b3cf40f0c384954f338b434765c155"
        ):
            errors.append("MCNet source reference commit is not pinned")
        if self.scales != (8, 4, 2):
            errors.append(f"mainline scales must be (8,4,2), got {self.scales}")
        if self.registered_scales != (8, 4, 2, 1):
            errors.append("registered scales must retain dormant D1")
        if tuple(self.raw["iteration"]["iterations_per_scale"]) != (2, 2, 2, 2):
            errors.append("each registered scale must retain two-update support")
        if self.raw["iteration"].get("primary_total_updates") != 6:
            errors.append("the D8/D4/D2 mainline must have six updates")
        pyramid = self.raw.get("pyramid", {})
        if (
            tuple(pyramid.get("mainline_scales", ())) != (8, 4, 2)
            or tuple(pyramid.get("registered_optional_scales", ())) != (1,)
            or pyramid.get("run_scale1_by_default") is not False
        ):
            errors.append("CGMDP mainline must stop at D2 while retaining optional D1")
        if self.radii != (4, 4, 3, 2):
            errors.append(f"correlation radii mismatch: {self.radii}")
        expected_candidates = tuple((2 * radius + 1) ** 2 for radius in self.radii)
        if tuple(self.raw["correlation"]["candidates"]) != expected_candidates:
            errors.append("candidate counts do not match radii")
        decoder = self.raw["decoder"]
        if decoder.get("family") != "MCNet_CorrelationDecoder_adapted_784":
            errors.append("decoder family is not the registered MCNet adaptation")
        if tuple(decoder.get("input_order", ())) != ("correlation",):
            errors.append("MCNet-faithful decoder must consume correlation only")
        if self.decoder_input_channels != expected_candidates:
            errors.append("decoder input channels must equal correlation candidates")
        if self.down_blocks != (6, 7, 8, 9):
            errors.append("784 pyramid requires 6/7/8/9 MCNet downsample blocks")
        if tuple(decoder.get("corner_grid_hw", ())) != (2, 2):
            errors.append("MCNet corner-logit grid must be 2x2")
        if decoder.get("down_block", {}).get("pool_ceil_mode") is not True:
            errors.append("non-power-of-two pyramid adaptation requires ceil-mode pooling")
        stem = decoder.get("stem", {})
        if (
            stem.get("kernel") != 1
            or stem.get("channels") != 64
            or stem.get("bias") is not True
            or stem.get("activation") != "none"
        ):
            errors.append("decoder stem must be MCNet Conv1x1(K,64,bias=True)")
        block = decoder.get("down_block", {})
        if (
            block.get("channels") != 64
            or block.get("conv_kernel") != 3
            or block.get("conv_stride") != 1
            or block.get("padding") != 1
            or block.get("conv_bias") is not True
            or block.get("group_norm_groups") != 8
            or block.get("activation") != "ReLU"
            or block.get("pool") != "MaxPool2d"
            or block.get("pool_kernel") != 2
            or block.get("pool_stride") != 2
        ):
            errors.append("decoder block must be Conv3/GN8/ReLU/MaxPool2")
        expected_paths = (
            (98, 49, 25, 13, 7, 4, 2),
            (196, 98, 49, 25, 13, 7, 4, 2),
            (392, 196, 98, 49, 25, 13, 7, 4, 2),
            (784, 392, 196, 98, 49, 25, 13, 7, 4, 2),
        )
        if tuple(tuple(path) for path in decoder.get("spatial_paths", ())) != expected_paths:
            errors.append("decoder spatial paths do not match the 784 adaptation")
        if decoder.get("adaptive_pool") is not False:
            errors.append("registered MCNet path must not contain adaptive pooling")
        output_conv = decoder.get("output_conv", {})
        if (
            output_conv.get("kernel") != 1
            or output_conv.get("channels") != 2
            or output_conv.get("bias") is not True
            or output_conv.get("grid_to_corner_order")
            != "permute_BCHW_to_BHWC_then_row_major_TL_TR_BL_BR"
        ):
            errors.append("decoder output must be Conv1x1(64,2) in TL/TR/BL/BR order")
        if decoder.get("final_projection_init") != "all_zero":
            errors.append("final two-channel projection must be zero initialized")
        if decoder.get("output_activation") != "tanh":
            errors.append("registered safety adaptation requires tanh-bounded deltas")
        if self.max_delta_px != (32.0, 16.0, 6.0, 2.0):
            errors.append("per-scale input-pixel residual bounds changed")
        if self.expected_new_parameters != 1_176_712:
            errors.append("MCNet-faithful MHINet new-parameter count mismatch")
        if self.raw.get("expected_mainline_trainable_new_parameters") != 833_222:
            errors.append("mainline D8/D4/D2 trainable new-parameter count mismatch")
        if self.raw.get("d1_status") != (
            "implemented_but_inactive_no_descriptor_decode_no_optimizer_membership"
        ):
            errors.append("D1 must be retained but inactive in the mainline")
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
