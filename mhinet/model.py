"""Top-level GHIM/CGMDP/MHIR model and training-profile ownership."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import (
    ArchitectureConfig,
    RuntimePaths,
    load_architecture_config,
    load_training_profiles,
)
from .feature_provider import SharedFeatureProvider, build_feature_provider
from .iterator import MultiScaleHIterator
from .modules import (
    EXPECTED_NEW_PARAMETERS,
    build_multiscale_modules,
    count_new_parameters,
)


class MHINet(nn.Module):
    """Run GHIM/CGMDP followed by MHIR's eight homography updates."""

    def __init__(
        self,
        feature_provider: SharedFeatureProvider,
        architecture: ArchitectureConfig | None = None,
    ) -> None:
        super().__init__()
        self.architecture = architecture or load_architecture_config()
        self.feature_provider = feature_provider
        adapters, decoders = build_multiscale_modules()
        geometry = self.architecture.raw["geometry"]
        correlation = self.architecture.raw["correlation"]
        self.iterator = MultiScaleHIterator(
            adapters,
            decoders,
            query_chunk_size=int(correlation["query_chunk_size"]),
            feature_epsilon=float(self.architecture.raw["adapter"]["epsilon"]),
            minimum_supported_queries=int(geometry["minimum_supported_queries"]),
            condition_max=float(geometry["condition_max"]),
            projection_denominator_min=float(
                geometry["projection_denominator_min"]
            ),
            validity_grid_hw=tuple(geometry["validity_grid_hw"]),
            target_hw=tuple(self.architecture.raw["input"]["size_hw"]),
        )
        if count_new_parameters(self.iterator.adapters, self.iterator.decoders) != EXPECTED_NEW_PARAMETERS:
            raise AssertionError("MHINet new parameter count changed from the v1 contract")
        self.training_profiles = load_training_profiles()
        self._training_phase = "heads"
        self.set_training_phase("heads")

    @property
    def adapters(self) -> nn.ModuleDict:
        return self.iterator.adapters

    @property
    def refinement_decoders(self) -> nn.ModuleDict:
        return self.iterator.decoders

    @property
    def training_phase(self) -> str:
        return self._training_phase

    def set_training_phase(self, profile: str) -> dict[str, Any]:
        if profile not in self.training_profiles:
            raise ValueError(
                f"Unknown training profile {profile!r}; expected {tuple(self.training_profiles)}"
            )
        active = set(self.training_profiles[profile])
        for parameter in self.adapters.parameters():
            parameter.requires_grad = "adapters" in active
        for parameter in self.refinement_decoders.parameters():
            parameter.requires_grad = "refinement_decoders" in active
        self.feature_provider.set_training_groups(
            dedode="dedode" in active,
            vgg="vgg" in active,
            mvt="mvt" in active,
        )
        self._training_phase = profile
        # Restore mode after changing group flags; requires_grad alone is not train().
        self.train(self.training)
        report = self.training_parameter_report()
        if report["dino_trainable_parameters"] != 0:
            raise AssertionError("DINOv3 must remain frozen")
        if report["stage1_head_trainable_parameters"] != 0:
            raise AssertionError("Stage1 head parameters must remain frozen")
        return report

    def train(self, mode: bool = True) -> "MHINet":
        super().train(mode)
        self.feature_provider.train(mode)
        return self

    def _all_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        provider_groups = self.feature_provider.parameter_groups()
        return {
            "new_modules": list(self.adapters.parameters())
            + list(self.refinement_decoders.parameters()),
            "dedode": provider_groups["dedode"],
            "vgg": provider_groups["vgg"],
            "mvt": provider_groups["mvt"],
            "dino": provider_groups["dino"],
            "stage1_head_parameters": provider_groups[
                "stage1_head_parameters"
            ],
        }

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            name: [parameter for parameter in parameters if parameter.requires_grad]
            for name, parameters in self._all_parameter_groups().items()
        }
        active = {name: values for name, values in groups.items() if values}
        seen: dict[int, str] = {}
        for name, parameters in active.items():
            for parameter in parameters:
                previous = seen.setdefault(id(parameter), name)
                if previous != name:
                    raise RuntimeError(
                        f"Parameter appears in optimizer groups {previous} and {name}"
                    )
        return active

    def training_parameter_report(self) -> dict[str, Any]:
        groups = self._all_parameter_groups()
        return {
            "profile": self._training_phase,
            "groups": {
                name: {
                    "parameters": sum(parameter.numel() for parameter in parameters),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in parameters
                        if parameter.requires_grad
                    ),
                    "tensor_count": len(parameters),
                    "trainable_tensor_count": sum(
                        parameter.requires_grad for parameter in parameters
                    ),
                    "optimizer_parameter_ids": [
                        id(parameter) for parameter in parameters if parameter.requires_grad
                    ],
                }
                for name, parameters in groups.items()
            },
            "total_trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "new_parameters": count_new_parameters(
                self.adapters, self.refinement_decoders
            ),
            "dino_trainable_parameters": sum(
                parameter.numel()
                for parameter in groups["dino"]
                if parameter.requires_grad
            ),
            "stage1_head_trainable_parameters": sum(
                parameter.numel()
                for parameter in groups["stage1_head_parameters"]
                if parameter.requires_grad
            ),
        }

    def optimizer_group_spec(self) -> list[dict[str, Any]]:
        """Return disjoint AdamW groups with protocol-v1.2 learning rates."""

        learning_rates = self.architecture.raw["training"]["learning_rates"]
        lr_names = {
            "new_modules": "new_modules",
            "dedode": "dedode",
            "vgg": "vgg",
            "mvt": "mvt",
        }
        groups = self.trainable_parameter_groups()
        unexpected = set(groups) - set(lr_names)
        if unexpected:
            raise AssertionError(f"Frozen groups unexpectedly trainable: {sorted(unexpected)}")
        return [
            {
                "name": name,
                "params": parameters,
                "lr": float(learning_rates[lr_names[name]]),
            }
            for name, parameters in groups.items()
        ]

    def forward(
        self,
        images: torch.Tensor,
        *,
        active_scales: tuple[int, ...] | None = None,
        iterations_per_scale: int = 2,
        cnn_autocast_enabled: bool = True,
    ) -> dict[str, Any]:
        selected_scales = (
            self.iterator.scales
            if active_scales is None
            else tuple(int(scale) for scale in active_scales)
        )
        if not selected_scales:
            raise ValueError("active_scales must contain at least one scale")
        finest_index = max(self.iterator.scales.index(scale) for scale in selected_scales)
        provider_prefix = self.iterator.scales[: finest_index + 1]
        shared = self.feature_provider(images, pyramid_scales=provider_prefix)
        refinement = self.iterator(
            shared["pyramid"],
            shared["H0_norm"],
            shared["stage1_valid"],
            target_hw=tuple(images.shape[-2:]),
            active_scales=selected_scales,
            iterations_per_scale=iterations_per_scale,
            cnn_autocast_enabled=cnn_autocast_enabled,
        )
        refinement.update(
            {
                "stage1_valid_correspondences": shared[
                    "stage1_valid_correspondences"
                ],
                "shared_call_counts": shared["call_counts"],
                "runtime_diagnostics": {
                    "shared_call_counts": shared["call_counts"],
                    "feature_valid_counts": refinement["feature_valid_counts"],
                },
            }
        )
        return refinement


def build_model(
    runtime: RuntimePaths,
    architecture_path: str | Path | None = None,
) -> tuple[MHINet, dict[str, Any]]:
    architecture = load_architecture_config(
        architecture_path
        if architecture_path is not None
        else Path(__file__).resolve().parents[1]
        / "MHINet_server_handoff_v1.2/configs/mhinet_v1.json"
    )
    provider, provider_report = build_feature_provider(runtime)
    model = MHINet(provider, architecture).to(torch.device(runtime.device))
    report = {
        "architecture_path": str(architecture.path),
        "architecture_sha256": architecture.sha256,
        "provider": provider_report,
        "training_parameters": model.training_parameter_report(),
    }
    return model, report


__all__ = ["MHINet", "build_model"]
