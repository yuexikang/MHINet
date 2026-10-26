"""Strict 512-only contracts for independently supervised v2 experiments."""
from dataclasses import dataclass, fields
import math

from mhinet.downstream.semidense import SemidenseConfig


@dataclass(frozen=True)
class SemidenseConfigV2(SemidenseConfig):
    input_size: int = 512
    geometry_mode: str = "flow_d16"
    fine_prior: str = "direct_d8"
    fine_target_side: int = 8
    flow_guidance_enabled: bool = True
    flow_lambda: float = 3.0
    flow_sigma_d8: float = 4.0
    flow_min_matchability: float = 0.0
    use_discrepancy: bool = False
    use_dgfo: bool = False
    dgfo_radius: int = 1
    dgfo_dim: int = 32
    max_output_matches: int = 2048
    coarse_mode: str = "chunked"
    fine_precision: str = "fp32"
    inference_iterations: int = 4
    qrru_early_stop: bool = True

    def __post_init__(self):
        if type(self.fine_target_side) is not int or self.fine_target_side not in (4, 8):
            raise ValueError("fine_target_side must be 4 or 8")
        if self.coarse_mode not in ("chunked", "full"):
            raise ValueError("coarse_mode must be chunked or full")
        if self.fine_precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("fine_precision must be fp32, bf16 or fp16")
        if self.inference_iterations not in (1, 2, 3, 4):
            raise ValueError("inference_iterations must be 1..4")
        for name in ("input_size", "coarse_queries", "fine_windows", "qrru_queries", "chunk",
                     "window_chunk", "inference_window_chunk", "iterations", "max_coarse",
                     "max_matches", "max_output_matches", "dgfo_radius", "dgfo_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("radius", "gamma", "noise_std", "noise_bound", "temperature",
                     "fine_gt_tolerance", "confidence_threshold", "lambda_c", "lambda_f", "lambda_q",
                     "flow_lambda", "flow_sigma_d8", "flow_min_matchability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        super().__post_init__()
        if type(self.input_size) is not int or self.input_size != 512:
            raise ValueError("MHI v2 only accepts input_size=512")
        if self.geometry_mode not in ("h0", "none", "flow_d16"):
            raise ValueError("Unknown geometry_mode")
        if self.fine_prior not in ("h0_residual", "direct_d8"):
            raise ValueError("Unknown fine_prior")
        if self.geometry_mode != "h0" and self.fine_prior != "direct_d8":
            raise ValueError("No-H modes require direct_d8 fine grids")
        if self.geometry_mode == "h0" and self.fine_prior != "h0_residual":
            raise ValueError("The legacy H0 control requires h0_residual fine grids")
        for name in ("flow_guidance_enabled", "use_discrepancy", "use_dgfo"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if type(self.qrru_early_stop) is not bool:
            raise ValueError('qrru_early_stop must be boolean')
        if self.use_dgfo and not self.use_discrepancy:
            raise ValueError("DGFO requires discrepancy features")
        for name in ("max_output_matches", "dgfo_radius", "dgfo_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dgfo_radius != 1 or self.dgfo_dim != 32:
            raise ValueError("This protocol fixes DGFO radius=1, dim=32")
        for name in ("flow_lambda", "flow_sigma_d8", "flow_min_matchability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(name)
        if self.flow_sigma_d8 == 0 or self.flow_min_matchability > 1:
            raise ValueError("Invalid Flow prior scale/probability")

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("Matcher configuration must be an object")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown matcher configuration fields: {sorted(unknown)}")
        return cls(**payload)


def variant_config(variant):
    """A-F share all budgets; only the registered mechanisms change."""
    if not isinstance(variant, str) or variant not in tuple("ABCDEF"):
        raise ValueError("Expected variant A-F")
    return SemidenseConfigV2(
        geometry_mode="h0" if variant == "A" else "none" if variant in "BF" else "flow_d16",
        fine_prior="h0_residual" if variant == "A" else "direct_d8",
        flow_guidance_enabled=variant in "CDE",
        use_discrepancy=variant in "DEF", use_dgfo=variant in "EF",
    )
