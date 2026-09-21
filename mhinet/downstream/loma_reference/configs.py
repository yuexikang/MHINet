"""Frozen T0 configuration and stable result identifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


SupportMode = Literal["oracle_gt_overlap", "predicted_stage1_overlap"]
CoarsePolicy = Literal["mutual", "dual_topk"]
FineSelection = Literal["local_mutual", "local_topk"]
FinePrior = Literal["h_residual", "direct_d8"]
FineMode = Literal["expanded_local", "h_warped_sym4"]


@dataclass(frozen=True)
class MatcherConfig:
    """All parameters that can affect a T0 match result."""

    result_id: str = "C3-P3"
    support_mode: SupportMode = "predicted_stage1_overlap"
    coarse_policy: CoarsePolicy = "dual_topk"
    coarse_temperature: float = 0.05
    coarse_confidence_threshold: float = 0.0
    max_coarse: int = 6000
    coarse_chunk_rows: int = 1024
    force_chunked_coarse: bool = True
    run_fine: bool = True
    fine_mode: FineMode = "expanded_local"
    fine_prior: FinePrior = "h_residual"
    fine_window: int = 9
    fine_temperature: float = 0.05
    fine_confidence_threshold: float = 0.0
    fine_selection: FineSelection = "local_mutual"
    local_topk: int = 2
    max_final_matches: int = 12000
    fine_geometry_radius: int | None = None
    mask_dilation_d8: int = 0
    image_size: int = 784

    def __post_init__(self) -> None:
        if self.result_id not in {
            "C3-P0", "C3-P1", "C3-P2", "C3-P3", "C3-P4", "C3-P5",
            "C3-P6", "C3-P7-D0", "C3-P7-D1", "C3-P7-D2",
        }:
            raise ValueError(f"Unknown stable result ID: {self.result_id}")
        if self.coarse_temperature <= 0 or self.fine_temperature <= 0:
            raise ValueError("Temperatures must be positive")
        if self.coarse_confidence_threshold < 0 or self.fine_confidence_threshold < 0:
            raise ValueError("Confidence thresholds must be non-negative")
        if self.max_coarse <= 0 or self.max_final_matches <= 0:
            raise ValueError("Match caps must be positive")
        if self.coarse_chunk_rows <= 0:
            raise ValueError("coarse_chunk_rows must be positive")
        if self.fine_window not in {5, 9}:
            raise ValueError("T0 fine_window must be 5 or 9")
        if self.fine_prior not in {"h_residual", "direct_d8"}:
            raise ValueError(f"Unknown fine prior: {self.fine_prior}")
        if self.fine_mode not in {"expanded_local", "h_warped_sym4"}:
            raise ValueError(f"Unknown fine mode: {self.fine_mode}")
        if self.local_topk <= 0:
            raise ValueError("local_topk must be positive")
        if self.fine_geometry_radius not in {None, 0, 1, 2}:
            raise ValueError("Fine geometry radius must be None, 0, 1, or 2")
        if self.image_size != 784:
            raise ValueError("V1 fixes the shared input size at 784")
        if self.mask_dilation_d8 not in {0, 1, 3, 6}:
            raise ValueError("Mask dilation must be a registered ablation")
        if self.result_id != "C3-P5" and self.mask_dilation_d8:
            raise ValueError("Only C3-P5 may use predicted-mask dilation")
        if self.result_id == "C3-P0" and self.support_mode != "oracle_gt_overlap":
            raise ValueError("C3-P0 is the explicitly GT-labeled oracle")
        if self.result_id != "C3-P0" and self.support_mode != "predicted_stage1_overlap":
            raise ValueError("Only C3-P0 may consume GT support/H")
        if self.result_id in {"C3-P0", "C3-P1", "C3-P2"} and self.run_fine:
            raise ValueError(f"{self.result_id} is a coarse-only condition")
        if not self.run_fine and self.fine_prior != "h_residual":
            raise ValueError("fine_prior ablations require an enabled fine stage")
        if not self.run_fine and self.fine_mode != "expanded_local":
            raise ValueError("fine_mode ablations require an enabled fine stage")
        if self.result_id == "C3-P3" and not self.run_fine:
            raise ValueError("C3-P3 is the primary discrete D2 condition")
        if self.result_id == "C3-P3" and self.fine_mode != "expanded_local":
            raise ValueError("C3-P3 remains the immutable expanded-local baseline")
        if self.result_id == "C3-P6" and not self.run_fine:
            raise ValueError("C3-P6 requires the D2 fine stage")
        if self.result_id == "C3-P6" and self.fine_mode != "h_warped_sym4":
            raise ValueError("C3-P6 is reserved for H-warped symmetric 4x4 fine")
        geometry_results = {
            "C3-P7-D0": 0,
            "C3-P7-D1": 1,
            "C3-P7-D2": 2,
        }
        if self.result_id in geometry_results:
            if not self.run_fine or self.fine_mode != "h_warped_sym4":
                raise ValueError("C3-P7 geometry ablations require H-warped fine")
            if self.fine_geometry_radius != geometry_results[self.result_id]:
                raise ValueError("C3-P7 result ID and geometry radius disagree")
        elif self.fine_geometry_radius is not None:
            raise ValueError("Only C3-P7 conditions may restrict fine geometry")
        if self.fine_mode == "h_warped_sym4" and self.fine_prior != "h_residual":
            raise ValueError("H-warped symmetric fine requires the H-residual geometry")

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        # Preserve every historical config hash when the new ablation is off.
        if output["fine_geometry_radius"] is None:
            output.pop("fine_geometry_radius")
        return output

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def registered_config(result_id: str, **overrides: object) -> MatcherConfig:
    """Construct a preregistered T0 condition without silent GT or fine changes."""

    defaults: dict[str, object] = {
        "result_id": result_id,
        "support_mode": "predicted_stage1_overlap",
        "coarse_policy": "dual_topk",
        "run_fine": result_id == "C3-P3",
    }
    if result_id == "C3-P0":
        defaults.update(support_mode="oracle_gt_overlap", coarse_policy="mutual", run_fine=False)
    elif result_id == "C3-P1":
        defaults.update(coarse_policy="mutual", run_fine=False)
    elif result_id == "C3-P2":
        defaults.update(coarse_policy="dual_topk", run_fine=False)
    elif result_id == "C3-P6":
        defaults.update(
            coarse_policy="mutual",
            run_fine=True,
            fine_mode="h_warped_sym4",
            fine_prior="h_residual",
        )
    elif result_id in {"C3-P7-D0", "C3-P7-D1", "C3-P7-D2"}:
        defaults.update(
            coarse_policy="mutual",
            run_fine=True,
            fine_mode="h_warped_sym4",
            fine_prior="h_residual",
            fine_geometry_radius=int(result_id[-1]),
        )
    defaults.update(overrides)
    return MatcherConfig(**defaults)
