"""Configuration for Pair-AFSS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PairAFSSConfig:
    """Hyperparameters that define scoring, state updates, and sampling."""

    warmup_rounds: int = 2
    refresh_interval: int = 5
    matchability_threshold: float = 0.3
    correct_threshold_px: float = 5.0
    hard_threshold: float = 0.55
    easy_threshold: float = 0.85
    easy_ratio: float = 0.02
    moderate_ratio: float = 0.40
    easy_review_rounds: int = 10
    moderate_coverage_rounds: int = 3
    score_ema_decay: float = 0.5
    model_size: int = 512
    seed: int = 42

    def __post_init__(self) -> None:
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative")
        if self.refresh_interval < 1:
            raise ValueError("refresh_interval must be positive")
        if not 0.0 <= self.matchability_threshold <= 1.0:
            raise ValueError("matchability_threshold must lie in [0, 1]")
        if self.correct_threshold_px <= 0.0:
            raise ValueError("correct_threshold_px must be positive")
        if not 0.0 <= self.hard_threshold < self.easy_threshold <= 1.0:
            raise ValueError("AFSS thresholds must satisfy 0 <= hard < easy <= 1")
        if not 0.0 <= self.easy_ratio <= 1.0:
            raise ValueError("easy_ratio must lie in [0, 1]")
        if not 0.0 <= self.moderate_ratio <= 1.0:
            raise ValueError("moderate_ratio must lie in [0, 1]")
        if self.easy_review_rounds < 1 or self.moderate_coverage_rounds < 1:
            raise ValueError("review and coverage intervals must be positive")
        if not 0.0 <= self.score_ema_decay < 1.0:
            raise ValueError("score_ema_decay must lie in [0, 1)")
        if self.model_size < 1:
            raise ValueError("model_size must be positive")

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairAFSSV2Config(PairAFSSConfig):
    """Conservative, task-calibrated AFSS while preserving the v1 schema."""

    correct_threshold_px: float = 1.0
    easy_ratio: float = 0.05
    moderate_ratio: float = 0.50
    threshold_mode: str = "calibrated"
    hard_quantile: float = 0.10
    easy_quantile: float = 0.80
    homography_auc_max_px: float = 3.0
    min_round_ratio: float = 0.45

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.threshold_mode != "calibrated":
            raise ValueError("Pair-AFSS v2 requires calibrated thresholds")
        if not 0.0 < self.hard_quantile < self.easy_quantile < 1.0:
            raise ValueError("AFSS quantiles must satisfy 0 < hard < easy < 1")
        if not 0.5 < self.homography_auc_max_px <= 10.0:
            raise ValueError("homography AUC max px must lie in (0.5, 10]")
        if not 0.0 < self.min_round_ratio <= 1.0:
            raise ValueError("minimum round ratio must lie in (0, 1]")
