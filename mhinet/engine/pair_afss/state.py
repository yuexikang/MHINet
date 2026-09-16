"""Persistent learning state for Pair-AFSS."""

from __future__ import annotations

import hashlib
from enum import IntEnum
from typing import Any, Iterable

import torch

from .config import PairAFSSConfig


class PairLearningState(IntEnum):
    UNKNOWN = 0
    EASY = 1
    MODERATE = 2
    HARD = 3


def pair_id_fingerprint(pair_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for pair_id in pair_ids:
        digest.update(pair_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class PairState:
    """Tensor-backed state table keyed by the dataset's stable item order."""

    VERSION = 1

    def __init__(self, pair_ids: list[str], config: PairAFSSConfig) -> None:
        if not pair_ids or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("Pair-AFSS requires non-empty, unique pair IDs")
        self.pair_count = len(pair_ids)
        self.pair_fingerprint = pair_id_fingerprint(pair_ids)
        self.config_digest = config.digest()
        self.raw_score = torch.zeros(self.pair_count, dtype=torch.float32)
        self.ema_score = torch.zeros(self.pair_count, dtype=torch.float32)
        self.precision_geo = torch.zeros(self.pair_count, dtype=torch.float32)
        self.recall_geo = torch.zeros(self.pair_count, dtype=torch.float32)
        self.homography_auc = torch.zeros(self.pair_count, dtype=torch.float32)
        self.learning_state = torch.full(
            (self.pair_count,), int(PairLearningState.UNKNOWN), dtype=torch.uint8
        )
        self.last_seen_round = torch.full(
            (self.pair_count,), -1, dtype=torch.int64
        )
        self.last_scored_round = torch.full(
            (self.pair_count,), -1, dtype=torch.int64
        )
        self.seen_count = torch.zeros(self.pair_count, dtype=torch.int64)
        self.fit_succeeded = torch.zeros(self.pair_count, dtype=torch.bool)
        # v2 calibrates absolute task thresholds once after warm-up. Optional
        # fields keep v1 checkpoints loadable without changing their version.
        self.calibrated_hard_threshold: float | None = None
        self.calibrated_easy_threshold: float | None = None

    def update_scores(
        self,
        *,
        round_id: int,
        score: torch.Tensor,
        precision_geo: torch.Tensor,
        recall_geo: torch.Tensor,
        homography_auc: torch.Tensor,
        fit_succeeded: torch.Tensor,
        scored: torch.Tensor,
        config: PairAFSSConfig,
    ) -> None:
        tensors = (
            score,
            precision_geo,
            recall_geo,
            homography_auc,
            fit_succeeded,
            scored,
        )
        if any(tensor.shape != (self.pair_count,) for tensor in tensors):
            raise ValueError("Full-state refresh tensors do not match the dataset")
        scored = scored.cpu().bool()
        if not bool(scored.all()):
            missing = int((~scored).sum().item())
            raise ValueError(f"Full Pair-AFSS refresh missed {missing} pairs")
        score = score.cpu().float().clamp(0.0, 1.0)
        previously_scored = self.last_scored_round >= 0
        ema = torch.where(
            previously_scored,
            config.score_ema_decay * self.ema_score
            + (1.0 - config.score_ema_decay) * score,
            score,
        )
        self.raw_score.copy_(score)
        self.ema_score.copy_(ema)
        self.precision_geo.copy_(precision_geo.cpu().float().clamp(0.0, 1.0))
        self.recall_geo.copy_(recall_geo.cpu().float().clamp(0.0, 1.0))
        self.homography_auc.copy_(homography_auc.cpu().float().clamp(0.0, 1.0))
        self.fit_succeeded.copy_(fit_succeeded.cpu().bool())
        self.last_scored_round.fill_(round_id)
        if getattr(config, "threshold_mode", "absolute") == "calibrated":
            if self.calibrated_hard_threshold is None:
                quantiles = torch.quantile(
                    ema,
                    torch.tensor(
                        [config.hard_quantile, config.easy_quantile],
                        dtype=ema.dtype,
                    ),
                )
                self.calibrated_hard_threshold = float(quantiles[0].item())
                self.calibrated_easy_threshold = float(quantiles[1].item())
            hard_threshold = self.calibrated_hard_threshold
            easy_threshold = self.calibrated_easy_threshold
        else:
            hard_threshold = config.hard_threshold
            easy_threshold = config.easy_threshold
        easy = ema > easy_threshold
        hard = (ema < hard_threshold) | ~self.fit_succeeded
        self.learning_state.fill_(int(PairLearningState.MODERATE))
        self.learning_state[easy] = int(PairLearningState.EASY)
        self.learning_state[hard] = int(PairLearningState.HARD)

    def thresholds(self, config: PairAFSSConfig) -> dict[str, float | str]:
        if self.calibrated_hard_threshold is not None:
            return {
                "mode": "calibrated",
                "hard": self.calibrated_hard_threshold,
                "easy": self.calibrated_easy_threshold,
            }
        return {
            "mode": "absolute",
            "hard": config.hard_threshold,
            "easy": config.easy_threshold,
        }

    def mark_seen(self, indices: list[int], round_id: int) -> None:
        if not indices:
            return
        unique = torch.as_tensor(sorted(set(indices)), dtype=torch.int64)
        self.last_seen_round[unique] = round_id
        self.seen_count[unique] += 1

    def counts(self) -> dict[str, int]:
        return {
            state.name.lower(): int((self.learning_state == int(state)).sum().item())
            for state in PairLearningState
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "pair_count": self.pair_count,
            "pair_fingerprint": self.pair_fingerprint,
            "config_digest": self.config_digest,
            "raw_score": self.raw_score,
            "ema_score": self.ema_score,
            "precision_geo": self.precision_geo,
            "recall_geo": self.recall_geo,
            "homography_auc": self.homography_auc,
            "learning_state": self.learning_state,
            "last_seen_round": self.last_seen_round,
            "last_scored_round": self.last_scored_round,
            "seen_count": self.seen_count,
            "fit_succeeded": self.fit_succeeded,
            "calibrated_hard_threshold": self.calibrated_hard_threshold,
            "calibrated_easy_threshold": self.calibrated_easy_threshold,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("version", -1)) != self.VERSION:
            raise ValueError("Unsupported Pair-AFSS state version")
        if int(payload.get("pair_count", -1)) != self.pair_count:
            raise ValueError("Pair-AFSS checkpoint dataset length does not match")
        if payload.get("pair_fingerprint") != self.pair_fingerprint:
            raise ValueError("Pair-AFSS checkpoint pair IDs/order do not match")
        if payload.get("config_digest") != self.config_digest:
            raise ValueError("Pair-AFSS checkpoint configuration does not match")
        for name in (
            "raw_score",
            "ema_score",
            "precision_geo",
            "recall_geo",
            "homography_auc",
            "learning_state",
            "last_seen_round",
            "last_scored_round",
            "seen_count",
            "fit_succeeded",
        ):
            current = getattr(self, name)
            saved = payload[name].cpu().to(dtype=current.dtype)
            if saved.shape != current.shape:
                raise ValueError(f"Pair-AFSS tensor shape mismatch for {name}")
            current.copy_(saved)
        hard_threshold = payload.get("calibrated_hard_threshold")
        easy_threshold = payload.get("calibrated_easy_threshold")
        self.calibrated_hard_threshold = (
            None if hard_threshold is None else float(hard_threshold)
        )
        self.calibrated_easy_threshold = (
            None if easy_threshold is None else float(easy_threshold)
        )
        if (self.calibrated_hard_threshold is None) != (
            self.calibrated_easy_threshold is None
        ):
            raise ValueError("Pair-AFSS calibrated thresholds are incomplete")
