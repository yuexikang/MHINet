"""Deterministic anti-forgetting pair selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import PairAFSSConfig
from .state import PairLearningState, PairState


@dataclass(frozen=True)
class PairSelection:
    indices: list[int]
    unique_indices: list[int]
    padding_count: int
    selected_by_state: dict[str, int]


class PairAFSSScheduler:
    def __init__(self, state: PairState, config: PairAFSSConfig) -> None:
        self.state = state
        self.config = config

    @staticmethod
    def _sample(pool: list[int], count: int, rng: random.Random) -> list[int]:
        if count <= 0:
            return []
        if count >= len(pool):
            output = list(pool)
            rng.shuffle(output)
            return output
        return rng.sample(pool, count)

    def _oldest(
        self,
        pool: list[int],
        count: int,
        round_id: int,
        rng: random.Random,
    ) -> list[int]:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        shuffled.sort(
            key=lambda index: round_id - int(self.state.last_seen_round[index]),
            reverse=True,
        )
        return shuffled[:count]

    def _lowest_sufficiency_oldest(
        self,
        pool: list[int],
        count: int,
        rng: random.Random,
    ) -> list[int]:
        """Prefer the weakest and then longest-unseen replay candidates."""
        shuffled = list(pool)
        rng.shuffle(shuffled)
        shuffled.sort(
            key=lambda index: (
                float(self.state.ema_score[index]),
                int(self.state.last_seen_round[index]),
            )
        )
        return shuffled[:count]

    def select(
        self,
        *,
        round_id: int,
        round_fraction: float,
        pad_to_multiple: int,
    ) -> PairSelection:
        if not 0.0 < round_fraction <= 1.0:
            raise ValueError("round_fraction must lie in (0, 1]")
        if pad_to_multiple < 1:
            raise ValueError("pad_to_multiple must be positive")
        rng = random.Random(self.config.seed + 1_000_003 * round_id)
        states = self.state.learning_state.tolist()
        pools = {
            state: [index for index, value in enumerate(states) if value == int(state)]
            for state in PairLearningState
        }

        if round_id < self.config.warmup_rounds:
            target = max(1, math.ceil(self.state.pair_count * round_fraction))
            unique = self._sample(list(range(self.state.pair_count)), target, rng)
        else:
            unknown_and_hard = (
                pools[PairLearningState.UNKNOWN] + pools[PairLearningState.HARD]
            )
            hard_target = max(
                1 if unknown_and_hard else 0,
                math.ceil(len(unknown_and_hard) * round_fraction),
            )
            selected_hard = self._sample(unknown_and_hard, hard_target, rng)

            moderate = pools[PairLearningState.MODERATE]
            moderate_target = math.ceil(
                len(moderate) * self.config.moderate_ratio * round_fraction
            )
            forced_moderate = [
                index
                for index in moderate
                if round_id - int(self.state.last_seen_round[index])
                >= self.config.moderate_coverage_rounds
            ]
            selected_moderate = self._oldest(
                forced_moderate, len(forced_moderate), round_id, rng
            )
            selected_moderate_set = set(selected_moderate)
            remaining_moderate = [
                index for index in moderate if index not in selected_moderate_set
            ]
            selected_moderate.extend(
                self._sample(
                    remaining_moderate,
                    max(moderate_target - len(selected_moderate), 0),
                    rng,
                )
            )

            easy = pools[PairLearningState.EASY]
            easy_target = math.ceil(
                len(easy) * self.config.easy_ratio * round_fraction
            )
            forced_easy = [
                index
                for index in easy
                if round_id - int(self.state.last_seen_round[index])
                >= self.config.easy_review_rounds
            ]
            forced_quota = min(len(forced_easy), math.ceil(easy_target * 0.5))
            selected_easy = self._oldest(
                forced_easy, forced_quota, round_id, rng
            )
            selected_easy_set = set(selected_easy)
            remaining_easy = [
                index for index in easy if index not in selected_easy_set
            ]
            selected_easy.extend(
                self._sample(
                    remaining_easy,
                    max(easy_target - len(selected_easy), 0),
                    rng,
                )
            )
            unique = selected_hard + selected_moderate + selected_easy

        unique = list(dict.fromkeys(unique))
        minimum_ratio = float(getattr(self.config, "min_round_ratio", 0.0))
        if round_id >= self.config.warmup_rounds and minimum_ratio > 0.0:
            minimum_target = max(
                1,
                math.ceil(
                    self.state.pair_count * minimum_ratio * round_fraction
                ),
            )
            selected = set(unique)
            deficit = max(minimum_target - len(unique), 0)
            # Strengthen the paper's short-term coverage before drawing more
            # easy pairs. This guard is inactive whenever native AFSS already
            # selects at least the conservative participation floor.
            for state in (
                PairLearningState.MODERATE,
                PairLearningState.EASY,
            ):
                if deficit <= 0:
                    break
                remaining = [
                    index for index in pools[state] if index not in selected
                ]
                added = self._lowest_sufficiency_oldest(
                    remaining, deficit, rng
                )
                unique.extend(added)
                selected.update(added)
                deficit -= len(added)
            if deficit > 0:
                raise RuntimeError(
                    "Pair-AFSS could not satisfy minimum round participation"
                )
        if not unique:
            raise RuntimeError("Pair-AFSS selected no training pairs")
        rng.shuffle(unique)
        selected_by_state = {
            state.name.lower(): sum(states[index] == int(state) for index in unique)
            for state in PairLearningState
        }
        indices = list(unique)
        padding_count = (-len(indices)) % pad_to_multiple
        if padding_count:
            indices.extend(rng.choice(unique) for _ in range(padding_count))
        return PairSelection(
            indices=indices,
            unique_indices=unique,
            padding_count=padding_count,
            selected_by_state=selected_by_state,
        )
