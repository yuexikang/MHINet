"""Round lifecycle and checkpoint state for Pair-AFSS."""

from __future__ import annotations

import math
from typing import Any

from .config import PairAFSSConfig
from .scheduler import PairAFSSScheduler, PairSelection
from .state import PairLearningState, PairState


class PairAFSSController:
    VERSION = 1

    def __init__(
        self,
        pair_ids: list[str],
        config: PairAFSSConfig,
        *,
        baseline_max_steps: int,
        baseline_steps_per_round: int,
    ) -> None:
        if baseline_max_steps < 1 or baseline_steps_per_round < 1:
            raise ValueError("Pair-AFSS baseline step counts must be positive")
        self.config = config
        self.state = PairState(pair_ids, config)
        self.scheduler = PairAFSSScheduler(self.state, config)
        self.baseline_max_steps = int(baseline_max_steps)
        self.baseline_steps_per_round = int(baseline_steps_per_round)
        self.total_rounds = self.baseline_max_steps / self.baseline_steps_per_round
        self.current_round = 0
        self.selected_indices: list[int] = []
        self.selected_unique_indices: list[int] = []
        self.selected_by_state: dict[str, int] = {}
        self.padding_count = 0
        self.consumed_global = 0

    def has_training_remaining(self) -> bool:
        return self.current_round < math.ceil(self.total_rounds)

    def round_fraction(self) -> float:
        return min(1.0, self.total_rounds - self.current_round)

    def prepare_round(self, *, pad_to_multiple: int) -> PairSelection:
        if self.selected_indices:
            return PairSelection(
                indices=self.selected_indices,
                unique_indices=self.selected_unique_indices,
                padding_count=self.padding_count,
                selected_by_state=self.selected_by_state,
            )
        selection = self.scheduler.select(
            round_id=self.current_round,
            round_fraction=self.round_fraction(),
            pad_to_multiple=pad_to_multiple,
        )
        self.selected_indices = selection.indices
        self.selected_unique_indices = selection.unique_indices
        self.selected_by_state = selection.selected_by_state
        self.padding_count = selection.padding_count
        self.consumed_global = 0
        self.state.mark_seen(selection.unique_indices, self.current_round)
        return selection

    def adopt_broadcast_indices(self, indices: list[int]) -> PairSelection:
        """Adopt rank 0's selection or verify an in-progress resumed selection."""
        if self.selected_indices:
            if indices != self.selected_indices:
                raise ValueError("Pair-AFSS DDP selection differs from checkpoint state")
        else:
            unique = list(dict.fromkeys(indices))
            states = self.state.learning_state.tolist()
            self.selected_indices = list(indices)
            self.selected_unique_indices = unique
            self.padding_count = len(indices) - len(unique)
            self.selected_by_state = {
                state.name.lower(): sum(
                    states[index] == int(state) for index in unique
                )
                for state in PairLearningState
            }
            self.consumed_global = 0
            self.state.mark_seen(unique, self.current_round)
        return PairSelection(
            indices=self.selected_indices,
            unique_indices=self.selected_unique_indices,
            padding_count=self.padding_count,
            selected_by_state=self.selected_by_state,
        )

    def advance(self, global_exposures: int) -> None:
        self.consumed_global += int(global_exposures)
        if self.consumed_global > len(self.selected_indices):
            raise RuntimeError("Pair-AFSS cursor advanced past the selected round")

    def round_complete(self) -> bool:
        return bool(self.selected_indices) and self.consumed_global == len(
            self.selected_indices
        )

    def baseline_equivalent_step(self) -> float:
        if not self.selected_indices:
            rounds = min(float(self.current_round), self.total_rounds)
        else:
            fraction_done = self.consumed_global / len(self.selected_indices)
            rounds = self.current_round + self.round_fraction() * fraction_done
        return min(
            float(self.baseline_max_steps),
            rounds * self.baseline_steps_per_round,
        )

    def planned_minimum_actual_steps(self) -> int:
        """Return the v2 cosine horizon implied by its participation floor."""
        minimum_ratio = float(getattr(self.config, "min_round_ratio", 0.0))
        if minimum_ratio <= 0.0:
            return self.baseline_max_steps
        warmup_steps = min(
            self.baseline_max_steps,
            self.config.warmup_rounds * self.baseline_steps_per_round,
        )
        remaining_steps = self.baseline_max_steps - warmup_steps
        return max(
            1,
            math.ceil(warmup_steps + minimum_ratio * remaining_steps),
        )

    def finish_round(self) -> None:
        if not self.round_complete():
            raise RuntimeError("Cannot finish an incomplete Pair-AFSS round")
        self.current_round += 1
        self.selected_indices = []
        self.selected_unique_indices = []
        self.selected_by_state = {}
        self.padding_count = 0
        self.consumed_global = 0

    def refresh_due(self) -> bool:
        completed = self.current_round
        return completed >= self.config.warmup_rounds and (
            completed - self.config.warmup_rounds
        ) % self.config.refresh_interval == 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "baseline_max_steps": self.baseline_max_steps,
            "baseline_steps_per_round": self.baseline_steps_per_round,
            "current_round": self.current_round,
            "selected_indices": self.selected_indices,
            "selected_unique_indices": self.selected_unique_indices,
            "selected_by_state": self.selected_by_state,
            "padding_count": self.padding_count,
            "consumed_global": self.consumed_global,
            "pair_state": self.state.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("version", -1)) != self.VERSION:
            raise ValueError("Unsupported Pair-AFSS controller version")
        if int(payload["baseline_max_steps"]) != self.baseline_max_steps:
            raise ValueError("Pair-AFSS max-step budget changed across resume")
        if int(payload["baseline_steps_per_round"]) != self.baseline_steps_per_round:
            raise ValueError("Pair-AFSS dataset/global-batch geometry changed across resume")
        self.state.load_state_dict(payload["pair_state"])
        self.current_round = int(payload["current_round"])
        self.selected_indices = [int(value) for value in payload["selected_indices"]]
        self.selected_unique_indices = [
            int(value) for value in payload["selected_unique_indices"]
        ]
        self.selected_by_state = {
            str(key): int(value)
            for key, value in payload.get("selected_by_state", {}).items()
        }
        self.padding_count = int(payload["padding_count"])
        self.consumed_global = int(payload["consumed_global"])
        if self.consumed_global > len(self.selected_indices):
            raise ValueError("Pair-AFSS resume cursor exceeds the selected round")
