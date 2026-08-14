"""Incremental rolling-form state, used to featurise simulated future
races the same way src/features.py featurises real ones.

features.py computes driver_avg_finish_last{3,5}, driver_points_last{3,5},
and driver_form_ewm from the full historical sequence via shift+rolling
and shift+ewm. Once simulating rounds that don't exist in that history
yet, those pandas calls have nothing to operate on — each simulated
race's outcome has to update these values by hand, one round at a time,
using the exact recurrence pandas uses internally.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

EWM_ALPHA = 0.3
MAX_WINDOW = 5


@dataclass
class RollingForm:
    """Rolling-form state for one driver or one constructor.

    finish_history/points_history hold at most the last 5 values seen
    (oldest dropped once full) — enough for both the last-3 and last-5
    windows features.py exposes.

    ewm_numerator/ewm_denominator jointly reproduce pandas'
    `.ewm(alpha=EWM_ALPHA, adjust=True).mean()` exactly:
        num_t = x_t + (1 - alpha) * num_{t-1}
        den_t = 1   + (1 - alpha) * den_{t-1}
        ewm_t = num_t / den_t
    Matching this exactly (not an approximation) matters: the classifier
    was trained on features.py's pandas-computed EWM values, and a
    different recurrence would create a train/simulation mismatch the
    model never saw during fitting.
    """

    finish_history: deque = field(default_factory=lambda: deque(maxlen=MAX_WINDOW))
    points_history: deque = field(default_factory=lambda: deque(maxlen=MAX_WINDOW))
    ewm_numerator: float = 0.0
    ewm_denominator: float = 0.0
    races_completed: int = 0

    def avg_finish(self, window: int) -> float:
        values = list(self.finish_history)[-window:]
        return sum(values) / len(values) if values else float("nan")

    def sum_points(self, window: int) -> float:
        return sum(list(self.points_history)[-window:])

    def ewm(self) -> float:
        if self.ewm_denominator == 0:
            return float("nan")
        return self.ewm_numerator / self.ewm_denominator

    def update(self, finish_position: float, points: float) -> None:
        """Fold in one more real or simulated race result."""
        self.finish_history.append(finish_position)
        self.points_history.append(points)
        self.ewm_numerator = finish_position + (1 - EWM_ALPHA) * self.ewm_numerator
        self.ewm_denominator = 1 + (1 - EWM_ALPHA) * self.ewm_denominator
        self.races_completed += 1


def _seed_states(history, group_col: str) -> dict[str, RollingForm]:
    states: dict[str, RollingForm] = {}
    ordered = history.sort_values(["season", "round"])
    for key, group in ordered.groupby(group_col):
        state = RollingForm()
        for row in group.itertuples():
            state.update(row.finish_position, row.points)
        states[key] = state
    return states


def seed_driver_states(history) -> dict[str, RollingForm]:
    """Replay real results to build each driver's state as of the last completed race."""
    return _seed_states(history, "driver_code")


def seed_constructor_states(history) -> dict[str, RollingForm]:
    """Replay real results to build each constructor's state as of the last completed race."""
    return _seed_states(history, "constructor")