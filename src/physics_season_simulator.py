"""Physics-based (v3) full-season Monte Carlo simulator: uses the
lap-by-lap race_simulator instead of classifier-score ranking. Mirrors
full_season_simulator.py's completed/future round structure and
qualifying-model batching, but swaps the per-race ranking mechanism.

Does NOT use the classifier-style season-level form_adjustment
(constructor/driver effects) -- the race simulator's own race-to-race
baseline variance (rng.normal(baseline_mean, baseline_std) per driver
per race, see src/race_simulator.py) is this variant's persistence
mechanism. Stacking both would double-count the same real-world
phenomenon (car/driver form varying race to race).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.config import ERA_COLUMNS
from src.full_season_simulator import build_future_round_features
from src.model import FEATURE_COLUMNS
from src.qualifying_model import QUALIFYING_FEATURE_COLUMNS
from src.race_calendar import get_lap_count_for_round
from src.race_simulator import simulate_race
from src.rolling_state import RollingForm
from src.season_simulator import POINTS_TABLE

CLASSIFIER_FEATURE_COLUMNS = FEATURE_COLUMNS + ERA_COLUMNS


def simulate_physics_season_mc(
    completed_schedule: list,
    completed_grid_by_round: dict[int, dict[str, int]],
    completed_event_names: dict[int, str],
    future_schedule: list,
    future_event_names: dict[int, str],
    drivers: list[str],
    driver_constructor_map: dict[str, str],
    initial_driver_states: dict[str, RollingForm],
    initial_constructor_states: dict[str, RollingForm],
    initial_season_race_counts: dict[str, int],
    qualifying_model,
    qualifying_residual_std: float,
    race_sim_params: dict,
    noise_pool: np.ndarray,
    lap_counts: dict,
    lap_count_fallback: int,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate n_simulations full seasons using real lap-by-lap physics
    for every race, completed and future."""
    rng = np.random.default_rng(seed)
    driver_index = {driver: i for i, driver in enumerate(drivers)}
    points_matrix = np.zeros((n_simulations, len(drivers)))

    # Completed rounds: real, fixed grid positions. Still re-simulated
    # per sim -- the race simulator's own noise/baseline-variance draws
    # give a distribution of plausible outcomes given what was actually
    # knowable pre-race, same design intent as v1's completed-rounds
    # simulator.
    for round_num in completed_schedule:
        total_laps = get_lap_count_for_round(
            round_num, completed_event_names[round_num], lap_counts, lap_count_fallback
        )
        grid_positions = completed_grid_by_round[round_num]
        for sim in range(n_simulations):
            positions = simulate_race(grid_positions, total_laps, race_sim_params, noise_pool, rng)
            for driver, position in positions.items():
                points = POINTS_TABLE.get(position)
                if points:
                    points_matrix[sim, driver_index[driver]] += points

    # Future rounds: recursive rolling state per sim, qualifying model
    # batched across sims (same reasoning as full_season_simulator.py --
    # sklearn's fixed per-call overhead makes batching worthwhile there;
    # the race simulator itself is cheap enough it doesn't need batching).
    driver_states = [copy.deepcopy(initial_driver_states) for _ in range(n_simulations)]
    constructor_states = [copy.deepcopy(initial_constructor_states) for _ in range(n_simulations)]
    season_race_counts = [dict(initial_season_race_counts) for _ in range(n_simulations)]

    for round_num in future_schedule:
        total_laps = get_lap_count_for_round(
            round_num, future_event_names[round_num], lap_counts, lap_count_fallback
        )

        batch_rows = []
        for sim in range(n_simulations):
            features = build_future_round_features(
                drivers, driver_constructor_map,
                driver_states[sim], constructor_states[sim], season_race_counts[sim],
            )
            features["sim"] = sim
            batch_rows.append(features)
        batch = pd.concat(batch_rows, ignore_index=True)

        predicted_grid = qualifying_model.predict(batch[QUALIFYING_FEATURE_COLUMNS])
        noisy_grid_score = predicted_grid + rng.normal(0, qualifying_residual_std, size=len(predicted_grid))
        batch["grid_position"] = (
            pd.Series(noisy_grid_score)
            .groupby(batch["sim"])
            .rank(method="first")
            .astype(int)
            .to_numpy()
        )

        for sim, group in batch.groupby("sim"):
            grid_positions = dict(zip(group["driver_code"], group["grid_position"]))
            positions = simulate_race(grid_positions, total_laps, race_sim_params, noise_pool, rng)
            for driver, position in positions.items():
                points = POINTS_TABLE.get(position, 0)
                if points:
                    points_matrix[sim, driver_index[driver]] += points
                constructor = driver_constructor_map[driver]
                driver_states[sim][driver].update(position, points)
                constructor_states[sim][constructor].update(position, points)
                season_race_counts[sim][driver] += 1

    return pd.DataFrame(points_matrix, columns=drivers)