"""Full-season Monte Carlo simulator covering completed and not-yet-run
rounds of a season."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import fastf1

from src.config import ERA_COLUMNS
from src.model import FEATURE_COLUMNS
from src.qualifying_model import QUALIFYING_FEATURE_COLUMNS, simulate_grid_positions
from src.rolling_state import RollingForm
from src.season_simulator import (
    POINTS_TABLE,
    compute_strength_scores,
    draw_season_form,
    simulate_single_race,
)

CLASSIFIER_FEATURE_COLUMNS = FEATURE_COLUMNS + ERA_COLUMNS


def get_future_rounds(season: int, completed_rounds: set[int]) -> list[int]:
    """Remaining calendar rounds not yet reflected in features.csv."""
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    all_rounds = set(schedule["RoundNumber"])
    return sorted(all_rounds - completed_rounds)


def build_future_round_features(
    drivers: list[str],
    driver_constructor_map: dict[str, str],
    driver_states: dict[str, RollingForm],
    constructor_states: dict[str, RollingForm],
    season_race_counts: dict[str, int],
) -> pd.DataFrame:
    """Assemble one future round's pre-race feature row per driver from current state."""
    driver_points_last5 = {d: driver_states[d].sum_points(5) for d in drivers}
    leader_points = max(driver_points_last5.values())

    rows = []
    for driver in drivers:
        constructor = driver_constructor_map[driver]
        d, c = driver_states[driver], constructor_states[constructor]
        rows.append(
            {
                "driver_code": driver,
                "started_from_pit_lane": 0,
                "driver_avg_finish_last3": d.avg_finish(3),
                "driver_avg_finish_last5": d.avg_finish(5),
                "driver_points_last3": d.sum_points(3),
                "driver_points_last5": driver_points_last5[driver],
                "driver_form_ewm": d.ewm(),
                "driver_races_completed": d.races_completed,
                "constructor_avg_finish_last3": c.avg_finish(3),
                "constructor_avg_finish_last5": c.avg_finish(5),
                "constructor_points_last3": c.sum_points(3),
                "constructor_points_last5": c.sum_points(5),
                "constructor_races_completed": c.races_completed,
                "is_new_team": int(c.races_completed == 0),
                "races_into_season": season_race_counts[driver],
                "points_gap_to_leader": leader_points - driver_points_last5[driver],
                "era_ground_effect_2022_2025": 0,
                "era_next_gen_2026": 1,
            }
        )
    return pd.DataFrame(rows)


def simulate_full_season_mc(
    completed_schedule: list,
    completed_race_strengths: dict,
    future_schedule: list,
    drivers: list[str],
    driver_constructor_map: dict[str, str],
    initial_driver_states: dict[str, RollingForm],
    initial_constructor_states: dict[str, RollingForm],
    initial_season_race_counts: dict[str, int],
    top10_classifier,
    qualifying_model,
    qualifying_residual_std: float,
    n_simulations: int = 10_000,
    driver_sigma: float = 0.15,
    constructor_sigma: float = 0.10,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate n_simulations full seasons: real completed rounds plus
    recursively simulated future rounds, batched across simulations."""
    rng = np.random.default_rng(seed)
    driver_index = {driver: i for i, driver in enumerate(drivers)}
    points_matrix = np.zeros((n_simulations, len(drivers)))

    # Completed rounds: precomputed-strength lookups only.
    form_adjustments = []
    for sim in range(n_simulations):
        form_adjustment = draw_season_form(
            driver_constructor_map, driver_sigma, constructor_sigma, rng
        )
        form_adjustments.append(form_adjustment)
        for race_id in completed_schedule:
            positions = simulate_single_race(
                completed_race_strengths[race_id], form_adjustment, rng
            )
            for driver, position in positions.items():
                points = POINTS_TABLE.get(position)
                if points:
                    points_matrix[sim, driver_index[driver]] += points

    # Future rounds: per-simulation state, but model calls batched per round.
    driver_states = [copy.deepcopy(initial_driver_states) for _ in range(n_simulations)]
    constructor_states = [copy.deepcopy(initial_constructor_states) for _ in range(n_simulations)]
    season_race_counts = [dict(initial_season_race_counts) for _ in range(n_simulations)]

    for _race_id in future_schedule:
        batch_rows = []
        for sim in range(n_simulations):
            features = build_future_round_features(
                drivers, driver_constructor_map,
                driver_states[sim], constructor_states[sim], season_race_counts[sim],
            )
            features["sim"] = sim
            batch_rows.append(features)
        batch = pd.concat(batch_rows, ignore_index=True)

        # One qualifying prediction for all sims x drivers this round.
        predicted_grid = qualifying_model.predict(batch[QUALIFYING_FEATURE_COLUMNS])
        noisy_grid_score = predicted_grid + rng.normal(0, qualifying_residual_std, size=len(predicted_grid))
        batch["grid_position"] = (
            pd.Series(noisy_grid_score)
            .groupby(batch["sim"])
            .rank(method="first")
            .astype(int)
            .to_numpy()
        )

        # One classifier prediction for all sims x drivers this round.
        probabilities = top10_classifier.predict_proba(batch[CLASSIFIER_FEATURE_COLUMNS])[:, 1]
        batch["strength"] = compute_strength_scores(probabilities)

        # Per-sim: cheap ranking + state update (no model calls here).
        for sim, group in batch.groupby("sim"):
            strength = pd.Series(group["strength"].to_numpy(), index=group["driver_code"])
            positions = simulate_single_race(strength, form_adjustments[sim], rng)
            for driver, position in positions.items():
                points = POINTS_TABLE.get(position, 0)
                if points:
                    points_matrix[sim, driver_index[driver]] += points
                constructor = driver_constructor_map[driver]
                driver_states[sim][driver].update(position, points)
                constructor_states[sim][constructor].update(position, points)
                season_race_counts[sim][driver] += 1

    return pd.DataFrame(points_matrix, columns=drivers)