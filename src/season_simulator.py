"""Monte Carlo season simulator with two-level variance propagation."""
from __future__ import annotations

import numpy as np
import pandas as pd

POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def compute_strength_scores(probabilities: np.ndarray) -> np.ndarray:
    """Convert P(top-10) into an unbounded log-odds skill score."""
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def precompute_race_strengths(
    schedule: list,
    race_features_by_id: dict[object, pd.DataFrame],
    classifier,
    feature_columns: list[str],
    id_column: str = "driver_code",
) -> dict[object, pd.Series]:
    """Run the classifier once per race, not once per simulation."""
    strengths = {}
    for race_id in schedule:
        features = race_features_by_id[race_id]
        probabilities = classifier.predict_proba(features[feature_columns])[:, 1]
        scores = compute_strength_scores(probabilities)
        strengths[race_id] = pd.Series(scores, index=features[id_column])
    return strengths


def draw_season_form(
    driver_constructor_map: dict[str, str],
    driver_sigma: float = 0.15,
    constructor_sigma: float = 0.10,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Draw one season-level form adjustment per driver (constructor effect + driver effect)."""
    rng = rng or np.random.default_rng()
    constructors = sorted(set(driver_constructor_map.values()))
    constructor_effect = {c: rng.normal(0, constructor_sigma) for c in constructors}
    return {
        driver: rng.normal(0, driver_sigma) + constructor_effect[constructor]
        for driver, constructor in driver_constructor_map.items()
    }


def simulate_single_race(
    strength: pd.Series,
    form_adjustment: dict[str, float],
    rng: np.random.Generator,
) -> pd.Series:
    """Simulate one race and return a valid 1..N finishing order (Gumbel-max ranking)."""
    adjustment = strength.index.map(form_adjustment).fillna(0.0).to_numpy()
    noise = rng.gumbel(size=len(strength))
    race_score = strength.to_numpy() + adjustment + noise

    order = np.argsort(-race_score)
    ranked_drivers = strength.index.to_numpy()[order]
    return pd.Series(np.arange(1, len(order) + 1), index=ranked_drivers)


def simulate_season_mc(
    schedule: list,
    race_strengths: dict[object, pd.Series],
    driver_constructor_map: dict[str, str],
    n_simulations: int = 10_000,
    driver_sigma: float = 0.15,
    constructor_sigma: float = 0.10,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run n_simulations full seasons, return one row per simulation (columns = driver codes)."""
    rng = np.random.default_rng(seed)
    drivers = sorted(driver_constructor_map)
    driver_index = {driver: i for i, driver in enumerate(drivers)}
    points_matrix = np.zeros((n_simulations, len(drivers)))

    for sim in range(n_simulations):
        form_adjustment = draw_season_form(
            driver_constructor_map, driver_sigma, constructor_sigma, rng
        )
        for race_id in schedule:
            positions = simulate_single_race(
                race_strengths[race_id], form_adjustment, rng
            )
            for driver, position in positions.items():
                points = POINTS_TABLE.get(position)
                if points:
                    points_matrix[sim, driver_index[driver]] += points

    return pd.DataFrame(points_matrix, columns=drivers)


def compute_championship_prob(points_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-simulation points into championship probabilities."""
    n_sims = len(points_df)
    winner_counts = points_df.idxmax(axis=1).value_counts()
    ranks = points_df.rank(axis=1, ascending=False, method="min")

    summary = pd.DataFrame(
        {
            "mean_points": points_df.mean(),
            "std_points": points_df.std(),
            "p5_points": points_df.quantile(0.05),
            "p95_points": points_df.quantile(0.95),
        }
    )
    summary["p_win_championship"] = (
        winner_counts.reindex(summary.index, fill_value=0) / n_sims
    )
    summary["p_top3_final"] = (ranks <= 3).mean()

    return summary.sort_values("p_win_championship", ascending=False)