"""Consolidate every Phase 1/2 fitted input into one params object for the race simulator."""
import pickle

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.tyre_model import HIERARCHICAL_COMPOUNDS, prepare_all_driver_pooled_laps

GRID_2026 = [
    "ALB", "ALO", "ANT", "BEA", "BOR", "BOT", "COL", "GAS", "HAD", "HAM",
    "HUL", "LAW", "LEC", "LIN", "NOR", "OCO", "PER", "PIA", "RUS", "SAI",
    "STR", "VER",
]


def main() -> None:
    with open(MODELS_DIR / "hierarchical_tyre_trace.pkl", "rb") as f:
        trace = pickle.load(f)

    laps = pd.read_csv(RAW_DATA_DIR / "lap_data_all.csv")
    pooled = prepare_all_driver_pooled_laps(laps, GRID_2026)

    # Recency-weighted baseline_pace: a flat career average would let old
    # seasons outweigh current form. RECENCY_DECAY gives 2026 races full
    # weight (1.0), decaying per year back.
    RECENCY_DECAY = 0.08

    race_driver_map = pooled[["race_id", "driver_id"]].drop_duplicates().set_index("race_id")["driver_id"]
    baseline_means = trace.posterior["baseline_pace"].mean(dim=["chain", "draw"]).values

    race_meta = pooled[["race_id", "season", "round"]].drop_duplicates().set_index("race_id")
    race_meta["baseline"] = baseline_means[race_meta.index]
    race_meta["driver_id"] = race_driver_map

    circuit_avg = race_meta.groupby(["season", "round"])["baseline"].transform("mean")
    race_meta["baseline_relative_to_circuit"] = race_meta["baseline"] - circuit_avg

    current_season = 2026
    race_meta["season_weight"] = RECENCY_DECAY ** (current_season - race_meta["season"])

    weight_by_season_driver = race_meta.groupby(["driver_id", "season"])["season_weight"].sum()
    print("\n2026 weight share for key drivers:")
    for driver_idx, driver_code in enumerate(["VER", "ANT", "LEC"]):
        if driver_code not in GRID_2026:
            continue
        idx = GRID_2026.index(driver_code)
        driver_weights = weight_by_season_driver.get(idx, pd.Series(dtype=float))
        total = driver_weights.sum()
        share_2026 = driver_weights.get(2026, 0) / total if total > 0 else 0
        print(f"  {driver_code}: 2026 share = {share_2026:.1%} (total weight across seasons: {driver_weights.to_dict()})")

    grid_wide_average_pace = float(race_meta["baseline"].mean())

    def weighted_mean(group: pd.DataFrame) -> float:
        return float(np.average(group["baseline_relative_to_circuit"], weights=group["season_weight"]))

    def weighted_std(group: pd.DataFrame) -> float:
        mean = weighted_mean(group)
        variance = np.average(
            (group["baseline_relative_to_circuit"] - mean) ** 2, weights=group["season_weight"]
        )
        return float(np.sqrt(variance))

    driver_relative_offset = race_meta.groupby("driver_id", group_keys=False).apply(weighted_mean)
    driver_relative_std = race_meta.groupby("driver_id", group_keys=False).apply(weighted_std)

    # Guard against zero variance for drivers with only one season fitted.
    population_avg_std = float(race_meta.groupby("driver_id", group_keys=False).apply(
        lambda g: g["baseline_relative_to_circuit"].std(ddof=0)
    ).mean())
    driver_relative_std = driver_relative_std.replace(0, population_avg_std).fillna(population_avg_std)

    driver_baseline = grid_wide_average_pace + driver_relative_offset
    driver_baseline_std = driver_relative_std

    degradation_means = trace.posterior["degradation_rate"].mean(dim=["chain", "draw"]).values
    fuel_means = trace.posterior["fuel_coefficient"].mean(dim=["chain", "draw"]).values
    first_lap_means = trace.posterior["first_lap_effect"].mean(dim=["chain", "draw"]).values

    grid_penalty = joblib.load(MODELS_DIR / "grid_penalty.joblib")
    pit_loss = joblib.load(MODELS_DIR / "pit_stop_loss.joblib")

    params = {
        "drivers": GRID_2026,
        "compounds": HIERARCHICAL_COMPOUNDS,
        "baseline_pace": {
            GRID_2026[i]: float(driver_baseline.get(i, driver_baseline.mean()))
            for i in range(len(GRID_2026))
        },
        "baseline_pace_std": {
            GRID_2026[i]: float(driver_baseline_std.get(i, population_avg_std))
            for i in range(len(GRID_2026))
        },
        "degradation_rate": {
            GRID_2026[i]: {
                HIERARCHICAL_COMPOUNDS[c]: float(degradation_means[i, c])
                for c in range(len(HIERARCHICAL_COMPOUNDS))
            }
            for i in range(len(GRID_2026))
        },
        "fuel_coefficient": {GRID_2026[i]: float(fuel_means[i]) for i in range(len(GRID_2026))},
        "first_lap_effect": {GRID_2026[i]: float(first_lap_means[i]) for i in range(len(GRID_2026))},
        "sigma": float(trace.posterior["sigma"].mean()),
        "skew_a": float(trace.posterior["skew_a"].mean()),
        "skew_b": float(trace.posterior["skew_b"].mean()),
        "laps_since_pit_mean": float(pooled["laps_since_pit"].mean()),
        "laps_since_pit_std": float(pooled["laps_since_pit"].std(ddof=0)) or 1.0,
        "fuel_remaining_mean": float(pooled["fuel_remaining_kg"].mean()),
        "fuel_remaining_std": float(pooled["fuel_remaining_kg"].std(ddof=0)) or 1.0,
        "seconds_per_grid_slot": grid_penalty["seconds_per_grid_slot"],
        "grid_penalty_intercept": grid_penalty["intercept"],
        "pit_stop_loss_seconds": pit_loss["pit_stop_loss_seconds"],
    }

    joblib.dump(params, MODELS_DIR / "race_simulator_params.joblib")
    print("Saved consolidated params to models/race_simulator_params.joblib")
    print(f"Baseline pace range: {min(params['baseline_pace'].values()):.1f}-{max(params['baseline_pace'].values()):.1f}s")
    print(f"SOFT degradation range: {min(v['SOFT'] for v in params['degradation_rate'].values()):.3f} to {max(v['SOFT'] for v in params['degradation_rate'].values()):.3f}")


if __name__ == "__main__":
    main()