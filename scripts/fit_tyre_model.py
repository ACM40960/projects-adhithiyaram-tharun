"""Fits the Tier 1 (OLS) and Tier 2 (pooled Bayesian) tyre degradation models."""

import pandas as pd
import arviz as az
from src import tyre_model
from src.config import RAW_DATA_DIR

DRIVER_CODE = "VER"
POOL_SEASONS = [2022, 2023, 2024, 2025]  # exclude 2026 — partial season, reserved for later prediction


def get_driver_races(laps: pd.DataFrame, driver_code: str, seasons: list[int]) -> list[tuple[int, int]]:
    """Return every (season, round) this driver actually has lap data for, across the given seasons."""
    subset = laps[(laps["driver_code"] == driver_code) & (laps["season"].isin(seasons))]
    races = subset[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    return list(races.itertuples(index=False, name=None))


def main() -> None:
    laps = pd.read_csv(RAW_DATA_DIR / "lap_data_all.csv")

    races = get_driver_races(laps, DRIVER_CODE, POOL_SEASONS)
    print(f"Found {len(races)} races for {DRIVER_CODE} across seasons {POOL_SEASONS}")

    print("\n=== Tier 1: single-race OLS (first race in the list, for comparison) ===")
    season, gp_round = races[0]
    race = tyre_model.prepare_race_laps(laps, DRIVER_CODE, season, gp_round)
    if not race.empty:
        fit = tyre_model.fit_tier1_model(race)
        model = fit["model"]
        for name, coef in zip(fit["feature_names"], model.coef_):
            print(f"  {name}: {coef:.4f}")
        print(f"  R²: {model.score(fit['X'], fit['y']):.4f}")

    print(f"\n=== Tier 2: pooled Bayesian fit — {DRIVER_CODE}, {len(races)} races ===")
    pooled = tyre_model.prepare_pooled_laps(laps, DRIVER_CODE, races)
    print(f"{len(pooled)} total pooled laps across {pooled['race_id'].nunique()} races")
    print(pooled["Compound"].value_counts())

    if pooled.empty:
        print("No usable pooled laps found.")
        return

    print("This will take a while...")
    tier2_model, trace = tyre_model.fit_pooled_tier2_model(pooled)

    summary = az.summary(trace, var_names=["baseline_pace", "degradation_rate", "fuel_coefficient",
                                             "first_lap_effect", "sigma", "skew_a", "skew_b"])
    print(summary)
    print(f"\nDivergences: {trace.sample_stats.diverging.sum().item()}")


if __name__ == "__main__":
    main()