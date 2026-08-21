"""Fit the grid-to-lap-1 time-loss penalty from real historical data."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def build_grid_vs_lap1_dataset(laps: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per driver-race: grid_position, and time gap to the race leader after lap 1."""
    lap1 = laps[laps["LapNumber"] == 1].copy()
    lap1["LapTime_seconds"] = lap1["LapTime"]  # already numeric seconds, no conversion needed

    merged = lap1.merge(
        results[["season", "round", "driver_code", "grid_position"]],
        on=["season", "round", "driver_code"],
    )

    # Gap to the fastest lap-1 time in that race.
    merged["lap1_gap_to_best"] = merged.groupby(["season", "round"])["LapTime_seconds"].transform(
        lambda s: s - s.min()
    )
    return merged[["season", "round", "driver_code", "grid_position", "lap1_gap_to_best"]].dropna()


def fit_grid_penalty(dataset: pd.DataFrame) -> dict:
    """Seconds lost per grid slot, from a simple linear fit."""
    X = dataset[["grid_position"]].values
    y = dataset["lap1_gap_to_best"].values

    model = LinearRegression()
    model.fit(X, y)

    return {
        "seconds_per_grid_slot": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "n_observations": len(dataset),
        "r_squared": model.score(X, y),
    }


def main() -> None:
    laps = pd.read_csv(RAW_DATA_DIR / "lap_data_all.csv")
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    results = features[["season", "round", "driver_code", "grid_position"]]

    dataset = build_grid_vs_lap1_dataset(laps, results)
    print(f"{len(dataset)} driver-race observations")

    fitted = fit_grid_penalty(dataset)
    print(fitted)

    import joblib
    from src.config import MODELS_DIR
    joblib.dump(fitted, MODELS_DIR / "grid_penalty.joblib")
    print(f"Saved to {MODELS_DIR / 'grid_penalty.joblib'}")


if __name__ == "__main__":
    main()