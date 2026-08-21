"""Fit average pit-stop time loss from real in-lap/out-lap timing data."""
import pandas as pd

from src.config import MODELS_DIR, RAW_DATA_DIR


def fit_pit_stop_loss(laps: pd.DataFrame) -> dict:
    """Estimate pit-stop cost as in/out-lap time minus typical green-flag pace, averaged across all pit events."""
    laps = laps.dropna(subset=["LapTime"]).copy()
    losses = []

    for (_driver, _season, _round), race in laps.groupby(["driver_code", "season", "round"]):
        race = race.sort_values("LapNumber")
        green = race[
            race["PitInTime"].isna() & race["PitOutTime"].isna()
            & (race["TrackStatus"].astype(str) == "1")
        ]
        if green.empty:
            continue
        typical_pace = green["LapTime"].median()

        for _, lap in race[race["PitInTime"].notna()].iterrows():
            losses.append(lap["LapTime"] - typical_pace)
        for _, lap in race[race["PitOutTime"].notna()].iterrows():
            losses.append(lap["LapTime"] - typical_pace)

    losses = pd.Series(losses).dropna()
    losses = losses[losses > 0]

    return {
        "pit_stop_loss_seconds": float(losses.median()),
        "mean": float(losses.mean()),
        "n_observations": int(len(losses)),
    }


def main() -> None:
    laps = pd.read_csv(RAW_DATA_DIR / "lap_data_all.csv")
    fitted = fit_pit_stop_loss(laps)
    print(fitted)

    import joblib
    joblib.dump(fitted, MODELS_DIR / "pit_stop_loss.joblib")
    print(f"Saved to {MODELS_DIR / 'pit_stop_loss.joblib'}")


if __name__ == "__main__":
    main()