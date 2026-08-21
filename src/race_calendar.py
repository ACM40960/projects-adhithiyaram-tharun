"""Per-circuit race distance (total laps), derived from historical lap
data. Falls back to the grid-wide average for circuits with no history."""
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Barcelona Grand Prix (2026 round 7) = historical "Spanish Grand Prix".
EVENT_NAME_ALIASES = {
    "Barcelona Grand Prix": "Spanish Grand Prix",
}

# Round 14 ("Gran Premio de España 2026", Madrid) is a different circuit
# from the historical "Spanish Grand Prix" (Barcelona) despite the similar name.
NEW_CIRCUITS_BY_ROUND_2026 = {14: "Madrid Grand Prix (new circuit, no historical data)"}


def build_circuit_lap_counts() -> dict:
    laps = pd.read_csv(RAW_DATA_DIR / "lap_data_all.csv")
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")

    race_lengths = laps.groupby(["season", "round"])["LapNumber"].max().reset_index()
    race_lengths.columns = ["season", "round", "total_laps"]
    named = race_lengths.merge(
        features[["season", "round", "event_name"]].drop_duplicates(),
        on=["season", "round"],
    )

    lap_counts = (
        named.groupby("event_name")["total_laps"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )

    grid_wide_average = int(round(named["total_laps"].mean()))

    return lap_counts, grid_wide_average


def get_lap_count_for_round(round_num: int, event_name: str, lap_counts: dict, fallback: int) -> int:
    """Resolve a 2026 round's lap count, checking the new-circuit list before any name match."""
    if round_num in NEW_CIRCUITS_BY_ROUND_2026:
        return fallback

    resolved_name = EVENT_NAME_ALIASES.get(event_name, event_name)
    return int(lap_counts.get(resolved_name, fallback))


if __name__ == "__main__":
    lap_counts, grid_wide_average = build_circuit_lap_counts()
    print(f"Grid-wide average (fallback): {grid_wide_average} laps\n")
    for event, laps in sorted(lap_counts.items()):
        print(f"{event}: {int(laps)} laps")
    print(f"\nRound 14 (Madrid, new circuit): uses fallback = {grid_wide_average} laps")