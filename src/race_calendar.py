"""Per-circuit race distance (total laps), derived from real historical
lap data rather than an external reference table -- consistent with how
every other number in this project has been grounded. Falls back to the
grid-wide average lap count for circuits with zero historical data.
"""
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Maps 2026 schedule event names to the historical event name used in
# past seasons' data, where the same physical circuit was renamed.
# Barcelona Grand Prix (2026 round 7) is the same circuit as the
# historical "Spanish Grand Prix" (Circuit de Catalunya) -- real alias.
EVENT_NAME_ALIASES = {
    "Barcelona Grand Prix": "Spanish Grand Prix",
}

# 2026 rounds that are GENUINELY new circuits with zero historical lap
# data, even if they share a name with an existing historical event.
# Round 14 is officially "Gran Premio de España 2026" at MADRID -- a
# different physical circuit from every historical "Spanish Grand Prix"
# (2022-2025), which was held at Barcelona's Circuit de Catalunya.
# Confirmed via fastf1's OfficialEventName + Location fields, not
# assumed from the event name alone (which would have wrongly matched
# it to Barcelona's 66-lap historical figure). These use the grid-wide
# average as an explicit, documented cold-start fallback -- same
# treatment as Cadillac's cold-start in Stage 2.
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
    """Resolve a 2026 round's lap count: explicit new-circuit fallback
    takes priority over any name-based match, since a shared name does
    not guarantee a shared circuit (see round 14)."""
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