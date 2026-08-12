"""Fetches lap-by-lap data for 2022-2026, required for Stage 4 tyre modelling.

Separate from `python -m src.data_fetch` (race results) since this is a
much larger, slower fetch, only needed from Stage 4 onward.
"""

from src.data_fetch import fetch_all_lap_data

if __name__ == "__main__":
    fetch_all_lap_data()
