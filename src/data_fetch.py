"""Fetch race results from fastf1 and save tidy per-season CSVs."""

import logging
import time

import fastf1
import pandas as pd

from src.config import (
    ALL_SEASONS,
    CACHE_DIR,
    CONSTRUCTOR_NAME_MAP,
    FASTF1_REQUEST_DELAY_SECONDS,
    RACE_SESSION,
    RAW_DATA_DIR,
    REGULATION_ERA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def enable_fastf1_cache() -> None:
    """Enable fastf1's on-disk cache. Must be called before any other fastf1 call."""
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    logger.info("fastf1 cache enabled at %s", CACHE_DIR)


def get_race_calendar(year: int) -> pd.DataFrame:
    """Return the race calendar for a season, excluding pre-season testing."""
    schedule = fastf1.get_event_schedule(year)
    return schedule[~schedule["EventFormat"].eq("testing")]


def get_completed_rounds(year: int) -> pd.DataFrame:
    """Return the calendar filtered to rounds whose race day is today or earlier.

    Uses the schedule's own EventDate rather than an API call per round, so
    future rounds are ruled out for free instead of via a failed request.
    Compares calendar dates only (not time-of-day), since EventDate marks
    the day of the race rather than its exact start time; fetch_race_results
    still falls back to skipping gracefully if a same-day race hasn't
    published results yet.
    """
    calendar = get_race_calendar(year)
    event_dates = pd.to_datetime(calendar["EventDate"])
    if event_dates.dt.tz is not None:
        event_dates = event_dates.dt.tz_localize(None)
    event_dates = event_dates.dt.normalize()

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return calendar[event_dates <= today]


def normalize_constructor(name: str) -> str:
    """Map a season-specific constructor name onto its current identity."""
    return CONSTRUCTOR_NAME_MAP.get(name, name)


def fetch_race_results(year: int, gp_round: int) -> pd.DataFrame | None:
    """Fetch and tidy the results table for a single race.

    Returns None if the session could not be loaded or has no results.
    """
    try:
        session = fastf1.get_session(year, gp_round, RACE_SESSION)
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        logger.warning("Could not load %s round %s (%s) — skipping", year, gp_round, exc)
        return None

    results = session.results
    if results is None or results.empty:
        logger.warning("No results available for %s round %s — skipping", year, gp_round)
        return None

    return pd.DataFrame(
        {
            "season": year,
            "round": gp_round,
            "event_name": session.event["EventName"],
            "regulation_era": REGULATION_ERA.get(year, "unknown"),
            "driver_code": results["Abbreviation"],
            "driver_name": results["FullName"],
            "constructor": results["TeamName"].apply(normalize_constructor),
            "constructor_raw": results["TeamName"],
            "grid_position": results["GridPosition"],
            "finish_position": results["Position"],
            "points": results["Points"],
            "status": results["Status"],
        }
    )


def fetch_season(year: int, request_delay_seconds: float = FASTF1_REQUEST_DELAY_SECONDS) -> pd.DataFrame:
    """Fetch every completed race in a season and combine into a single DataFrame.

    A delay is applied between rounds to stay under jolpica-f1's rate
    limit, which matters most on a cold cache where a 429 has no cached
    response to fall back to.
    """
    calendar = get_race_calendar(year)
    completed = get_completed_rounds(year)
    skipped = len(calendar) - len(completed)
    logger.info(
        "Fetching %s completed races for the %s season (%s not yet run, skipped without an API call)",
        len(completed),
        year,
        skipped,
    )

    season_results = []
    rounds = list(completed["RoundNumber"])
    for i, gp_round in enumerate(rounds):
        race_df = fetch_race_results(year, gp_round)
        if race_df is not None:
            season_results.append(race_df)
        if request_delay_seconds and i < len(rounds) - 1:
            time.sleep(request_delay_seconds)

    if not season_results:
        return pd.DataFrame()

    return pd.concat(season_results, ignore_index=True)


def fetch_all_seasons(
    seasons: list[int] = ALL_SEASONS,
    request_delay_seconds: float = FASTF1_REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """Fetch all configured seasons, saving one CSV per season plus a combined CSV.

    Rounds that haven't been run yet are filtered out using the calendar's
    own event dates before any API call is made. fetch_race_results still
    keeps its own try/except as a fallback for the rare case where a race
    has finished but results aren't published yet, so this can be re-run
    throughout the season to pick up new results.
    """
    enable_fastf1_cache()

    all_seasons_results = []
    for year in seasons:
        season_df = fetch_season(year, request_delay_seconds=request_delay_seconds)
        if season_df.empty:
            continue

        out_path = RAW_DATA_DIR / f"race_results_{year}.csv"
        season_df.to_csv(out_path, index=False)
        logger.info("Saved %s rows to %s", len(season_df), out_path)

        all_seasons_results.append(season_df)

    combined = pd.concat(all_seasons_results, ignore_index=True) if all_seasons_results else pd.DataFrame()
    if not combined.empty:
        combined_path = RAW_DATA_DIR / "race_results_all.csv"
        combined.to_csv(combined_path, index=False)
        logger.info("Saved combined file: %s rows to %s", len(combined), combined_path)

    return combined


if __name__ == "__main__":
    fetch_all_seasons()
