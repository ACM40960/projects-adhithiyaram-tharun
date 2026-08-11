"""Leakage-safe feature engineering for the F1 race predictor.

Every function computes features using only information available
strictly before the race being featured. Column names match
src/data_fetch.py's actual output: driver_code, constructor, season,
round (there is no explicit race_date column — season+round is already
a valid chronological sort key since round numbers are assigned in
calendar order).

Sprint races are excluded entirely: data_fetch.py only ever pulls the
'R' (main race) session, so there is nothing to filter here.
"""

import pandas as pd

ROLLING_WINDOWS = [3, 5]
EWM_ALPHA = 0.3

SORT_KEYS = ["season", "round"]


def _rolling_shifted(series: pd.Series, groups: pd.Series, window: int, agg: str) -> pd.Series:
    """Shift a series by one row within each group, then take a rolling aggregate.

    Shifting before rolling is what guarantees race i's feature only
    reflects races < i.
    """
    shifted = series.groupby(groups).shift(1)
    rolled = shifted.groupby(groups).rolling(window, min_periods=1)
    return getattr(rolled, agg)().reset_index(level=0, drop=True)


def _ewm_shifted(series: pd.Series, groups: pd.Series, alpha: float) -> pd.Series:
    """Shift by one row within each group, then take an exponentially weighted mean."""
    shifted = series.groupby(groups).shift(1)
    return (
        shifted.groupby(groups)
        .apply(lambda s: s.ewm(alpha=alpha, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


def drop_unclassified(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no official classification (no finish_position at all).

    A small number of rows (e.g. did-not-start) have no finish_position
    because they were never classified, not because of a formatting issue.
    These can't sensibly contribute to or receive a rolling-form value, so
    they're dropped before feature engineering rather than imputed.
    """
    before = len(df)
    df = df.dropna(subset=["finish_position"]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"drop_unclassified: removed {dropped} unclassified row(s)")
    return df


def add_driver_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and EWM driver-form features, computed per driver_code."""
    df = df.sort_values(["driver_code", *SORT_KEYS]).copy()

    for window in ROLLING_WINDOWS:
        df[f"driver_avg_finish_last{window}"] = _rolling_shifted(
            df["finish_position"], df["driver_code"], window, "mean"
        )
        df[f"driver_points_last{window}"] = _rolling_shifted(
            df["points"], df["driver_code"], window, "sum"
        )

    df["driver_form_ewm"] = _ewm_shifted(df["finish_position"], df["driver_code"], EWM_ALPHA)
    df["driver_races_completed"] = df.groupby("driver_code").cumcount()
    return df


def add_constructor_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling constructor-form features on the normalized constructor name.

    Uses `constructor` (already normalized by CONSTRUCTOR_NAME_MAP in
    data_fetch.py) so a Kick Sauber -> Audi rebrand doesn't reset history.
    """
    df = df.sort_values(["constructor", *SORT_KEYS]).copy()

    for window in ROLLING_WINDOWS:
        df[f"constructor_avg_finish_last{window}"] = _rolling_shifted(
            df["finish_position"], df["constructor"], window, "mean"
        )
        df[f"constructor_points_last{window}"] = _rolling_shifted(
            df["points"], df["constructor"], window, "sum"
        )

    df["constructor_races_completed"] = df.groupby("constructor").cumcount()
    return df


def add_cold_start_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with no constructor history and impute form columns with a neutral default.

    A team with zero history (e.g. Cadillac in 2026) gets is_new_team=True
    and its rolling-form columns filled with the season-wide average rather
    than left as NaN or defaulted to 0, since 0 would read as "best possible
    result" to a model instead of "no information."
    """
    df = df.copy()
    df["is_new_team"] = df["constructor_races_completed"] == 0

    form_cols = [c for c in df.columns if "avg_finish" in c or "_form_ewm" in c]
    for col in form_cols:
        season_avg = df.groupby("season")[col].transform("mean")
        df[col] = df[col].fillna(season_avg)

    points_cols = [c for c in df.columns if "points_last" in c]
    df[points_cols] = df[points_cols].fillna(0)

    return df


def add_season_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Add as-of-race-day season indicators: races so far, points gap to leader."""
    df = df.sort_values(SORT_KEYS).copy()
    df["races_into_season"] = df.groupby(["season", "driver_code"]).cumcount()

    season_leader_points = df.groupby(["season", "round"])["driver_points_last5"].transform("max")
    df["points_gap_to_leader"] = season_leader_points - df["driver_points_last5"]
    return df


def build_feature_table(races: pd.DataFrame) -> pd.DataFrame:
    """Run the full leakage-safe feature pipeline in order."""
    df = drop_unclassified(races)
    df = add_driver_rolling_form(df)
    df = add_constructor_rolling_form(df)
    df = add_cold_start_flag(df)
    df = add_season_progress(df)
    return df