"""Tier 1: linear tyre degradation model, fit via ordinary least squares.
Tier 2: Bayesian state-space version with skewed-t noise, fit via MCMC.

prepare_race_laps() is shared by both tiers and produces both raw and
standardized (z-scored) versions of the continuous predictors. Tier 1
uses the raw versions (coefficients stay physically interpretable, e.g.
seconds per lap). Tier 2 uses the standardized versions: an initial
unstandardized version produced severe sampling problems (r_hat 2-3.6,
effective sample sizes in single digits, every chain hitting max tree
depth) caused by wildly different parameter scales. Standardizing fixed
part of this, but pooling more data (see prepare_pooled_laps) and, most
importantly, replacing a hand-built skew via pt.switch (a discontinuous,
non-differentiable construction that HMC/NUTS cannot sample efficiently)
with PyMC's built-in pm.SkewStudentT (a smooth, proper skewed-t
distribution) were both needed before sampling converged cleanly.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

FUEL_MASS_START_KG = 110.0  # regulatory maximum fuel load

if not hasattr(pm, "SkewStudentT"):
    raise ImportError(
        "This PyMC version has no pm.SkewStudentT. Check `python -c \"import pymc as pm; "
        "print(pm.__version__)\"` and either upgrade PyMC or fall back to plain pm.StudentT "
        "(symmetric, drop the skew_a parameter) in build_tier2_model / build_pooled_tier2_model."
    )


def prepare_race_laps(laps: pd.DataFrame, driver_code: str, season: int, gp_round: int) -> pd.DataFrame:
    """Filter to one driver's laps in one race, drop pit in/out laps, add derived columns.

    Pit-entry and pit-exit laps are dominated by the pit lane speed limit
    rather than tyre performance, so they're excluded before fitting.
    Laps run under any non-green flag condition (safety car, VSC, yellow
    flag) are also excluded, since these run at an artificially slower,
    compressed pace unrelated to genuine tyre degradation.
    """
    race = laps[
        (laps["driver_code"] == driver_code)
        & (laps["season"] == season)
        & (laps["round"] == gp_round)
    ].copy()

    race = race[race["PitInTime"].isna() & race["PitOutTime"].isna()]
    race = race.dropna(subset=["LapTime", "Compound"]).sort_values("LapNumber").reset_index(drop=True)

    if race.empty:
        return race

    total_laps = race["LapNumber"].max()

    race["TrackStatus"] = race["TrackStatus"].astype(str)
    race = race[race["TrackStatus"] == "1"]

    if race.empty:
        return race

    race["is_first_lap"] = (race["LapNumber"] == 1).astype(int)

    fuel_remaining_fraction = 1 - (race["LapNumber"] - 1) / total_laps
    race["fuel_remaining_kg"] = fuel_remaining_fraction * FUEL_MASS_START_KG

    stint_lap = np.zeros(len(race))
    counter = 0
    previous_stint = race["Stint"].iloc[0]
    for i in range(len(race)):
        if race["Stint"].iloc[i] != previous_stint:
            counter = 0
            previous_stint = race["Stint"].iloc[i]
        stint_lap[i] = counter
        counter += 1
    race["laps_since_pit"] = stint_lap

    # Standardized (z-scored) versions for Tier 2's MCMC sampling only.
    # ddof=0 avoids a divide-by-zero if a race somehow has 1 usable lap;
    # std of 0 (no variation at all, e.g. laps_since_pit constant) falls
    # back to 1 to avoid a divide-by-zero, leaving the column as all 0s.
    for col in ["laps_since_pit", "fuel_remaining_kg"]:
        std = race[col].std(ddof=0)
        std = std if std > 0 else 1.0
        race[f"{col}_z"] = (race[col] - race[col].mean()) / std

    return race


def fit_tier1_model(race: pd.DataFrame) -> dict:
    """Fit an OLS regression: lap_time ~ compound-specific degradation + fuel weight + first-lap effect."""
    encoder = OneHotEncoder(sparse_output=False)
    compound_dummies = encoder.fit_transform(race[["Compound"]])
    compound_names = encoder.categories_[0]

    degradation_features = compound_dummies * race[["laps_since_pit"]].values

    X = np.column_stack([
        degradation_features,
        race["fuel_remaining_kg"].values,
        race["is_first_lap"].values,
    ])
    feature_names = [f"degradation_{c}" for c in compound_names] + ["fuel_remaining_kg", "is_first_lap"]

    y = race["LapTime"].values

    model = LinearRegression()
    model.fit(X, y)

    return {
        "model": model,
        "feature_names": feature_names,
        "compound_names": compound_names,
        "X": X,
        "y": y,
    }


def build_tier2_model(race: pd.DataFrame) -> pm.Model:
    """Bayesian state-space tyre degradation model with skewed-t noise (single race).

    Uses standardized (z-scored) laps_since_pit and fuel_remaining_kg
    (see prepare_race_laps) to keep all parameters on a comparable numeric
    scale. Noise uses pm.SkewStudentT — a proper, smooth skewed Student-t
    distribution — rather than a hand-built switch-based construction,
    which produced a non-differentiable likelihood that broke NUTS's
    gradient-based sampling (every chain hit max tree depth, r_hat 2-4,
    despite zero divergences).
    """
    compounds = race["Compound"].unique()
    compound_idx = race["Compound"].map({c: i for i, c in enumerate(compounds)}).values

    with pm.Model() as model:
        baseline_pace = pm.Normal("baseline_pace", mu=race["LapTime"].mean(), sigma=5)

        degradation_rate = pm.Normal("degradation_rate", mu=0, sigma=1, shape=len(compounds))
        fuel_coefficient = pm.Normal("fuel_coefficient", mu=0, sigma=1)
        first_lap_effect = pm.Normal("first_lap_effect", mu=5, sigma=10)

        expected_pace = (
            baseline_pace
            + degradation_rate[compound_idx] * race["laps_since_pit_z"].values
            + fuel_coefficient * race["fuel_remaining_kg_z"].values
            + first_lap_effect * race["is_first_lap"].values
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        # a and b are the Jones-Faddy skew-t's two shape parameters.
        # a == b gives a symmetric Student-t (heavier tails for smaller
        # values); a != b introduces skew, with the direction and degree
        # set by their difference. Separate weakly-informative Gamma
        # priors let the data pull a and b apart if the lap-time errors
        # are genuinely asymmetric, without forcing it.
        skew_a = pm.Gamma("skew_a", alpha=2, beta=0.5)
        skew_b = pm.Gamma("skew_b", alpha=2, beta=0.5)

        pm.SkewStudentT(
            "observed_lap_time",
            mu=expected_pace,
            sigma=sigma,
            a=skew_a,
            b=skew_b,
            observed=race["LapTime"].values,   # (or pooled["LapTime"].values in the pooled version)
        )

    return model


def fit_tier2_model(race: pd.DataFrame, draws: int = 1000, tune: int = 1000) -> tuple[pm.Model, az.InferenceData]:
    """Fit the single-race Tier 2 model via MCMC sampling."""
    model = build_tier2_model(race)
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=0.9, random_seed=42, progressbar=True)
    return model, trace


# Fixed, known universe of compounds, so compound indices stay consistent
# across every race pooled together — a per-race np.unique() would give
# inconsistent indices if different races happen to use different subsets
# of compounds.
ALL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def prepare_pooled_laps(laps: pd.DataFrame, driver_code: str, races: list[tuple[int, int]]) -> pd.DataFrame:
    """Build a combined lap table for one driver across several races.

    races is a list of (season, round) tuples. Each race is processed
    through prepare_race_laps individually (so all the existing filtering
    — pit laps, non-green flag laps — still applies per race), then
    concatenated and tagged with a race_id for the pooled model.
    Standardization is recomputed across the POOLED data, not per race,
    so laps_since_pit_z and fuel_remaining_kg_z are on one consistent
    scale across every race, rather than each race being standardized
    against its own separate mean/std.
    """
    all_races = []
    for race_id, (season, gp_round) in enumerate(races):
        race = prepare_race_laps(laps, driver_code, season, gp_round)
        if race.empty:
            print(f"Warning: no usable laps for {driver_code} {season} round {gp_round} — skipping")
            continue
        race["race_id"] = race_id
        race["season"] = season
        race["round"] = gp_round
        all_races.append(race)

    if not all_races:
        return pd.DataFrame()

    pooled = pd.concat(all_races, ignore_index=True)

    for col in ["laps_since_pit", "fuel_remaining_kg"]:
        std = pooled[col].std(ddof=0)
        std = std if std > 0 else 1.0
        pooled[f"{col}_z"] = (pooled[col] - pooled[col].mean()) / std

    return pooled


def build_pooled_tier2_model(pooled: pd.DataFrame) -> pm.Model:
    """Bayesian tyre degradation model pooling multiple races for one driver.

    degradation_rate, fuel_coefficient, first_lap_effect, and the noise
    parameters (nu, sigma, skew_a) are SHARED across all pooled races,
    treated as stable properties of this driver's tyre/car behaviour.
    baseline_pace is fit separately PER RACE, since different circuits
    and car setups produce genuinely different baseline lap times that
    shouldn't be forced to share one value. Noise uses pm.SkewStudentT
    (see build_tier2_model's docstring for why the earlier hand-built
    switch-based skew construction was replaced).
    """
    compound_idx = pooled["Compound"].map({c: i for i, c in enumerate(ALL_COMPOUNDS)}).values
    race_idx = pooled["race_id"].values
    n_races = pooled["race_id"].nunique()

    with pm.Model() as model:
        baseline_pace = pm.Normal("baseline_pace", mu=pooled["LapTime"].mean(), sigma=5, shape=n_races)

        degradation_rate = pm.Normal("degradation_rate", mu=0, sigma=1, shape=len(ALL_COMPOUNDS))
        fuel_coefficient = pm.Normal("fuel_coefficient", mu=0, sigma=1)
        first_lap_effect = pm.Normal("first_lap_effect", mu=5, sigma=10)

        expected_pace = (
            baseline_pace[race_idx]
            + degradation_rate[compound_idx] * pooled["laps_since_pit_z"].values
            + fuel_coefficient * pooled["fuel_remaining_kg_z"].values
            + first_lap_effect * pooled["is_first_lap"].values
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        # a and b are the Jones-Faddy skew-t's two shape parameters.
        # a == b gives a symmetric Student-t (heavier tails for smaller
        # values); a != b introduces skew, with the direction and degree
        # set by their difference. Separate weakly-informative Gamma
        # priors let the data pull a and b apart if the lap-time errors
        # are genuinely asymmetric, without forcing it.
        skew_a = pm.Gamma("skew_a", alpha=2, beta=0.5)
        skew_b = pm.Gamma("skew_b", alpha=2, beta=0.5)

        pm.SkewStudentT(
            "observed_lap_time",
            mu=expected_pace,
            sigma=sigma,
            a=skew_a,
            b=skew_b,
            observed=pooled["LapTime"].values,
        )

    return model


def fit_pooled_tier2_model(pooled: pd.DataFrame, draws: int = 1000, tune: int = 1000) -> tuple[pm.Model, az.InferenceData]:
    """Fit the pooled Tier 2 model via MCMC sampling."""
    model = build_pooled_tier2_model(pooled)
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=0.9, random_seed=42, progressbar=True)
    return model, trace