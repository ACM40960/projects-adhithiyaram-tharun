"""Tier 1: linear tyre degradation model, fit via OLS. Tier 2: Bayesian
state-space version with skewed-t noise, fit via MCMC (pooled-per-driver
and full-grid hierarchical variants)."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

FUEL_MASS_START_KG = 110.0  # regulatory maximum fuel load

MIN_LAPS_FOR_OWN_PARAMS = 1000  # below this, a driver is flagged "substitute"

ALL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

WET_COMPOUNDS = ["INTERMEDIATE", "WET"]

# Dry compounds only -- WET/INTERMEDIATE degradation is too data-sparse
# for a reliable per-driver estimate (see fit_pooled_wet_model instead).
HIERARCHICAL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Grid-wide wet/intermediate fit (fit_pooled_wet_model) came out physically
# implausible, so this fixed multiplier is used instead. See README Limitations.
WET_FALLBACK_MULTIPLIER = 1.5  # not fitted; placeholder pending a safety-car-aware wet-lap filter

if not hasattr(pm, "SkewStudentT"):
    raise ImportError(
        "This PyMC version has no pm.SkewStudentT. Check `python -c \"import pymc as pm; "
        "print(pm.__version__)\"` and either upgrade PyMC or fall back to plain pm.StudentT "
        "(symmetric, drop the skew_a parameter) in build_pooled_tier2_model."
    )


def prepare_race_laps(laps: pd.DataFrame, driver_code: str, season: int, gp_round: int) -> pd.DataFrame:
    """Filter to one driver's laps in one race, drop pit in/out laps and non-green-flag laps."""
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


def prepare_pooled_laps(laps: pd.DataFrame, driver_code: str, races: list[tuple[int, int]]) -> pd.DataFrame:
    """Build a combined lap table for one driver across several (season, round) races."""
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
    baseline_pace is fit per race; degradation/fuel/first-lap effects are shared."""
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


def build_driver_race_index(laps: pd.DataFrame, min_laps: int = MIN_LAPS_FOR_OWN_PARAMS) -> pd.DataFrame:
    """One row per driver: total lap count and primary/substitute tier."""
    counts = laps.groupby("driver_code").size().rename("n_laps").reset_index()
    counts["is_primary"] = counts["n_laps"] >= min_laps
    return counts.sort_values("n_laps", ascending=False).reset_index(drop=True)


def prepare_all_driver_pooled_laps(laps: pd.DataFrame, drivers: list[str]) -> pd.DataFrame:
    """Build one combined, cleaned lap table across every driver and every race they ran."""
    all_rows = []
    driver_race_id = 0
    for driver_idx, driver_code in enumerate(drivers):
        driver_laps = laps[laps["driver_code"] == driver_code]
        race_keys = driver_laps[["season", "round"]].drop_duplicates().itertuples(index=False)
        for season, gp_round in race_keys:
            race = prepare_race_laps(laps, driver_code, season, gp_round)
            if race.empty:
                continue
            race["driver_id"] = driver_idx
            race["driver_code"] = driver_code
            race["race_id"] = driver_race_id
            driver_race_id += 1
            all_rows.append(race)

    if not all_rows:
        return pd.DataFrame()

    pooled = pd.concat(all_rows, ignore_index=True)
    pooled = pooled[pooled["Compound"].isin(HIERARCHICAL_COMPOUNDS)].reset_index(drop=True)

    pooled["race_id"] = pooled["race_id"].factorize()[0]

    for col in ["laps_since_pit", "fuel_remaining_kg"]:
        std = pooled[col].std(ddof=0)
        std = std if std > 0 else 1.0
        pooled[f"{col}_z"] = (pooled[col] - pooled[col].mean()) / std

    return pooled


def build_hierarchical_tier2_model(pooled: pd.DataFrame, n_drivers: int) -> pm.Model:
    """Full-grid Bayesian tyre degradation model with partial pooling across drivers
    (dry compounds only). Non-centered parameterization for reliable NUTS sampling."""
    compound_idx = pooled["Compound"].map({c: i for i, c in enumerate(HIERARCHICAL_COMPOUNDS)}).values
    race_idx = pooled["race_id"].values
    driver_idx = pooled["driver_id"].values
    n_races = pooled["race_id"].nunique()
    n_compounds = len(HIERARCHICAL_COMPOUNDS)

    with pm.Model() as model:
        baseline_pace = pm.Normal("baseline_pace", mu=pooled["LapTime"].mean(), sigma=5, shape=n_races)

        mu_degradation = pm.Normal("mu_degradation", mu=0, sigma=1, shape=n_compounds)
        sigma_degradation = pm.HalfNormal("sigma_degradation", sigma=0.5, shape=n_compounds)
        degradation_raw = pm.Normal("degradation_raw", mu=0, sigma=1, shape=(n_drivers, n_compounds))
        degradation_rate = pm.Deterministic(
            "degradation_rate", mu_degradation + sigma_degradation * degradation_raw
        )

        mu_fuel = pm.Normal("mu_fuel", mu=0, sigma=1)
        sigma_fuel = pm.HalfNormal("sigma_fuel", sigma=0.5)
        fuel_raw = pm.Normal("fuel_raw", mu=0, sigma=1, shape=n_drivers)
        fuel_coefficient = pm.Deterministic("fuel_coefficient", mu_fuel + sigma_fuel * fuel_raw)

        mu_first_lap = pm.Normal("mu_first_lap", mu=5, sigma=10)
        sigma_first_lap = pm.HalfNormal("sigma_first_lap", sigma=3)
        first_lap_raw = pm.Normal("first_lap_raw", mu=0, sigma=1, shape=n_drivers)
        first_lap_effect = pm.Deterministic(
            "first_lap_effect", mu_first_lap + sigma_first_lap * first_lap_raw
        )

        expected_pace = (
            baseline_pace[race_idx]
            + degradation_rate[driver_idx, compound_idx] * pooled["laps_since_pit_z"].values
            + fuel_coefficient[driver_idx] * pooled["fuel_remaining_kg_z"].values
            + first_lap_effect[driver_idx] * pooled["is_first_lap"].values
        )

        sigma = pm.HalfNormal("sigma", sigma=1)
        skew_a = pm.Gamma("skew_a", alpha=8, beta=9.2)   # mean ~0.87
        skew_b = pm.Gamma("skew_b", alpha=6, beta=12.2)  # mean ~0.49

        pm.SkewStudentT(
            "observed_lap_time",
            mu=expected_pace,
            sigma=sigma,
            a=skew_a,
            b=skew_b,
            observed=pooled["LapTime"].values,
        )

    return model


def fit_hierarchical_tier2_model(
    pooled: pd.DataFrame, n_drivers: int, draws: int = 1000, tune: int = 1000
) -> tuple[pm.Model, az.InferenceData]:
    """Fit the full-grid hierarchical model via MCMC (target_accept=0.95)."""
    model = build_hierarchical_tier2_model(pooled, n_drivers)
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=0.95, random_seed=42, progressbar=True)
    return model, trace


def summarize_driver_degradation(
    trace: az.InferenceData, drivers: list[str], driver_race_index: pd.DataFrame
) -> pd.DataFrame:
    """Per-driver, per-compound (dry only) posterior mean degradation rate, tagged primary/substitute."""
    means = trace.posterior["degradation_rate"].mean(dim=["chain", "draw"]).values
    primary_lookup = dict(zip(driver_race_index["driver_code"], driver_race_index["is_primary"]))
    rows = [
        {
            "driver_code": driver_code,
            "compound": compound,
            "degradation_rate": means[driver_idx, compound_idx],
            "is_primary": primary_lookup.get(driver_code, False),
        }
        for driver_idx, driver_code in enumerate(drivers)
        for compound_idx, compound in enumerate(HIERARCHICAL_COMPOUNDS)
    ]
    return pd.DataFrame(rows)


def prepare_pooled_wet_laps(laps: pd.DataFrame, drivers: list[str]) -> pd.DataFrame:
    """Combined wet/intermediate lap table across the full grid (grid-wide, not hierarchical)."""
    all_rows = []
    for driver_code in drivers:
        driver_laps = laps[laps["driver_code"] == driver_code]
        race_keys = driver_laps[["season", "round"]].drop_duplicates().itertuples(index=False)
        for season, gp_round in race_keys:
            race = prepare_race_laps(laps, driver_code, season, gp_round)
            if race.empty:
                continue
            race = race[race["Compound"].isin(WET_COMPOUNDS)]
            if race.empty:
                continue
            race["driver_code"] = driver_code
            all_rows.append(race)

    if not all_rows:
        return pd.DataFrame()

    pooled = pd.concat(all_rows, ignore_index=True)
    for col in ["laps_since_pit", "fuel_remaining_kg"]:
        std = pooled[col].std(ddof=0)
        std = std if std > 0 else 1.0
        pooled[f"{col}_z"] = (pooled[col] - pooled[col].mean()) / std
    return pooled


def fit_pooled_wet_model(pooled_wet: pd.DataFrame) -> dict:
    """Grid-wide (non-hierarchical) OLS wet/intermediate degradation fit.
    Not used by Phase 2's simulator -- see WET_FALLBACK_MULTIPLIER instead."""
    encoder = OneHotEncoder(sparse_output=False)
    compound_dummies = encoder.fit_transform(pooled_wet[["Compound"]])
    compound_names = encoder.categories_[0]
    degradation_features = compound_dummies * pooled_wet[["laps_since_pit"]].values

    X = np.column_stack([
        degradation_features,
        pooled_wet["fuel_remaining_kg"].values,
        pooled_wet["is_first_lap"].values,
    ])
    feature_names = [f"degradation_{c}" for c in compound_names] + ["fuel_remaining_kg", "is_first_lap"]
    y = pooled_wet["LapTime"].values

    model = LinearRegression()
    model.fit(X, y)

    return {
        "model": model,
        "feature_names": feature_names,
        "compound_names": compound_names,
        "n_laps": len(pooled_wet),
        "n_drivers": pooled_wet["driver_code"].nunique(),
    }