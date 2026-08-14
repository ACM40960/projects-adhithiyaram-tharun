"""Tier 1: linear tyre degradation model, fit via ordinary least squares.
Tier 2: Bayesian state-space version with skewed-t noise, fit via MCMC —
single-driver, pooled-per-driver, and full-grid hierarchical variants.

prepare_race_laps() is shared by every tier/variant and produces both raw
and standardized (z-scored) versions of the continuous predictors. Tier 1
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

MIN_LAPS_FOR_OWN_PARAMS = 1000  # below this, a driver is flagged "substitute"
# for display purposes; they are still fully included in the hierarchical
# fit and still get a real posterior estimate, just one that leans more
# heavily on the population-level prior (wide uncertainty, near the mean).

# Fixed, known universe of compounds, so compound indices stay consistent
# across every race pooled together — a per-race np.unique() would give
# inconsistent indices if different races happen to use different subsets
# of compounds. Used by the single-driver and pooled-per-driver Tier 2
# functions (build_tier2_model, build_pooled_tier2_model) — this is the
# 5-compound set the original VER fit was built and validated against.
ALL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

WET_COMPOUNDS = ["INTERMEDIATE", "WET"]

# Dry compounds only, used exclusively by the full-grid HIERARCHICAL
# model (build_hierarchical_tier2_model onward) — kept as a SEPARATE
# constant from ALL_COMPOUNDS (not a reassignment of it) specifically so
# the two never collide. WET/INTERMEDIATE degradation was already
# flagged as out of reliable scope in the single-driver Stage 4 fit (237
# and 13 laps respectively, vs 600-1,900+ for dry compounds) -- pooling
# across multiple drivers doesn't add data to those compounds, it just
# spreads the same sparsity across more weakly-informed per-driver
# estimates, which destabilized mu_degradation[INTERMEDIATE] and skew_a
# at n=15 drivers (r_hat 1.35 and 1.09 respectively). Hierarchical scope
# stays dry-compound; wet/intermediate stints instead use the separate
# grid-wide pooled fallback (see fit_pooled_wet_model) or, since that fit
# also produced physically implausible coefficients, WET_FALLBACK_MULTIPLIER.
HIERARCHICAL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Grid-wide OLS fit (fit_pooled_wet_model, 22-driver 2026 grid, 3,422 laps)
# produced physically implausible coefficients: negative degradation for
# both INTERMEDIATE and WET, and a first-lap effect of -7.4s (a "faster
# than normal" standing start) -- directly contradicting the dry-compound
# fit's +10.17s/+2.21s finding for the same effect. Most likely cause:
# wet races have disproportionate safety-car/VSC running, and the
# existing TrackStatus == "1" filter likely leaves a biased,
# unrepresentative subset of "green flag" wet laps. Rather than use a
# fitted-but-wrong number, Phase 2's simulator falls back to the HARD
# compound's dry degradation rate scaled by this fixed multiplier -- an
# explicit, documented simplification, not a validated wet-tyre model.
# See PROJECT_STATUS.md Stage 4/7 limitations.
WET_FALLBACK_MULTIPLIER = 1.5  # not fitted; placeholder pending a safety-car-aware wet-lap filter

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
    """Build one combined, cleaned lap table across every driver and every race they ran.

    Each driver's races are individually filtered via prepare_race_laps
    (same pit-lap / non-green-flag exclusions as the single-driver
    version), then concatenated and restricted to HIERARCHICAL_COMPOUNDS
    (dry compounds only — see that constant's docstring for why).
    driver_id and race_id are global integer indices used by the
    hierarchical model to index into per-driver and per-(driver,race)
    parameters.
    """
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

    # Re-densify race_id post-filter — see docstring above.
    pooled["race_id"] = pooled["race_id"].factorize()[0]

    for col in ["laps_since_pit", "fuel_remaining_kg"]:
        std = pooled[col].std(ddof=0)
        std = std if std > 0 else 1.0
        pooled[f"{col}_z"] = (pooled[col] - pooled[col].mean()) / std

    return pooled


def build_hierarchical_tier2_model(pooled: pd.DataFrame, n_drivers: int) -> pm.Model:
    """Full-grid Bayesian tyre degradation model with partial pooling across drivers.

    Dry compounds only (HIERARCHICAL_COMPOUNDS) — see that constant's
    docstring for why WET/INTERMEDIATE are excluded here specifically.

    degradation_rate, fuel_coefficient, and first_lap_effect are drawn
    per driver from shared population-level Normal distributions rather
    than fit independently — a low-lap-count driver's estimate shrinks
    toward the population mean with wide uncertainty, a high-lap-count
    driver's estimate stays close to their own data with tight
    uncertainty. Same shrinkage idea as Stage 5's driver/constructor
    form-adjustment split, applied here inside the tyre-physics model.

    Non-centered parameterization (raw ~ Normal(0,1), then
    param = mu + sigma * raw) throughout: NUTS samples this form far more
    reliably than the direct (centered) version once some groups have
    little data, where the centered form produces the classic
    hard-to-sample "funnel" posterior geometry.

    baseline_pace stays per (driver, race), not hierarchical: circuit and
    weekend car setup dominate it far more than a general "driver pace"
    trait would. Noise (sigma, skew_a, skew_b) stays shared across the
    whole grid — a stated simplification, not a tested claim that noise
    is genuinely uniform across drivers.
    """
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
        # Gamma(alpha, beta) with mean = alpha/beta. Centered near the
        # Stage 4 single-driver (VER) estimates (skew_a ~0.87, skew_b
        # ~0.49), with alpha chosen for a moderately tight but not rigid
        # spread -- informative enough to break the multimodality seen
        # when pooling 6 drivers' heterogeneous residuals under one
        # shared skew shape (original weak prior gave skew_a r_hat=1.92,
        # ess_bulk=5), not so tight it can't move if the true
        # population value differs from one driver's estimate.
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
    """Fit the full-grid hierarchical model via MCMC.

    target_accept raised to 0.95 (vs. 0.9 for the single-driver model) --
    non-centered hierarchies with sparse groups often need a higher
    target acceptance rate to avoid divergences even when the underlying
    geometry is otherwise fine, since NUTS has to take smaller, more
    careful steps near the funnel-prone regions.
    """
    model = build_hierarchical_tier2_model(pooled, n_drivers)
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=0.95, random_seed=42, progressbar=True)
    return model, trace


def summarize_driver_degradation(
    trace: az.InferenceData, drivers: list[str], driver_race_index: pd.DataFrame
) -> pd.DataFrame:
    """Per-driver, per-compound (dry only) posterior mean degradation rate, tagged primary/substitute.

    Built for a results view where primary (high-lap-count) drivers are
    shown by default and substitute drivers are available on request.
    """
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
    """Combined wet/intermediate lap table across the full grid.

    Grid-wide, not hierarchical: wet-condition laps per driver are too
    sparse (single digits to low hundreds) to support an individual
    per-driver estimate, so all drivers share one fitted rate here
    rather than each getting their own value the data can't actually
    distinguish. (This fit's coefficients turned out unusable anyway —
    see WET_FALLBACK_MULTIPLIER.)
    """
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

    Deliberately simple (Tier 1 style, not MCMC). Produced physically
    implausible coefficients when run on the 2026 grid (negative
    degradation, negative first-lap effect) — kept in the codebase for
    transparency/reproducibility, but NOT used by Phase 2's simulator.
    See WET_FALLBACK_MULTIPLIER for what's actually used instead.
    """
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