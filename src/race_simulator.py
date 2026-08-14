"""Single-race lap-by-lap Monte Carlo simulator, using the fitted
hierarchical tyre model, fitted grid penalty, and fitted pit-stop loss.
Multi-strategy: each driver samples one strategy from a fixed menu.

Noise is sampled from a large PRE-DRAWN pool (see scripts/
build_noise_pool.py), not generated live via PyMC -- an initial version
called pm.draw() once per driver per race, which recompiles PyTensor's
sampler from scratch every call (~6.6s for one 22-driver race, almost
entirely compilation overhead). Since sigma/skew_a/skew_b are fixed
population-wide constants, drawing a large iid pool once and indexing
into it with numpy during simulation is statistically equivalent and
removes PyTensor from the hot path entirely.

No overtaking dynamics -- each driver's race time is computed as if
racing alone on track. Wet/intermediate compounds use HARD's dry
degradation rate x WET_FALLBACK_MULTIPLIER (see src/tyre_model.py),
since the fitted wet-compound coefficients were physically implausible.
"""
from __future__ import annotations

import numpy as np

from src.tyre_model import WET_COMPOUNDS, WET_FALLBACK_MULTIPLIER

FUEL_MASS_START_KG = 110.0

STRATEGY_MENU = [
    [("MEDIUM", 0.55), ("HARD", 0.45)],
    [("SOFT", 0.35), ("HARD", 0.65)],
    [("SOFT", 0.30), ("MEDIUM", 0.35), ("SOFT", 0.35)],
]


def build_stints(strategy: list[tuple[str, float]], total_laps: int) -> list[tuple[str, int]]:
    stints = []
    laps_assigned = 0
    for i, (compound, fraction) in enumerate(strategy):
        if i == len(strategy) - 1:
            stint_laps = total_laps - laps_assigned
        else:
            stint_laps = round(fraction * total_laps)
        stints.append((compound, stint_laps))
        laps_assigned += stint_laps
    return stints


def get_degradation_rate(driver_code: str, compound: str, params: dict) -> float:
    if compound in WET_COMPOUNDS:
        return params["degradation_rate"][driver_code]["HARD"] * WET_FALLBACK_MULTIPLIER
    return params["degradation_rate"][driver_code][compound]


def sample_noise(n: int, noise_pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample n noise values from the pre-drawn pool, with replacement."""
    idx = rng.integers(0, len(noise_pool), size=n)
    return noise_pool[idx]


def simulate_driver_race(
    driver_code: str,
    grid_position: int,
    strategy: list[tuple[str, float]],
    total_laps: int,
    params: dict,
    noise_pool: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Simulate one driver's full race, return total race time in seconds."""
    stints = build_stints(strategy, total_laps)
    baseline_mean = params["baseline_pace"][driver_code]
    baseline_std = params["baseline_pace_std"][driver_code]
    baseline = rng.normal(baseline_mean, baseline_std)
    fuel_coef = params["fuel_coefficient"][driver_code]
    first_lap_effect = params["first_lap_effect"][driver_code]

    lap_times = []
    global_lap = 0
    for stint_idx, (compound, stint_laps) in enumerate(stints):
        deg_rate = get_degradation_rate(driver_code, compound, params)
        for laps_since_pit in range(stint_laps):
            fuel_remaining = FUEL_MASS_START_KG * (1 - global_lap / total_laps)
            is_first_lap = 1 if global_lap == 0 else 0

            z_laps = (laps_since_pit - params["laps_since_pit_mean"]) / params["laps_since_pit_std"]
            z_fuel = (fuel_remaining - params["fuel_remaining_mean"]) / params["fuel_remaining_std"]

            expected_pace = (
                baseline
                + deg_rate * z_laps
                + fuel_coef * z_fuel
                + first_lap_effect * is_first_lap
            )
            lap_times.append(expected_pace)
            global_lap += 1

        if stint_idx < len(stints) - 1:
            lap_times.append(params["pit_stop_loss_seconds"])

    noise = sample_noise(total_laps, noise_pool, rng)
    total_time = sum(lap_times) + noise.sum()

    grid_penalty = params["grid_penalty_intercept"] + params["seconds_per_grid_slot"] * grid_position
    return total_time + grid_penalty


def simulate_race(
    grid_positions: dict[str, int],
    total_laps: int,
    params: dict,
    noise_pool: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Simulate one full race for every driver, return finishing positions (1..N)."""
    race_times = {}
    for driver_code, grid_position in grid_positions.items():
        strategy = STRATEGY_MENU[rng.integers(len(STRATEGY_MENU))]
        race_times[driver_code] = simulate_driver_race(
            driver_code, grid_position, strategy, total_laps, params, noise_pool, rng
        )

    ranked = sorted(race_times, key=race_times.get)
    return {driver: position for position, driver in enumerate(ranked, start=1)}