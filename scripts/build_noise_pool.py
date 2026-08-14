"""Pre-draw a large pool of skewed-t noise samples once, so the race
simulator never has to compile/call PyTensor's SkewStudentT sampler
inside the simulation loop. The one-race test showed pm.draw() being
called 22 times (once per driver) each recompiling from scratch --
~6.6s for a single race, almost entirely compilation overhead, not
simulation cost. sigma/skew_a/skew_b are fixed population-wide
constants, so a large iid pool drawn once and sampled-with-replacement
during simulation is statistically valid and removes this cost entirely.
"""
import joblib
import numpy as np
import pymc as pm

from src.config import MODELS_DIR

POOL_SIZE = 2_000_000  # far more than any single run will need to draw from


def main() -> None:
    params = joblib.load(MODELS_DIR / "race_simulator_params.joblib")

    dist = pm.SkewStudentT.dist(
        mu=0, sigma=params["sigma"], a=params["skew_a"], b=params["skew_b"]
    )
    print(f"Drawing {POOL_SIZE} noise samples (one-time compile cost)...")
    pool = pm.draw(dist, draws=POOL_SIZE, random_seed=42)

    # Center to mean zero: expected_pace already represents the fitted
    # average lap time (baseline_pace etc. were fit to match observed
    # LapTime including whatever mean-shift the skew introduces at mu=0).
    # Adding this pool uncentered would double-count that shift as a
    # second systematic offset on every simulated lap.
    pool = pool - pool.mean()

    np.save(MODELS_DIR / "noise_pool.npy", pool)
    print(f"Saved pool: mean={pool.mean():.4f}, std={pool.std():.4f}")


if __name__ == "__main__":
    main()