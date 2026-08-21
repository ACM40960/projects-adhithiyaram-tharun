"""Pre-draw a large pool of skewed-t noise samples once, so the race
simulator never has to compile/call PyTensor's SkewStudentT sampler
inside the simulation loop."""
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

    pool = pool - pool.mean()  # center to mean zero

    np.save(MODELS_DIR / "noise_pool.npy", pool)
    print(f"Saved pool: mean={pool.mean():.4f}, std={pool.std():.4f}")


if __name__ == "__main__":
    main()