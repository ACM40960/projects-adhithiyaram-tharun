"""Single-race test: correctness check plus timing, before wiring into Stage 5's simulation loop."""
import time

import joblib
import numpy as np

from src.config import MODELS_DIR
from src.race_simulator import simulate_race

TOTAL_LAPS = 58


def main() -> None:
    params = joblib.load(MODELS_DIR / "race_simulator_params.joblib")
    noise_pool = np.load(MODELS_DIR / "noise_pool.npy")
    drivers = params["drivers"]

    rng = np.random.default_rng(42)
    grid_positions = {driver: i + 1 for i, driver in enumerate(rng.permutation(drivers))}

    start = time.time()
    positions = simulate_race(grid_positions, TOTAL_LAPS, params, noise_pool, rng)
    elapsed = time.time() - start

    print(f"Single race simulated in {elapsed:.5f}s")
    for driver, position in sorted(positions.items(), key=lambda x: x[1]):
        print(f"  P{position:2d}  {driver}  (started P{grid_positions[driver]})")

    n_sims = 10_000
    n_races = 23
    projected = elapsed * n_sims * n_races
    print(f"\nProjected cost at {n_sims} sims x {n_races} races: {projected:.0f}s ({projected/60:.1f} min)")


if __name__ == "__main__":
    main()