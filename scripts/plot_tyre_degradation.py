"""Plot expected lap-time-vs-laps-since-pit curves per compound, using
the fitted hierarchical tyre model's population-level degradation rates.

Anchored so all three compounds start at the same fresh-tyre pace at
lap 0 and diverge from there -- the model only differentiates
DEGRADATION RATE by compound, not starting pace, so plotting the raw
z-scored expected_pace formula directly makes the lines cross near the
pool's mean laps-since-pit (~lap 12) rather than fan out from a common
origin, which reads as physically backwards to anyone unfamiliar with
the z-scoring. This version re-centers each compound's line so lap 0
is the shared reference point, matching the physical intuition that a
fresh tyre's pace is the same regardless of which compound it is --
only the WEAR RATE differs.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COMPOUND_RATES = {"SOFT": 0.598, "MEDIUM": 0.463, "HARD": 0.364}
COLORS = {"SOFT": "red", "MEDIUM": "gold", "HARD": "gray"}


def main() -> None:
    params = joblib.load(MODELS_DIR / "race_simulator_params.joblib")

    avg_baseline = np.mean(list(params["baseline_pace"].values()))
    laps_std = params["laps_since_pit_std"]
    laps_mean = params["laps_since_pit_mean"]

    laps = np.arange(0, 30)
    laps_z = (laps - laps_mean) / laps_std
    laps_z_at_zero = (0 - laps_mean) / laps_std  # z-value corresponding to lap 0

    fig, ax = plt.subplots(figsize=(7, 5))
    for compound, rate in COMPOUND_RATES.items():
        # Degradation added relative to lap 0, not relative to the pool
        # mean -- this is what anchors all three lines to a shared
        # starting point instead of letting them cross mid-plot.
        lap_times = avg_baseline + rate * (laps_z - laps_z_at_zero)
        ax.plot(laps, lap_times, label=compound, color=COLORS[compound], linewidth=2)

    ax.set_xlabel("Laps since pit stop")
    ax.set_ylabel("Expected lap time (s)")
    ax.set_title("Tyre Degradation by Compound (22-driver hierarchical fit)")
    ax.legend()
    ax.text(
        0.02, 0.98,
        "Curves anchored to a common lap-0 pace;\nonly the wear rate differs by compound.",
        transform=ax.transAxes, va="top", fontsize=8, color="gray",
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tyre_degradation_curves.png", dpi=200)
    print(f"Saved to {FIGURES_DIR / 'tyre_degradation_curves.png'}")


if __name__ == "__main__":
    main()