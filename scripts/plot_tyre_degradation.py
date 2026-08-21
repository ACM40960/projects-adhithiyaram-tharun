"""Plot expected lap-time-vs-laps-since-pit curves per compound, using
the fitted hierarchical tyre model's population-level degradation rates.
Curves are re-centered so lap 0 is a shared reference point across compounds."""
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
        # Degradation added relative to lap 0, not the pool mean.
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