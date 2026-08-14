"""Plot predicted vs. actual calibration curve from the 2025 holdout check."""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DATA_DIR

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    calibration = pd.read_csv(PROCESSED_DATA_DIR / "calibration_2025.csv")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.plot(
        calibration["predicted_probability"],
        calibration["actual_top10_rate"],
        marker="o",
        color="steelblue",
        label="Model (2025 holdout)",
    )

    for _, row in calibration.iterrows():
        ax.annotate(
            f"n={int(row['n_observations'])}",
            (row["predicted_probability"], row["actual_top10_rate"]),
            textcoords="offset points",
            xytext=(6, -8),
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("Predicted P(top-10)")
    ax.set_ylabel("Actual top-10 rate")
    ax.set_title("Classifier Calibration — 2025 Holdout")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_2025.png", dpi=200)
    print(f"Saved to {FIGURES_DIR / 'calibration_2025.png'}")


if __name__ == "__main__":
    main()