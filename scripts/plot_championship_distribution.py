"""Violin plot of full Monte Carlo championship point distributions —
shows variance propagation directly, not just summary statistics.
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DATA_DIR

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_TOP_DRIVERS = 8  # keep the plot readable; backmarkers add clutter, not signal


def main() -> None:
    points_df = pd.read_csv(PROCESSED_DATA_DIR / "physics_points_distribution.csv")

    mean_points = points_df.mean().sort_values(ascending=False)
    top_drivers = mean_points.head(N_TOP_DRIVERS).index.tolist()
    plot_data = points_df[top_drivers]

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(
        [plot_data[d].values for d in top_drivers],
        showmeans=True,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor("steelblue")
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")

    # Overlay p5-p95 as explicit whiskers, since default violin extrema
    # are full min/max (noisier, less standard than percentile bands).
    for i, driver in enumerate(top_drivers, start=1):
        p5, p95 = plot_data[driver].quantile([0.05, 0.95])
        ax.plot([i, i], [p5, p95], color="black", linewidth=1.2, alpha=0.7)
        ax.plot([i - 0.06, i + 0.06], [p5, p5], color="black", linewidth=1.2)
        ax.plot([i - 0.06, i + 0.06], [p95, p95], color="black", linewidth=1.2)

    ax.set_xticks(range(1, len(top_drivers) + 1))
    ax.set_xticklabels(top_drivers)
    ax.set_ylabel("Simulated championship points (10,000 seasons)")
    ax.set_title("Championship Points Distribution — Physics-Based Simulation (v3)")
    ax.text(
        0.98, 0.02,
        "Black bars: 5th–95th percentile range",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray",
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "championship_points_distribution.png", dpi=200)
    print(f"Saved to {FIGURES_DIR / 'championship_points_distribution.png'}")


if __name__ == "__main__":
    main()