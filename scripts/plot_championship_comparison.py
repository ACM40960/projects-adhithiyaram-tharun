"""Plot model championship predictions against real-world market consensus."""
import matplotlib.pyplot as plt
import numpy as np

from src.config import PROCESSED_DATA_DIR

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model (v3, physics-based) vs. real-world 2026 betting-market consensus.
# Real-world figures sourced from public prediction markets/sportsbooks
# (Polymarket, DraftKings, mid-August 2026) -- see PROJECT_STATUS.md for
# the search that produced these; not derived from this project's own data.
DRIVERS = ["Antonelli", "Hamilton"]
MODEL_PROB = [69.0, 7.0]
MARKET_PROB = [76.0, 10.0]


def main() -> None:
    x = np.arange(len(DRIVERS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, MODEL_PROB, width, label="Our Model (v3, physics-based)", color="steelblue")
    market_plot = [v if v is not None else 0 for v in MARKET_PROB]
    bars = ax.bar(x + width / 2, market_plot, width, label="Real-World Market Consensus", color="darkorange")

    for i, v in enumerate(MARKET_PROB):
        if v is None:
            bars[i].set_alpha(0.15)
            ax.annotate("n/a", (x[i] + width / 2, 2), ha="center", fontsize=8, color="gray")

    ax.set_ylabel("Title probability (%)")
    ax.set_title("2026 Championship Prediction: Model vs. Real-World Consensus")
    ax.set_xticks(x)
    ax.set_xticklabels(DRIVERS)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "championship_model_vs_market.png", dpi=200)
    print(f"Saved to {FIGURES_DIR / 'championship_model_vs_market.png'}")


if __name__ == "__main__":
    main()