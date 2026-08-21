"""Plot model championship predictions against real-world market consensus."""
import matplotlib.pyplot as plt
import numpy as np

from src.config import PROCESSED_DATA_DIR

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model (v3, physics-based, 2026_championship_predictions_physics.csv) vs.
# real-world market consensus (Polymarket, via CryptoSlate, synced late
# July 2026) -- external reference data, not a project output. Driver
# order follows the model's own predicted ranking, not the market's.
DRIVERS = ["Antonelli", "Leclerc", "Russell", "Hamilton", "Norris"]
MODEL_PROB = [69.0, 14.3, 7.1, 7.0, 1.8]
MARKET_PROB = [75.2, 2.4, 5.5, 9.7, 3.0]


def main() -> None:
    x = np.arange(len(DRIVERS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    model_bars = ax.bar(x - width / 2, MODEL_PROB, width, label="Our Model (v3, physics-based)", color="steelblue")
    market_bars = ax.bar(x + width / 2, MARKET_PROB, width, label="Real-World Market Consensus", color="darkorange")

    for i, v in enumerate(MODEL_PROB):
        ax.annotate(f"{v:.1f}%", (x[i] - width / 2, v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(MARKET_PROB):
        ax.annotate(f"{v:.1f}%", (x[i] + width / 2, v), ha="center", va="bottom", fontsize=9)

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