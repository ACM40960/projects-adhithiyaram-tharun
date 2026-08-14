"""Plot Random Forest feature importance for the top-10 and podium targets."""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.model import get_feature_importance

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")

    top10_importance = get_feature_importance(df, "target_top10", "random_forest").head(10)
    podium_importance = get_feature_importance(df, "target_top3", "random_forest").head(10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].barh(top10_importance.index[::-1], top10_importance["importance"][::-1], color="steelblue")
    axes[0].set_title("Feature Importance — Top-10 Target")
    axes[0].set_xlabel("Importance")

    axes[1].barh(podium_importance.index[::-1], podium_importance["importance"][::-1], color="firebrick")
    axes[1].set_title("Feature Importance — Podium Target")
    axes[1].set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
    print(f"Saved to {FIGURES_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()