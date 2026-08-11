"""Trains and compares LogReg / Random Forest / Extra Trees on top10 and podium targets."""

import pandas as pd

from src import model
from src.config import PROCESSED_DATA_DIR


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")

    print("=== Top-10 finish prediction (precision@10 per race) ===")
    top10_results = model.run_comparison(df, target_col="target_top10", k=10)
    print(top10_results.round(3))

    print("\n=== Podium (top-3) prediction (precision@3 per race) ===")
    top3_results = model.run_comparison(df, target_col="target_top3", k=3)
    print(top3_results.round(3))

    print("\n=== Feature importance — Random Forest, top10 target ===")
    print(model.get_feature_importance(df, target_col="target_top10", model_name="random_forest").round(4))

    print("\n=== Feature importance — Random Forest, podium target ===")
    print(model.get_feature_importance(df, target_col="target_top3", model_name="random_forest").round(4))


if __name__ == "__main__":
    main()