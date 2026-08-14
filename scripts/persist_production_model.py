"""Fit and save the production model used by the season simulator.

Kept separate from train_classifier.py on purpose: that script's job is
model comparison/diagnostics across four algorithms and two targets; this
script's job is producing the one persisted artifact Stage 5 depends on.
"""
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.model import fit_final_model

TARGET = "target_top10"
MODEL_NAME = "random_forest"


def main() -> None:
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    _, output_path = fit_final_model(features, TARGET, MODEL_NAME)
    print(f"Saved {MODEL_NAME} ({TARGET}) to {output_path}")


if __name__ == "__main__":
    main()