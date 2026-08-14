"""Fit and save the grid-position (qualifying) model."""
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.qualifying_model import fit_qualifying_model


def main() -> None:
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    _, residual_std, output_path = fit_qualifying_model(features)
    print(f"Saved qualifying model to {output_path} (residual_std={residual_std:.3f})")


if __name__ == "__main__":
    main()