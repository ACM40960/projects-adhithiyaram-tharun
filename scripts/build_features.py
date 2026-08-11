"""Builds the Stage 2 leakage-safe feature table from cleaned race data."""

import pandas as pd

from src import features
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main() -> None:
    races = pd.read_csv(RAW_DATA_DIR / "race_results_all.csv")

    feature_table = features.build_feature_table(races)

    output_path = PROCESSED_DATA_DIR / "features.csv"
    feature_table.to_csv(output_path, index=False)
    print(f"Saved {len(feature_table)} rows, {len(feature_table.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()