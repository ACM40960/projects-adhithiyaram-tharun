"""Calibration check: do the classifier's predicted P(top-10)
probabilities match actual top-10 rates on held-out 2025 data?"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.config import PROCESSED_DATA_DIR
from src.model import build_models, get_feature_columns, prepare_model_frame, temporal_split

TARGET = "target_top10"
MODEL_NAME = "random_forest"
N_BINS = 10


def compute_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Fit on 2022-2024, evaluate calibration against the real 2025 outcomes."""
    prepared = prepare_model_frame(df)
    feature_cols = get_feature_columns(prepared)
    train, val = temporal_split(prepared)

    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    positive_class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = build_models(positive_class_ratio)[MODEL_NAME]
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    actual_rate, predicted_rate = calibration_curve(y_val, y_prob, n_bins=N_BINS, strategy="uniform")

    bin_edges = np.linspace(0, 1, N_BINS + 1)
    bin_counts = pd.cut(y_prob, bins=bin_edges).value_counts().sort_index().to_numpy()
    # calibration_curve silently drops empty bins.
    nonempty_counts = bin_counts[bin_counts > 0]

    return pd.DataFrame(
        {
            "predicted_probability": predicted_rate,
            "actual_top10_rate": actual_rate,
            "n_observations": nonempty_counts,
        }
    )


def main() -> None:
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    calibration = compute_calibration(features)
    print(calibration.round(3))

    max_gap = (calibration["predicted_probability"] - calibration["actual_top10_rate"]).abs().max()
    print(f"\nMax gap between predicted and actual: {max_gap:.3f}")

    output_path = PROCESSED_DATA_DIR / "calibration_2025.csv"
    calibration.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()