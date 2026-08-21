"""Confusion matrices and ROC curves for the production classifier (Random Forest), on both targets."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from src.config import PROCESSED_DATA_DIR
from src.model import build_models, get_feature_columns, prepare_model_frame, temporal_split

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "random_forest"
TARGETS = {
    "target_top10": "Top-10 Finish",
    "target_top3": "Podium Finish",
}


def fit_and_score(df: pd.DataFrame, target_col: str):
    """Fit Random Forest on the standard temporal split, return y_true, y_pred, y_prob."""
    prepared = prepare_model_frame(df)
    feature_cols = get_feature_columns(prepared)
    train, val = temporal_split(prepared)

    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]

    positive_class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = build_models(positive_class_ratio)[MODEL_NAME]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    return y_val, y_pred, y_prob


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")

    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))
    fig_roc, axes_roc = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (target_col, label) in enumerate(TARGETS.items()):
        y_true, y_pred, y_prob = fit_and_score(df, target_col)

        # Row-normalized: each row sums to 1.
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not " + label, label])
        disp.plot(ax=axes_cm[ax_idx], cmap="Blues", values_format=".2f", colorbar=False)
        axes_cm[ax_idx].set_title(f"Confusion Matrix — {label}\n(Random Forest, 2025 holdout)")

        # ROC curve
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes_roc[ax_idx], name="Random Forest")
        axes_roc[ax_idx].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
        axes_roc[ax_idx].set_title(f"ROC Curve — {label}\n(Random Forest, 2025 holdout)")
        axes_roc[ax_idx].legend()

    fig_cm.tight_layout()
    fig_cm.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200)
    print(f"Saved to {FIGURES_DIR / 'confusion_matrices.png'}")

    fig_roc.tight_layout()
    fig_roc.savefig(FIGURES_DIR / "roc_curves.png", dpi=200)
    print(f"Saved to {FIGURES_DIR / 'roc_curves.png'}")


if __name__ == "__main__":
    main()