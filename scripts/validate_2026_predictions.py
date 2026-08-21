"""Stage 6: walk-forward validation of the top-10 classifier against
real, completed 2026 rounds."""
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.model import (
    ERA_COLUMNS,
    FEATURE_COLUMNS,
    build_models,
    prepare_model_frame,
    walk_forward_split,
)
from sklearn.metrics import roc_auc_score

TARGET = "target_top10"
MODEL_NAME = "random_forest"


def get_completed_2026_rounds(df: pd.DataFrame) -> list[int]:
    return sorted(df[df["season"] == 2026]["round"].unique())


def run_walk_forward_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Fit-and-score one round at a time, walking forward through the real 2026 season."""
    prepared = prepare_model_frame(df)
    feature_cols = FEATURE_COLUMNS + ERA_COLUMNS
    rounds = get_completed_2026_rounds(prepared)

    all_rows = []
    for target_round in rounds:
        train, target = walk_forward_split(prepared, target_round)

        X_train, y_train = train[feature_cols], train[TARGET]
        X_target = target[feature_cols]

        positive_class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        model = build_models(positive_class_ratio)[MODEL_NAME]
        model.fit(X_train, y_train)

        predicted_prob = model.predict_proba(X_target)[:, 1]

        result = target[["driver_code", "grid_position", "finish_position"]].copy()
        result["round"] = target_round
        result["n_training_rows"] = len(train)
        result["predicted_prob"] = predicted_prob
        result["actual_top10"] = (target["finish_position"] <= 10).astype(int)
        all_rows.append(result)

        print(f"Round {target_round}: trained on {len(train)} rows, scored {len(target)} drivers")

    return pd.concat(all_rows, ignore_index=True)

def summarize(results: pd.DataFrame) -> dict:
    results["correct_at_0.5"] = (results["predicted_prob"] >= 0.5) == results["actual_top10"]
    return {
        "n_rounds": results["round"].nunique(),
        "n_driver_races": len(results),
        "accuracy_at_0.5_threshold": results["correct_at_0.5"].mean(),
        "roc_auc": roc_auc_score(results["actual_top10"], results["predicted_prob"]),
        "mean_predicted_prob_when_actual_top10": results.loc[
            results["actual_top10"] == 1, "predicted_prob"
        ].mean(),
        "mean_predicted_prob_when_actual_not_top10": results.loc[
            results["actual_top10"] == 0, "predicted_prob"
        ].mean(),
    }


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    results = run_walk_forward_validation(df)

    output_path = PROCESSED_DATA_DIR / "walk_forward_2026_validation.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved per-driver-race results to {output_path}")

    summary = summarize(results)
    print("\n=== Summary across all completed 2026 rounds ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Per-round accuracy ===")
    per_round = results.groupby("round", group_keys=False)[["predicted_prob", "actual_top10"]].apply(
        lambda g: (g["predicted_prob"] >= 0.5).eq(g["actual_top10"]).mean()
    )
    print(per_round.round(3))


if __name__ == "__main__":
    main()