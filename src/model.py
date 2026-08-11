"""
Follows Alahmadi et al.'s framing (binary classification on pre-race
features) rather than Bansal et al.'s raw points regression, to avoid the
circularity of using final race position as an input.
"""

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TRAIN_SEASONS = [2022, 2023, 2024]
VALIDATION_SEASON = 2025

# Every column here must be knowable strictly before the race starts.
# finish_position, points, and status are the current race's *outcome* —
# never allowed in this list.
FEATURE_COLUMNS = [
    "grid_position",
    "started_from_pit_lane",
    "driver_avg_finish_last3",
    "driver_avg_finish_last5",
    "driver_points_last3",
    "driver_points_last5",
    "driver_form_ewm",
    "driver_races_completed",
    "constructor_avg_finish_last3",
    "constructor_avg_finish_last5",
    "constructor_points_last3",
    "constructor_points_last5",
    "constructor_races_completed",
    "is_new_team",
    "races_into_season",
    "points_gap_to_leader",
]

MODELS = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=350, class_weight="balanced", random_state=42
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=350, class_weight="balanced", random_state=42
    ),
}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary top10 and top3 (podium) targets from finish_position."""
    df = df.copy()
    df["target_top10"] = (df["finish_position"] <= 10).astype(int)
    df["target_top3"] = (df["finish_position"] <= 3).astype(int)
    return df


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add targets, one-hot encode regulation_era, and cast is_new_team to int.

    regulation_era is categorical text ('ground_effect_2022_2025' /
    'next_gen_2026'); models need it as numbers, so it's one-hot encoded
    rather than dropped, since it is genuinely known before the race.
    """
    df = add_targets(df)
    df["is_new_team"] = df["is_new_team"].astype(int)
    era_dummies = pd.get_dummies(df["regulation_era"], prefix="era")
    df = pd.concat([df, era_dummies], axis=1)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Base feature list plus any one-hot regulation_era columns present."""
    era_cols = [c for c in df.columns if c.startswith("era_")]
    return FEATURE_COLUMNS + era_cols


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train (2022-2024) and validation (2025) — no shuffling.

    A random shuffle would let a 2025 race leak into training while a 2024
    race ends up in validation, which is exactly the temporal leakage
    problem Bansal et al.'s random k-fold caused.
    """
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val = df[df["season"] == VALIDATION_SEASON].copy()
    return train, val


def evaluate_row_level(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict:
    """Standard row-level classification metrics.

    zero_division=0 avoids a crash if a model predicts the positive class
    zero times on a given fold — reports 0 instead of erroring out.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def evaluate_race_level(val_df: pd.DataFrame, y_prob: pd.Series, k: int) -> float:
    """Precision@k per race, averaged across all validation races.

    For each race, rank drivers by predicted probability and take the
    top k. Compare against the actual top k finishers. This is the
    adaptation of Alahmadi et al.'s per-race Top-1/Top-3 ranking metric
    to a top-10/top-3 SET target rather than a single winner — row-level
    accuracy alone can't tell you whether the model correctly identifies
    *which* drivers make up the group for a given race.
    """
    scored = val_df.copy()
    scored["predicted_prob"] = y_prob

    race_scores = []
    for (season, rnd), race_group in scored.groupby(["season", "round"]):
        predicted_top_k = set(
            race_group.sort_values("predicted_prob", ascending=False).head(k)["driver_code"]
        )
        actual_top_k = set(
            race_group.sort_values("finish_position").head(k)["driver_code"]
        )
        overlap = len(predicted_top_k & actual_top_k) / k
        race_scores.append(overlap)

    return sum(race_scores) / len(race_scores) if race_scores else float("nan")


def run_comparison(df: pd.DataFrame, target_col: str, k: int) -> pd.DataFrame:
    """Train and evaluate all MODELS for one target, return a results table."""
    prepared = prepare_model_frame(df)
    feature_cols = get_feature_columns(prepared)
    train, val = temporal_split(prepared)

    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]

    rows = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = evaluate_row_level(y_val, y_pred, y_prob)
        metrics["precision_at_k"] = evaluate_race_level(val, y_prob, k)
        metrics["model"] = name
        rows.append(metrics)

    results = pd.DataFrame(rows).set_index("model")
    return results[["accuracy", "precision", "recall", "f1", "roc_auc", "precision_at_k"]]

def get_feature_importance(df: pd.DataFrame, target_col: str, model_name: str = "random_forest") -> pd.DataFrame:
    """Fit one model on the full training set and return its feature importances, sorted descending.

    Only tree-based models (random_forest, extra_trees) expose
    .feature_importances_; Logistic Regression's coefficients aren't
    directly comparable on the same scale and aren't handled here.
    """
    prepared = prepare_model_frame(df)
    feature_cols = get_feature_columns(prepared)
    train, _ = temporal_split(prepared)

    X_train, y_train = train[feature_cols], train[target_col]

    fitted_model = MODELS[model_name]
    fitted_model.fit(X_train, y_train)

    importances = pd.Series(fitted_model.feature_importances_, index=feature_cols)
    return importances.sort_values(ascending=False).to_frame("importance")