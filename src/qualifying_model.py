"""Predicts grid_position from pre-qualifying rolling-form features.

Needed only for simulating rounds that haven't happened: real grid
positions come from qualifying, which for future rounds hasn't occurred.
Trained on the same leakage-safe features as the top-10 classifier, but
excludes grid_position and started_from_pit_lane — both are qualifying
OUTCOMES, not inputs to it.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.config import ERA_COLUMNS, MODELS_DIR
from src.model import TRAIN_SEASONS, VALIDATION_SEASON, prepare_model_frame

QUALIFYING_FEATURE_COLUMNS = [
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
] + ERA_COLUMNS


def fit_qualifying_model(df: pd.DataFrame):
    """Fit a grid-position regressor and report its held-out residual std.

    residual_std becomes the noise scale used to simulate grid position
    under genuine uncertainty later — a point prediction alone would
    silently assume qualifying is deterministic given recent form, which
    it obviously isn't.
    """
    prepared = prepare_model_frame(df)
    train = prepared[prepared["season"].isin(TRAIN_SEASONS)]
    val = prepared[prepared["season"] == VALIDATION_SEASON]

    X_train, y_train = train[QUALIFYING_FEATURE_COLUMNS], train["grid_position"]
    X_val, y_val = val[QUALIFYING_FEATURE_COLUMNS], val["grid_position"]

    reg = RandomForestRegressor(n_estimators=350, random_state=42)
    reg.fit(X_train, y_train)

    val_pred = reg.predict(X_val)
    residual_std = float(np.sqrt(mean_squared_error(y_val, val_pred)))

    output_path = MODELS_DIR / "qualifying_position_model.joblib"
    joblib.dump({"model": reg, "residual_std": residual_std}, output_path)
    return reg, residual_std, output_path


def simulate_grid_positions(
    drivers: list[str],
    features: pd.DataFrame,
    model,
    residual_std: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Simulate one qualifying session: a valid 1..N grid order.

    Adding N(0, residual_std) noise to the model's point prediction and
    re-ranking — rather than rounding the point estimate directly — is
    what keeps grid position genuinely uncertain race to race, instead
    of collapsing to the model's single best guess every simulation.
    """
    predicted_grid = model.predict(features[QUALIFYING_FEATURE_COLUMNS])
    noisy = predicted_grid + rng.normal(0, residual_std, size=len(predicted_grid))
    order = np.argsort(noisy)  # ascending: smallest predicted slot = pole
    grid_positions = np.empty(len(order), dtype=int)
    grid_positions[order] = np.arange(1, len(order) + 1)
    return dict(zip(features["driver_code"], grid_positions))