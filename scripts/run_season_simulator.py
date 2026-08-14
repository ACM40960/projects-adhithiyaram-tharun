"""Entry point: simulate the 2026 season and save championship predictions."""
import joblib
import pandas as pd

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.model import prepare_model_frame, get_feature_columns
from src.season_simulator import (
    precompute_race_strengths,
    simulate_season_mc,
    compute_championship_prob,
)

TARGET = "target_top10"
MODEL_NAME = "random_forest"


def load_2026_race_features() -> dict[int, pd.DataFrame]:
    """Run 2026 rows through the same prep as training, then split by round.

    Using prepare_model_frame here (not the raw CSV) is what guarantees
    the era dummy columns match what the fitted model expects.
    """
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    prepared = prepare_model_frame(features)
    season_2026 = prepared[prepared["season"] == 2026]
    return {
        round_id: group.reset_index(drop=True)
        for round_id, group in season_2026.groupby("round")
    }


def main() -> None:
    classifier = joblib.load(MODELS_DIR / f"{MODEL_NAME}_{TARGET}.joblib")

    race_features_by_id = load_2026_race_features()
    schedule = sorted(race_features_by_id)
    feature_cols = get_feature_columns(race_features_by_id[schedule[0]])

    driver_constructor_map = {
        row.driver_code: row.constructor
        for race in race_features_by_id.values()
        for row in race.itertuples()
    }

    race_strengths = precompute_race_strengths(
        schedule, race_features_by_id, classifier, feature_cols
    )

    points_df = simulate_season_mc(
        schedule, race_strengths, driver_constructor_map,
        n_simulations=10_000, seed=42,
    )
    results = compute_championship_prob(points_df)

    output_path = PROCESSED_DATA_DIR / "2026_championship_predictions.csv"
    results.to_csv(output_path)
    print(results)


if __name__ == "__main__":
    main()