"""Full end-to-end 2026 season Monte Carlo simulation: real completed
rounds plus recursively simulated remaining rounds.
"""
import joblib
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.full_season_simulator import get_future_rounds, simulate_full_season_mc
from src.model import get_feature_columns, prepare_model_frame
from src.rolling_state import seed_constructor_states, seed_driver_states
from src.season_simulator import compute_championship_prob, precompute_race_strengths

TARGET = "target_top10"
MODEL_NAME = "random_forest"


def main() -> None:
    classifier = joblib.load(MODELS_DIR / f"{MODEL_NAME}_{TARGET}.joblib")
    qualifying = joblib.load(MODELS_DIR / "qualifying_position_model.joblib")

    full_history = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    prepared_all = prepare_model_frame(full_history)
    season_2026 = prepared_all[prepared_all["season"] == 2026]

    completed_rounds = set(season_2026["round"].unique())
    future_rounds = get_future_rounds(2026, completed_rounds)
    completed_schedule = sorted(completed_rounds)

    race_features_by_id = {
        r: season_2026[season_2026["round"] == r].reset_index(drop=True)
        for r in completed_schedule
    }
    feature_cols = get_feature_columns(season_2026)
    completed_race_strengths = precompute_race_strengths(
        completed_schedule, race_features_by_id, classifier, feature_cols
    )

    last_round = max(completed_rounds)
    latest_snapshot = season_2026[season_2026["round"] == last_round]
    drivers = sorted(latest_snapshot["driver_code"].unique())
    driver_constructor_map = dict(
        zip(latest_snapshot["driver_code"], latest_snapshot["constructor"])
    )

    driver_states = seed_driver_states(prepared_all)
    constructor_states = seed_constructor_states(prepared_all)
    season_race_counts = {
        driver: int(
            season_2026[season_2026["driver_code"] == driver]["round"].nunique()
        )
        for driver in drivers
    }

    print(f"Completed rounds: {completed_schedule}")
    print(f"Future rounds to simulate recursively: {future_rounds}")

    points_df = simulate_full_season_mc(
        completed_schedule, completed_race_strengths,
        future_rounds, drivers, driver_constructor_map,
        driver_states, constructor_states, season_race_counts,
        classifier, qualifying["model"], qualifying["residual_std"],
        n_simulations=500, seed=42,
    )
    results = compute_championship_prob(points_df)

    output_path = PROCESSED_DATA_DIR / "2026_championship_predictions_full.csv"
    results.to_csv(output_path)
    print(results)


if __name__ == "__main__":
    main()