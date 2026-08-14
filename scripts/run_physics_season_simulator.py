"""Entry point: v3 physics-based full-season 2026 simulation."""
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.full_season_simulator import get_future_rounds
from src.model import prepare_model_frame
from src.physics_season_simulator import simulate_physics_season_mc
from src.race_calendar import build_circuit_lap_counts
from src.rolling_state import seed_constructor_states, seed_driver_states
from src.season_simulator import compute_championship_prob

N_SIMULATIONS = 10_000
SEED = 42


def main() -> None:
    race_sim_params = joblib.load(MODELS_DIR / "race_simulator_params.joblib")
    noise_pool = np.load(MODELS_DIR / "noise_pool.npy")
    qualifying = joblib.load(MODELS_DIR / "qualifying_position_model.joblib")

    full_history = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    prepared_all = prepare_model_frame(full_history)
    season_2026 = prepared_all[prepared_all["season"] == 2026]

    completed_rounds = sorted(season_2026["round"].unique())
    future_rounds = get_future_rounds(2026, set(completed_rounds))

    completed_grid_by_round = {
        r: dict(zip(
            season_2026[season_2026["round"] == r]["driver_code"],
            season_2026[season_2026["round"] == r]["grid_position"].astype(int),
        ))
        for r in completed_rounds
    }
    completed_event_names = {
        r: season_2026[season_2026["round"] == r]["event_name"].iloc[0]
        for r in completed_rounds
    }

    import fastf1
    schedule = fastf1.get_event_schedule(2026, include_testing=False)
    future_event_names = {
        r: schedule[schedule["RoundNumber"] == r]["EventName"].iloc[0]
        for r in future_rounds
    }

    lap_counts, lap_count_fallback = build_circuit_lap_counts()

    last_round = max(completed_rounds)
    latest_snapshot = season_2026[season_2026["round"] == last_round]
    drivers = sorted(latest_snapshot["driver_code"].unique())
    driver_constructor_map = dict(zip(latest_snapshot["driver_code"], latest_snapshot["constructor"]))

    driver_states = seed_driver_states(prepared_all)
    constructor_states = seed_constructor_states(prepared_all)
    season_race_counts = {
        driver: int(season_2026[season_2026["driver_code"] == driver]["round"].nunique())
        for driver in drivers
    }

    print(f"Completed rounds: {completed_rounds}")
    print(f"Future rounds: {future_rounds}")
    print(f"Running {N_SIMULATIONS} simulations...")

    points_df = simulate_physics_season_mc(
        completed_rounds, completed_grid_by_round, completed_event_names,
        future_rounds, future_event_names,
        drivers, driver_constructor_map,
        driver_states, constructor_states, season_race_counts,
        qualifying["model"], qualifying["residual_std"],
        race_sim_params, noise_pool, lap_counts, lap_count_fallback,
        n_simulations=N_SIMULATIONS, seed=SEED,
    )
    points_df.to_csv(PROCESSED_DATA_DIR / "physics_points_distribution.csv", index=False)
    results = compute_championship_prob(points_df)

    output_path = PROCESSED_DATA_DIR / "2026_championship_predictions_physics.csv"
    results.to_csv(output_path)
    print(results)


if __name__ == "__main__":
    main()