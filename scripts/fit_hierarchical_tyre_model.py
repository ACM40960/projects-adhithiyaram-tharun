"""Fit the full-grid (2026 roster) hierarchical dry-compound tyre model."""
import time

import joblib
import arviz as az
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.tyre_model import fit_hierarchical_tier2_model, prepare_all_driver_pooled_laps

GRID_2026 = [
    "ALB", "ALO", "ANT", "BEA", "BOR", "BOT", "COL", "GAS", "HAD", "HAM",
    "HUL", "LAW", "LEC", "LIN", "NOR", "OCO", "PER", "PIA", "RUS", "SAI",
    "STR", "VER",
]


def main() -> None:
    laps = pd.read_csv(RAW_DATA_DIR / "lap_data_all.csv")
    pooled = prepare_all_driver_pooled_laps(laps, GRID_2026)
    print(
        f"{len(pooled)} laps, {pooled['race_id'].nunique()} races, "
        f"max race_id: {pooled['race_id'].max()}"
    )

    start = time.time()
    model, trace = fit_hierarchical_tier2_model(
        pooled, n_drivers=len(GRID_2026), draws=500, tune=500
    )
    print(f"Elapsed: {time.time() - start:.0f}s")

    # Print and save the summary FIRST, before the fragile netcdf write --
    # a save-format failure must never again cost a completed 13+ minute
    # sampling run (see earlier crash: ValueError, missing h5netcdf,
    # trace lost because it crashed before this point was ever reached).
    summary = az.summary(
        trace, var_names=["mu_degradation", "sigma_degradation", "sigma", "skew_a", "skew_b"]
    )
    print(summary)
    summary.to_csv(PROCESSED_DATA_DIR / "hierarchical_tyre_summary.csv")
    print("Summary saved.")

    # Full per-driver degradation summary too, saved before the risky part.
    from src.tyre_model import build_driver_race_index, summarize_driver_degradation
    driver_race_index = build_driver_race_index(laps)
    driver_summary = summarize_driver_degradation(trace, GRID_2026, driver_race_index)
    driver_summary.to_csv(PROCESSED_DATA_DIR / "driver_degradation_summary.csv", index=False)
    print("Per-driver degradation summary saved.")

    joblib.dump({"drivers": GRID_2026}, MODELS_DIR / "hierarchical_tyre_meta.joblib")

    try:
        trace.to_netcdf(MODELS_DIR / "hierarchical_tyre_trace.nc")
        print("Trace saved to netcdf.")
    except Exception as e:
        print(f"netcdf save failed ({e}); falling back to pickle.")
        import pickle
        with open(MODELS_DIR / "hierarchical_tyre_trace.pkl", "wb") as f:
            pickle.dump(trace, f)
        print("Trace saved to pickle instead.")

    print("DONE")


if __name__ == "__main__":
    main()