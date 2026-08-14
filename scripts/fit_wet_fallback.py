"""Grid-wide wet/intermediate degradation fallback.

Supplements the dry-compound hierarchical model (scripts/
fit_hierarchical_tyre_model.py): wet/intermediate laps are too sparse
per driver to support hierarchical per-driver estimates (see
src/tyre_model.py ALL_COMPOUNDS docstring), so this fits one shared
rate across the full grid instead.
"""
import joblib
import pandas as pd

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.tyre_model import fit_pooled_wet_model, prepare_pooled_wet_laps

GRID_2026 = [
    "ALB", "ALO", "ANT", "BEA", "BOR", "BOT", "COL", "GAS", "HAD", "HAM",
    "HUL", "LAW", "LEC", "LIN", "NOR", "OCO", "PER", "PIA", "RUS", "SAI",
    "STR", "VER",
]


def main() -> None:
    laps = pd.read_csv(RAW_DATA_DIR / "lap_data_all.csv")
    pooled_wet = prepare_pooled_wet_laps(laps, GRID_2026)

    print(f"{len(pooled_wet)} wet/intermediate laps, {pooled_wet['driver_code'].nunique()} drivers represented")
    print(pooled_wet["Compound"].value_counts())

    fitted = fit_pooled_wet_model(pooled_wet)
    for name, coef in zip(fitted["feature_names"], fitted["model"].coef_):
        print(f"{name}: {coef:.3f}")

    joblib.dump(fitted, MODELS_DIR / "wet_degradation_fallback.joblib")
    print(f"\nSaved to {MODELS_DIR / 'wet_degradation_fallback.joblib'}")


if __name__ == "__main__":
    main()