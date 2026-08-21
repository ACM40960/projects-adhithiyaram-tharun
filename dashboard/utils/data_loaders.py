"""Cached data-loading utilities for the F1 Predictor dashboard."""
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.race_calendar import build_circuit_lap_counts

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "2026_championship_predictions_physics.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
TYRE_SUMMARY_PATH = PROCESSED_DATA_DIR / "hierarchical_tyre_summary.csv"
DRIVER_DEGRADATION_PATH = PROCESSED_DATA_DIR / "driver_degradation_summary.csv"
CALIBRATION_PATH = PROCESSED_DATA_DIR / "calibration_2025.csv"
WALK_FORWARD_PATH = PROCESSED_DATA_DIR / "walk_forward_2026_validation.csv"
POINTS_DIST_PATH = PROCESSED_DATA_DIR / "physics_points_distribution.csv"
RACE_SIM_PARAMS_PATH = MODELS_DIR / "race_simulator_params.joblib"
NOISE_POOL_PATH = MODELS_DIR / "noise_pool.npy"

# Real-world 2026 title-probability consensus (Polymarket, via CryptoSlate, late July 2026).
MARKET_TITLE_PROBABILITY = {"ANT": 75.2, "LEC": 2.4, "RUS": 5.5, "HAM": 9.7, "NOR": 3.0}


@st.cache_data
def load_championship_predictions() -> pd.DataFrame:
    """Load and tidy the v3 physics-based 2026 championship predictions."""
    df = pd.read_csv(PREDICTIONS_PATH)
    df = df.rename(columns={"Unnamed: 0": "driver"})
    return df.sort_values("p_win_championship", ascending=False).reset_index(drop=True)


def figure_path(filename: str) -> Path:
    """Resolve a filename to its path under the figures directory."""
    return FIGURES_DIR / filename


def predictions_last_updated() -> str:
    """Human-readable last-modified timestamp for the predictions file."""
    if not PREDICTIONS_PATH.exists():
        return "not yet generated"
    return dt.datetime.fromtimestamp(PREDICTIONS_PATH.stat().st_mtime).strftime("%d %b %Y, %H:%M")

@st.cache_data
def load_features() -> pd.DataFrame:
    """Load the leakage-safe feature table, sorted chronologically."""
    df = pd.read_csv(FEATURES_PATH)
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["race_label"] = df["season"].astype(str) + " R" + df["round"].astype(str)
    return df

@st.cache_data
def load_driver_degradation_summary() -> pd.DataFrame:
    """Per-driver, per-compound fitted degradation rates."""
    return pd.read_csv(DRIVER_DEGRADATION_PATH)


@st.cache_data
def load_hierarchical_tyre_summary() -> pd.DataFrame:
    """Population-level PyMC posterior summary (r_hat, ess, etc.)."""
    df = pd.read_csv(TYRE_SUMMARY_PATH)
    return df.rename(columns={"Unnamed: 0": "parameter"})


@st.cache_data
def load_calibration_2025() -> pd.DataFrame:
    return pd.read_csv(CALIBRATION_PATH)


@st.cache_data
def load_walk_forward_validation() -> pd.DataFrame:
    return pd.read_csv(WALK_FORWARD_PATH)


@st.cache_data
def load_points_distribution() -> pd.DataFrame:
    """Full 10,000-season simulated points, one column per driver."""
    return pd.read_csv(POINTS_DIST_PATH)


@st.cache_resource
def load_race_sim_params() -> dict:
    """Consolidated lap-by-lap physics params (baseline pace, degradation rates, etc.)."""
    return joblib.load(RACE_SIM_PARAMS_PATH)


@st.cache_resource
def load_noise_pool() -> np.ndarray:
    """Pre-drawn skewed-t noise pool the race simulator samples from."""
    return np.load(NOISE_POOL_PATH)


@st.cache_data
def load_driver_lookup() -> dict:
    """driver_code -> {driver_name, constructor, grid_position}, from the most recent 2026 round."""
    df = pd.read_csv(FEATURES_PATH)
    df_2026 = df[df["season"] == 2026]
    if df_2026.empty:
        return {}
    last_round = df_2026["round"].max()
    latest = df_2026[df_2026["round"] == last_round]
    return {
        row.driver_code: {
            "driver_name": row.driver_name,
            "constructor": row.constructor,
            "grid_position": int(row.grid_position),
        }
        for row in latest.itertuples()
    }


@st.cache_data
def load_circuit_lap_counts() -> tuple[dict, int]:
    """Per-circuit total lap count, plus the grid-wide average fallback."""
    return build_circuit_lap_counts()