"""F1 Race Predictor -- dashboard entry point (Overview page)."""
import streamlit as st

from utils.data_loaders import (
    MARKET_TITLE_PROBABILITY,
    load_championship_predictions,
    load_points_distribution,
    predictions_last_updated,
)
from utils.methodology_diagram import build_methodology_figure
from utils.styling import MARKET_COLOR, MODEL_COLOR, NAV_CARD_COLORS

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

# Scoped to this page only.
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    .kpi-accent {
        height: 4px;
        border-radius: 2px;
        margin: -0.25rem 0 0.75rem 0;
    }
    .card-header {
        padding: 0.55rem 0.9rem;
        border-radius: 8px;
        margin: -1rem -1rem 0.85rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .card-header-title {
        color: white;
        font-weight: 700;
        font-size: 1.02rem;
    }
    .card-badge {
        background: rgba(255, 255, 255, 0.28);
        color: white;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 0.12rem 0.45rem;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("F1 Race Predictor")
st.caption("ACM40960 -- Project in Maths Modelling, University College Dublin")

predictions = load_championship_predictions()
leader = predictions.iloc[0]
n_simulations = len(load_points_distribution())

st.subheader("At a glance")
k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, MODEL_COLOR, "🏆 Model title favourite", leader["driver"], f"{leader['p_win_championship']*100:.1f}% win probability"),
    (k2, MARKET_COLOR, "📈 Real-world market consensus", "ANT", f"{MARKET_TITLE_PROBABILITY['ANT']:.1f}% (Polymarket)"),
    (k3, "#1baf7a", "✅ Walk-forward ROC-AUC (2026)", "0.794", "-0.042 vs. 2025 fixed-split"),
    (k4, "#4a3aa7", "🎲 Simulated seasons", f"{n_simulations:,}", "22-driver 2026 grid, physics-based (v3)"),
]
for col, color, label, value, sub in kpis:
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-accent" style="background:{color};"></div>', unsafe_allow_html=True)
            st.caption(label)
            st.markdown(f"<div style='font-size:1.7rem; font-weight:700; line-height:1.2;'>{value}</div>", unsafe_allow_html=True)
            st.caption(sub)

st.divider()
st.subheader("Pipeline overview")
st.caption("Hover a stage for details. Two independent tracks -- classifier-ranked and physics-based -- share a data stage and converge at validation.")
st.plotly_chart(build_methodology_figure(), use_container_width=True, config={"displayModeBar": False})

st.divider()
st.subheader("Explore")

NAV_ITEMS = [
    ("pages/1_Championship_Predictions.py", "Championship Predictions", "🏆", "Full 2026 grid, model vs. market, uncertainty ranges", None),
    ("pages/2_Driver_Deep_Dive.py", "Driver Deep-Dive", "🔍", "Rolling form, grid-vs-finish, per-driver history", None),
    ("pages/3_Tyre_Degradation_Explorer.py", "Tyre Degradation", "🛞", "Compound wear curves and model diagnostics", None),
    ("pages/4_Model_Validation.py", "Model Validation", "📊", "Confusion matrices, ROC curves, calibration", None),
    ("pages/5_What_If_Simulator.py", "What-If Simulator", "🎛️", "Simulate a race under a chosen strategy", "LIVE"),
    ("pages/6_Methodology.py", "Methodology", "📚", "Literature review, season split, limitations", None),
    ("pages/7_Data_Reproducibility.py", "Data & Reproducibility", "🗂️", "Dataset sizes, timestamps, setup", None),
]

row1 = st.columns(4)
row2 = st.columns(4)
nav_slots = row1 + row2

for (path, title, icon, description, badge), color, slot in zip(NAV_ITEMS, NAV_CARD_COLORS, nav_slots):
    with slot:
        with st.container(border=True):
            badge_html = f'<span class="card-badge">{badge}</span>' if badge else ""
            st.markdown(
                f"""
                <div class="card-header" style="background:{color};">
                    <span class="card-header-title">{icon} {title}</span>
                    {badge_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(description)
            st.page_link(path, label="Open page", icon="➡️")

st.divider()
st.caption(f"Championship predictions last regenerated: {predictions_last_updated()}")
