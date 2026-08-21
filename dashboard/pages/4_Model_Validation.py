"""Model Validation page."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

from utils.data_loaders import load_calibration_2025, load_walk_forward_validation, figure_path
from utils.styling import PLOTLY_TEMPLATE

st.set_page_config(page_title="Model Validation", page_icon="📊", layout="wide")
st.title("Model Validation")

# Metrics from the validated Stage 3 model comparison (see README.md Results).
TOP10_METRICS = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": 0.752, "Precision": 0.798, "Recall": 0.675, "F1": 0.731, "ROC-AUC": 0.840, "Precision@10": 0.754},
    {"Model": "Random Forest", "Accuracy": 0.770, "Precision": 0.780, "Recall": 0.754, "F1": 0.767, "ROC-AUC": 0.836, "Precision@10": 0.758},
    {"Model": "Extra Trees", "Accuracy": 0.764, "Precision": 0.782, "Recall": 0.733, "F1": 0.757, "ROC-AUC": 0.831, "Precision@10": 0.750},
    {"Model": "XGBoost", "Accuracy": 0.758, "Precision": 0.801, "Recall": 0.688, "F1": 0.740, "ROC-AUC": 0.831, "Precision@10": 0.767},
])
PODIUM_METRICS = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": 0.868, "Precision": 0.538, "Recall": 0.889, "F1": 0.670, "ROC-AUC": 0.941, "Precision@3": 0.708},
    {"Model": "Random Forest", "Accuracy": 0.919, "Precision": 0.732, "Recall": 0.722, "F1": 0.727, "ROC-AUC": 0.938, "Precision@3": 0.708},
    {"Model": "Extra Trees", "Accuracy": 0.896, "Precision": 0.844, "Recall": 0.375, "F1": 0.519, "ROC-AUC": 0.945, "Precision@3": 0.722},
    {"Model": "XGBoost", "Accuracy": 0.887, "Precision": 0.645, "Recall": 0.556, "F1": 0.597, "ROC-AUC": 0.912, "Precision@3": 0.611},
])

st.subheader("Classifier comparison (2025 held-out validation)")
target = st.radio("Target", ["Top-10 finish", "Podium finish"], horizontal=True)
metrics_df = TOP10_METRICS if target == "Top-10 finish" else PODIUM_METRICS
st.dataframe(metrics_df.style.format(precision=3), use_container_width=True)
st.caption("Random Forest is the production model -- most balanced across both targets rather than top scorer on any single metric.")

st.divider()
st.subheader("Confusion matrices & ROC curves (production model)")
c1, c2 = st.columns(2)
with c1:
    p = figure_path("confusion_matrices.png")
    if p.exists():
        st.image(str(p), use_container_width=True)
with c2:
    p = figure_path("roc_curves.png")
    if p.exists():
        st.image(str(p), use_container_width=True)

st.divider()
st.subheader("Calibration curve (2025 holdout)")
calibration = load_calibration_2025()
fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(
    x=calibration["predicted_probability"], y=calibration["actual_top10_rate"],
    mode="markers+lines", name="Observed",
    marker=dict(size=calibration["n_observations"] / calibration["n_observations"].max() * 30 + 6),
    text=[f"n={n}" for n in calibration["n_observations"]],
    hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}<extra></extra>",
))
fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect calibration"))
fig_cal.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Predicted P(top-10)", yaxis_title="Actual top-10 rate")
st.plotly_chart(fig_cal, use_container_width=True)
st.caption("Marker size reflects the number of observations in that bin.")

st.divider()
st.subheader("Walk-forward validation against real 2026 rounds")
walk_forward = load_walk_forward_validation()

round_auc = (
    walk_forward.groupby("round")
    .apply(lambda g: roc_auc_score(g["actual_top10"], g["predicted_prob"]) if g["actual_top10"].nunique() > 1 else None)
    .reset_index(name="roc_auc")
    .dropna()
)
overall_auc = roc_auc_score(walk_forward["actual_top10"], walk_forward["predicted_prob"])

st.metric("Overall walk-forward ROC-AUC (all rounds combined)", f"{overall_auc:.3f}")

fig_wf = px.line(round_auc, x="round", y="roc_auc", markers=True, template=PLOTLY_TEMPLATE)
fig_wf.add_hline(y=overall_auc, line_dash="dash", line_color="gray", annotation_text="Overall")
fig_wf.update_layout(xaxis_title="2026 round", yaxis_title="ROC-AUC")
st.plotly_chart(fig_wf, use_container_width=True)
st.caption("Per-round AUC is a new view not in the current README -- shows whether accuracy is stable or trending across the season, not just the final aggregate.")