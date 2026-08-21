"""Tyre Degradation Explorer page."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loaders import load_driver_degradation_summary, load_hierarchical_tyre_summary
from utils.styling import PLOTLY_TEMPLATE

st.set_page_config(page_title="Tyre Degradation Explorer", page_icon="🛞", layout="wide")
st.title("Tyre Degradation Explorer")

degradation = load_driver_degradation_summary()
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]

st.subheader("Population-level degradation rate by compound")
compound_avg = degradation.groupby("compound")["degradation_rate"].mean().reset_index()
present_order = [c for c in COMPOUND_ORDER if c in compound_avg["compound"].values]
compound_avg["compound"] = pd.Categorical(compound_avg["compound"], categories=present_order, ordered=True)
compound_avg = compound_avg.sort_values("compound")

fig_bar = px.bar(compound_avg, x="compound", y="degradation_rate", color="compound", text_auto=".3f", template=PLOTLY_TEMPLATE)
fig_bar.update_layout(showlegend=False, yaxis_title="Mean degradation rate (s/lap, averaged across drivers)")
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()
st.subheader("Relative time loss over a stint")
st.caption(
    "Cumulative time loss from tyre wear only (degradation_rate x laps since pit). "
    "Baseline pace and fuel-burn effects aren't in this summary file, so this shows "
    "relative wear only, not absolute lap time."
)

max_laps = st.slider("Stint length (laps)", 5, 40, 25)
compounds_selected = st.multiselect("Compounds", present_order, default=present_order)

laps = np.arange(0, max_laps + 1)
fig_curve = go.Figure()
for compound in compounds_selected:
    rate = compound_avg.loc[compound_avg["compound"] == compound, "degradation_rate"]
    if rate.empty:
        continue
    fig_curve.add_trace(go.Scatter(x=laps, y=rate.iloc[0] * laps, mode="lines", name=compound))
fig_curve.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Laps since pit stop", yaxis_title="Cumulative time loss (s, relative)")
st.plotly_chart(fig_curve, use_container_width=True)

st.divider()
st.subheader("Per-driver degradation rates")
driver_list = sorted(degradation["driver_code"].unique())
selected_drivers = st.multiselect("Filter drivers", driver_list, default=driver_list[:5])
filtered = degradation[degradation["driver_code"].isin(selected_drivers)] if selected_drivers else degradation
st.dataframe(filtered.style.format({"degradation_rate": "{:.3f}"}), use_container_width=True)

st.divider()
with st.expander("Model convergence diagnostics"):
    summary = load_hierarchical_tyre_summary()
    deg_rows = summary[summary["parameter"].str.startswith("mu_degradation")]
    st.write("Population-level degradation parameters (index order as fit; see per-driver table above for compound labels):")
    st.dataframe(
        deg_rows[["parameter", "mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].style.format(
            {"mean": "{:.4f}", "sd": "{:.4f}", "r_hat": "{:.4f}", "ess_bulk": "{:.0f}", "ess_tail": "{:.0f}"}
        ),
        use_container_width=True,
    )
    st.metric("Max r_hat across all parameters", f"{summary['r_hat'].max():.4f}", help="Values at or below 1.01 indicate good convergence")