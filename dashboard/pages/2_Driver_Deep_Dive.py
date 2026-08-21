"""Driver Deep-Dive page -- rolling form, grid-vs-finish, constructor lineage."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loaders import load_features
from utils.styling import PLOTLY_TEMPLATE

st.set_page_config(page_title="Driver Deep-Dive", page_icon="🔍", layout="wide")
st.title("Driver Deep-Dive")

features = load_features()

drivers = sorted(features["driver_code"].unique())
selected_driver = st.selectbox("Select a driver", drivers, index=drivers.index("HAM") if "HAM" in drivers else 0)

driver_df = features[features["driver_code"] == selected_driver].copy()
driver_name = driver_df["driver_name"].iloc[-1]

st.subheader(f"{driver_name} ({selected_driver})")

constructors_seen = driver_df["constructor"].unique().tolist()
if len(constructors_seen) > 1:
    lineage = " → ".join(
        f"{c} ({driver_df[driver_df['constructor'] == c]['season'].min()}–{driver_df[driver_df['constructor'] == c]['season'].max()})"
        for c in driver_df.drop_duplicates("constructor", keep="first")["constructor"]
    )
    st.info(f"Team history: {lineage}. Rolling form below carries across this change by design (tracked by driver identity, not team).")

st.divider()
st.subheader("Rolling form over time")

form_metric = st.radio(
    "Metric", ["driver_form_ewm", "driver_avg_finish_last5", "driver_points_last5"],
    horizontal=True, format_func=lambda x: {
        "driver_form_ewm": "Exponentially-weighted form",
        "driver_avg_finish_last5": "Avg finish (last 5)",
        "driver_points_last5": "Points (last 5)",
    }[x],
)

fig = px.line(driver_df, x="race_label", y=form_metric, template=PLOTLY_TEMPLATE, markers=True)
fig.update_layout(xaxis_title="Race", yaxis_title=form_metric)
fig.update_xaxes(tickangle=45, nticks=20)
st.plotly_chart(fig, use_container_width=True)
st.caption("Computed via shift(1) then rolling/EWM -- a race's own result never leaks into its own row's rolling average.")

st.divider()
st.subheader("Grid position vs. finish position")

col1, col2 = st.columns([1, 3])
with col1:
    scope = st.radio("Scope", ["This driver only", "All drivers, same era"])
    era = st.selectbox("Regulation era", sorted(features["regulation_era"].unique())) if scope == "All drivers, same era" else None

with col2:
    if scope == "This driver only":
        plot_df = driver_df.dropna(subset=["grid_position", "finish_position"])
        color_col = "season"
    else:
        plot_df = features[features["regulation_era"] == era].dropna(subset=["grid_position", "finish_position"])
        color_col = None

    fig2 = px.scatter(
        plot_df, x="grid_position", y="finish_position", color=color_col,
        template=PLOTLY_TEMPLATE, opacity=0.7,
        hover_data=["driver_name", "event_name", "season"] if scope != "This driver only" else ["event_name", "season"],
    )
    max_pos = max(plot_df["grid_position"].max(), plot_df["finish_position"].max())
    fig2.add_trace(go.Scatter(x=[0, max_pos], y=[0, max_pos], mode="lines", line=dict(dash="dash", color="gray"), name="No change", showlegend=True))
    fig2.update_layout(xaxis_title="Grid position", yaxis_title="Finish position")
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Points below the dashed line indicate positions gained on race day; points above indicate positions lost.")