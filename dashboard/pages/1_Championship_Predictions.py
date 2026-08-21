"""Championship Predictions page -- full 2026 grid, model vs. market, uncertainty."""
import plotly.graph_objects as go
import streamlit as st

from utils.data_loaders import MARKET_TITLE_PROBABILITY, load_championship_predictions
from utils.styling import MODEL_COLOR, MARKET_COLOR, PLOTLY_TEMPLATE

st.set_page_config(page_title="Championship Predictions", page_icon="🏆", layout="wide")
st.title("2026 Championship Predictions")

predictions = load_championship_predictions()

st.subheader("Full grid, v3 physics-based simulator (10,000 seasons)")
st.dataframe(
    predictions.style.format({
        "p_win_championship": "{:.1%}",
        "p_top3_final": "{:.1%}",
        "mean_points": "{:.0f}",
        "std_points": "{:.1f}",
        "p5_points": "{:.0f}",
        "p95_points": "{:.0f}",
    }),
    use_container_width=True,
)

st.divider()
st.subheader("Model vs. real-world market consensus (top 5)")

top5 = predictions.head(5).copy()
top5["market_prob"] = top5["driver"].map(MARKET_TITLE_PROBABILITY)

fig = go.Figure()
fig.add_bar(name="Our Model (v3)", x=top5["driver"], y=top5["p_win_championship"] * 100, marker_color=MODEL_COLOR)
fig.add_bar(name="Market Consensus", x=top5["driver"], y=top5["market_prob"], marker_color=MARKET_COLOR)
fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE, yaxis_title="Title probability (%)")
st.plotly_chart(fig, use_container_width=True)
st.caption("Market consensus: Polymarket, via CryptoSlate, synced late July 2026.")

st.divider()
st.subheader("Uncertainty range explorer")

selected = st.selectbox("Select a driver", predictions["driver"].tolist())
row = predictions[predictions["driver"] == selected].iloc[0]

m1, m2, m3 = st.columns(3)
m1.metric("Title probability", f"{row['p_win_championship']*100:.1f}%")
m2.metric("Top-3 finish probability", f"{row['p_top3_final']*100:.1f}%")
m3.metric("Mean simulated points", f"{row['mean_points']:.0f}")
st.write(f"5th-95th percentile range: **{row['p5_points']:.0f} – {row['p95_points']:.0f} points**")