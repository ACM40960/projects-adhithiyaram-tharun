"""What-If Simulator page -- runs the lap-by-lap Monte Carlo race
simulator live with a user-chosen grid, circuit, and tyre strategy."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# utils imported before src -- see dashboard/utils/__init__.py.
from utils.data_loaders import (
    load_circuit_lap_counts,
    load_driver_lookup,
    load_noise_pool,
    load_race_sim_params,
)
from utils.styling import MODEL_COLOR, PLOTLY_TEMPLATE, SEQUENTIAL_SCALE

from src.race_simulator import STRATEGY_MENU, simulate_driver_race
from src.season_simulator import POINTS_TABLE

st.set_page_config(page_title="What-If Simulator", page_icon="🎛️", layout="wide")
st.title("What-If Simulator")
st.caption(
    "Runs live Monte Carlo race simulations with the fitted lap-by-lap physics model -- "
    "baseline pace, tyre degradation, fuel burn, grid penalty, pit-stop loss. Every other "
    "page on this dashboard is a viewer over a saved result; this one computes a new answer "
    "for whatever grid and strategy you set below."
)

params = load_race_sim_params()
noise_pool = load_noise_pool()
driver_lookup = load_driver_lookup()
lap_counts, fallback_laps = load_circuit_lap_counts()
drivers = params["drivers"]

STRATEGY_LABELS = {
    "Random (menu default)": None,
    "Medium -> Hard (55/45)": STRATEGY_MENU[0],
    "Soft -> Hard (35/65)": STRATEGY_MENU[1],
    "Soft -> Medium -> Soft (30/35/35)": STRATEGY_MENU[2],
}

st.subheader("Race setup")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    circuit_options = sorted(lap_counts.keys())
    circuit = st.selectbox("Circuit", circuit_options)
with c2:
    total_laps = st.number_input(
        "Total laps", min_value=20, max_value=90, value=int(lap_counts.get(circuit, fallback_laps))
    )
with c3:
    n_simulations = st.slider("Simulations", 100, 5000, 1000, step=100)

st.subheader("Grid & strategy")
st.caption(
    "Defaults are each driver's most recent 2026 qualifying position, all on the random "
    "menu strategy (matching how the season simulator itself behaves). Edit grid slots, "
    "drop drivers, or pin a strategy to test a scenario."
)

editor_source = pd.DataFrame([
    {
        "include": True,
        "driver": d,
        "name": driver_lookup.get(d, {}).get("driver_name", d),
        "constructor": driver_lookup.get(d, {}).get("constructor", "-"),
        "grid_position": driver_lookup.get(d, {}).get("grid_position", i + 1),
        "strategy": "Random (menu default)",
    }
    for i, d in enumerate(drivers)
]).sort_values("grid_position").reset_index(drop=True)

edited = st.data_editor(
    editor_source,
    column_config={
        "include": st.column_config.CheckboxColumn("In race"),
        "driver": st.column_config.TextColumn("Code", disabled=True),
        "name": st.column_config.TextColumn("Driver", disabled=True),
        "constructor": st.column_config.TextColumn("Team", disabled=True),
        "grid_position": st.column_config.NumberColumn("Grid", min_value=1, max_value=22, step=1),
        "strategy": st.column_config.SelectboxColumn("Strategy", options=list(STRATEGY_LABELS.keys())),
    },
    hide_index=True,
    use_container_width=True,
    height=350,
)

active = edited[edited["include"]].reset_index(drop=True)
run = st.button("Run simulation", type="primary", disabled=len(active) < 2)
if len(active) < 2:
    st.warning("Include at least 2 drivers to run a race.")

if run:
    if active["grid_position"].duplicated().any():
        st.warning("Two or more drivers share a grid slot -- their grid-penalty term will be identical.")

    driver_codes = active["driver"].tolist()
    grid_map = dict(zip(active["driver"], active["grid_position"]))
    strategy_map = dict(zip(active["driver"], active["strategy"]))
    n_drivers = len(driver_codes)

    rng = np.random.default_rng()
    position_counts = {d: np.zeros(n_drivers, dtype=int) for d in driver_codes}
    points_total = {d: 0.0 for d in driver_codes}

    progress = st.progress(0.0)
    update_every = max(1, n_simulations // 20)
    for sim in range(n_simulations):
        race_times = {}
        for d in driver_codes:
            fixed_strategy = STRATEGY_LABELS[strategy_map[d]]
            strategy = fixed_strategy if fixed_strategy is not None else STRATEGY_MENU[rng.integers(len(STRATEGY_MENU))]
            race_times[d] = simulate_driver_race(
                d, int(grid_map[d]), strategy, int(total_laps), params, noise_pool, rng
            )
        ranked = sorted(race_times, key=race_times.get)
        for pos, d in enumerate(ranked, start=1):
            position_counts[d][pos - 1] += 1
            points_total[d] += POINTS_TABLE.get(pos, 0)
        if sim % update_every == 0:
            progress.progress((sim + 1) / n_simulations)
    progress.progress(1.0)
    progress.empty()

    st.session_state["whatif_results"] = {
        "position_counts": position_counts,
        "points_total": points_total,
        "driver_codes": driver_codes,
        "n_simulations": n_simulations,
        "circuit": circuit,
        "total_laps": int(total_laps),
    }

results = st.session_state.get("whatif_results")
if results:
    driver_codes = results["driver_codes"]
    n_drivers = len(driver_codes)
    n_simulations = results["n_simulations"]
    positions = np.arange(1, n_drivers + 1)

    summary_rows = []
    for d in driver_codes:
        counts = results["position_counts"][d]
        win_p = counts[0] / n_simulations
        podium_p = counts[:3].sum() / n_simulations
        mean_finish = float(np.average(positions, weights=counts))
        mean_points = results["points_total"][d] / n_simulations
        info = driver_lookup.get(d, {})
        summary_rows.append({
            "driver": d,
            "name": info.get("driver_name", d),
            "constructor": info.get("constructor", "-"),
            "win_probability": win_p,
            "podium_probability": podium_p,
            "mean_finish": mean_finish,
            "mean_points": mean_points,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("win_probability", ascending=False).reset_index(drop=True)

    st.divider()
    st.subheader(f"Results -- {results['circuit']} ({results['total_laps']} laps, {n_simulations:,} simulations)")

    st.dataframe(
        summary_df.style.format({
            "win_probability": "{:.1%}",
            "podium_probability": "{:.1%}",
            "mean_finish": "{:.1f}",
            "mean_points": "{:.1f}",
        }),
        use_container_width=True,
        column_config={
            "driver": "Code", "name": "Driver", "constructor": "Team",
            "win_probability": "Win %", "podium_probability": "Podium %",
            "mean_finish": "Mean finish", "mean_points": "Mean points",
        },
    )

    st.subheader("Win probability")
    fig_win = go.Figure(go.Bar(
        x=summary_df["driver"], y=summary_df["win_probability"] * 100,
        marker_color=MODEL_COLOR,
        text=[f"{v:.1%}" for v in summary_df["win_probability"]],
        textposition="outside",
    ))
    fig_win.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="P(win) %", xaxis_title=None)
    st.plotly_chart(fig_win, use_container_width=True)

    st.subheader("Finishing position distribution")
    st.caption("Probability of each finishing position, drivers ordered by win probability (best first).")
    heat_order = summary_df["driver"].tolist()
    heat_matrix = np.array([
        results["position_counts"][d] / n_simulations * 100 for d in heat_order
    ])
    fig_heat = go.Figure(go.Heatmap(
        z=heat_matrix, x=[str(p) for p in positions], y=heat_order,
        colorscale=SEQUENTIAL_SCALE, colorbar=dict(title="%"),
        hovertemplate="Driver: %{y}<br>Finish: P%{x}<br>Probability: %{z:.1f}%<extra></extra>",
    ))
    fig_heat.update_layout(
        template=PLOTLY_TEMPLATE, xaxis_title="Finishing position",
        yaxis=dict(autorange="reversed"), height=max(400, n_drivers * 22),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Driver detail")
    focus = st.selectbox("Driver", heat_order, format_func=lambda d: f"{d} -- {driver_lookup.get(d, {}).get('driver_name', d)}")
    focus_row = summary_df[summary_df["driver"] == focus].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win probability", f"{focus_row['win_probability']:.1%}")
    m2.metric("Podium probability", f"{focus_row['podium_probability']:.1%}")
    m3.metric("Mean finish", f"{focus_row['mean_finish']:.1f}")
    m4.metric("Mean points", f"{focus_row['mean_points']:.1f}")

    fig_focus = go.Figure(go.Bar(
        x=[str(p) for p in positions],
        y=results["position_counts"][focus] / n_simulations * 100,
        marker_color=MODEL_COLOR,
    ))
    fig_focus.update_layout(
        template=PLOTLY_TEMPLATE, xaxis_title="Finishing position", yaxis_title="Probability (%)",
    )
    st.plotly_chart(fig_focus, use_container_width=True)
else:
    st.info("Set up the grid above and click **Run simulation** to see results.")
