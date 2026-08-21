"""Interactive Plotly version of the pipeline flowchart. Background is
fully transparent so it sits on whatever the Streamlit theme is,
light or dark, without needing to detect it."""
import plotly.graph_objects as go

CLASSIFIER_COLOR = "#2a78d6"
PHYSICS_COLOR = "#1baf7a"
SHARED_COLOR = "#52514e"
FINAL_COLOR = "#2E2E30"

# (id, label, hover text, x, y, w, h, color)
BOXES = [
    (
        "stage1", "Stage 1\nData Pipeline & EDA",
        "Stage 1 -- Data Pipeline & EDA<br>fastf1 fetch, rate-limited, two-layer caching,<br>"
        "regulation-era tagging across 2022-2026.",
        1.0, 4.5, 2.3, 1.1, SHARED_COLOR,
    ),
    (
        "stage2", "Stage 2\nFeature Engineering\n(leakage-safe)",
        "Stage 2 -- Feature Engineering<br>Leakage-safe rolling/EWM form via shift(1) then<br>"
        "rolling(): a race's own result never leaks into<br>its own row's features.",
        4.1, 6.7, 2.3, 1.1, CLASSIFIER_COLOR,
    ),
    (
        "stage3", "Stage 3\nML Classifier\n(LogReg/RF/ET/XGB)",
        "Stage 3 -- ML Classifier<br>Logistic Regression, Random Forest, Extra Trees,<br>"
        "XGBoost compared on top-10/podium targets.<br>Random Forest selected: ROC-AUC 0.836 (2025 holdout).",
        7.5, 6.7, 2.3, 1.1, CLASSIFIER_COLOR,
    ),
    (
        "stage5", "Stage 5\nSeason Monte Carlo\n(v1/v2, Gumbel-max)",
        "Stage 5 -- Season Monte Carlo (v1/v2)<br>Classifier-score ranking, Gumbel-max trick,<br>"
        "two-level driver/constructor variance propagation<br>(Demsyn-Jones, 2019).",
        10.9, 6.7, 2.3, 1.1, CLASSIFIER_COLOR,
    ),
    (
        "stage4", "Stage 4 / Phase 1\nTyre Degradation\n(OLS -> Bayesian\nhierarchical)",
        "Stage 4 / Phase 1 -- Tyre Degradation<br>OLS baseline, then Bayesian hierarchical fit<br>"
        "(PyMC, skewed-t noise), full 22-driver 2026 grid.<br>"
        "Degradation: Soft 0.598, Medium 0.463, Hard 0.364 s/lap.",
        4.1, 2.3, 2.8, 1.6, PHYSICS_COLOR,
    ),
    (
        "phase2", "Phase 2\nLap-by-Lap\nRace Simulator",
        "Phase 2 -- Lap-by-Lap Race Simulator<br>Single-race Monte Carlo: fitted grid penalty,<br>"
        "pit-stop loss, multi-strategy pit menu.",
        7.5, 2.3, 2.3, 1.1, PHYSICS_COLOR,
    ),
    (
        "phase3", "Phase 3\nPhysics Season\nMonte Carlo (v3)",
        "Phase 3 -- Physics Season Monte Carlo (v3)<br>Replaces classifier-score ranking with simulated<br>"
        "race times. This is the production model.",
        10.9, 2.3, 2.3, 1.1, PHYSICS_COLOR,
    ),
    (
        "stage6", "Stage 6\nValidation\n(walk-forward +\ncalibration)",
        "Stage 6 -- Validation<br>Walk-forward against real, unseen 2026 rounds:<br>"
        "ROC-AUC 0.794. Calibration: 9/10 bins within 0.09<br>of the diagonal (2025 holdout).",
        14.3, 4.5, 2.6, 1.6, SHARED_COLOR,
    ),
    (
        "final", "2026 Championship\nPredictions",
        "2026 Championship Predictions<br>v3 physics-based simulator, 10,000 seasons:<br>"
        "Antonelli 69.0% title probability -- matches the<br>real-world market favourite, found independently.",
        17.6, 4.5, 2.3, 1.1, FINAL_COLOR,
    ),
]

ARROWS = [
    (1.0, 5.0, 4.1, 6.7),    # stage1 (corner) -> stage2
    (1.0, 4.0, 4.1, 2.3),    # stage1 (corner) -> stage4
    (6.4, 6.7, 7.5, 6.7),    # stage2 right edge -> stage3
    (9.8, 6.7, 10.9, 6.7),   # stage3 right edge -> stage5
    (6.9, 2.3, 7.5, 2.3),    # stage4 right edge -> phase2
    (9.8, 2.3, 10.9, 2.3),   # phase2 right edge -> phase3
    (13.2, 6.7, 14.3, 4.9),  # stage5 right edge -> stage6
    (13.2, 2.3, 14.3, 4.1),  # phase3 right edge -> stage6
    (16.9, 4.5, 17.6, 4.5),  # stage6 right edge -> final
]


def build_methodology_figure() -> go.Figure:
    """Build the interactive pipeline flowchart. Hover a stage for details."""
    fig = go.Figure()

    shapes = []
    for _id, label, _hover, x, y, w, h, color in BOXES:
        shapes.append(dict(
            type="rect", x0=x, y0=y - h / 2, x1=x + w, y1=y + h / 2,
            line=dict(width=0), fillcolor=color, layer="below",
        ))
        bold_label = "<b>" + label.replace("\n", "</b><br><b>") + "</b>"
        fig.add_annotation(
            x=x + w / 2, y=y, text=bold_label,
            showarrow=False, font=dict(color="white", size=11),
            align="center",
        )

    for x0, y0, x1, y1 in ARROWS:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=1.6,
            arrowcolor="#8A8A8E", standoff=4,
        )

    # Invisible large markers at each box center: shapes/annotations don't
    # support hover, so this is what actually makes the boxes interactive.
    fig.add_trace(go.Scatter(
        x=[x + w / 2 for _id, _l, _h, x, y, w, h, c in BOXES],
        y=[y for _id, _l, _h, x, y, w, h, c in BOXES],
        mode="markers",
        marker=dict(size=[max(w, h) * 34 for _id, _l, _h, x, y, w, h, c in BOXES], opacity=0),
        hovertemplate="%{text}<extra></extra>",
        text=[hover for _id, _l, hover, x, y, w, h, c in BOXES],
        showlegend=False,
    ))

    fig.add_annotation(
        x=0.2, y=7.8, text="<b>Classifier-ranked track</b>", showarrow=False,
        font=dict(color=CLASSIFIER_COLOR, size=13), xanchor="left",
    )
    fig.add_annotation(
        x=0.2, y=0.9, text="<b>Physics-based track (v3, production)</b>", showarrow=False,
        font=dict(color=PHYSICS_COLOR, size=13), xanchor="left",
    )

    fig.update_layout(
        shapes=shapes,
        xaxis=dict(visible=False, range=[0, 20.2], fixedrange=True),
        yaxis=dict(visible=False, range=[0, 8.5], fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        hoverlabel=dict(bgcolor="#2E2E30", font_color="white", font_size=12),
    )
    return fig
