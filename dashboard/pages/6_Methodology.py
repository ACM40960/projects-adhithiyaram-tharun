"""Methodology page -- literature review summary, season split, limitations."""
import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="📚", layout="wide")
st.title("Methodology")

st.subheader("Season split")
st.markdown(
    """
| Role | Seasons |
|---|---|
| Training | 2022, 2023, 2024 |
| Validation | 2025 |
| Prediction target | 2026 |

2022–2025 is a single stable regulatory generation. 2026 is a full
regulatory reset, so training/validation stays within one era and 2026
predictions carry explicitly wider uncertainty.
"""
)

st.subheader("Literature foundations")
papers = [
    ("Cappello & Hoegh (2026)", "Bayesian state-space tyre degradation model — basis for the project's hierarchical tyre model, including its skewed-t noise extension."),
    ("Alahmadi et al. (2026)", "Leakage-safe race-winner classification — basis for the project's shift-then-rolling feature construction and temporal train/validation split."),
    ("Bansal et al. (2024)", "Large-scale historical ML benchmark — used cautiously; its near-perfect R² from including final race position is treated as a cautionary example, not a target."),
    ("Demsyn-Jones (2019)", "Monte Carlo uncertainty propagation — basis for two-level variance propagation, preventing overconfident 0%/100% championship probabilities."),
]
for name, desc in papers:
    with st.expander(name):
        st.write(desc)

st.subheader("Limitations")
st.markdown(
    """
- No DNF or reliability modelling.
- Wet/intermediate tyre degradation uses a single grid-wide rate, not per-driver.
- No overtaking dynamics — each driver's race time is computed independently.
- Cadillac cold start — no 2022–2025 history; relies on `is_new_team` and season-average imputation.
- Fixed pit-strategy menu, not optimised per race or driver.
"""
)
