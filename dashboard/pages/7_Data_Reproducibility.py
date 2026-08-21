"""Data & Reproducibility page -- dataset status and setup instructions."""
import streamlit as st

from utils.data_loaders import load_championship_predictions, predictions_last_updated

st.set_page_config(page_title="Data & Reproducibility", page_icon="🗂️", layout="wide")
st.title("Data & Reproducibility")

st.subheader("Dataset status")
predictions = load_championship_predictions()
st.metric("Championship predictions rows", len(predictions))
st.caption(f"Last regenerated: {predictions_last_updated()}")

st.subheader("Setup")
st.code(
    """python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_fetch
python -m scripts.build_features
python -m scripts.train_classifier
python -m scripts.fit_hierarchical_tyre_model
python -m scripts.run_physics_season_simulator
streamlit run dashboard/Home.py""",
    language="bash",
)