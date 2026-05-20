"""
src/app/setup.py
-----------------
Auto-trains the model if best_model.pkl is missing.
Called at Streamlit app startup so the cloud deployment
works without pre-committed model artifacts.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def ensure_model_exists():
    model_path = Path("models/best_model.pkl")
    if model_path.exists():
        return

    logger.info("Model not found — training now (first boot)...")
    import streamlit as st

    with st.spinner("First boot: training model... (~30 seconds)"):
        from src.models.train import run_training

        run_training()
    st.success("Model ready!")
