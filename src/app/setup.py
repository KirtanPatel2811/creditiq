"""
src/app/setup.py
-----------------
Auto-trains the model if best_model.pkl is missing.
Works both locally and on Streamlit Cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path regardless of where this file is called from
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_model_exists() -> None:
    model_path = ROOT / "models" / "best_model.pkl"
    if model_path.exists():
        return

    import streamlit as st

    with st.spinner("First boot: training model (~30 seconds)..."):
        # imports happen here, after path is fixed
        from src.data.loader import load_dataset
        from src.data.splitter import split_data
        from src.features.preprocessor import build_preprocessor
        from src.models.train import run_training

        (ROOT / "models").mkdir(exist_ok=True)
        run_training()

    st.success("✅ Model ready!")
    st.rerun()
