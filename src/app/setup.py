from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

# Force both root and src onto path
for p in [str(ROOT), str(SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_model_exists() -> None:
    model_path = ROOT / "models" / "best_model.pkl"
    if model_path.exists():
        return

    import streamlit as st

    with st.spinner("First boot: training model (~30 seconds)..."):
        # Import using the resolved sys.path — no 'src.' prefix needed
        from data.loader import load_dataset
        from data.splitter import split_data
        from features.preprocessor import build_preprocessor
        from models.train import run_training

        (ROOT / "models").mkdir(exist_ok=True)
        run_training()

    st.success("✅ Model ready!")
    st.rerun()
