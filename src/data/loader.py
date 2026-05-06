"""
src/data/loader.py
------------------
Loads and prepares datasets for CreditIQ.

Design decision: the public `load_dataset` function is the only entry point
other modules should use. It returns a clean (X, y) pair regardless of which
dataset is underneath. This means models/train.py never needs to know which
dataset it's working with.

Currently supported:
  - german_credit  (1 000 rows, UCI — good for rapid prototyping)
  - home_credit    (300 K rows, Kaggle — production target)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
GERMAN_CREDIT_FILE = RAW_DIR / "german_credit.csv"


# ── public API ────────────────────────────────────────────────────────────────
def load_dataset(name: str = "german_credit") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a credit risk dataset and return (features, target).

    Parameters
    ----------
    name : str
        One of {"german_credit", "home_credit"}.

    Returns
    -------
    X : pd.DataFrame
        Raw feature matrix (no target column).
    y : pd.Series
        Binary target: 1 = default, 0 = no default.
    """
    loaders = {
        "german_credit": _load_german_credit,
        "home_credit": _load_home_credit,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders)}")

    logger.info("Loading dataset: %s", name)
    X, y = loaders[name]()
    logger.info(
        "Loaded %d rows, %d features. Default rate: %.2f%%",
        len(X),
        X.shape[1],
        y.mean() * 100,
    )
    return X, y


# ── private loaders ───────────────────────────────────────────────────────────
def _load_german_credit() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load German Credit dataset.

    Tries the local CSV first. If the CSV exists but has no target column
    (a common issue with some Kaggle uploads), falls back to fetching from
    OpenML — which always includes the target. Requires internet on first run;
    sklearn caches it locally after that.
    """
    if GERMAN_CREDIT_FILE.exists():
        df = pd.read_csv(GERMAN_CREDIT_FILE)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        if "Risk" in df.columns:
            y = (df["Risk"] == "bad").astype(int).rename("target")
            return df.drop(columns=["Risk"]), y
        elif "class" in df.columns:
            y = (df["class"] == 2).astype(int).rename("target")
            return df.drop(columns=["class"]), y
        else:
            logger.warning(
                "Local CSV has no target column (columns: %s). "
                "Falling back to OpenML fetch.",
                df.columns.tolist(),
            )

    # Fetch from OpenML — sklearn caches this in ~/scikit_learn_data/
    logger.info("Fetching German Credit from OpenML (cached after first run)...")
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")

    X = data.data.copy()
    # Target is 'good'/'bad' strings — recode: bad=1 (default), good=0
    y = (data.target == "bad").astype(int).rename("target")

    logger.info("OpenML fetch complete. Shape: %s", X.shape)
    return X, y


def _load_home_credit() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Home Credit Default Risk dataset (Kaggle).

    This joins the seven related tables into a single flat feature matrix.
    Expects the raw Kaggle CSVs to be in data/raw/.

    Phase 2 feature: will be implemented after German Credit pipeline
    is fully validated end-to-end.
    """
    raise NotImplementedError(
        "Home Credit loader coming in Phase 2. "
        "Start with load_dataset('german_credit') to validate the pipeline."
    )
