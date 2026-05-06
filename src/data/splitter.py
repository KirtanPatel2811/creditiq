"""
src/data/splitter.py
---------------------
Splits data into train / validation / test sets.

Design decision: we use a stratified split to preserve the class imbalance
ratio in every split. With ~30% default rate in German Credit (much higher
than the 8% in Home Credit), this matters less here but is critical later.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Stratified train / val / test split.

    Parameters
    ----------
    X, y         : full feature matrix and target.
    val_size     : fraction of data for validation (default 15%).
    test_size    : fraction of data for test (default 15%).
    random_state : reproducibility seed.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First cut off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Then split remaining into train / val
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=adjusted_val,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d | default rate: train=%.1f%% val=%.1f%% test=%.1f%%",
        len(X_train),
        len(X_val),
        len(X_test),
        y_train.mean() * 100,
        y_val.mean() * 100,
        y_test.mean() * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
