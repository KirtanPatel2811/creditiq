"""Unit tests for model training and evaluation."""

import numpy as np
import pytest
import joblib
from pathlib import Path

from src.data.loader import load_dataset
from src.data.splitter import split_data
from src.models.evaluate import evaluate_model, _ks_statistic


@pytest.fixture(scope="module")
def pipeline():
    path = Path("models/best_model.pkl")
    if not path.exists():
        pytest.skip("best_model.pkl not found — run train.py first")
    return joblib.load(path)


@pytest.fixture(scope="module")
def val_data():
    X, y = load_dataset("german_credit")
    _, X_val, _, _, y_val, _ = split_data(X, y)
    return X_val, y_val


def test_model_loads(pipeline):
    assert pipeline is not None
    assert hasattr(pipeline, "predict_proba")


def test_predict_proba_range(pipeline, val_data):
    X_val, _ = val_data
    probs = pipeline.predict_proba(X_val)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_roc_auc_above_baseline(pipeline, val_data):
    """Model must beat random (0.5) by a healthy margin."""
    X_val, y_val = val_data
    metrics = evaluate_model(pipeline, X_val, y_val)
    assert metrics["roc_auc"] > 0.65, f"ROC-AUC too low: {metrics['roc_auc']}"


def test_ks_statistic_positive(pipeline, val_data):
    X_val, y_val = val_data
    metrics = evaluate_model(pipeline, X_val, y_val)
    assert metrics["ks_statistic"] > 0.0


def test_all_metrics_present(pipeline, val_data):
    X_val, y_val = val_data
    metrics = evaluate_model(pipeline, X_val, y_val)
    required = {"roc_auc", "ks_statistic", "gini", "pr_auc", "brier_score"}
    assert required.issubset(metrics.keys())
