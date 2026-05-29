"""
src/monitoring/drift_detector.py
---------------------------------
Detects data drift and model performance degradation using Evidently AI.

Why this matters for interviews:
  In production, the world changes — customer behaviour shifts, economic
  conditions change, new loan products appear. A model trained in 2023 on
  pre-recession data may silently degrade in 2024. Evidently catches this
  before it becomes a business problem.

Two reports generated:
  1. Data drift report   — are feature distributions changing?
  2. Model quality report — is performance degrading on new data?
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.report import Report

from src.data.loader import load_dataset
from src.data.splitter import split_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models")
REF_DIR = Path("data/reference")
REF_DIR.mkdir(exist_ok=True)


def _transform_to_df(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessor and return a named DataFrame."""
    preprocessor = pipeline.named_steps["preprocessor"]
    X_np = preprocessor.transform(X)
    cols = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(X_np, columns=cols, index=X.index)


def simulate_drift(X: pd.DataFrame, drift_strength: float = 0.3) -> pd.DataFrame:
    """
    Simulate production data drift for demonstration purposes.

    In a real deployment, this function is replaced by live data from
    your scoring API. Here we artificially shift numeric columns so
    Evidently has something meaningful to detect.

    drift_strength : fraction of rows where we inject noise
    """
    X_drift = X.copy()
    rng = np.random.default_rng(seed=99)

    numeric_cols = X_drift.select_dtypes(include=["number"]).columns.tolist()
    n_drift = int(len(X_drift) * drift_strength)
    drift_idx = rng.choice(len(X_drift), size=n_drift, replace=False)

    for col in numeric_cols:
        std = X_drift[col].std()
        noise = rng.normal(loc=std * 1.5, scale=std * 0.5, size=n_drift)
        X_drift[col] = X_drift[col].astype(float)   # ← cast first
        X_drift.iloc[drift_idx, X_drift.columns.get_loc(col)] += noise


    logger.info(
        "Simulated drift on %d/%d rows across %d numeric features",
        n_drift,
        len(X_drift),
        len(numeric_cols),
    )
    return X_drift


def run_data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path = REPORTS_DIR / "data_drift_report.html",
) -> None:
    """
    Generate an Evidently data drift HTML report.

    reference_df : training data (what the model learned from)
    current_df   : new incoming data (what the model is scoring now)
    """
    logger.info("Running data drift analysis...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(str(output_path))

    logger.info("Data drift report saved → %s", output_path)


def run_model_quality_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    pipeline,
    output_path: Path = REPORTS_DIR / "model_quality_report.html",
) -> None:
    """
    Generate an Evidently model quality report.

    Compares model performance on reference vs current data.
    Requires a 'target' and 'prediction' column in both dataframes.
    """
    logger.info("Running model quality analysis...")

    def add_predictions(
        df_raw: pd.DataFrame, df_transformed: pd.DataFrame
    ) -> pd.DataFrame:
        classifier = pipeline.named_steps["classifier"]
        probs = classifier.predict_proba(df_transformed)[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = df_transformed.copy()
        out["prediction"] = preds
        return out

    ref_with_preds = add_predictions(reference_df, reference_df)
    cur_with_preds = add_predictions(current_df, current_df)

    col_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        pos_label=1,
    )

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=ref_with_preds,
        current_data=cur_with_preds,
        column_mapping=col_mapping,
    )
    report.save_html(str(output_path))
    logger.info("Model quality report saved → %s", output_path)


def run_monitoring() -> None:
    """Full monitoring run — drift + model quality reports."""

    # Load data
    X, y = load_dataset("german_credit")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pipeline = joblib.load(MODELS_DIR / "best_model.pkl")

    # Transform splits to the feature space the model operates in
    X_train_t = _transform_to_df(pipeline, X_train)
    X_val_t = _transform_to_df(pipeline, X_val)

    # Add target column for model quality report
    X_train_t["target"] = y_train.values
    X_val_t["target"] = y_val.values

    # Save reference dataset (training distribution) for future comparisons
    ref_path = REF_DIR / "reference_data.parquet"
    X_train_t.to_parquet(ref_path, index=False)
    logger.info("Reference dataset saved → %s", ref_path)

    # ── Report 1: data drift (val vs train) ──────────────────────────────────
    run_data_drift_report(
        reference_df=X_train_t.drop(columns=["target"]),
        current_df=X_val_t.drop(columns=["target"]),
        output_path=REPORTS_DIR / "data_drift_report.html",
    )

    # ── Report 2: simulated production drift ─────────────────────────────────
    X_drifted_raw = simulate_drift(X_val)
    X_drifted_t = _transform_to_df(pipeline, X_drifted_raw)
    X_drifted_t["target"] = y_val.values

    run_data_drift_report(
        reference_df=X_train_t.drop(columns=["target"]),
        current_df=X_drifted_t.drop(columns=["target"]),
        output_path=REPORTS_DIR / "data_drift_simulated.html",
    )

    print("\n✅  Monitoring reports saved to reports/")
    print("    data_drift_report.html      — val vs train (should be minimal drift)")
    print(
        "    data_drift_simulated.html   — simulated production drift (should be visible)"
    )
    print("\nOpen either HTML file in your browser to explore the interactive report.")


if __name__ == "__main__":
    run_monitoring()
