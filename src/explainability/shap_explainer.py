"""
src/explainability/shap_explainer.py
--------------------------------------
SHAP explanations for the best trained model.

Two levels of explanation:
  1. Global  — which features matter most across all applicants?
               → bar chart + beeswarm plot saved to reports/
  2. Local   — why did THIS specific applicant get their score?
               → waterfall plot (used later in Streamlit dashboard)

Design decision: we use TreeExplainer (not the generic KernelExplainer)
because our best models are all tree-based. TreeExplainer is ~100× faster
and gives exact SHAP values rather than approximations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.loader import load_dataset
from src.data.splitter import split_data
from src.features.preprocessor import build_preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models")


def load_pipeline(model_name: str = "best_model"):
    path = MODELS_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    return joblib.load(path)


def get_shap_values(pipeline, X_raw: pd.DataFrame):
    """
    Compute SHAP values for a tree-based pipeline.

    Extracts the preprocessor and classifier from the pipeline,
    transforms X, then runs TreeExplainer on the classifier directly.

    Returns
    -------
    explainer    : shap.TreeExplainer
    shap_values  : np.ndarray, shape (n_samples, n_features)
                   Values for the positive class (default=1)
    X_transformed: pd.DataFrame with feature names (for readable plots)
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform to numpy, then wrap back with feature names for readable plots
    X_np = preprocessor.transform(X_raw)
    feature_names = list(preprocessor.get_feature_names_out())
    X_transformed = pd.DataFrame(X_np, columns=feature_names)

    logger.info("Building TreeExplainer...")
    explainer = shap.TreeExplainer(classifier)

    logger.info("Computing SHAP values for %d samples...", len(X_transformed))
    shap_values = explainer.shap_values(X_transformed)

    # Random forests return one array per class — take class 1 (default)
    # Random forests: older shap returns list per class, newer returns 3D array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # take class 1 (default)

    return explainer, shap_values, X_transformed


def plot_global_importance(shap_values, X_transformed: pd.DataFrame, top_n: int = 15):
    """Bar chart of mean |SHAP| per feature — global importance."""
    logger.info("Plotting global feature importance (top %d)...", top_n)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    plt.title("Global feature importance (mean |SHAP value|)", fontsize=13)
    plt.tight_layout()
    out = REPORTS_DIR / "shap_global_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out)


def plot_beeswarm(shap_values, X_transformed: pd.DataFrame, top_n: int = 15):
    """
    Beeswarm plot — shows both direction and magnitude of each feature's effect.
    Red = high feature value pushed prediction toward default.
    Blue = low feature value pushed prediction away from default.
    """
    logger.info("Plotting beeswarm...")

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_transformed,
        max_display=top_n,
        show=False,
    )
    plt.title("SHAP beeswarm — feature impact on default probability", fontsize=13)
    plt.tight_layout()
    out = REPORTS_DIR / "shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out)


def plot_waterfall(
    explainer,
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    sample_idx: int = 0,
    expected_value: float = None,
):
    """
    Waterfall plot for a single applicant — shows exactly which features
    pushed the prediction up or down from the base rate.

    This is the plot shown per-applicant in the Streamlit dashboard.
    """
    logger.info("Plotting waterfall for sample index %d...", sample_idx)

    ev = (
        expected_value
        if expected_value is not None
        else (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
    )

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=ev,
        data=X_transformed.iloc[sample_idx].values,
        feature_names=X_transformed.columns.tolist(),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"Why did applicant #{sample_idx} get this score?", fontsize=12)
    plt.tight_layout()
    out = REPORTS_DIR / f"shap_waterfall_sample_{sample_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out)


def run_explainability() -> None:
    """Full explainability run — generates all three plot types."""

    # Load data + pipeline
    X, y = load_dataset("german_credit")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pipeline = load_pipeline("best_model")

    # Compute SHAP values on validation set
    explainer, shap_values, X_transformed = get_shap_values(pipeline, X_val)

    # Global plots
    plot_global_importance(shap_values, X_transformed)
    plot_beeswarm(shap_values, X_transformed)

    # Local waterfall for 3 example applicants
    for idx in [0, 1, 2]:
        plot_waterfall(explainer, shap_values, X_transformed, sample_idx=idx)

    # Print top 5 most important features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(mean_abs_shap, index=X_transformed.columns)
    top_features = top_features.sort_values(ascending=False).head(5)

    print("\n── Top 5 features by mean |SHAP| ──────────────────────────")
    for feat, val in top_features.items():
        print(f"  {feat:<35} {val:.4f}")

    print(f"\n✅  SHAP plots saved to reports/")
    print("    shap_global_importance.png")
    print("    shap_beeswarm.png")
    print("    shap_waterfall_sample_0/1/2.png")


if __name__ == "__main__":
    run_explainability()
