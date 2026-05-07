import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
"""
src/app/streamlit_app.py
-------------------------
Streamlit dashboard — ties together all CreditIQ components.

Run with:  streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from src.data.loader import load_dataset
from src.data.splitter import split_data

# ── config ────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

st.set_page_config(
    page_title="CreditIQ",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return joblib.load(MODELS_DIR / "best_model.pkl")


@st.cache_data
def load_data():
    X, y = load_dataset("german_credit")
    return split_data(X, y)


@st.cache_resource
def load_explainer(_pipeline):
    clf = _pipeline.named_steps["classifier"]
    return shap.TreeExplainer(clf)


# ── sidebar nav ──────────────────────────────────────────────────────────
st.sidebar.title("💳 CreditIQ")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🔮 Predict",
        "📊 Model Performance",
        "🔍 Feature Importance",
        "📡 Data Drift",
    ],
)

pipeline = load_pipeline()
X_train, X_val, X_test, y_train, y_val, y_test = load_data()
explainer = load_explainer(pipeline)

# ── Home ─────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("CreditIQ — Credit Risk Scoring System")
    st.markdown("End-to-end ML pipeline with explainability and drift monitoring.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training samples", f"{len(X_train):,}")
    col2.metric("Features", pipeline.named_steps["preprocessor"].n_features_in_)
    col3.metric("Default rate", f"{y_train.mean():.1%}")
    col4.metric("Best model", type(pipeline.named_steps["classifier"]).__name__)

    st.divider()
    st.subheader("Pipeline architecture")
    st.code(
        """
Raw data (German Credit / Home Credit)
    ↓
Data pipeline  →  loader · preprocessor · splitter
    ↓
Modelling      →  LR · RF · XGBoost · LightGBM  (MLflow tracked)
    ↓
Explainability →  SHAP global + per-applicant waterfall
    ↓
MLOps          →  DVC versioning · Evidently drift monitoring
    ↓
Serving        →  FastAPI /predict  ·  This Streamlit dashboard
    """,
        language="text",
    )

# ── Predict ───────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.title("Predict default probability")
    st.markdown(
        "Fill in applicant details to get a real-time risk score with SHAP explanation."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        checking_status = st.selectbox(
            "Checking status", ["no checking", "0<=X<200", "<0", ">=200"]
        )
        duration = st.slider("Loan duration (months)", 4, 72, 24)
        credit_amount = st.number_input("Credit amount (DM)", 250, 20000, 5000)
        purpose = st.selectbox(
            "Purpose",
            [
                "new car",
                "used car",
                "furniture/equipment",
                "radio/tv",
                "domestic appliance",
                "repairs",
                "education",
                "retraining",
                "business",
                "other",
            ],
        )
    with col2:
        savings_status = st.selectbox(
            "Savings status",
            ["< 100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
        )
        employment = st.selectbox(
            "Employment", ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
        )
        age = st.slider("Age", 18, 75, 35)
        housing = st.selectbox("Housing", ["own", "free", "rent"])
    with col3:
        credit_history = st.selectbox(
            "Credit history",
            [
                "all paid",
                "critical/other existing credit",
                "delayed previously",
                "existing paid",
                "no credits/all paid",
            ],
        )
        job = st.selectbox(
            "Job",
            [
                "high qualif/self emp/mgmt",
                "skilled",
                "unskilled resident",
                "unemp/unskilled non res",
            ],
        )
        personal_status = st.selectbox(
            "Personal status",
            ["female div/dep/mar", "male div/sep", "male mar/wid", "male single"],
        )
        foreign_worker = st.selectbox("Foreign worker", ["yes", "no"])

    if st.button("Score this applicant", type="primary"):
        row = pd.DataFrame(
            [
                {
                    "checking_status": checking_status,
                    "duration": duration,
                    "credit_history": credit_history,
                    "purpose": purpose,
                    "credit_amount": credit_amount,
                    "savings_status": savings_status,
                    "employment": employment,
                    "installment_commitment": 3,
                    "personal_status": personal_status,
                    "other_parties": "none",
                    "residence_since": 2,
                    "property_magnitude": "real estate",
                    "age": age,
                    "other_payment_plans": "none",
                    "housing": housing,
                    "existing_credits": 1,
                    "job": job,
                    "num_dependents": 1,
                    "own_telephone": "yes",
                    "foreign_worker": foreign_worker,
                }
            ]
        )

        prob = float(pipeline.predict_proba(row)[0, 1])
        tier = "🟢 Low" if prob < 0.3 else ("🟡 Medium" if prob < 0.6 else "🔴 High")

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Default probability", f"{prob:.1%}")
        c2.metric("Risk tier", tier)
        st.progress(prob)

        # SHAP waterfall
        st.subheader("Why this score? (SHAP explanation)")
        preprocessor = pipeline.named_steps["preprocessor"]
        X_t = preprocessor.transform(row)
        feature_names = list(preprocessor.get_feature_names_out())
        sv = explainer.shap_values(X_t)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]

        ev = (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )

        explanation = shap.Explanation(
            values=sv[0],
            base_values=ev,
            data=X_t[0],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(plt.gcf())
        plt.close()

# ── Model Performance ─────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("Model performance")

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    gini = 2 * roc - 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{roc:.4f}")
    c2.metric("PR-AUC", f"{pr:.4f}")
    c3.metric("Gini", f"{gini:.4f}")
    c4.metric("Test rows", len(X_test))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        ax.set_title(f"ROC curve (AUC={roc:.3f})")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.subheader("Precision-Recall curve")
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
        ax.set_title(f"PR curve (AUC={pr:.3f})")
        st.pyplot(fig)
        plt.close()

# ── Feature Importance ────────────────────────────────────────────────────
elif page == "🔍 Feature Importance":
    st.title("Global feature importance (SHAP)")

    preprocessor = pipeline.named_steps["preprocessor"]
    X_val_t = preprocessor.transform(X_val)
    feature_names = list(preprocessor.get_feature_names_out())

    sv = explainer.shap_values(X_val_t)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    X_val_df = pd.DataFrame(X_val_t, columns=feature_names)

    tab1, tab2 = st.tabs(["Bar chart", "Beeswarm"])
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, X_val_df, plot_type="bar", max_display=15, show=False)
        st.pyplot(plt.gcf())
        plt.close()
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_val_df, max_display=15, show=False)
        st.pyplot(plt.gcf())
        plt.close()

# ── Data Drift ────────────────────────────────────────────────────────────
elif page == "📡 Data Drift":
    st.title("Data drift monitoring")
    st.markdown(
        "Reports generated by Evidently AI. Open the HTML files for full interactive dashboards."
    )

    for report_file, label in [
        ("data_drift_report.html", "Val vs Train (expected: minimal drift)"),
        (
            "data_drift_simulated.html",
            "Simulated production drift (expected: visible drift)",
        ),
    ]:
        path = REPORTS_DIR / report_file
        if path.exists():
            with open(path) as f:
                content = f.read()
            st.subheader(label)
            st.components.v1.html(content, height=600, scrolling=True)
        else:
            st.warning(f"{report_file} not found. Run drift_detector.py first.")
