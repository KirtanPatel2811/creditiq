"""
src/api/main.py
---------------
FastAPI app exposing CreditIQ predictions via REST.

Endpoints:
  GET  /health              — liveness check
  POST /predict             — default probability + risk tier
  POST /predict/explain     — same + top-10 SHAP values
  GET  /model/info          — version and performance metadata
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.api.schemas import ApplicantFeatures, PredictionResponse, ExplainResponse

app = FastAPI(
    title="CreditIQ API",
    description="Credit risk scoring with explainability",
    version="1.0.0",
)

MODELS_DIR = Path("models")

# ── load model once at startup ─────────────────────────────────────────────
pipeline = None
explainer = None


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global pipeline, explainer
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise RuntimeError("best_model.pkl not found. Run train.py first.")
    pipeline = joblib.load(model_path)
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    yield
    # cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="CreditIQ API",
    description="Credit risk scoring with explainability",
    version="1.0.0",
    lifespan=lifespan,
)


def _predict(features: ApplicantFeatures):
    """Shared prediction logic used by both /predict endpoints."""
    row = pd.DataFrame([features.model_dump()])
    prob = float(pipeline.predict_proba(row)[0, 1])
    tier = "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High")
    return prob, tier, row


def _risk_tier(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.6:
        return "Medium"
    return "High"


# ── endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_type": (
            type(pipeline.named_steps["classifier"]).__name__ if pipeline else None
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ApplicantFeatures):
    try:
        prob, tier, _ = _predict(features)
        return PredictionResponse(
            default_probability=round(prob, 4),
            risk_tier=tier,
            model_version="1.0.0",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain", response_model=ExplainResponse)
def predict_explain(features: ApplicantFeatures):
    try:
        prob, tier, row = _predict(features)

        # Transform for SHAP
        preprocessor = pipeline.named_steps["preprocessor"]
        X_t = preprocessor.transform(row)
        feature_names = list(preprocessor.get_feature_names_out())

        sv = explainer.shap_values(X_t)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]

        # Top 10 features by absolute SHAP value
        shap_series = pd.Series(sv[0], index=feature_names)
        top10 = shap_series.abs().nlargest(10).index
        top_features = {k: round(float(shap_series[k]), 4) for k in top10}

        return ExplainResponse(
            default_probability=round(prob, 4),
            risk_tier=tier,
            model_version="1.0.0",
            top_features=top_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    clf = pipeline.named_steps["classifier"]
    return {
        "model_type": type(clf).__name__,
        "model_version": "1.0.0",
        "n_features": clf.n_features_in_,
        "dataset": "German Credit (OpenML)",
    }
