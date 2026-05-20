"""Unit tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

SAMPLE_APPLICANT = {
    "checking_status": "no checking",
    "duration": 24,
    "credit_history": "existing paid",
    "purpose": "new car",
    "credit_amount": 5000,
    "savings_status": "< 100",
    "employment": "1<=X<4",
    "installment_commitment": 3,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "real estate",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes",
}


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probability():
    r = client.post("/predict", json=SAMPLE_APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_tier"] in {"Low", "Medium", "High"}


def test_predict_explain_has_shap():
    r = client.post("/predict/explain", json=SAMPLE_APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert "top_features" in body
    assert len(body["top_features"]) == 10


def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "model_type" in r.json()
