"""
src/api/schemas.py
------------------
Pydantic request/response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ApplicantFeatures(BaseModel):
    """Raw applicant data — mirrors the German Credit feature set."""

    checking_status: str = Field(..., example="no checking")
    duration: int = Field(..., example=24, ge=1)
    credit_history: str = Field(..., example="existing paid")
    purpose: str = Field(..., example="new car")
    credit_amount: int = Field(..., example=5000, ge=1)
    savings_status: str = Field(..., example="< 100")
    employment: str = Field(..., example="1<=X<4")
    installment_commitment: int = Field(..., example=3, ge=1, le=4)
    personal_status: str = Field(..., example="male single")
    other_parties: str = Field(..., example="none")
    residence_since: int = Field(..., example=2, ge=1, le=4)
    property_magnitude: str = Field(..., example="real estate")
    age: int = Field(..., example=35, ge=18)
    other_payment_plans: str = Field(..., example="none")
    housing: str = Field(..., example="own")
    existing_credits: int = Field(..., example=1, ge=1)
    job: str = Field(..., example="skilled")
    num_dependents: int = Field(..., example=1, ge=1)
    own_telephone: str = Field(..., example="yes")
    foreign_worker: str = Field(..., example="yes")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier: str  # "Low" | "Medium" | "High"
    model_version: str


class ExplainResponse(PredictionResponse):
    top_features: dict  # {feature_name: shap_value} for top 10
