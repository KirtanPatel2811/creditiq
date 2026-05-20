from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class ApplicantFeatures(BaseModel):
    checking_status: str = Field(..., json_schema_extra={"example": "no checking"})
    duration: int = Field(..., ge=1)
    credit_history: str = Field(..., json_schema_extra={"example": "existing paid"})
    purpose: str = Field(..., json_schema_extra={"example": "new car"})
    credit_amount: int = Field(..., ge=1)
    savings_status: str = Field(..., json_schema_extra={"example": "< 100"})
    employment: str = Field(..., json_schema_extra={"example": "1<=X<4"})
    installment_commitment: int = Field(..., ge=1, le=4)
    personal_status: str = Field(..., json_schema_extra={"example": "male single"})
    other_parties: str = Field(..., json_schema_extra={"example": "none"})
    residence_since: int = Field(..., ge=1, le=4)
    property_magnitude: str = Field(..., json_schema_extra={"example": "real estate"})
    age: int = Field(..., ge=18)
    other_payment_plans: str = Field(..., json_schema_extra={"example": "none"})
    housing: str = Field(..., json_schema_extra={"example": "own"})
    existing_credits: int = Field(..., ge=1)
    job: str = Field(..., json_schema_extra={"example": "skilled"})
    num_dependents: int = Field(..., ge=1)
    own_telephone: str = Field(..., json_schema_extra={"example": "yes"})
    foreign_worker: str = Field(..., json_schema_extra={"example": "yes"})


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    default_probability: float
    risk_tier: str
    model_version: str


class ExplainResponse(PredictionResponse):
    top_features: dict
