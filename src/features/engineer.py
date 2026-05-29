"""
src/features/engineer.py
-------------------------
Domain-specific feature engineering for credit risk.
These hand-crafted features are what separate DS candidates — anyone
can call XGBoost, not everyone thinks about what features to create.
"""

from __future__ import annotations

import pandas as pd


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-relevant features from raw applicant data.
    Works on both German Credit and Home Credit datasets.
    """
    df = X.copy()

    # ── Ratio features ─────────────────────────────────────────────────────
    if "credit_amount" in df.columns and "duration" in df.columns:
        # Monthly repayment burden
        df["monthly_repayment"] = df["credit_amount"] / df["duration"]

    if "credit_amount" in df.columns and "age" in df.columns:
        # Credit relative to age — young person with huge loan is riskier
        df["credit_per_age_year"] = df["credit_amount"] / df["age"]

    if "duration" in df.columns and "age" in df.columns:
        # Loan duration as fraction of remaining working life (assume retire at 65)
        df["duration_to_working_life"] = df["duration"] / (
            (65 - df["age"]).clip(lower=1) * 12
        )

    # ── Risk indicator flags ────────────────────────────────────────────────
    if "checking_status" in df.columns:
        df["no_checking_account"] = (df["checking_status"] == "no checking").astype(int)

    if "savings_status" in df.columns:
        df["low_savings"] = (
            df["savings_status"].isin(["< 100", "no known savings"]).astype(int)
        )

    if "credit_history" in df.columns:
        df["bad_credit_history"] = (
            df["credit_history"]
            .isin(["critical/other existing credit", "delayed previously"])
            .astype(int)
        )

    # ── Employment stability ────────────────────────────────────────────────
    if "employment" in df.columns:
        df["unemployed"] = (df["employment"] == "unemployed").astype(int)
        # Map employment duration to ordinal score
        emp_map = {"unemployed": 0, "<1": 1, "1<=X<4": 2, "4<=X<7": 3, ">=7": 4}
        df["employment_score"] = df["employment"].map(emp_map).fillna(0)

    return df
