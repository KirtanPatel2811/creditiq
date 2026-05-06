"""
src/features/preprocessor.py
-----------------------------
Builds a scikit-learn preprocessing Pipeline that can be fit on training data
and applied identically to validation and test sets.

Design decision: we use a ColumnTransformer so numeric and categorical columns
get different treatment without manual splitting in calling code. The whole
pipeline is serialisable with joblib — the same object that's fit during
training is saved and reloaded at inference time, guaranteeing identical
transformations.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Inspect X, identify column types, and return a fitted-ready ColumnTransformer.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature matrix (before any transformation).

    Returns
    -------
    preprocessor : ColumnTransformer
        Unfitted transformer — call preprocessor.fit_transform(X_train).
    numeric_cols : list[str]
    categorical_cols : list[str]
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(
        "Detected %d numeric, %d categorical features",
        len(numeric_cols),
        len(categorical_cols),
    )

    # Numeric pipeline: median imputation (robust to outliers) → z-score scaling
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: most-frequent imputation → ordinal encoding
    # Why OrdinalEncoder and not OneHotEncoder?
    # Tree models (XGBoost, LightGBM, RF) handle ordinal integers natively and
    # don't need one-hot expansion. This keeps the feature matrix dense and fast.
    # If we add a Logistic Regression baseline that needs OHE, we'll branch there.
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",  # drop any columns we haven't accounted for
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract output feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Must be already fitted.

    Returns
    -------
    list[str]
        Feature names in the same order as the transformed matrix columns.
    """
    return list(preprocessor.get_feature_names_out())
