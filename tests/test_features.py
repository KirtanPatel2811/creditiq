"""Unit tests for the feature preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_openml

from src.features.preprocessor import build_preprocessor, get_feature_names


@pytest.fixture(scope="module")
def sample_data():
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    X = data.data.copy()
    return X.head(200)


def test_preprocessor_no_nulls(sample_data):
    prep, _, _ = build_preprocessor(sample_data)
    X_t = prep.fit_transform(sample_data)
    assert not np.isnan(X_t).any(), "Transformed data should have no NaNs"


def test_preprocessor_shape(sample_data):
    prep, num_cols, cat_cols = build_preprocessor(sample_data)
    X_t = prep.fit_transform(sample_data)
    assert X_t.shape[0] == len(sample_data)
    assert X_t.shape[1] == len(num_cols) + len(cat_cols)


def test_feature_names_match_columns(sample_data):
    prep, _, _ = build_preprocessor(sample_data)
    prep.fit(sample_data)
    names = get_feature_names(prep)
    X_t = prep.transform(sample_data)
    assert len(names) == X_t.shape[1]


def test_unseen_categories_handled(sample_data):
    prep, _, _ = build_preprocessor(sample_data)
    prep.fit(sample_data)
    # Inject an unseen category — should not raise
    X_new = sample_data.head(5).copy()
    cat_col = sample_data.select_dtypes(include="object").columns[0]
    X_new[cat_col] = "UNSEEN_CATEGORY_XYZ"
    result = prep.transform(X_new)
    assert not np.isnan(result).any()
