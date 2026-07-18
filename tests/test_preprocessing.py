import numpy as np
import pandas as pd
import pytest

from src.churn_classification.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    compute_scale_pos_weight,
    split_X_y,
)


def test_split_X_y_shapes_and_types(clean_df_sample):
    X, y = split_X_y(clean_df_sample)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(X.columns) == 19
    assert set(y.unique()) == {0, 1}
    assert "customerID" not in X.columns


def test_preprocessor_output_has_no_nans(clean_df_sample):
    X, _ = split_X_y(clean_df_sample)
    Xt = build_preprocessor().fit_transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(Xt).any()


def test_preprocessor_handles_unknown_category(clean_df_sample):
    """handle_unknown="ignore" is the serving safety net: a category value
    never seen in training must not raise at transform time.
    """
    X, _ = split_X_y(clean_df_sample)
    pre = build_preprocessor().fit(X)
    X_new = X.head(3).copy()
    X_new["Contract"] = "Decade-long"
    pre.transform(X_new)  # must not raise


def test_preprocessor_imputes_missing_values(clean_df_sample):
    X, _ = split_X_y(clean_df_sample)
    pre = build_preprocessor().fit(X)
    X_new = X.head(3).copy()
    X_new.loc[X_new.index[0], "tenure"] = np.nan
    X_new.loc[X_new.index[1], "Contract"] = np.nan
    Xt = pre.transform(X_new)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(Xt).any()


def test_compute_scale_pos_weight():
    y = pd.Series([0] * 30 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(3.0)
