import numpy as np

from src.churn_classification.final_model import build_final_pipeline
from src.churn_classification.preprocessing import split_X_y


def test_pipeline_step_names_are_the_serving_contract():
    pipeline = build_final_pipeline(scale_pos_weight=1.0)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]


def test_pipeline_fits_and_predicts_probabilities(clean_df_sample):
    X, y = split_X_y(clean_df_sample)
    pipeline = build_final_pipeline(scale_pos_weight=1.0)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))
