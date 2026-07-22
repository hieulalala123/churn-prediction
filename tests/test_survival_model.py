from src.survival_analysis.cox_model import (
    DURATION_COL,
    EVENT_COL,
    build_model_frame,
    fit_cox_model,
)


def test_build_model_frame_collapses_addon_categories(clean_df_sample):
    model_df = build_model_frame(clean_df_sample)

    assert DURATION_COL in model_df.columns
    assert EVENT_COL in model_df.columns
    # "No internet/phone service" must have been collapsed into "No" (dropped
    # by drop_first=True alongside it) instead of getting its own dummy column.
    addon_dummy_cols = [c for c in model_df.columns if c.startswith("OnlineSecurity_")]
    assert addon_dummy_cols == ["OnlineSecurity_Yes"]


def test_fit_cox_model_produces_a_usable_concordance(clean_df_sample):
    cph = fit_cox_model(clean_df_sample)

    assert 0.0 <= cph.concordance_index_ <= 1.0
    predicted = cph.predict_partial_hazard(build_model_frame(clean_df_sample))
    assert len(predicted) == len(clean_df_sample)
