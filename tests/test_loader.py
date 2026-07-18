import numpy as np

from src.data import clean


def test_clean_drops_not_yet_billed_rows(raw_df_sample):
    out = clean(raw_df_sample)
    assert len(out) == len(raw_df_sample) - 2
    assert (out["tenure"] > 0).all()


def test_clean_coerces_totalcharges_to_numeric(raw_df_sample):
    out = clean(raw_df_sample)
    assert np.issubdtype(out["TotalCharges"].dtype, np.number)
    assert out["TotalCharges"].notna().all()


def test_clean_normalizes_seniorcitizen(raw_df_sample):
    out = clean(raw_df_sample)
    assert set(out["SeniorCitizen"].unique()) <= {"No", "Yes"}


def test_clean_keeps_customer_id(raw_df_sample):
    assert "customerID" in clean(raw_df_sample).columns


def test_clean_does_not_mutate_input(raw_df_sample):
    before = raw_df_sample.copy()
    clean(raw_df_sample)
    assert raw_df_sample.equals(before)
