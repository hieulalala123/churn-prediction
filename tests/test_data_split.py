import pandas as pd
import pytest

from src.churn_classification import data_split


@pytest.fixture()
def patched_source(clean_df_sample, tmp_split_paths, monkeypatch):
    monkeypatch.setattr(data_split, "load_clean", lambda: clean_df_sample)
    return tmp_split_paths


def test_split_sizes_and_stratification(patched_source, clean_df_sample):
    train_df, test_df = data_split.get_split()
    n = len(clean_df_sample)
    assert len(train_df) + len(test_df) == n
    assert len(test_df) == round(n * data_split.TEST_SIZE)
    overall_rate = (clean_df_sample[data_split.TARGET] == "Yes").mean()
    test_rate = (test_df[data_split.TARGET] == "Yes").mean()
    assert abs(test_rate - overall_rate) < 0.2


def test_split_has_no_row_overlap(patched_source):
    train_df, test_df = data_split.get_split()
    assert set(train_df["customerID"]).isdisjoint(test_df["customerID"])


def test_split_is_deterministic(patched_source):
    train_a, test_a = data_split._create_split()
    train_b, test_b = data_split._create_split()
    pd.testing.assert_frame_equal(train_a.reset_index(drop=True), train_b.reset_index(drop=True))
    pd.testing.assert_frame_equal(test_a.reset_index(drop=True), test_b.reset_index(drop=True))


def test_get_split_reloads_cached_files(patched_source):
    train_a, test_a = data_split.get_split()
    assert data_split.TRAIN_PATH.exists() and data_split.TEST_PATH.exists()
    train_b, test_b = data_split.get_split()
    pd.testing.assert_frame_equal(train_a.reset_index(drop=True), train_b)
    pd.testing.assert_frame_equal(test_a.reset_index(drop=True), test_b)
