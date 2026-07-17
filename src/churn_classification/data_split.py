"""Train/holdout split for the churn classification task, persisted to disk.

The holdout set must stay identical across every notebook and every re-run
from here through Pha 4 — recomputing train_test_split() fresh in each
notebook risks silently drawing a different holdout (different sklearn/
pandas version, changed row order after a data update, etc.), which would
let previously-held-out customers leak into training in a later phase.
Persisting the split once and loading it thereafter removes that risk.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_clean

SPLIT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "churn_classification"
TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

# 15% holdout, not the more common 20-25%: with only 7032 rows and a 26.5%
# positive rate, a larger holdout would shrink the 5-fold CV train pool
# (Pha 3 model selection) more than it would improve the precision of the
# final holdout metric. 15% still leaves ~280 positive cases in the holdout
# — enough for a reasonably stable PR-AUC estimate at Pha 4.
TEST_SIZE = 0.15
RANDOM_STATE = 42
TARGET = "Churn"


def _create_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_clean()
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    return train_df, test_df


def get_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted train/holdout split, creating it on first call.

    On every call after the first, this returns exactly the cached CSVs —
    changes to TEST_SIZE/RANDOM_STATE or to the raw source data are NOT
    picked up automatically. Delete data/processed/churn_classification/
    and re-run to force a fresh split if either changes.
    """
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        print(
            f"Loading cached split from {SPLIT_DIR} (created earlier). "
            "Delete this folder and re-run to regenerate with current TEST_SIZE/RANDOM_STATE."
        )
        return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    return _create_split()
