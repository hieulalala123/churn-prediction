"""Shared loading/cleaning for the Telco Customer Churn dataset.

Every task built on this dataset (churn classification, and later survival
analysis) should go through `load_clean` rather than re-deriving these fixes,
so raw-data quirks are handled in exactly one place. See docs/docs.md
(Pha 1, section 1.3) for how each quirk was diagnosed.
"""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the dataset exactly as shipped, with no fixes applied."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the data-hygiene fixes established during Pha 1 EDA.

    customerID is kept (not dropped) so it stays available for traceability
    — e.g. joining error-analysis rows back to a real customer at Pha 4, or
    detecting accidental row overlap when a train/test split is persisted.
    It has no predictive value, so every task must exclude it explicitly
    from its own feature list rather than relying on it being absent here.
    """
    df = df.copy()

    # Stored as object because 11 rows hold whitespace instead of a number;
    # those rows all coincide with tenure == 0 (not-yet-billed new
    # customers), so they are dropped below rather than imputed.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.loc[df["tenure"] > 0].reset_index(drop=True)

    # Normalize to "No"/"Yes" to match every other binary categorical column.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def load_clean(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the dataset and apply the standard cleaning steps."""
    return clean(load_raw(path))
