"""Shared fixtures: a small synthetic frame matching the raw Kaggle schema,
so the test suite never needs the real CSV (CI runs without the dataset).
"""

import numpy as np
import pandas as pd
import pytest

from src.churn_classification import data_split

N_ROWS = 40


@pytest.fixture()
def raw_df_sample() -> pd.DataFrame:
    """Synthetic raw-schema frame: 40 valid rows + 2 not-yet-billed rows
    (tenure == 0, whitespace TotalCharges) that clean() must drop.
    """
    rng = np.random.default_rng(0)
    tenure = rng.integers(1, 72, N_ROWS)
    monthly = rng.uniform(20, 118, N_ROWS).round(2)
    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(N_ROWS)],
            "gender": rng.choice(["Male", "Female"], N_ROWS),
            "SeniorCitizen": rng.choice([0, 1], N_ROWS),
            "Partner": rng.choice(["Yes", "No"], N_ROWS),
            "Dependents": rng.choice(["Yes", "No"], N_ROWS),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], N_ROWS),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], N_ROWS),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], N_ROWS),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], N_ROWS),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], N_ROWS),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], N_ROWS),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], N_ROWS),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], N_ROWS),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], N_ROWS),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], N_ROWS),
            "PaperlessBilling": rng.choice(["Yes", "No"], N_ROWS),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                N_ROWS,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": (tenure * monthly).round(2).astype(str),
            # Deterministic alternating target so both classes always appear
            # with a stable 50/50 balance regardless of the rng draws above.
            "Churn": ["Yes" if i % 2 == 0 else "No" for i in range(N_ROWS)],
        }
    )
    not_yet_billed = df.head(2).copy()
    not_yet_billed["customerID"] = ["9998-NEW", "9999-NEW"]
    not_yet_billed["tenure"] = 0
    not_yet_billed["TotalCharges"] = " "
    return pd.concat([df, not_yet_billed], ignore_index=True)


@pytest.fixture()
def clean_df_sample(raw_df_sample) -> pd.DataFrame:
    from src.data import clean

    return clean(raw_df_sample)


@pytest.fixture()
def tmp_split_paths(tmp_path, monkeypatch):
    """Redirect the persisted-split paths into tmp_path so tests never touch
    (or depend on) data/processed/churn_classification/.
    """
    split_dir = tmp_path / "split"
    monkeypatch.setattr(data_split, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(data_split, "TRAIN_PATH", split_dir / "train.csv")
    monkeypatch.setattr(data_split, "TEST_PATH", split_dir / "test.csv")
    return split_dir
