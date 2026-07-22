"""Cox proportional-hazards model for time-to-churn.

Selected in notebooks/survival_analysis/survival_analysis.ipynb (section 4)
over Exponential/Weibull/Log-Logistic (no covariates, compared by AIC) and
Weibull AFT (worse log-likelihood, and a stronger distributional assumption
than Cox's proportional-hazards one): unstratified Cox PH has the highest
concordance (0.8663).

Kept **unstratified** even though `Contract` fails the proportional-hazards
check (section 3.2 of the notebook) — stratifying it away (the textbook fix)
drops concordance to 0.7205, because `Contract` carries most of the model's
discriminative power. For a model whose purpose is business interpretation
("who churns, and roughly when"), keeping that signal outweighs strict PH
compliance; the violation is a documented, accepted limitation instead.

This module is an analysis/interpretation aid, not a second production
model: it is intentionally not wired into MLflow registry or FastAPI
serving (see notebook section 5).
"""

import pandas as pd
from lifelines import CoxPHFitter

DURATION_COL = "tenure"
EVENT_COL = "event"
TARGET = "Churn"
POSITIVE_LABEL = "Yes"

FEATURE_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
]

# "No internet/phone service" duplicates InternetService=="No"/PhoneService=="No"
# exactly — collapsed to "No" to avoid multicollinearity between these columns
# (same cleanup as notebooks/survival_analysis/survival_analysis.ipynb section 2.2).
INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw/cleaned Kaggle-schema df -> one-hot encoded frame ready for CoxPHFitter.fit().

    `TotalCharges` and `customerID` are excluded: TotalCharges is
    near-collinear with tenure * MonthlyCharges (the duration column itself),
    customerID has no predictive value.
    """
    features = df[FEATURE_COLS].copy()
    features[INTERNET_ADDON_COLS] = features[INTERNET_ADDON_COLS].replace(
        "No internet service", "No"
    )
    features["MultipleLines"] = features["MultipleLines"].replace("No phone service", "No")

    model_df = pd.get_dummies(features, drop_first=True)
    model_df[DURATION_COL] = df[DURATION_COL]
    model_df[EVENT_COL] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return model_df


def fit_cox_model(df: pd.DataFrame) -> CoxPHFitter:
    """Fit the unstratified Cox PH model on a raw/cleaned Kaggle-schema df."""
    cph = CoxPHFitter()
    cph.fit(build_model_frame(df), duration_col=DURATION_COL, event_col=EVENT_COL)
    return cph
