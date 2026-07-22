"""Preprocessing pipeline for the churn classification task.

Task-specific (not shared with other future tasks on this dataset, e.g.
survival analysis) — the column typing and imbalance-handling convention
here are tied to how *this* classification problem is framed. Dataset-level
cleaning stays in src/data/loader.py and is reused unchanged.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
POSITIVE_LABEL = "Yes"  # Churn=Yes is the positive class throughout this task.

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
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
]

# Valid values per categorical feature in the cleaned dataset. The serving
# schema's Literal types must match these exactly (asserted in tests) so an
# out-of-vocabulary value is rejected at the API boundary instead of being
# silently all-zeros-encoded by OneHotEncoder(handle_unknown="ignore").
_YES_NO = ("Yes", "No")
_INTERNET_ADDON = ("Yes", "No", "No internet service")
CATEGORY_VALUES: dict[str, tuple[str, ...]] = {
    "gender": ("Male", "Female"),
    "SeniorCitizen": _YES_NO,
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: numeric -> median-impute + scale; categorical -> impute + one-hot.

    Imputers are a defensive safety net for future/production data — the
    current cleaned training data has zero missing values (see Pha 1 EDA).
    Median (not mean) for numeric because MonthlyCharges/TotalCharges are
    right-skewed (established in Pha 1), so median is the more robust
    central estimate under skew. StandardScaler is applied even though
    tree models don't need it, because it's a monotonic per-feature
    transform that never changes tree splits — so one preprocessor safely
    serves both the linear baseline and the tree-based models planned for
    Pha 3.
    """
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def compute_scale_pos_weight(y: pd.Series) -> float:
    """neg/pos ratio, for gradient-boosting APIs' manual imbalance param
    (XGBoost/LightGBM `scale_pos_weight`).

    NOT needed for sklearn's `class_weight="balanced"` (LogisticRegression,
    RandomForest, ...) — that option computes its own internal weighting
    from `y` and takes no manual ratio. Only pass this value where the
    library's API explicitly asks for a `scale_pos_weight`-style parameter.

    Must be computed on the TRAIN fold/split only, never on the full
    dataset or the held-out test set — leaking test-set class balance into
    a training-time hyperparameter is still leakage even though it looks
    like "just a ratio".
    """
    counts = y.value_counts()
    return counts[0] / counts[1]
