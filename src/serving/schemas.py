"""Request/response schemas for the churn serving API.

Field names match the raw dataframe columns exactly (including case), so a
validated request converts straight into the one-row frame the sklearn
pipeline expects — no renaming layer to drift out of sync.

Categorical fields are Literal-typed on purpose: the preprocessor's
OneHotEncoder(handle_unknown="ignore") would silently encode an unseen value
(e.g. "Fiber" instead of "Fiber optic") as all-zeros and return a biased
prediction with HTTP 200. Literal turns that silent failure into a 422 at
the API boundary. tests/test_api.py asserts these Literals stay in sync with
the dataset categories declared in preprocessing.CATEGORY_VALUES.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.churn_classification.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool
    threshold: float
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    model_uri: str
