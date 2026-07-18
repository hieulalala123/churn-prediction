"""Request/response schemas for the churn serving API.

Field names match the raw dataframe columns exactly (including case), so a
validated request converts straight into the one-row frame the sklearn
pipeline expects — no renaming layer to drift out of sync.
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.churn_classification.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

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
