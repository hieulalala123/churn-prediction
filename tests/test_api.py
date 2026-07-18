import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.churn_classification.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.serving import app as app_module
from src.serving.schemas import CustomerFeatures

VALID_CUSTOMER = {
    "tenure": 2,
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0,
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


class StubPipeline:
    def __init__(self, proba: float):
        self._proba = proba

    def predict_proba(self, X):
        return np.tile([1 - self._proba, self._proba], (len(X), 1))


@pytest.fixture()
def client(monkeypatch):
    def make(proba: float = 0.9, threshold: float = 0.465):
        service = app_module.service
        monkeypatch.setattr(service, "pipeline", StubPipeline(proba))
        monkeypatch.setattr(service, "threshold", threshold)
        monkeypatch.setattr(service, "model_version", "stub-1")
        # TestClient without triggering lifespan (which would hit MLflow)
        return TestClient(app_module.app)

    return make


def test_schema_matches_feature_lists():
    assert set(CustomerFeatures.model_fields) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_predict_happy_path(client):
    resp = client(proba=0.9).post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["churn"] is True
    assert body["churn_probability"] == pytest.approx(0.9)
    assert body["threshold"] == pytest.approx(0.465)
    assert body["model_version"] == "stub-1"


def test_predict_threshold_boundary(client):
    assert client(proba=0.464).post("/predict", json=VALID_CUSTOMER).json()["churn"] is False
    assert client(proba=0.466).post("/predict", json=VALID_CUSTOMER).json()["churn"] is True


def test_predict_missing_field_is_422(client):
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "Contract"}
    assert client().post("/predict", json=payload).status_code == 422


def test_predict_unknown_field_is_422(client):
    assert client().post("/predict", json={**VALID_CUSTOMER, "extra": 1}).status_code == 422


def test_health(client):
    body = client().get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "stub-1"


def test_prediction_logging(client, tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.csv"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    c = client(proba=0.7)
    c.post("/predict", json=VALID_CUSTOMER)
    c.post("/predict", json=VALID_CUSTOMER)
    import pandas as pd

    log = pd.read_csv(log_path)
    assert len(log) == 2
    assert "churn_probability" in log.columns
