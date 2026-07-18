"""FastAPI serving app for the churn model.

Run locally:  uvicorn src.serving.app:app --port 8000
(needs MLFLOW_TRACKING_URI pointing at the registry that holds @champion)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.serving.model import MODEL_NAME, ModelService
from src.serving.schemas import CustomerFeatures, ModelInfo, PredictResponse

service = ModelService()

# Score-distribution histogram: drift of served probabilities over time is
# the serving-side complement to the Evidently feature-drift check.
PREDICTION_PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Distribution of predicted churn probabilities",
    buckets=[i / 10 for i in range(11)],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(title="Telco Churn API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {
        "status": "ok" if service.is_loaded else "model_not_loaded",
        "model_version": service.model_version,
    }


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    _require_loaded()
    return ModelInfo(
        model_name=MODEL_NAME,
        model_version=service.model_version,
        threshold=service.threshold,
        model_uri=service.model_uri,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerFeatures):
    _require_loaded()
    proba, labels = service.predict(customer.to_frame())
    PREDICTION_PROBABILITY.observe(proba[0])
    return PredictResponse(
        churn_probability=proba[0],
        churn=labels[0],
        threshold=service.threshold,
        model_version=service.model_version,
    )


@app.post("/reload")
def reload_model():
    """Re-resolve @champion so a retrain+promote is picked up without restart."""
    service.load()
    return {"status": "reloaded", "model_version": service.model_version}


def _require_loaded():
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
