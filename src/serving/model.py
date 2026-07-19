"""Model loading/prediction service, kept separate from the FastAPI app so it
can be unit-tested with a stub and hot-swapped via the /reload endpoint.
"""

import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

MODEL_NAME = os.environ.get("MODEL_NAME", "telco-churn-catboost")
DEFAULT_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
FALLBACK_THRESHOLD = 0.5


class ModelService:
    def __init__(self) -> None:
        self.pipeline = None
        self.threshold: float = FALLBACK_THRESHOLD
        self.model_version: str = "unknown"
        self.model_uri: str = ""
        # _state_lock guards the swap in load() and reads in predict() so a
        # /reload can never expose a half-updated (pipeline, threshold,
        # version) combination to an in-flight request. _log_lock serializes
        # prediction-log appends within this process (single-worker
        # assumption; multi-worker deployments need per-worker files or a
        # queue-based writer instead of a shared CSV).
        self._state_lock = threading.Lock()
        self._log_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        """Resolve and load the serving model, then swap state atomically.

        Everything is resolved into locals first; the service's visible state
        is only updated once the new model is fully loaded, so a failed
        reload leaves the previous model serving untouched.

        Model: env MODEL_URI overrides; default is the registry alias
        models:/<MODEL_NAME>@<alias> against MLFLOW_TRACKING_URI.
        Threshold: env DECISION_THRESHOLD > registry tag on the version > 0.5.
        """
        import mlflow

        model_uri = os.environ.get("MODEL_URI", f"models:/{MODEL_NAME}@{DEFAULT_ALIAS}")
        pipeline = mlflow.sklearn.load_model(model_uri)

        model_version = "unknown"
        version_tags: dict[str, str] = {}
        if model_uri.startswith("models:/") and "@" in model_uri:
            client = mlflow.MlflowClient()
            name, alias = model_uri.removeprefix("models:/").split("@", 1)
            version = client.get_model_version_by_alias(name, alias)
            model_version = str(version.version)
            version_tags = version.tags

        env_threshold = os.environ.get("DECISION_THRESHOLD")
        if env_threshold is not None:
            threshold = float(env_threshold)
        elif "decision_threshold" in version_tags:
            threshold = float(version_tags["decision_threshold"])
        else:
            threshold = FALLBACK_THRESHOLD

        with self._state_lock:
            self.pipeline = pipeline
            self.threshold = threshold
            self.model_version = model_version
            self.model_uri = model_uri

    def predict(self, X: pd.DataFrame) -> tuple[list[float], list[bool]]:
        with self._state_lock:
            pipeline = self.pipeline
            threshold = self.threshold
            model_version = self.model_version
        proba = pipeline.predict_proba(X)[:, 1]
        labels = proba >= threshold
        self._log_predictions(X, proba, model_version)
        return proba.tolist(), labels.tolist()

    def _log_predictions(self, X: pd.DataFrame, proba, model_version: str) -> None:
        """Append served rows to PREDICTION_LOG_PATH (CSV) — the "current"
        dataset for drift monitoring. No-op when the env var is unset.
        """
        log_path = os.environ.get("PREDICTION_LOG_PATH")
        if not log_path:
            return
        record = X.copy()
        record["churn_probability"] = proba
        record["predicted_at"] = datetime.now(UTC).isoformat()
        record["model_version"] = model_version
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            record.to_csv(path, mode="a", header=not path.exists(), index=False)
