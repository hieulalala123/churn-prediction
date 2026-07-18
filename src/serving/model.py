"""Model loading/prediction service, kept separate from the FastAPI app so it
can be unit-tested with a stub and hot-swapped via the /reload endpoint.
"""

import os
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

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        """Resolve and load the serving model.

        Model: env MODEL_URI overrides; default is the registry alias
        models:/<MODEL_NAME>@<alias> against MLFLOW_TRACKING_URI.
        Threshold: env DECISION_THRESHOLD > registry tag on the version > 0.5.
        """
        import mlflow

        self.model_uri = os.environ.get("MODEL_URI", f"models:/{MODEL_NAME}@{DEFAULT_ALIAS}")
        self.pipeline = mlflow.sklearn.load_model(self.model_uri)

        version_tags: dict[str, str] = {}
        if self.model_uri.startswith("models:/") and "@" in self.model_uri:
            client = mlflow.MlflowClient()
            name, alias = self.model_uri.removeprefix("models:/").split("@", 1)
            version = client.get_model_version_by_alias(name, alias)
            self.model_version = str(version.version)
            version_tags = version.tags

        env_threshold = os.environ.get("DECISION_THRESHOLD")
        if env_threshold is not None:
            self.threshold = float(env_threshold)
        elif "decision_threshold" in version_tags:
            self.threshold = float(version_tags["decision_threshold"])
        else:
            self.threshold = FALLBACK_THRESHOLD

    def predict(self, X: pd.DataFrame) -> tuple[list[float], list[bool]]:
        proba = self.pipeline.predict_proba(X)[:, 1]
        labels = proba >= self.threshold
        self._log_predictions(X, proba)
        return proba.tolist(), labels.tolist()

    def _log_predictions(self, X: pd.DataFrame, proba) -> None:
        """Append served rows to PREDICTION_LOG_PATH (CSV) — the "current"
        dataset for drift monitoring. No-op when the env var is unset.
        """
        log_path = os.environ.get("PREDICTION_LOG_PATH")
        if not log_path:
            return
        record = X.copy()
        record["churn_probability"] = proba
        record["predicted_at"] = datetime.now(UTC).isoformat()
        record["model_version"] = self.model_version
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record.to_csv(path, mode="a", header=not path.exists(), index=False)
