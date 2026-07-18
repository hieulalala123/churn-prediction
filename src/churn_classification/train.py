"""Reproducible training entrypoint: fit the final pipeline on the persisted
train split, evaluate once on the holdout, log everything to MLflow, and
optionally promote the new model version to the serving alias.

Orchestration only — the split, preprocessing, and model definition all come
from the existing modules (data_split / preprocessing / final_model), so this
script can never drift from what the notebooks established.

Usage:
    python -m src.churn_classification.train --config configs/train.yaml [--promote]
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.churn_classification import data_split
from src.churn_classification.data_split import RANDOM_STATE, get_split
from src.churn_classification.final_model import BEST_PARAMS, build_final_pipeline
from src.churn_classification.preprocessing import compute_scale_pos_weight, split_X_y


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(pipeline: Pipeline, X, y, threshold: float) -> dict[str, float]:
    """Holdout metrics: ranking quality (PR-AUC primary, per notebook 03) plus
    operating-point metrics at the configured decision threshold.
    """
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "precision_at_threshold": precision_score(y, pred),
        "recall_at_threshold": recall_score(y, pred),
        "f1_at_threshold": f1_score(y, pred),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(config_path: str | Path, promote: bool = False) -> str:
    cfg = load_config(config_path)
    threshold = float(cfg["decision_threshold"])

    train_df, test_df = get_split()
    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    pipeline = build_final_pipeline(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test, threshold)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    registered_name = cfg["mlflow"]["registered_model_name"]

    with mlflow.start_run() as run:
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_params(
            {
                "scale_pos_weight": scale_pos_weight,
                "decision_threshold": threshold,
                "random_state": RANDOM_STATE,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_data_sha256": _file_sha256(data_split.TRAIN_PATH),
            }
        )
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_train.head(), pipeline.predict_proba(X_train.head())),
            input_example=X_train.head(3),
            registered_model_name=registered_name,
            # mlflow>=3 serializes sklearn models with skops, which requires
            # explicitly trusting non-sklearn classes inside the pipeline.
            skops_trusted_types=["catboost.core.CatBoostClassifier", "numpy.dtype"],
        )

    client = mlflow.MlflowClient()
    version = model_info.registered_model_version
    client.set_model_version_tag(registered_name, version, "decision_threshold", str(threshold))
    if promote:
        client.set_registered_model_alias(registered_name, cfg["mlflow"]["promote_alias"], version)

    metrics_path = Path(cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(
        f"Run {run.info.run_id}: registered {registered_name} v{version}"
        f"{' -> alias @' + cfg['mlflow']['promote_alias'] if promote else ''}"
    )
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}))
    return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="point the serving alias at the newly registered version",
    )
    args = parser.parse_args()
    main(args.config, promote=args.promote)
