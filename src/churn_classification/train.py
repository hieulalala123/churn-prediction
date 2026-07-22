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
import os
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

# Captured once at import time, before any mlflow.set_tracking_uri() call can
# run. mlflow.set_tracking_uri() sets this same env var as a side effect (so
# subprocesses inherit it) — reading os.environ live inside main() would pick
# up that leftover value from a previous main() call in the same process
# (e.g. two tests, or a notebook retraining twice) instead of the config's
# own tracking_uri, silently pointing a later call at an earlier run's store.
_ENV_MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")


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


def _promote_if_not_worse(client, cfg: dict, version: str, metrics: dict[str, float]) -> bool:
    """Quality gate: only move the serving alias if the new model's holdout
    PR-AUC is not materially worse than the current champion's.

    Without this, --promote (and the automated drift->retrain hook calling
    it) would ship whatever the last run produced — including a model
    trained on corrupted data. The current champion stays serving when the
    gate refuses; a human can still promote manually via the MLflow UI.
    """
    import mlflow

    registered_name = cfg["mlflow"]["registered_model_name"]
    alias = cfg["mlflow"]["promote_alias"]
    max_regression = float(cfg.get("promote_max_regression", 0.02))

    try:
        champion = client.get_model_version_by_alias(registered_name, alias)
    except mlflow.exceptions.MlflowException:
        champion = None  # no champion yet -> first promotion is free

    if champion is not None and str(champion.version) != str(version):
        champion_pr_auc = client.get_run(champion.run_id).data.metrics.get("pr_auc")
        floor = champion_pr_auc * (1 - max_regression)
        if metrics["pr_auc"] < floor:
            print(
                f"PROMOTE REFUSED: new pr_auc {metrics['pr_auc']:.4f} < floor {floor:.4f} "
                f"(champion v{champion.version} pr_auc {champion_pr_auc:.4f}, "
                f"max_regression {max_regression}). Champion unchanged."
            )
            return False

    client.set_registered_model_alias(registered_name, alias, version)
    return True


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

    # Env var wins over config so `MLFLOW_TRACKING_URI=http://... make train-promote`
    # can target the docker-compose registry without editing the yaml.
    mlflow.set_tracking_uri(_ENV_MLFLOW_TRACKING_URI or cfg["mlflow"]["tracking_uri"])
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
    promoted = promote and _promote_if_not_worse(client, cfg, version, metrics)

    metrics_path = Path(cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(
        f"Run {run.info.run_id}: registered {registered_name} v{version}"
        f"{' -> alias @' + cfg['mlflow']['promote_alias'] if promoted else ''}"
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
