import mlflow
import pytest
import yaml

from src.churn_classification import data_split, train


@pytest.fixture()
def tmp_config(tmp_path):
    cfg = {
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
            "experiment_name": "test-churn",
            "registered_model_name": "test-model",
            "promote_alias": "champion",
        },
        "decision_threshold": 0.465,
        "metrics_path": str(tmp_path / "metrics.json"),
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_main_logs_run_and_registers_model(
    tmp_config, clean_df_sample, tmp_split_paths, monkeypatch
):
    config_path, cfg = tmp_config
    monkeypatch.setattr(data_split, "load_clean", lambda: clean_df_sample)

    run_id = train.main(config_path, promote=True)

    client = mlflow.MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    run = client.get_run(run_id)
    assert "pr_auc" in run.data.metrics
    assert 0.0 <= run.data.metrics["pr_auc"] <= 1.0
    assert run.data.params["decision_threshold"] == "0.465"

    version = client.get_model_version_by_alias("test-model", "champion")
    assert version.tags["decision_threshold"] == "0.465"

    assert (tmp_split_paths.parent / "metrics.json").exists()
