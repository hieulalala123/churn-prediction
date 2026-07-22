#!/usr/bin/env bash
# Retraining hook: run the drift check; on drift (exit 1), retrain + promote
# a new @champion and hot-reload the serving API.
# Manual-trigger by design for this local setup — in production this would be
# a scheduled job (cron/Airflow) with alerting instead of an immediate retrain.
set -uo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

if uv run python -m src.monitoring.drift "$@"; then
    echo "No drift — nothing to do."
    exit 0
fi

echo "Drift detected — retraining and promoting a new champion..."
uv run python -m src.churn_classification.train --config configs/train.yaml --promote
curl -sf -X POST "$API_URL/reload" && echo && echo "API reloaded."
