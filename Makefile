.PHONY: setup download-data lint fmt test cov train train-promote mlflow-ui serve compose-up compose-down \
	check-drift simulate-drift retrain-if-drift

setup:
	uv sync --all-groups

download-data:
	uv run python scripts/download_dataset.py

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fmt:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

test:
	uv run pytest -q

cov:
	uv run pytest --cov=src --cov-report=term-missing

train:
	uv run python -m src.churn_classification.train --config configs/train.yaml

train-promote:
	uv run python -m src.churn_classification.train --config configs/train.yaml --promote

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

serve:
	MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db} \
		uv run uvicorn src.serving.app:app --port 8000

compose-up:
	# Pre-create bind-mount sources as the host user first — otherwise Docker
	# auto-creates missing ones as root, and the api container (non-root
	# appuser) can't write predictions.csv into a root-owned prediction-logs/.
	mkdir -p prediction-logs mlflow-data
	docker compose up -d --build

compose-down:
	docker compose down

check-drift:
	uv run python -m src.monitoring.drift

simulate-drift:
	uv run python -m src.monitoring.simulate_drift

retrain-if-drift:
	bash scripts/retrain_if_drift.sh
