.PHONY: setup lint fmt test cov train train-promote mlflow-ui

setup:
	uv sync --all-groups

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
