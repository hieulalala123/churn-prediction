.PHONY: setup lint fmt test cov

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
