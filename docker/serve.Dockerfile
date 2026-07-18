# Serving image: core deps + serve group only (no notebooks/train tooling).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group serve --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-default-groups --group serve

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
