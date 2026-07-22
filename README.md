# Telco Customer Churn — Data Science + MLOps (local stack)

Dự đoán khách hàng rời bỏ dịch vụ (Telco Customer Churn, IBM/Kaggle) — làm trọn vòng đời:
từ CRISP-ML(Q) notebook analysis đến một MLOps stack chạy được bằng một lệnh `docker compose up`.

## Kết quả chính

- **Model**: CatBoost (Optuna-tuned) trong sklearn Pipeline — holdout **PR-AUC 0.6365**, ROC-AUC 0.8369.
- **Decision threshold 0.465**: chọn bằng expected retention value tính bằng tiền (không phải 0.5 mặc định hay F2-optimal) — xem notebook 05.
- **MLOps**: training tái lập được, MLflow registry với alias `@champion`, FastAPI serving đọc threshold từ registry tag, DVC data versioning, Evidently drift monitoring với vòng retrain tự động, Prometheus + Grafana.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `notebooks/churn_classification/` | 5 notebooks CRISP-ML(Q): EDA → data prep → modeling → evaluation → business value & SHAP |
| `notebooks/survival_analysis/` | Pha mở rộng: time-to-churn (Kaplan-Meier, Cox PH) — bổ sung "khi nào" cho model classification "có/không", xem `docs/docs.md` |
| `src/` | Module tái sử dụng: data loading, preprocessing, persisted split, final model, `train.py`, serving, monitoring, `survival_analysis/` (Cox PH) |
| `configs/` | `train.yaml` (MLflow, threshold), `monitoring.yaml` (drift) |
| `tests/` + `.github/workflows/` | pytest (fixtures synthetic, không cần data thật) + CI matrix Python 3.12/3.14 |
| `docker/` + `docker-compose.yml` | mlflow, api, prometheus, grafana |
| `docs/docs.md` | Writeup data science theo từng pha |
| `docs/mlops.md` | Kiến trúc MLOps, runbook, design decisions, integration gotchas |

## Chạy nhanh

```bash
make setup                 # uv sync --all-groups (cần uv + Python 3.12)
uv run dvc pull            # lấy data (DVC remote)
docker compose up -d mlflow
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 make train-promote
docker compose up -d --build
curl localhost:8000/docs   # API; Grafana :3000, MLflow :5000, Prometheus :9090
```

Toàn bộ lệnh vận hành (train, drift check, retrain hook, ...) xem runbook trong [`docs/mlops.md`](docs/mlops.md).

## Vòng monitoring → retrain

Serving log mỗi request vào `prediction-logs/`; `make check-drift` so phân phối feature với train reference (Evidently), exit 1 khi >30% cột drift; `make retrain-if-drift` nối tiếp: retrain → promote `@champion` mới → `POST /reload` để API hot-swap không cần restart.
