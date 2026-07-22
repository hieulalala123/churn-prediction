# CLAUDE.md — Telco Customer Churn Classification

Hướng dẫn làm việc trong repo này. Đọc trước khi sửa code trong `notebooks/`, `src/`, hoặc tạo pipeline mới.

## Bối cảnh dự án

- **Bài toán hiện tại**: binary classification dự đoán khách hàng rời bỏ dịch vụ (`Churn`: Yes/No) từ bộ dữ liệu Telco Customer Churn (IBM sample data, nguồn Kaggle `blastchar/telco-customer-churn`, tải qua `download_dataset.py`).
- **Tên repo là `survival-analysis`** — đây là chủ đích: `tenure` (số tháng gắn bó) + `Churn` (event) là cặp duration/event kinh điển cho survival analysis (Kaplan-Meier, Cox PH). Task classification hiện tại là bước đầu; khi bàn tới modeling nâng cao, cân nhắc đề xuất hướng survival nếu phù hợp với câu hỏi business (ví dụ "khi nào khách sẽ rời đi" thay vì chỉ "có rời đi không"), nhưng **không tự ý chuyển hướng** — task đang yêu cầu là classification.
- **Business framing cần giữ trong đầu**: false negative (bỏ sót khách sắp churn) thường tốn kém hơn false positive (chăm sóc nhầm khách trung thành) vì chi phí giữ chân < chi phí mất khách + acquisition cost mới. Điều này ảnh hưởng tới lựa chọn metric và threshold, không chỉ optimize accuracy.

## Dữ liệu

File: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7043 dòng, 21 cột.

Đã xác nhận từ EDA trong `notebooks/churn_classification/01_business_and_data_understanding.ipynb`:
- `customerID`: định danh, không có giá trị dự đoán — đã drop.
- `TotalCharges`: đọc vào là `object` do có 11 dòng chuỗi rỗng, cần `pd.to_numeric(errors='coerce')`. 11 dòng này trùng với 11 dòng có `tenure == 0` (khách mới, chưa có billing cycle) — hợp lý để loại khỏi tập train vì không đại diện cho pattern churn thực sự, không phải missing-at-random cần impute.
- `SeniorCitizen`: encode dạng 0/1, nên map về "No"/"Yes" cho nhất quán với các cột categorical binary khác trước khi encode lại cho model.
- Target `Churn` mất cân bằng: **73.5% No / 26.5% Yes** — bắt buộc phải xử lý imbalance (xem phần Modeling bên dưới), accuracy đơn thuần là metric gây hiểu lầm.
- 16 cột categorical (phần lớn Yes/No/No-<service>-service dạng 3 mức), 3 cột numeric (`tenure`, `MonthlyCharges`, `TotalCharges`).

## Trạng thái hiện tại

Phần data science đã **hoàn tất** theo CRISP-ML(Q), 5 notebooks trong `notebooks/churn_classification/` (01 EDA → 05 business value & SHAP; chi tiết trong `docs/docs.md`). Model cuối: CatBoost tuned bằng Optuna (holdout PR-AUC 0.6365, ROC-AUC 0.8369), decision threshold cost-based **0.465** chọn ở notebook 05.

Phần MLOps đã triển khai (chi tiết vận hành trong `docs/mlops.md`): training pipeline tái lập được (`train.py` + MLflow registry, alias `@champion`), FastAPI serving (`src/serving/`), DVC cho data + pipeline split→train, Evidently drift monitoring + retraining hook, Prometheus/Grafana, pytest + GitHub Actions CI, docker-compose đầy đủ stack.

## Tiêu chuẩn kỹ thuật khi làm modeling (expert-level)

- **Không leakage**: encode/scale phải fit trên train, transform trên test — dùng `sklearn.pipeline.Pipeline` + `ColumnTransformer`, không fit encoder/scaler trên toàn bộ `df` rồi mới split.
- **Split**: `train_test_split(..., stratify=y)` bắt buộc vì target imbalance.
- **Encoding categorical**: OneHotEncoder cho nominal categories (không có thứ tự); tránh LabelEncoder cho features nhiều mức (nó áp đặt ordinal giả). LabelEncoder chỉ hợp lý cho target binary hoặc feature thực sự ordinal (ví dụ Contract: month-to-month < one year < two year có thể encode ordinal nếu muốn giữ thông tin thứ tự).
- **Xử lý imbalance**: ưu tiên `class_weight='balanced'` (LogisticRegression, RandomForest, SVC) hoặc `scale_pos_weight` (XGBoost/CatBoost) trước khi nhảy sang resampling (SMOTE) — đơn giản hơn, ít rủi ro overfit synthetic samples. Nếu dùng SMOTE, phải áp dụng **sau** khi split, chỉ trên tập train, trong pipeline (dùng `imblearn.pipeline.Pipeline` để tránh leakage qua CV folds).
- **Metric**: không dùng accuracy làm tiêu chí chính. Ưu tiên ROC-AUC, PR-AUC (informative hơn với imbalance), recall/precision cho lớp Yes (churn), và F1 hoặc F-beta nếu cần cân bằng theo chi phí business đã nêu ở trên. Luôn xem confusion matrix, không chỉ điểm số tổng hợp.
- **Cross-validation**: `StratifiedKFold` khi so sánh model hoặc tune hyperparameter, không chỉ đánh giá trên một train/test split.
- **Model selection**: baseline nên là LogisticRegression (interpretable, nhanh) để có điểm so sánh, sau đó thử tree-based ensemble (RandomForest, XGBoost, CatBoost — CatBoost xử lý categorical trực tiếp, đáng cân nhắc là candidate mạnh cho bộ data này vì nhiều cột categorical). Không cần chạy toàn bộ danh sách model đã import — chọn có chủ đích, so sánh bằng CV, không phải "thử hết rồi lấy số cao nhất" (dễ overfit trên leaderboard).
- **Interpretability**: sau khi có model tốt, dùng feature importance (tree-based) hoặc SHAP để giải thích driver của churn — đây là bài toán business cần insight, không chỉ predictive score.
- **Threshold tuning**: mặc định 0.5 hiếm khi tối ưu với imbalanced target — cân nhắc chọn threshold theo precision-recall curve dựa trên chi phí business thực tế nếu có thông tin, hoặc tối ưu theo F-beta.

## Cấu trúc repo

- `notebooks/churn_classification/` — 5 notebooks CRISP-ML(Q) đã hoàn tất.
- `src/data/` — load/clean dataset dùng chung (`load_clean`).
- `src/churn_classification/` — preprocessing, persisted split (`get_split`, seed 42), final model (`build_final_pipeline`), training entrypoint (`train.py`).
- `src/serving/` — FastAPI app + `ModelService` (load `@champion` từ MLflow registry).
- `src/monitoring/` — Evidently drift check + drift simulator.
- `configs/` — `train.yaml` (threshold 0.465, MLflow config), `monitoring.yaml`.
- `tests/` — pytest với fixtures synthetic (không cần CSV thật); CI chạy matrix 3.12/3.14.
- `docker/` + `docker-compose.yml` — mlflow, api, prometheus, grafana.
- `data/` — raw CSV do **DVC** quản lý (`dvc pull` để lấy về; git chỉ giữ `.dvc` pointer).
- `docs/docs.md` — writeup CRISP-ML(Q); `docs/mlops.md` — kiến trúc + runbook MLOps.

## Môi trường & quy ước MLOps

- **`uv`** quản lý deps qua `pyproject.toml` + `uv.lock` (không còn requirements.txt). Groups: `dev`/`train`/`serve`/`monitor`/`survival`/`notebooks`. `make setup` = `uv sync --all-groups`.
- **Python pin 3.12** (`.python-version`, Docker) vì shap→numba/llvmlite chưa có wheel 3.14 — xem `docs/mlops.md` phần tương thích.
- Mọi thao tác thường dùng đều có Make target — xem runbook trong `docs/mlops.md`.
- Model artifacts nằm trong MLflow registry (`mlflow.db`/`mlruns` local hoặc container `mlflow-data/`), **không** commit vào git; `models/` không dùng.
- Threshold 0.465 là source of truth ở `configs/train.yaml`, stamp lên registry tag mỗi lần train, serving tự đọc — đừng hardcode 0.5 ở bất kỳ đâu.
