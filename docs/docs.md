# Project Documentation — Telco Customer Churn Classification

Tài liệu sống, cập nhật theo tiến độ CRISP-ML(Q). Mỗi Pha được chốt (đã thảo luận và thống nhất với stakeholder) mới được ghi vào đây — không viết trước nội dung của các Pha chưa thực hiện.

---

## Pha 1 — Business & Data Understanding

### 1.1 Bài toán

- **Loại bài toán**: Binary classification. Dự đoán $P(\text{Churn}=1 \mid X)$ cho từng khách hàng tại một thời điểm snapshot.
- **Không phải** time-to-event/survival analysis ở giai đoạn này (dù tên repo gợi ý hướng đó) — `tenure`/`Churn` để dành cho một pha mở rộng sau này nếu câu hỏi business chuyển từ "có rời đi không" sang "khi nào rời đi".
- **Đơn vị quan sát**: 1 dòng = 1 khách hàng, tại 1 thời điểm duy nhất. Đã xác nhận bằng dữ liệu: `customerID` có 7043/7043 giá trị unique trên 7043 dòng — không có khách hàng lặp lại theo thời gian. Đây là cơ sở để dùng Stratified K-Fold thông thường ở Pha 2 (không cần time-series split hay group-aware split theo customer).

### 1.2 Use case & Cost matrix

Khi model gắn cờ một khách hàng là "rủi ro churn cao", team CS gửi ưu đãi/khuyến mãi (chi phí trung bình — không miễn phí nhưng cũng không phải can thiệp cấp cao đắt đỏ).

|                       | Thực tế: No Churn                                                                                                     | Thực tế: Churn                                                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dự đoán: No Churn** | True Negative — đúng                                                                                                  | **False Negative** — mất khách thật, mất LTV còn lại + chi phí acquisition khách thay thế. **Đắt hơn FP nhiều lần** (đã xác nhận với stakeholder). |
| **Dự đoán: Churn**    | **False Positive** — tốn 1 ưu đãi cho khách vốn đã trung thành, lãng phí ngân sách marketing nhưng không nghiêm trọng | True Positive — đúng, có cơ hội giữ khách                                                                                                          |

Cost ratio chính xác (FN/FP) chưa có số liệu cụ thể từ business — sẽ ước lượng gián tiếp qua Precision-Recall curve ở Pha 4 thay vì áp một con số giả định ngay từ đầu. **Cập nhật**: đến hết Pha 4 việc này vẫn **chưa thực hiện** (để ngỏ có chủ đích — xem mục 4.3), do ưu tiên đi sâu error analysis trước.

### 1.3 Data snapshot đã xác nhận

Nguồn: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (Kaggle `blastchar/telco-customer-churn`, tải qua `scripts/download_dataset.py`).

- Kích thước: 7043 dòng × 21 cột.
- Target `Churn`: **imbalanced — 73.5% No / 26.5% Yes**.
- `customerID`: định danh, unique 100%, không có giá trị dự đoán. **Cập nhật ở Pha 2**: giữ lại trong dataframe (không drop trong `clean()`) để phục vụ leak-check theo identity thật và truy vết ở Pha 4 error analysis; mỗi task tự loại nó khỏi feature set X một cách tường minh (xem `split_X_y()` trong `src/churn_classification/preprocessing.py`) thay vì dựa vào việc loader đã xóa sẵn.
- `TotalCharges`: bị đọc vào dưới dạng `object` do 11 dòng chứa chuỗi rỗng (whitespace) thay vì số — cần `pd.to_numeric(errors='coerce')`. 11 dòng này trùng khớp hoàn toàn với 11 dòng có `tenure == 0` (khách hàng mới, chưa qua billing cycle đầu tiên) → không phải missing ngẫu nhiên, mà là artefact của logic nghiệp vụ (chưa có total charge vì chưa từng bị tính phí).
- `SeniorCitizen`: encode 0/1 thay vì "No"/"Yes" như các cột binary categorical khác trong bộ dữ liệu — cần chuẩn hóa để nhất quán trước khi đưa vào pipeline encode.
- 16 cột categorical (phần lớn dạng Yes/No/"No <service>"), 3 cột numeric: `tenure`, `MonthlyCharges`, `TotalCharges`.

### 1.4 Primary metric

- **Accuracy bị loại bỏ khỏi vai trò primary metric**: baseline "luôn dự đoán No Churn" đạt 73.5% accuracy mà bỏ sót 100% khách churn thật — tối đa hóa đúng loại lỗi tốn kém nhất (FN) mà accuracy không có khả năng phân biệt.
- **ROC-AUC bị loại khỏi vai trò primary metric**: FPR = FP/(FP+TN) có mẫu số lớn do lớp âm chiếm 73.5% và dễ đoán đúng → ROC-AUC có xu hướng optimistic bias khi lớp dương là thiểu số, không phản ánh đúng độ khó thực sự của việc tìm khách churn. Vẫn được **báo cáo như metric tham chiếu phụ**, không dùng để chọn model.
- **Primary metric để so sánh/chọn model (threshold-independent)**: **PR-AUC (Average Precision)**. Precision = TP/(TP+FP) không có TN trong mẫu số → nhạy trực tiếp với loại lỗi cần theo dõi (FP trong nhóm bị gắn cờ churn), và baseline ngẫu nhiên của PR-AUC bằng đúng tỷ lệ lớp dương (0.265) — cho một mốc so sánh trung thực với chính bộ dữ liệu này.
- **Metric để chọn operating threshold khi triển khai**: **F2-score** (F-beta, β=2) trên lớp Churn — trọng số recall gấp đôi precision, phản ánh "FN đắt hơn FP nhiều lần" nhưng vẫn phạt FP vì ưu đãi không miễn phí. Giá trị β sẽ được tinh chỉnh lại ở Pha 4 khi có Precision-Recall curve thực tế và có thể mô phỏng chi phí theo từng threshold.
- **Metric giám sát/chẩn đoán phụ** (không dùng để ra quyết định): Confusion Matrix, Recall/Precision riêng lẻ cho lớp Churn.

---

## Pha 2 — Data Preparation & Validation Strategy

### 2.1 Validation strategy

- Đơn vị quan sát là 1 khách hàng/1 thời điểm, không có time dimension, không có customer lặp lại (đã xác
  nhận ở Pha 1) → không cần time-series split hay group-aware split. **StratifiedKFold** là lựa chọn đúng.
- **Holdout test set tách riêng khỏi CV**: cắt 15% dữ liệu (stratified theo `Churn`, `random_state=42`) làm
  holdout **không đụng tới cho tới Pha 4**. Lý do: nếu dùng chung CV score vừa để chọn model/tune
  hyperparameter vừa để báo cáo kết quả cuối, con số cuối sẽ lạc quan giả tạo (selection bias). 15% (thay vì
  20–25% phổ biến hơn) vì dataset chỉ 7032 dòng — holdout lớn hơn sẽ thu hẹp train pool dùng cho CV nhiều hơn
  mức cải thiện được độ chính xác của metric holdout; 15% vẫn để lại ~280 mẫu dương trong holdout, đủ cho một
  ước lượng PR-AUC tương đối ổn định.
- Split được **persist ra đĩa** (`data/processed/churn_classification/{train,test}.csv`, gitignored — tái
  tạo được từ raw + seed cố định) để không bị vô tình đổi khác giữa các lần chạy/notebook. `get_split()`
  trong `src/churn_classification/data_split.py` tự tạo lần đầu, các lần sau load lại y nguyên và **in cảnh
  báo rõ ràng** nếu đang dùng bản cache (tránh bẫy: đổi `TEST_SIZE`/`RANDOM_STATE` sau này mà tưởng có tác
  dụng ngay).
- Trong train pool: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` cho mọi model
  selection/hyperparameter tuning ở Pha 3. Đã verify thực tế: pos-rate mỗi fold dao động 0.2653–0.2661 (rất
  sát tỷ lệ tổng thể 0.2659).
- `customerID` được **giữ lại** trong dataframe qua toàn bộ pipeline load → split (không drop trong
  `clean()` như dự định ban đầu ở Pha 1) để dùng làm định danh leak-check thật và để truy vết ở Pha 4 error
  analysis. Mỗi task tự loại nó khỏi feature set X một cách tường minh (`split_X_y()` trong
  `src/churn_classification/preprocessing.py`), không dựa vào việc loader đã xóa sẵn.
  - Lưu ý quan trọng phát hiện được: leak-check ban đầu dựa trên so khớp toàn bộ dòng giữa train/test đã báo
    lỗi giả, vì dataset categorical-heavy này có một số khách hàng khác nhau trùng y hệt feature profile một
    cách ngẫu nhiên (7 cặp, đã verify). Leak-check đúng phải dựa trên identity (`customerID`), không phải
    so khớp giá trị dòng.

### 2.2 Feature typing & preprocessing pipeline

- Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`. Categorical: 16 cột còn lại (khai báo **tường minh**
  trong `preprocessing.py`, không dùng `df.select_dtypes` tự động, để tránh pipeline âm thầm đổi hành vi nếu
  dtype của một cột đổi ở lần load dữ liệu sau).
- **Phát hiện cấu trúc quan trọng**: 6 cột (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
  `TechSupport`, `StreamingTV`, `StreamingMovies`) dùng giá trị `"No internet service"` thay vì `"No"` bất cứ
  khi nào `InternetService == "No"` → collinear hoàn hảo với nhau và với `InternetService` tại các dòng đó.
  Quyết định: không xử lý đặc biệt (dựa vào L2 regularization mặc định của LogisticRegression, tree-based
  models không bị ảnh hưởng) — chỉ ghi chú để không hiểu sai feature importance/coefficient ở Pha 3.
- Pipeline (`build_preprocessor()`): numeric → `SimpleImputer(median)` + `StandardScaler`; categorical →
  `SimpleImputer(most_frequent)` + `OneHotEncoder(drop='if_binary', handle_unknown='ignore')`. Median (không
  phải mean) vì `MonthlyCharges`/`TotalCharges` lệch phải (đã xác nhận ở Pha 1). Imputer là safety net phòng
  dữ liệu tương lai có missing, dữ liệu train hiện tại không còn missing sau `clean()`. `StandardScaler` áp
  dụng thống nhất kể cả cho tree-based model vì đây là transform đơn điệu theo từng feature, không đổi tree
  split — một preprocessor phục vụ được cả baseline tuyến tính lẫn tree-based ở Pha 3.
- **Nguyên tắc bắt buộc ở Pha 3**: `preprocessor` phải được fit **bên trong từng CV fold** (bọc chung với
  model qua `sklearn.pipeline.Pipeline`), không fit một lần trên toàn bộ train pool rồi tái sử dụng cho mọi
  fold — đó vẫn là leakage dù nhìn qua tưởng vô hại (thống kê impute/scale của fold validation sẽ rò vào lúc
  fit).

### 2.3 Xử lý class imbalance

- Ratio neg/pos (`scale_pos_weight`) tính **trên train pool** (không phải toàn bộ `df` hay test set, kể cả
  một con số tỷ lệ cũng là leakage nếu tính trên test): hiện tại ≈ 2.76.
- Quyết định: dùng `class_weight='balanced'` (LogisticRegression/RandomForest — sklearn tự tính nội bộ,
  không cần con số thủ công) hoặc `scale_pos_weight` (XGBoost/LightGBM/CatBoost — cần truyền con số vừa
  tính) ở Pha 3. **Cập nhật ở Pha 3**: `scale_pos_weight` được tính **một lần trên toàn bộ train pool** rồi
  dùng cố định cho mọi fold CV, thay vì tính lại riêng từng fold như dự kiến ban đầu ở đây — vì tỷ lệ lớp
  dương giữa các fold gần như đồng nhất (đã verify ở mục 2.1: 0.2653–0.2661), nên con số tổng train pool là
  xấp xỉ rất tốt cho mọi fold, và tính lại riêng từng fold chỉ thêm phức tạp mà không đổi kết quả đáng kể.
  **Không dùng SMOTE**, vì: (1) SMOTE nội suy tuyến tính giữa
  các one-hot vector trên dữ liệu phần lớn categorical tạo ra điểm "không tồn tại trong thực tế nghiệp vụ",
  rủi ro overfit vào artefact resampling; (2) `class_weight` chỉ đổi trọng số trong loss, không tạo/xóa dữ
  liệu, đơn giản hơn và không cần `imblearn.pipeline.Pipeline` để tránh resampling leak qua CV fold; (3)
  cost-sensitivity đã được xử lý nhất quán ở tầng metric/threshold (F2-score, Pha 1) — xử lý thêm ở tầng loss
  là hợp lý hơn là bóp méo phân phối dữ liệu bằng resampling.
- Nếu Pha 3 cho thấy `class_weight` không đủ cải thiện recall lớp Yes, sẽ cân nhắc SMOTE như phương án thứ
  hai — không áp dụng đồng thời cả hai để giữ khả năng quy kết nguyên nhân khi so sánh kết quả. **Cập nhật**:
  không cần fallback này — recall lớp Churn trên test set đạt 94% (Pha 4, mục 4.2) với `scale_pos_weight`
  đơn thuần.

## Pha 3 — Baseline & Iterative Modeling

Toàn bộ so sánh dưới đây chỉ dùng `train_df` (train pool, 5977 dòng) qua `StratifiedKFold(5)`; `test_df`
(holdout) không được nhìn tới ở Pha này. Metric chọn model: **PR-AUC** (theo `docs/docs.md` Pha 1).

### 3.1 Baseline & sanity check

- `DummyClassifier(strategy="stratified")`: PR-AUC = 0.2654 ≈ prevalence (0.2659) — xác nhận thực nghiệm
  đúng lý thuyết đã lập luận ở Pha 1 ("PR-AUC baseline ngẫu nhiên = tỷ lệ lớp dương"). Dùng làm sanity check
  cho toàn bộ pipeline CV/scoring trước khi tin bất kỳ model thật nào.
- `LogisticRegression(class_weight="balanced")`: PR-AUC = 0.6648, overfit_gap ≈ 0.006 (rất ổn định) — baseline
  thật, làm mốc so sánh cho các model phức tạp hơn.

### 3.2 Ba vòng so sánh có kiểm soát (cùng preprocessing, CV, scoring — chỉ đổi model/hyperparameter)

1. **Vòng 1 — tree model tay chọn, chưa tune** (`max_depth=6, n_estimators=300`): LightGBM/XGBoost/CatBoost
   đều **thua** Logistic Regression trên PR-AUC, và overfit_gap rất lớn (0.14–0.26) — cấu hình quá mạnh so
   với train pool chỉ ~6000 dòng, không phải bằng chứng "tree không phù hợp với bài toán này".
2. **Vòng 2 — regularize thủ công** (`max_depth=4, n_estimators=150`, thêm L1/L2, `min_child_samples`/
   `min_data_in_leaf` cao hơn, `subsample=0.8`): overfit_gap giảm mạnh (CatBoost 0.145→0.027, LightGBM
   0.226→0.073, XGBoost 0.264→0.073); CatBoost (reg) vượt Logistic Regression, PR-AUC = 0.6764.
3. **Vòng 3 — Optuna (TPE sampler, 25 trial/model, tối ưu PR-AUC qua `cross_val_score`)**: tune **cả 4 họ
   model** (kể cả Logistic Regression) để đảm bảo so sánh công bằng — không chỉ tune lại phần "trông có vẻ
   kém" ở vòng trước. Không dùng pruning (ngân sách search khiêm tốn, dataset nhỏ nên mỗi trial đã nhanh —
   quyết định phạm vi có chủ đích).

### 3.3 Kết quả cuối — bảng so sánh đầy đủ (PR-AUC, CV mean)

| Model | PR-AUC | overfit_gap |
|---|---|---|
| **CatBoost (Optuna-tuned)** | **0.6768** | 0.0257 |
| CatBoost (regularize thủ công) | 0.6764 | 0.0271 |
| XGBoost (Optuna-tuned) | 0.6759 | 0.0300 |
| LightGBM (Optuna-tuned) | 0.6718 | 0.0487 |
| LogisticRegression (Optuna-tuned) | 0.6661 | 0.0037 |
| LogisticRegression (balanced, mặc định) | 0.6648 | 0.0058 |
| XGBoost (regularize thủ công) | 0.6708 | 0.0726 |
| LightGBM (regularize thủ công) | 0.6703 | 0.0727 |
| CatBoost (vòng 1, chưa tune) | 0.6610 | 0.1450 |
| LightGBM (vòng 1, chưa tune) | 0.6503 | 0.2263 |
| XGBoost (vòng 1, chưa tune) | 0.6473 | 0.2637 |
| Dummy (no-skill) | 0.2654 | 0.0009 |

Optuna **xác nhận** cấu hình tay chọn cho CatBoost đã gần tối ưu (0.6764→0.6768, chênh trong sai số CV);
cải thiện thật sự đáng kể ở XGBoost và LightGBM — PR-AUC tăng **và** overfit_gap giảm đồng thời (không phải
đánh đổi), chứng tỏ search tìm ra điểm cân bằng tốt hơn phỏng đoán tay, không chỉ học thuộc train pool tốt
hơn.

### 3.4 Model cuối cùng được chọn

**CatBoost**, PR-AUC (CV) = 0.6768, overfit_gap = 0.0257. Hyperparameter (từ Optuna, xem
`src/churn_classification/final_model.py::BEST_PARAMS`): `max_depth=4, n_estimators=239,
learning_rate≈0.0267, l2_leaf_reg≈1.046, min_data_in_leaf=35, subsample≈0.868`, cộng
`bootstrap_type="Bernoulli"` và `scale_pos_weight` tính trên train pool (~2.76, tính lại động, không hardcode
— xem `compute_scale_pos_weight`).

### 3.5 Feature importance & đối chiếu EDA

Top feature của model cuối: `tenure`, `Contract_Month-to-month`/`Two year`, `TotalCharges`,
`InternetService_Fiber optic`, `MonthlyCharges`, `TechSupport_No`, `OnlineSecurity_No`,
`PaymentMethod_Electronic check`, `PaperlessBilling_Yes` — khớp nhất quán với quan sát trực quan ở Pha 1
(khách churn: tenure thấp, hợp đồng ngắn hạn, fiber optic, electronic check, thiếu dịch vụ bảo vệ). Không
phát hiện bất thường cần điều tra thêm.

**Lưu ý diễn giải quan trọng** (phát hiện được khi Logistic Regression còn dẫn đầu bảng, vẫn đáng ghi nhớ
nếu sau này quay lại đọc hệ số tuyến tính): hệ số `MonthlyCharges` của Logistic Regression mang dấu **âm**
(ngược với xu hướng dương thấy ở KDE plot Pha 1) — không phải lỗi, mà do multicollinearity với
`InternetService`/`Contract`/`TotalCharges`: hệ số hồi quy phản ánh hiệu ứng riêng phần sau khi đã kiểm soát
các biến tương quan, khác với hiệu ứng biên (marginal) mà EDA đơn biến cho thấy. Cẩn trọng khi diễn giải hệ
số tuyến tính cho business trong bối cảnh nhiều feature tương quan.

## Pha 4 — Offline Evaluation & Error Analysis

`test_df` (1055 dòng, holdout) được chạm tới **đúng một lần** trong toàn bộ dự án, ở Pha này. Model đánh giá:
CatBoost (tuned) đã chốt ở Pha 3, refit trên toàn bộ `train_df` (không giới hạn theo fold).

### 4.1 Chọn operating threshold — không dùng test set

Threshold chọn bằng cách quét trên xác suất **out-of-fold** (`cross_val_predict` trên `train_df`, cùng
`StratifiedKFold(5)` đã dùng ở Pha 3), tối đa hóa **F2** (β=2, quyết định đã chốt ở Pha 1). Không dùng
`test_df` để chọn threshold — nếu không, đó vẫn là một dạng nhìn trộm test set dưới vỏ bọc khác.

- **Threshold đã chọn: 0.295** (thấp hơn nhiều so với mặc định 0.5 — hệ quả tự nhiên của việc tối ưu F2 kết
  hợp `scale_pos_weight` đã áp lúc train).

### 4.2 Kết quả trên test set (một lần, không lặp lại)

| Metric | CV (Pha 3) | Test | z-score |
|---|---|---|---|
| PR-AUC | 0.6768 ± 0.0260 | 0.6365 | -1.55 |
| ROC-AUC | 0.8504 ± 0.0136 | 0.8369 | -0.99 |

(z-score = (Test − CV mean) / CV std — số độ lệch chuẩn mà kết quả test lệch khỏi trung bình 5-fold CV;
|z| ≤ ~2 được coi là nằm trong biến động ngẫu nhiên bình thường, không phải dấu hiệu bất thường.)

ROC-AUC lệch ~1 std — bình thường, trong phạm vi biến động fold-to-fold. PR-AUC lệch nhiều hơn (thấp hơn cả
fold thấp nhất từng quan sát ở Pha 3: 0.6505), nhưng z=-1.55 vẫn trong 2 độ lệch chuẩn — chưa đủ bằng chứng
nghi ngờ leakage (đã kiểm tra kỹ ở Pha 2-3), nhiều khả năng là biến động tự nhiên của holdout set không lớn
(1055 dòng, chỉ 280 mẫu dương — PR-AUC vốn nhạy hơn ROC-AUC với cỡ mẫu dương nhỏ).

Classification report tại threshold=0.295: **recall lớp Churn = 94%**, **precision = 41%**.

### 4.3 Vấn đề thực tiễn chưa giải quyết — tỷ lệ gắn cờ quá cao

Confusion matrix: TP=263, FP=372, FN=17, TN=403 → **635/1055 khách hàng (60%) bị gắn cờ "rủi ro cao"**,
trong đó 372/635 (59%) thực ra không rời đi. Đây là hệ quả trực tiếp của việc tối ưu F2 với β=2 (ưu tiên
recall gấp 4 lần precision) — đúng theo nguyên tắc "FN đắt hơn FP" đã chốt ở Pha 1, nhưng con số 60% có thể
vượt ngân sách/năng lực vận hành thực tế của team CS.

**Chưa giải quyết, để ngỏ có chủ đích**: giá trị β chưa được tinh chỉnh lại theo số liệu Pha 4 (đã đề xuất
nhưng stakeholder chọn ưu tiên phân tích FN chi tiết trước). Nếu quay lại, cách làm đúng là quét lại β/
threshold trên **out-of-fold train pool** (không dùng test set) rồi mới áp 1 lần lên test — không được chọn
β bằng cách thử nhiều giá trị trực tiếp trên test set.

### 4.4 Error analysis — 17 khách hàng False Negative

Chia 2 nhóm theo khoảng cách `churn_proba` tới threshold:

- **Near-miss (8 khách, proba 0.22–0.28)**: model gần đúng, chỉ chưa đủ tự tin — nhiễu quanh ranh giới quyết
  định, không phải vấn đề nghiêm trọng.
- **Confident miss (9 khách, proba 0.04–0.20)**: nhóm đáng chú ý hơn — model gần như chắc chắn (và sai) rằng
  những khách này sẽ ở lại.

**Pattern chung trên toàn bộ 17 khách FN** (đã đếm lại bằng code, không phải ước lượng bằng mắt):
`TechSupport = "Yes"` ở **13/17** khách (76%, và 7/9 nếu chỉ tính riêng nhóm confident-miss) —
đây đúng là feature model đã học thành "tín hiệu an toàn" mạnh (từ feature importance Pha 3, `TechSupport_No`
là chỉ báo churn dương) — nhóm này "trông trung thành" trên giấy tờ nhưng vẫn rời đi, model bị chính tín
hiệu tưởng là bảo vệ đánh lừa. (`OnlineSecurity = "Yes"` riêng lẻ chỉ 10/17; cả hai cột cùng "Yes" đồng thời
chỉ 9/17 — hai điều kiện không trùng nhau hoàn toàn, tách riêng ra để tránh phóng đại pattern.)

**Outlier đáng chú ý nhất**: khách `4464-JCOLN` — tenure 2 tháng, `InternetService=No`, gói rẻ nhất
(`MonthlyCharges=19.85`), dự đoán sai tự tin nhất trong cả nhóm (`churn_proba=0.038`). Cột
`"No internet service"` được model học là tín hiệu an toàn mạnh (dựa trên đa số dòng có giá trị này là
khách gắn bó lâu, ít dùng dịch vụ phát sinh), nhưng ở một khách hàng mới/cam kết thấp, "không dùng internet"
nhiều khả năng mang ý nghĩa ngược lại — chi phí chuyển đổi thấp, dễ rời đi — model không phân biệt được hai
ý nghĩa khác nhau của cùng một giá trị cột.

**Kết luận**: cả hai pattern trên là **giới hạn của feature hiện có** trong bộ dữ liệu công khai này (không
có lịch sử thay đổi giá, sự kiện gia hạn hợp đồng, ticket hỗ trợ/khiếu nại, dữ liệu cạnh tranh khu vực) —
không phải lỗi có thể sửa bằng tune thêm model/threshold.

### 4.5 Calibration — không cần cho cách dùng hiện tại

PR-AUC/ROC-AUC/threshold F2-optimal đều bất biến với phép biến đổi đơn điệu của điểm số, và threshold được
tune thực nghiệm trực tiếp (không giả định `churn_proba` là xác suất thật) → calibration không ảnh hưởng
tới quy trình ra quyết định hiện tại (gắn cờ/không gắn cờ theo threshold cố định).

**Lưu ý cho tương lai**: `scale_pos_weight` (kỹ thuật reweight loss dùng để xử lý imbalance, Pha 2) có tác
dụng phụ đã biết là làm `predict_proba` lệch lên (model báo xác suất churn cao hơn thực tế). Nếu sau này cần
dùng `churn_proba` cho việc đòi hỏi con số xác suất có ý nghĩa thật (ví dụ tính "giá trị kỳ vọng rủi ro" =
proba × LTV để ưu tiên nguồn lực), cần áp `CalibratedClassifierCV` (Platt scaling/isotonic) trước.

### 4.6 Kết thúc chu trình CRISP-ML(Q) Pha 1–4

Theo đúng phạm vi đã thống nhất từ đầu: **MLflow/experiment tracking/monitoring không nằm trong giai đoạn
này** — đây là điểm dừng của một baseline + experiment vững chắc, có tài liệu hóa đầy đủ quyết định và lý do
qua từng Pha, sẵn sàng cho bước tiếp theo (tracking/deployment, hoặc mở rộng sang survival analysis theo tên
repo) khi cần.

---

## Pha 5 — Business Value & Explainability

Mở rộng sau Pha 4 (ngoài CRISP-ML(Q) gốc, nối tiếp tự nhiên khi baseline đã vững). Notebook:
`notebooks/churn_classification/05_business_impact_and_explainability.ipynb`. Dùng lại nguyên `final_pipeline`
(CatBoost tuned, Pha 3), không train lại/đổi model.

### 5.1 Cost-based threshold — thay F-score bằng giá trị kỳ vọng bằng tiền

Dataset không có số liệu chi phí/lợi ích thật (đã ghi nhận từ Pha 1) → xây mô hình giá trị với giả định
**minh bạch, kiểm tra độ nhạy**, thay vì tin một con số duy nhất:

- `OFFER_COST = $50` (chi phí ưu đãi giữ chân), `RETAINED_MONTHS = 12`, `RETENTION_SUCCESS_RATE = 0.30`
  (xác suất ưu đãi thành công khi liên hệ đúng khách sắp churn) — giả định minh họa, không phải số liệu
  thật của bộ dữ liệu.
- `MonthlyCharges` từng khách dùng làm proxy doanh thu → giá trị kỳ vọng khác nhau theo từng khách, không
  còn là tỷ lệ FN/FP cố định chung cho mọi người như F-beta.
- Giá trị kỳ vọng theo outcome: TP = `success_rate × retained_months × MonthlyCharges − offer_cost`;
  FP = `−offer_cost`; không gắn cờ (FN/TN) = `0` (so với baseline "không làm gì").

**Threshold chọn trên OOF (train pool), không dùng test set** — đúng nguyên tắc đã chốt ở Pha 4.

| Threshold | Giá trị kỳ vọng (OOF) |
|---|---|
| **0.465 (tối ưu $)** | **$232,017** |
| 0.5 (mặc định) | $230,102 |
| 0.295 (F2-optimal, Pha 4) | $225,603 |

**Phát hiện quan trọng nhất của Pha 5**: threshold F2 chọn ở Pha 4 (0.295) thực ra kém hơn cả threshold mặc
định 0.5 về mặt tài chính — F2 (β=2) áp đặt "recall quan trọng gấp 4 lần precision" một cách trừu tượng,
không biết mỗi FP tốn $50 thật và mỗi TP chỉ đáng giá khi nhân với xác suất thành công 30%. Gắn cờ quá tay
(60% khách hàng như ở Pha 4) phá hủy giá trị nhiều hơn là tạo ra. **Bài học**: tối ưu một metric thống kê
không đảm bảo tối ưu giá trị kinh doanh.

**Sensitivity analysis** (quét `success_rate` × `offer_cost`): threshold tối ưu tăng khi ưu đãi đắt hơn,
giảm khi ưu đãi hiệu quả hơn — biến thiên có trật tự, đúng logic kinh tế, không dao động bất thường → tăng
độ tin cậy vào cách tiếp cận dù giả định gốc có thể chưa chính xác.

**Test set (một lần)** tại threshold=0.465: model tạo thêm **$38,290** so với không làm gì, và **$16,453**
nhiều hơn so với gắn cờ toàn bộ khách hàng (không dùng model để lọc) — con số cụ thể, dễ trình bày cho
business hơn nhiều so với "PR-AUC = 0.68".

### 5.2 Explainable AI — 5 phương pháp độc lập, kiểm định chéo lẫn nhau

| Phương pháp | Loại | Cơ chế |
|---|---|---|
| `feature_importances_` (Pha 3) | Global | Gain khi split, nội bộ cây |
| SHAP (`TreeExplainer`) | Global + Local | Giá trị Shapley (game theory), dựa cấu trúc cây |
| Permutation Importance | Global | Xáo trộn feature gốc, đo PR-AUC giảm — model-agnostic, không đụng nội bộ model |
| LIME | Local | Fit model tuyến tính cục bộ quanh 1 điểm dữ liệu — model-agnostic |
| PDP/ICE | Global (hình dạng quan hệ) | Quan hệ hàm số feature ↔ dự đoán trung bình, giữ các feature khác cố định |

**Top feature theo SHAP** (mean |SHAP|, train pool): `tenure` (0.469), `Contract_Month-to-month` (0.425),
`InternetService_Fiber optic` (0.270), `Contract_Two year` (0.248), `PaymentMethod_Electronic check` (0.192),
`TechSupport_No` (0.188), `OnlineSecurity_No` (0.180), `MonthlyCharges` (0.178), `PaperlessBilling_Yes`
(0.158), `TotalCharges` (0.149) — khớp gần như hoàn toàn thứ hạng với `feature_importances_` gốc (Pha 3).

**Permutation Importance** (trên feature gốc, trước one-hot): `tenure` (0.110) và `Contract` (0.101) bỏ xa
phần còn lại — gấp ~3 lần feature thứ 3 (`InternetService`, 0.032). Top-3 trùng khớp hoàn toàn với SHAP —
đây là bằng chứng chéo mạnh nhất vì cơ chế đo tách biệt hoàn toàn khỏi nội bộ CatBoost (chỉ quan sát
input/output, không dùng cấu trúc cây).

**Đồng thuận 3/3 phương pháp global** (`feature_importances_`, SHAP, Permutation Importance) về top driver
churn: `tenure`, `Contract`, `InternetService`, `TechSupport`/`OnlineSecurity` — 3 cơ chế toán học độc lập
đồng ý, giảm đáng kể rủi ro đây là artefact riêng của một phương pháp đo cụ thể.

**Giải thích cá nhân (SHAP waterfall + LIME)** cho 2 khách hàng từ Pha 4 error analysis:
- `4464-JCOLN` (FN, confident-miss, tenure=2 tháng, `InternetService=No`, proba=0.117): cả SHAP lẫn LIME đều
  xác nhận định lượng `InternetService=No`/các cột `"No internet service"` là lực kéo dự đoán xuống lớn
  nhất — đúng cơ chế đã suy luận định tính ở Pha 4, giờ có bằng chứng định lượng.
- `5178-LMXOP` (TP tự tin nhất, proba=0.962): `Contract=Month-to-month` và `tenure` thấp kéo mạnh dự đoán
  lên — hồ sơ "churn điển hình" model học rất tốt.
- SHAP và LIME đồng thuận về feature đứng đầu ở cả 2 khách dù xuất phát từ 2 cơ chế toán học khác nhau
  (game theory vs local linear surrogate) — trọng số feature phụ có xê dịch (bình thường, LIME nhạy với
  vùng lân cận cụ thể hơn SHAP).

**PDP/ICE**: rủi ro churn theo `tenure` giảm dốc nhất ở khoảng 0–20 tháng rồi thoải dần (không tuyến tính
đều) — gợi ý ưu tiên nguồn lực retention vào ~20 tháng đầu thay vì dàn trải đều theo thời gian. `Contract`
(PDP dạng categorical) định lượng trực tiếp mức chênh dự đoán giữa 3 loại hợp đồng, rõ ràng hơn một con số
importance đơn thuần. Đường ICE tách khá rộng ở vùng tenure thấp — hiệu ứng `tenure` phụ thuộc vào các
feature khác của từng khách (tương tác thật), không đồng nhất toàn bộ dữ liệu.

### 5.3 Môi trường

Thêm `optuna`, `shap`, `lime` vào `requirements.txt` (không có trong bản gốc). Cài `shap` làm numpy hạ từ
2.5.1 xuống 2.4.6 — đã verify toàn bộ pipeline (pandas/sklearn/catboost) vẫn chạy đúng sau khi hạ cấp.

---

## Pha mở rộng — Survival Analysis (time-to-churn)

Pha 1 đã chốt rõ: bài toán chính là classification tại 1 thời điểm snapshot, **không phải** time-to-event
(xem Pha 1.1). Pha mở rộng này trả lời thêm câu hỏi "nếu rời đi thì khi nào", bổ sung cho câu hỏi "có rời đi
không" đã có — không thay thế model classification, không tích hợp vào serving/registry (xem quyết định
cuối mục dưới). Toàn bộ phân tích ở `notebooks/survival_analysis/survival_analysis.ipynb`; lý thuyết nền ở
`docs/survival_analysis_theory.md`; model tái sử dụng ở `src/survival_analysis/cox_model.py`.

**Duration/event**: `tenure` (số tháng đã là khách) làm duration, `Churn == "Yes"` làm event; khách còn ở
lại (`Churn == "No"`) là right-censored — chỉ biết họ "sống" ít nhất đến `tenure` hiện tại.

**So sánh model** (non-parametric KM/Nelson-Aalen làm baseline mô tả, không dùng để so sánh vì không có
covariates):

| Model | Metric | Giá trị |
|---|---|---|
| Exponential / Weibull / Log-Logistic (no covariates) | AIC | 21696.97 / 21156.57 / 21139.91 |
| Weibull AFT (có covariates) | log-likelihood | -8915.63 |
| **Cox PH (có covariates, không stratify)** | **concordance** | **0.8663** |
| Cox PH (stratified by `Contract`) | concordance | 0.7205 |

**Model chọn: Cox PH không stratify.** Concordance cao nhất, giả định phân phối nhẹ hơn Weibull AFT, hệ số
(hazard ratio) dễ diễn giải cho business. `Contract` vi phạm giả định proportional hazards (p < 5e-05,
`check_assumptions`) — cách sách vở khuyên là `strata=['Contract']`, nhưng làm vậy loại `Contract` (yếu tố
chi phối tốc độ churn, xác nhận bằng log-rank test) khỏi linear predictor và làm rơi concordance xuống
0.7205. Với mục tiêu là diễn giải business chứ không phải suy luận thống kê chặt chẽ về hazard ratio không
đổi, đánh đổi đó không đáng — vi phạm PH của `Contract` được **ghi nhận là giới hạn đã biết** thay vì xử lý.
8 covariate vi phạm nhẹ hơn còn lại cũng để nguyên vì cùng lý do.

**Phát hiện chính dùng được cho retention**: khách `Contract=Month-to-month` có hazard cao nhất và tập
trung ở giai đoạn đầu vòng đời (khớp với PDP theo `tenure` ở Pha 5.2: rủi ro giảm dốc nhất ở 0–20 tháng) —
gợi ý một cửa sổ can thiệp cụ thể (vài tháng đầu) thay vì rải đều ưu đãi theo tenure.

**Quyết định phạm vi**: không tích hợp vào FastAPI serving/MLflow registry — đây là một phân tích bổ sung
(exploratory extension), không phải model production thứ hai. Random Survival Forest (`scikit-survival`,
có sẵn trong dependency group `notebooks`) là hướng mở rộng hợp lý tiếp theo, cố ý để ngoài phạm vi hiện
tại.
