# Báo cáo kết quả nghiên cứu — Telco Customer Churn

Tổng hợp **kết quả và insight** từ toàn bộ phân tích data science trong repo: 5 notebook
`notebooks/churn_classification/` (CRISP-ML(Q): EDA → data prep → modeling → evaluation → business value/XAI)
và `notebooks/survival_analysis/survival_analysis.ipynb` (time-to-churn). Mọi số liệu trong tài liệu này lấy
trực tiếp từ output đã chạy của các notebook đó — không tính toán lại, không làm tròn/suy diễn thêm số nào
ngoài số đã in ra. Phần quyết định/lý do đầy đủ theo từng pha xem `docs/docs.md`; tài liệu này chỉ trình bày
**kết quả và insight rút ra được**.

Dataset: Telco Customer Churn (IBM/Kaggle, `blastchar/telco-customer-churn`), 7043 khách hàng × 21 cột, 1
dòng = 1 khách hàng tại 1 thời điểm snapshot.

---

## Phần 1 — Classification: khách nào sẽ rời đi

### 1.1 Dữ liệu

- Sau khi loại 11 dòng `tenure == 0` (khách mới, `TotalCharges` rỗng — không phải missing ngẫu nhiên, là
  artefact nghiệp vụ vì khách chưa qua billing cycle đầu): **7032 dòng**, không còn missing value nào khác.
- Target imbalance: **73.42% No / 26.58% Yes**. Không có outlier numeric nào bị loại (0% theo luật 1.5×IQR ở
  cả 3 cột `tenure`/`MonthlyCharges`/`TotalCharges`) — phân phối lệch phải là đặc trưng tự nhiên của dữ liệu
  billing, không phải lỗi đo lường.
- 6 cột add-on internet (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`,
  `StreamingMovies`) collinear hoàn hảo với `InternetService` tại các dòng `InternetService == "No"` (giá trị
  `"No internet service"`) — ghi nhận để không hiểu sai feature importance/hệ số sau này, không xử lý gì thêm
  vì cả Logistic Regression (L2) lẫn tree model đều không bị ảnh hưởng bởi collinearity.
- EDA trực quan (chưa định lượng): khách churn thiên về `tenure` thấp, `MonthlyCharges` cao, `Contract`
  month-to-month, `PaymentMethod` electronic check, thiếu `OnlineSecurity`/`TechSupport` — các quan sát này
  sau đó được xác nhận lại bằng feature importance định lượng ở mục 1.3.

### 1.2 Model progression (train pool, 5-fold CV, PR-AUC là metric chọn model)

| Vòng | Model tốt nhất | PR-AUC (CV) | overfit_gap |
|---|---|---|---|
| Baseline no-skill | Dummy (stratified) | 0.2654 | 0.0009 |
| Vòng 1 (tay chọn, chưa regularize) | CatBoost | 0.6610 | 0.1450 |
| Vòng 2 (tay chọn, regularize thủ công) | CatBoost (reg) | 0.6764 | 0.0271 |
| Vòng 3 (Optuna TPE, 25 trial/model, tune công bằng cả 4 họ model) | **CatBoost (tuned)** | **0.6768** | **0.0257** |

Baseline no-skill ra đúng PR-AUC ≈ prevalence (0.2654 ≈ 0.2658) — xác nhận scoring/CV không có bug trước khi
tin các con số model thật. Từ vòng 1 sang vòng 2, `overfit_gap` của 3 tree model giảm mạnh (CatBoost:
0.1450 → 0.0271) sau khi giảm độ sâu/số cây và thêm regularization — bằng chứng regularization có tác dụng
thật, không phải nhiễu ngẫu nhiên. Optuna (vòng 3) chỉ cải thiện thêm rất ít so với vòng 2 tay chọn
(0.6768 vs 0.6764) — vòng tay chọn đã tình cờ gần vùng tối ưu; điểm quan trọng hơn con số tuyệt đối là
`overfit_gap` ổn định ở mức thấp (0.017–0.049) trên toàn bộ model đã tune.

**Model chọn: CatBoost (tuned)** — `max_depth=4, n_estimators=239, learning_rate≈0.0267, l2_leaf_reg≈1.046,
min_data_in_leaf=35, subsample≈0.868` (params thật trong `src/churn_classification/final_model.py`).

### 1.3 Đánh giá trên holdout (test set 1055 dòng, 280 dương — chạm đúng 1 lần)

| Metric | CV mean (Pha 3) | Test (holdout) |
|---|---|---|
| PR-AUC | 0.6768 ± 0.0260 | **0.6365** (z ≈ −1.55) |
| ROC-AUC | 0.8504 ± 0.0136 | **0.8369** (z ≈ −0.99) |

PR-AUC test lệch nhiều hơn ROC-AUC (z=-1.55 so với z=-0.99) — thấp hơn cả fold CV thấp nhất từng quan sát
(0.6505), nhưng vẫn trong 2 độ lệch chuẩn, không đủ bằng chứng nghi ngờ leakage. PR-AUC vốn nhạy hơn với cỡ
mẫu dương nhỏ (test chỉ 280 dương so với ~1194 trung bình mỗi fold train pool) — biến động này nhiều khả năng
chỉ là phương sai tự nhiên của một holdout không quá lớn.

Threshold vận hành chọn bằng out-of-fold predictions trên train pool (không đụng test set), tối đa F2:
**threshold = 0.295**, F2(OOF) = 0.7482 (so với F2 = 0.7220 tại threshold mặc định 0.5). Tại threshold này
trên test set:

```
              precision    recall  f1-score   support
    No Churn       0.96      0.52      0.67       775
       Churn       0.41      0.94      0.57       280
```

Recall 0.94 cho lớp Churn — đúng định hướng "FN đắt hơn FP" đã chốt từ Pha 1, đổi lại bằng precision 0.41
(nhiều FP — chấp nhận được vì FP chỉ tốn 1 ưu đãi, không phải mất khách thật).

### 1.4 Error analysis (test set, tại threshold 0.295)

Phân rã: TN 403, FP 372, TP 263, **FN 17**.

- **FN vs TP** (median): FN có `tenure` cao hơn hẳn (40 tháng vs 9), `TotalCharges` cao hơn nhiều (2896 vs
  574), `churn_proba` thấp (0.201 vs 0.756). Theo `Contract`: 92.4% nhóm TP là month-to-month, trong khi nhóm
  FN chỉ 17.6% month-to-month (52.9% one-year, 29.4% two-year) — **model đang bỏ sót đúng nhóm khách "trông
  trung thành theo mọi tín hiệu có trong dataset"** (tenure dài, hợp đồng dài hạn) nhưng vẫn rời đi.
- Trong 17 FN, chia 2 loại theo `churn_proba` so với threshold: **9 "confident miss"** (proba rất thấp, model
  không hề nghi ngờ) và **8 "near-miss"** (proba ≥ 70% threshold, model đã nghi ngờ đúng hướng nhưng chưa đủ
  tự tin). Nhóm confident-miss là dấu hiệu **giới hạn của chính bộ dữ liệu** (nguyên nhân churn thật — ví dụ
  tăng giá lúc gia hạn, đối thủ chào giá tốt hơn, sự cố dịch vụ cụ thể — không được ghi nhận trong feature
  set hiện có), không phải lỗi có thể sửa bằng tune thêm model/threshold.
- **FP vs TN** (median): FP có `tenure` thấp hơn nhiều (18 vs 51), `MonthlyCharges` cao hơn (75.4 vs 47.9) —
  model gắn cờ đúng kiểu hồ sơ "trông giống sắp churn" dù thực tế khách này ở lại; chi phí của lỗi này chỉ là
  1 ưu đãi lãng phí, không phải mất khách.

### 1.5 Cost-based threshold (thay F2 bằng giá trị kỳ vọng $)

Value model: chi phí ưu đãi `$50`, giữ được `12` tháng doanh thu nếu thành công, xác suất ưu đãi thành công
`30%`, `MonthlyCharges` làm proxy doanh thu/khách (giả định minh bạch, kiểm tra sensitivity chứ không tin một
con số cố định).

Quét threshold trên OOF train pool: **threshold tối ưu theo $ = 0.465**, giá trị kỳ vọng $232,017 — cao hơn cả
threshold F2 0.295 ($225,603) **và** threshold mặc định 0.5 ($230,102). **Phát hiện đáng chú ý: threshold tối
ưu theo F2 (một metric thống kê) thực ra kém hơn cả threshold mặc định 0.5 về mặt giá trị kinh doanh** — tối
ưu một proxy thống kê không tự động đảm bảo tối ưu giá trị thực tế.

Sensitivity analysis (lưới `success_rate` × `offer_cost`) cho threshold tối ưu dao động có trật tự (0.095 →
0.950): tăng khi ưu đãi đắt hơn, giảm khi ưu đãi hiệu quả hơn — đúng logic kinh tế, không phải nhiễu ngẫu
nhiên, tăng độ tin cậy vào cách tiếp cận.

Áp `threshold = 0.465` một lần lên test set: giá trị model tạo ra **$38,290** so với "không làm gì" ($0), và
**$16,453** so với "gắn cờ toàn bộ không dùng model" ($21,837) — model có giá trị thực sự cao hơn cả 2
baseline.

### 1.6 Explainability — 4 phương pháp độc lập đồng thuận

| Phương pháp | Cơ chế | Top feature (thứ tự) |
|---|---|---|
| `feature_importances_` (Pha 3) | Gain khi split (nội bộ CatBoost) | tenure, Contract_Month-to-month, TotalCharges, InternetService_Fiber optic, Contract_Two year |
| SHAP (`TreeExplainer`, Pha 5) | Giá trị Shapley (game theory) | tenure (0.469), Contract_Month-to-month (0.425), InternetService_Fiber optic (0.270), Contract_Two year (0.248), PaymentMethod_Electronic check (0.192), TechSupport_No (0.188), OnlineSecurity_No (0.180) |
| Permutation Importance (feature gốc, không one-hot) | Xáo trộn input, đo PR-AUC giảm | tenure (0.1098), Contract (0.1014), InternetService (0.0320) |
| LIME | Local linear surrogate | đồng thuận top feature với SHAP cho 2 case cụ thể (mục dưới) |

4 cơ chế toán học độc lập (gain-based, Shapley, model-agnostic permutation, local-surrogate) đều xác nhận
cùng nhóm driver: **tenure, Contract, InternetService, TechSupport/OnlineSecurity** — bằng chứng chéo mạnh
rằng đây là driver churn thật, không phải artefact riêng của CatBoost.

**Giải thích cá nhân** cho 2 khách hàng từ mục 1.4: `4464-JCOLN` (FN, confident-miss, tenure=2 tháng,
`InternetService=No`, proba=0.117) — SHAP và LIME đều xác nhận `InternetService=No` là lực kéo dự đoán xuống
lớn nhất, đúng cơ chế đã suy luận định tính ở mục 1.4. `5178-LMXOP` (TP tự tin nhất, proba=0.962) —
`Contract=Month-to-month` và tenure thấp kéo mạnh dự đoán lên.

**PDP/ICE**: rủi ro churn theo `tenure` giảm dốc nhất ở khoảng 0–20 tháng rồi thoải dần (không tuyến tính đều)
— gợi ý ưu tiên nguồn lực retention vào ~20 tháng đầu thay vì dàn trải đều theo thời gian.

---

## Phần 2 — Survival analysis: nếu rời đi thì khi nào

Bổ sung cho Phần 1 (trả lời "có/không") bằng câu hỏi "khi nào" — cùng dataset, `tenure` làm duration,
`Churn=="Yes"` làm event, `Churn=="No"` là right-censored.

### 2.1 Non-parametric

- Kaplan-Meier trên toàn bộ 7032 khách: **median survival time = inf** — S(t) không bao giờ chạm 0.5 trong
  khoảng quan sát, vì tỷ lệ churn tổng thể chỉ 26.58% (dưới 50%).
- Log-rank test giữa `Contract=Month-to-month` và `Contract=One year`: test statistic 926.06, **p < 0.005** —
  khác biệt survival curve giữa 2 nhóm cực kỳ có ý nghĩa thống kê, không phải ngẫu nhiên.

### 2.2 So sánh model (điểm chọn model chính)

| Model | Loại | Metric | Giá trị |
|---|---|---|---|
| Exponential | Parametric, no covariates | AIC | 21696.97 |
| Weibull | Parametric, no covariates | AIC | 21156.57 |
| Log-Logistic | Parametric, no covariates | AIC | 21139.91 |
| Weibull AFT | Parametric, có covariates | log-likelihood | −8915.63 |
| **Cox PH (không stratify)** | Semi-parametric, có covariates | concordance | **0.8663** |
| Cox PH (stratified by `Contract`) | Semi-parametric, có covariates | concordance | 0.7205 |

**Model chọn: Cox PH không stratify** (concordance 0.8663). `Contract` vi phạm proportional-hazards
assumption (`check_assumptions`, p < 5e-05) — thử fix theo khuyến nghị chuẩn (`strata=['Contract']`) cho ra
một phát hiện ngược trực giác: **concordance rơi xuống 0.7205**, vì `Contract` mang phần lớn sức mạnh phân
biệt của model (khớp với log-rank test ở mục 2.1). Quyết định: giữ bản không stratify, ghi nhận vi phạm PH
của `Contract` là giới hạn đã biết thay vì xử lý — đánh đổi 0.146 điểm concordance để tuân thủ chặt giả định
thống kê không xứng đáng với mục tiêu diễn giải business ở đây. 8 covariate vi phạm nhẹ hơn còn lại
(`Partner`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `TechSupport`, `StreamingTV`, `PaymentMethod`,
p trong khoảng 0.0017–0.0503) cũng để nguyên vì cùng lý do.

### 2.3 Insight chính

- **`Contract` chi phối tốc độ churn** (log-rank p<0.005, hazard ratio lớn nhất trong Cox PH): khách
  month-to-month có hazard cao nhất, khách one/two-year gần như không đổi và cực thấp.
- **Hazard không cố định theo thời gian** — chính là lý do `Contract` vi phạm PH: rủi ro rời đi của nhóm
  month-to-month tập trung ở giai đoạn đầu vòng đời rồi giảm dần nếu khách ở lại, khớp với PDP theo `tenure`
  ở mục 1.6 (rủi ro giảm dốc nhất ở 0–20 tháng).
- Model classification cho một xác suất tại 1 thời điểm; model survival cho thêm một đường cong theo thời
  gian — dùng để ước tính "còn bao lâu" cho khách đã bị model classification gắn cờ rủi ro cao, phục vụ xếp
  hạng ưu tiên trong hàng đợi can thiệp của team CS.

---

## Tổng kết

| Câu hỏi | Trả lời | Bằng chứng |
|---|---|---|
| Model nào, hiệu năng bao nhiêu? | CatBoost (Optuna-tuned) | PR-AUC 0.6365 / ROC-AUC 0.8369 (holdout, 1 lần) |
| Threshold vận hành? | 0.465 (cost-based) | Giá trị kỳ vọng cao hơn threshold F2 và 0.5 mặc định — $232,017 vs $225,603 vs $230,102 (OOF); tạo thêm $38,290 so với không làm gì trên test set |
| Driver churn chính? | tenure, Contract, InternetService, TechSupport/OnlineSecurity | 4 phương pháp XAI độc lập đồng thuận (mục 1.6) |
| Model còn thiếu gì? | Nhóm khách "trông trung thành" (tenure dài, hợp đồng dài) vẫn churn — nguyên nhân ngoài phạm vi feature hiện có | 9/17 FN là confident-miss (mục 1.4) |
| Khi nào khách rời đi? | Sớm nhất và tập trung nhất ở nhóm month-to-month, giai đoạn đầu vòng đời | Cox PH concordance 0.8663, log-rank p<0.005 (Phần 2) |

Toàn bộ số liệu trên tái tạo được bằng cách chạy lại 6 notebook theo thứ tự trong `notebooks/`; các quyết
định/lý do đầy đủ (không lặp lại ở đây) xem `docs/docs.md`.
