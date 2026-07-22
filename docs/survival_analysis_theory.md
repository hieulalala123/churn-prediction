# Survival Analysis — Lý thuyết

Tài liệu này giải thích lý thuyết đứng sau các phương pháp dùng trong
`notebooks/survival_analysis/survival_analysis.ipynb`. Bài toán: dự đoán
"khi nào" khách hàng sẽ churn, thay vì chỉ "có churn hay không" (đó là bài
toán classification, xem `docs/docs.md`).

## Khái niệm cơ bản

- **Duration (T)**: thời gian từ lúc bắt đầu quan sát đến khi event xảy ra
  hoặc đến khi ngừng quan sát. Ở đây là `tenure` — số tháng khách đã là
  khách hàng.
- **Event**: sự kiện quan tâm xảy ra. Ở đây là churn (`Churn == "Yes"`).
- **Censoring (right-censoring)**: khi kết thúc quan sát mà event chưa xảy
  ra — khách hàng `Churn == "No"` vẫn đang là khách hàng, ta chỉ biết họ
  "sống" ít nhất đến `tenure` tháng, không biết họ sẽ rời đi khi nào. Bỏ qua
  các dòng censored (chỉ dùng dòng đã churn) sẽ làm sai lệch ước lượng —
  đây là lý do cần các phương pháp survival analysis riêng thay vì hồi quy
  thông thường.
- **Survival function S(t) = P(T > t)**: xác suất còn "sống" (chưa churn)
  sau thời điểm t.
- **Hazard function h(t)**: tốc độ churn tức thời tại thời điểm t, với điều
  kiện đã sống đến t. `S(t)` và `h(t)` là hai cách biểu diễn tương đương của
  cùng một phân phối thời gian sống.

## 1. Non-parametric

Không giả định S(t) hay h(t) theo dạng hàm số cụ thể nào — ước lượng trực
tiếp từ dữ liệu quan sát.

### Kaplan-Meier (KM)

Ước lượng S(t) bằng tích các xác suất sống sót có điều kiện tại mỗi thời
điểm có event xảy ra:

```
S(t) = Π (1 - d_i / n_i)   với mọi t_i <= t
```

trong đó `n_i` là số đối tượng còn "at risk" (chưa churn, chưa bị censor)
ngay trước thời điểm `t_i`, `d_i` là số event xảy ra tại `t_i`. Đây là
đường cong bậc thang (step function), là ước lượng non-parametric chuẩn của
survival function.

**KM theo nhóm + log-rank test**: fit KM riêng cho từng nhóm (ví dụ theo
`Contract`) rồi so sánh trực quan. Log-rank test là kiểm định thống kê xem
hai đường KM có khác nhau có ý nghĩa hay không (null hypothesis: hai nhóm
có cùng hazard function).

### Nelson-Aalen

Ước lượng cumulative hazard `H(t) = ∫h(u)du` (từ 0 đến t) trực tiếp:

```
H(t) = Σ (d_i / n_i)   với mọi t_i <= t
```

Về bản chất tương đương KM (`S(t) = exp(-H(t))` là xấp xỉ), nhưng Nelson-Aalen
ước lượng hazard tích lũy ổn định hơn khi số lượng event tại mỗi thời điểm
nhỏ.

## 2. Parametric

Giả định T theo một phân phối xác suất cụ thể. Nếu giả định đúng (hoặc gần
đúng), model parametric cho ước lượng mượt hơn KM và ngoại suy được ra
ngoài khoảng thời gian đã quan sát — điều KM không làm được vì nó chỉ định
nghĩa tại các điểm có dữ liệu.

- **Exponential**: hazard không đổi theo thời gian, `h(t) = λ`. Đơn giản
  nhất nhưng thường không thực tế (giả định "không có trí nhớ" — rủi ro
  churn ở tháng 1 giống hệt tháng 50).
- **Weibull**: hazard đơn điệu tăng hoặc giảm theo thời gian,
  `h(t) = (ρ/λ)(t/λ)^(ρ-1)`. Tham số `ρ` (shape) quyết định hazard tăng
  (`ρ > 1`, khách càng lâu càng dễ rời đi) hay giảm (`ρ < 1`, khách càng lâu
  càng gắn bó — hiệu ứng thường gặp với churn). Exponential là trường hợp
  đặc biệt của Weibull khi `ρ = 1`.
- **Log-Logistic**: hazard có thể tăng rồi giảm (non-monotonic) — linh hoạt
  hơn Weibull, phù hợp khi rủi ro churn tăng dần rồi giảm sau một mốc thời
  gian nhất định.

So sánh 3 phân phối bằng **AIC** (thấp hơn = fit tốt hơn, đã phạt theo số
tham số) thay vì chỉ nhìn log-likelihood.

### AFT (Accelerated Failure Time) — bản có covariates

`WeibullAFTFitter` mở rộng Weibull với covariates: mỗi covariate "kéo dài"
hoặc "rút ngắn" thời gian sống theo một hệ số nhân (`exp(coef)`), tương tự
hồi quy tuyến tính nhưng trên log(T) thay vì T. Hệ số dương → covariate đó
làm khách hàng ở lại lâu hơn; âm → rời đi nhanh hơn.

## 3. Semi-parametric: Cox Proportional Hazards (Cox PH)

Không giả định dạng phân phối cho `S(t)`/`h(t)` (giống non-parametric),
nhưng giả định hazard của mỗi cá thể là một hằng số nhân so với baseline
hazard `h_0(t)` (không phụ thuộc covariates):

```
h(t | X) = h_0(t) * exp(β_1 X_1 + β_2 X_2 + ... )
```

`exp(β_i)` là **hazard ratio**: covariate tăng 1 đơn vị làm hazard (rủi ro
churn tức thời) nhân lên bao nhiêu lần, độc lập với thời gian t — đây chính
là giả định **proportional hazards**: tỷ lệ hazard giữa hai cá thể bất kỳ
không đổi theo thời gian.

Ưu điểm so với parametric: không cần đoán đúng phân phối baseline. Ưu điểm
so với non-parametric: đưa được covariates vào mô hình để định lượng ảnh
hưởng của từng biến.

### Kiểm tra giả định Proportional Hazards

Giả định PH có thể bị vi phạm — nghĩa là ảnh hưởng thực của covariate đó
thay đổi theo thời gian chứ không phải hệ số cố định. `cph.check_assumptions`
kiểm định điều này cho từng covariate (dựa trên scaled Schoenfeld residuals).

Trong notebook, `Contract` là covariate vi phạm rõ nhất (p-value < 0.00005)
— hợp lý về mặt business: hiệu ứng "khóa hợp đồng" của Contract dài hạn lên
churn không cố định theo thời gian (ví dụ: tác động mạnh nhất ở giai đoạn
đầu hợp đồng, giảm dần khi gần hết hạn). Khi vi phạm, hướng xử lý phổ biến
là stratify theo covariate đó (`strata=[...]` trong `cph.fit`) hoặc thêm
time-interaction term, thay vì bỏ qua vi phạm.

## Tài liệu tham khảo

- [lifelines documentation](https://lifelines.readthedocs.io/)
- [Proportional Hazards Assumption — lifelines](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html)
