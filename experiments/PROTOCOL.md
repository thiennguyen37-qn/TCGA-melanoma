# Protocol đánh giá

**Trạng thái:** đã khoá ngày 2026-11-11 bằng tag git `protocol-v1` (commit `5d55c14`), trước mọi lần đánh giá external. Mọi thay đổi về sau phải ghi vào mục 9 (Sai lệch so với protocol) kèm ngày và lý do.

## 1. Dữ liệu

| Tập | Nguồn | Số ảnh | Melanoma | Dùng để |
|---|---|---|---|---|
| Train | `data/resplit/split.csv`, split = train | 12.684 | 5.037 | Huấn luyện |
| Val | split = val | 2.718 | 1.079 | Chọn λ, early stopping, chọn ngưỡng |
| Test nội bộ | split = test | 2.717 | 1.079 | Báo cáo hiệu năng nội bộ |
| BALD | `experiments/external_ids.csv` | 660 | 140 | External 1 (dermoscopy acral) |
| PAD-UFES-20 | `experiments/external_ids.csv` | 296 | 52 | External 2 (ảnh smartphone) |

- SHA-256 của `data/resplit/split.csv`: `43fb3d44414e502f84b3b87fc841bd91e7e110e8d362a8f2e30f38a07490ba22`.
- SHA-256 của `experiments/external_ids.csv`: `65eb23b3e313b471fbfd61d1c5f0307180a511464586fcab1efb58ca33aef494`.
- External không được dùng cho bất kỳ quyết định nào: kiến trúc, siêu tham số, λ, ngưỡng, số epoch, chọn checkpoint.
- Mask tổn thương: `data/masks_unet/` (ground truth ISIC 2018 khi có, còn lại U-Net). Ảnh có `qc_flag = 1` trong `index.csv` được đặt λ = 0.

## 2. Cấu hình chung cho mọi biến thể

- **Đầu vào:** chỉ ảnh, không dùng metadata. Lý do: trên PAD, việc thiếu `sex` gần như lộ nhãn.
- **Model:** EfficientNet-B0 pretrained ImageNet, một logit melanoma.
- **Ảnh:** resize 224×224, chuẩn hoá ImageNet.
- **Augmentation mặc định:** lật ngang, lật dọc, xoay ±20°, ColorJitter (brightness, contrast, saturation 0,2), theo notebook 04.
- **Loss phân loại:** sigmoid focal loss, α = tỉ lệ nevus trong train, γ = 2.
- **Tối ưu:** Adam, lr 1e-4, ReduceLROnPlateau (theo val AUC, factor 0,5, patience 1), tối đa 15 epoch, early stopping patience 3 theo val AUC, batch 32.
- **Chọn checkpoint:** epoch có val AUC cao nhất.
- **Đánh giá:** TTA 4 góc nhìn (gốc, lật ngang, lật dọc, lật cả hai), lấy trung bình xác suất.
- **Seed:** 0, 1, 2, 3, 4. Seed điều khiển khởi tạo head, thứ tự dữ liệu, augmentation và cách gán mask ngẫu nhiên. Mọi biến thể dùng cùng 5 seed để so sánh theo cặp.

## 3. Các biến thể

| Mã | Tên | Khác với V0 |
|---|---|---|
| V0 | Baseline | Không |
| V1 | Mask-guided | Loss = L_focal + λ·L_att |
| V2 | Mask ngẫu nhiên | Như V1, nhưng mỗi ảnh nhận mask của một ảnh train khác |
| V3 | Augmentation chống chuyển miền | Thay augmentation mặc định bằng policy `mix` của Kozachok 2026a |
| V4 | Augmentation + mask | V3 cộng L_att như V1 |

**Chi tiết:**
- **L_att (V1, V2, V4):** L_att = 1 − Σ(A⊙M) / ΣA.
  - A là Grad-CAM của logit melanoma ở tầng conv cuối, qua ReLU, nội suy lên 224×224.
  - M là mask nhị phân, biến đổi hình học cùng với ảnh.
  - L_att tính trên mọi ảnh trong batch (cả melanoma lẫn nevus), trừ ảnh có `qc_flag = 1`.
- **V2:** mỗi seed tạo một hoán vị cố định trên ảnh train. Ảnh i nhận mask của ảnh π(i), giữ nguyên `qc_flag` của mask được nhận. Cùng λ với V1.
- **V3:** policy `mix` theo đúng Bảng A6 của Kozachok 2026a (albumentations 2.0.8), áp trước khi resize. Riêng chuẩn hoá vẫn dùng ImageNet như mọi biến thể khác.
- **V4:** dùng augmentation của V3, mask biến đổi hình học cùng ảnh, cùng λ với V1.

## 4. Chọn λ

- **Lưới giá trị:** {0,1; 0,3; 1; 3}, chạy V1 với seed 0.
- **Quy tắc:** chọn λ lớn nhất có val AUC ≥ val AUC của V0 seed 0 trừ 0,005. Nếu không có λ nào đạt, chọn λ có val AUC cao nhất.
- λ đã chọn được dùng cho V1, V2, V4 ở mọi seed. Các lần chạy dùng để chọn λ không được tính vào kết quả.

## 5. Ngưỡng phân loại

Với mỗi model (biến thể × seed), ngưỡng là giá trị lớn nhất mà sensitivity trên val nội bộ vẫn ≥ 0,90 (xác suất có TTA). Ngưỡng này áp nguyên sang test nội bộ và external.

## 6. Endpoint

**Chính (RQ1):** ΔAUC = AUC(V1) − AUC(V0), tính riêng trên BALD và trên PAD-UFES-20, lấy trung bình qua 5 seed.

**Phụ:**
- **RQ2:** V1 − V2 (AUC external). V1 cũng phải vượt V2 về tỉ lệ attention trong mask.
- **RQ3:** V1 − V3, và V4 − V3 (AUC external).
- **RQ4:** tỉ lệ năng lượng Grad-CAM++ nằm trong mask, đo trên 334 ảnh test nội bộ có mask chuẩn (chính) và trên ảnh BALD có `qc_flag = 0` (phụ). Sanity check ngẫu nhiên hoá trọng số theo tầng (Adebayo 2018) trên seed 0 của V0, V1, V2: báo tương quan hạng Spearman giữa bản đồ gốc và bản đồ sau khi ngẫu nhiên hoá.
- **RQ5:** AUC(model) − AUC(reader) trên BALD, với reader score = answer_mel_count / num_rated.
- **Với mọi biến thể:** AUC test nội bộ; sensitivity và specificity tại ngưỡng ở mục 5.

## 7. Thống kê

- **Khoảng tin cậy:** bootstrap phân cấp, 2.000 lần. Mỗi lần lấy lại cụm (bệnh nhân ở PAD và nội bộ, ảnh ở BALD) có hoàn lại; với mẫu đó tính ΔAUC cho từng seed rồi lấy trung bình 5 seed. CI 95% theo phân vị.
- **Bội số:** endpoint chính có 2 phép so sánh (BALD, PAD), hiệu chỉnh Holm. Endpoint phụ báo CI, không hiệu chỉnh, và được gọi là thăm dò.
- **Kết luận "cải thiện"** chỉ khi CI 95% của ΔAUC (sau Holm với endpoint chính) không chứa 0.
- **Kết quả null** vẫn được báo cáo đầy đủ, kèm mức hiệu ứng nhỏ nhất mà dữ liệu loại trừ được (cận trên của CI).
- **Hiệu ứng nhỏ nhất đáng quan tâm:** ΔAUC = +0,02. Với PAD chỉ có 52 melanoma, CI dự kiến rộng; điểm này được nêu trong phần hạn chế.
- **Báo cáo:** mean ± SD qua seed cho từng tập; không gộp BALD và PAD.

## 8. Quy tắc chạy

1. Khoá protocol (tag `protocol-v1`) trước mọi lần đánh giá external.
2. Mỗi biến thể × seed chỉ được đánh giá trên external đúng một lần, bằng checkpoint đã chọn theo val.
3. Sau lần đánh giá external đầu tiên, không được đổi code huấn luyện, siêu tham số hay cách tiền xử lý. Nếu bắt buộc phải đổi (ví dụ sửa lỗi), ghi vào mục 9 và chạy lại toàn bộ các biến thể bị ảnh hưởng.
4. Lưu cho mỗi lần chạy: checkpoint, xác suất trên val/test/external, ngưỡng, log huấn luyện, commit hash của code.

## 9. Sai lệch so với protocol

Mọi mục dưới đây được phát hiện hoặc quyết định sau khi đã chạy xong 25 run và chấm external. Không mục nào làm thay đổi dữ liệu, code huấn luyện, λ, ngưỡng hay checkpoint.

1. **ToGray trong policy `mix` (V3, V4), phát hiện 2026-11-12 khi đối chiếu lại Bảng A6 của Kozachok 2026a.**
   - Bảng A6 đặt `ToGray` (trọng số 0,001) là một thành viên của khối OneOf photometric. `experiments/augment.py` lại áp `ToGray(p=0.001)` thành một bước riêng, sau khối đó.
   - Hệ quả: xác suất một ảnh bị chuyển xám là 0,001 thay vì khoảng 0,0001 (0,7 × 0,001 / tổng trọng số ≈ 5,8 của khối). Trên khoảng 190.000 lượt ảnh mỗi run (12.684 ảnh × tối đa 15 epoch), đó là khoảng 190 ảnh xám thay vì khoảng 23.
   - Quyết định: không chạy lại V3, V4. Tác động được đánh giá là không đáng kể, và việc này được nêu trong phần Methods.
2. **Cụm bootstrap ở test nội bộ.** Mục 7 ghi "bệnh nhân". Vì khoảng 72% ảnh không có `patient_id`, `experiments/stats.py` dùng `group_id` của `data/resplit/split.csv`, tức là cụm liên thông theo lesion, bệnh nhân và ảnh gần trùng. Đây là cụm thô hơn, nên chặt hơn so với cụm bệnh nhân. Endpoint chính (BALD, PAD) không bị ảnh hưởng.
3. **Phân tích bổ sung, ngoài protocol, báo cáo là thăm dò:**
   - RQ4: ngoài Grad-CAM++, báo thêm tỉ lệ năng lượng trong mask của CAM thường, vì với head GAP + Linear, trọng số Grad-CAM++ gần như không phụ thuộc vào trọng số của head.
   - Độ trùng giữa mask thật và mask hoán vị của V2 (`experiments/mask_overlap.py`), là số liệu mô tả, không dùng đầu ra của model.
   - Mức lộ nhãn do thiếu dữ liệu `sex` trên PAD-UFES-20 (AUC của chỉ báo "có giá trị sex" = 0,846), dùng để giải thích lựa chọn chỉ dùng ảnh ở mục 2.
   - RQ4: tỉ lệ năng lượng trong mask của các bản đồ cố định không phụ thuộc ảnh (hình tròn giữa ảnh có diện tích bằng diện tích mask trung vị của train, mask trung bình của train), làm mốc "chỉ nhìn vào giữa ảnh" (`experiments/centre_prior.py`, thêm ngày 2026-11-13).
   - Phân tích theo nhóm trên BALD: vị trí acral (lòng bàn chân, lòng bàn tay) và mức xâm lấn của melanoma (in situ, xâm lấn), AUC và sensitivity/specificity tại ngưỡng val, so với reader (`experiments/subgroup_bald.py`, thêm ngày 2026-11-13). Không có kiểm định giữa các biến thể trong nhóm.
   - Calibration của mọi run trên val, test, BALD, PAD-UFES-20: Brier, calibration intercept và slope, ECE 10 bin, điểm reliability (`experiments/calibration.py`, thêm ngày 2026-11-13 theo hướng dẫn báo cáo AI của JAMA Dermatology). Mô tả theo trung bình (SD) qua seed, không kiểm định.
