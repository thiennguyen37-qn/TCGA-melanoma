# Giải thích các biến trong `metadata.csv`

Dữ liệu từ ISIC (International Skin Imaging Collaboration). Mỗi dòng tương ứng với 1 ảnh dermoscopy trong `images/`, khớp qua `isic_id`.

## 1. Định danh & nguồn gốc

| Cột | Giải thích |
|---|---|
| `isic_id` | Mã định danh ảnh, khớp với tên file `images/ISIC_XXXXXXX.jpg` |
| `lesion_id` | Mã tổn thương (lesion). Một lesion có thể được chụp nhiều ảnh (nhiều góc/thời điểm) |
| `patient_id` | Mã bệnh nhân. Một bệnh nhân có thể có nhiều lesion. Dùng để split train/val/test tránh data leakage |
| `attribution` | Tổ chức/cá nhân đóng góp ảnh (xem `docs/attribution.txt`) |
| `copyright_license` | Giấy phép sử dụng ảnh: `CC-0`, `CC-BY`, hoặc `CC-BY-NC` (xem `docs/`) |

## 2. Nhãn chẩn đoán (nhãn chính cho classification)

| Cột | Giải thích |
|---|---|
| `diagnosis` | Chẩn đoán chính: `melanoma` hoặc `nevus` — **nhãn dùng cho binary classification** |
| `benign_malignant` | Tính chất lành/ác: `benign`, `malignant` (một số dòng có giá trị mơ hồ `indeterminate`, `indeterminate/benign`, `indeterminate/malignant`, hoặc rỗng) |
| `diagnosis_1` → `diagnosis_5` | Cây phân loại chẩn đoán (taxonomy ISIC) từ tổng quát → cụ thể, ví dụ: `Malignant` → `Malignant melanocytic proliferations (Melanoma)` → `Melanoma, NOS` |
| `diagnosis_confirm_type` | Phương pháp xác nhận chẩn đoán, ví dụ `histopathology` (sinh thiết mô bệnh học — độ tin cậy cao nhất) |
| `melanocytic` | Tổn thương có nguồn gốc từ tế bào melanocyte hay không (`True`/`False`) |
| `image_type` / `dermoscopic_type` | Loại ảnh chụp — trong dataset này toàn bộ là ảnh soi da (`dermoscopic`) |

## 3. Đặc điểm riêng của melanoma

Chỉ có giá trị khi `diagnosis = melanoma`, các lesion khác sẽ để trống.

| Cột | Giải thích |
|---|---|
| `mel_class` | Phân loại melanoma (ví dụ: invasive, in situ...) |
| `mel_type` | Loại mô học của melanoma (ví dụ: superficial spreading, nodular...) |
| `mel_thick_mm` | Độ dày khối u — **Breslow thickness** (mm), yếu tố tiên lượng quan trọng nhất của melanoma |
| `mel_ulcer` | Khối u có bị loét không (`True`/`False`) — yếu tố tiên lượng xấu nếu có |
| `mel_mitotic_index` | Chỉ số phân bào (số lượng tế bào đang phân chia/mm²) — phản ánh độ ác tính |

## 4. Đặc điểm riêng của nevus

| Cột | Giải thích |
|---|---|
| `nevus_type` | Loại nốt ruồi (ví dụ: common, dysplastic, blue nevus...) |

## 5. Thông tin bệnh nhân (demographic)

| Cột | Giải thích |
|---|---|
| `age_approx` | Tuổi gần đúng của bệnh nhân tại thời điểm chụp |
| `sex` | Giới tính (`male`/`female`) |
| `fitzpatrick_skin_type` | Loại da theo thang Fitzpatrick (I–VI), từ da rất sáng đến rất tối — ảnh hưởng đến màu sắc/độ tương phản tổn thương trên ảnh |
| `family_hx_mm` | Tiền sử gia đình có melanoma (`True`/`False`) |
| `personal_hx_mm` | Tiền sử cá nhân từng mắc melanoma (`True`/`False`) |

## 6. Thông tin tổn thương & chụp ảnh

| Cột | Giải thích |
|---|---|
| `anatom_site_general` | Vị trí giải phẫu tổng quát (ví dụ: `upper extremity`, `posterior torso`, `lower extremity`...) |
| `anatom_site_special` | Vị trí giải phẫu đặc biệt (ví dụ: lòng bàn tay/chân, móng, niêm mạc...) |
| `clin_size_long_diam_mm` | Đường kính dài nhất của tổn thương (mm), đo trên lâm sàng |
| `acquisition_day` | Ngày chụp ảnh (thường đã được ẩn danh hóa/làm tròn) |
| `concomitant_biopsy` | Có thực hiện sinh thiết đồng thời với lần chụp này không (`True`/`False`) |

## Lưu ý khi dùng cho machine learning

- **Nhãn binary chuẩn:** dùng `diagnosis` (chỉ 2 giá trị: `melanoma`/`nevus`), sạch hơn so với `benign_malignant` (có vài giá trị mơ hồ cần lọc).
- **Tránh data leakage:** split train/val/test theo `patient_id` (hoặc `lesion_id`), không theo từng ảnh riêng lẻ, vì nhiều ảnh có thể cùng 1 bệnh nhân/lesion.
- **Mất cân bằng lớp:** `nevus` (~10,931) nhiều hơn `melanoma` (~7,202) — cân nhắc class weighting/oversampling.
- **Cột chỉ áp dụng cho melanoma** (`mel_*`) sẽ là `NaN` với các dòng `nevus`, và ngược lại với `nevus_type`.
- **Bias cần kiểm tra:** phân bố `fitzpatrick_skin_type`, `anatom_site_general`, `age_approx`, `sex` giữa 2 lớp để đánh giá model có bị thiên lệch theo nhóm nhân khẩu học không.
