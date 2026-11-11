# Cấu trúc phân cấp dữ liệu: Patient → Lesion → Image

Bộ dữ liệu này có cấu trúc phân cấp 3 tầng, thể hiện qua 3 cột khóa trong `metadata.csv`: `patient_id`, `lesion_id`, `isic_id`.

## Khái niệm

- **Patient (`patient_id`)**: một bệnh nhân.
- **Lesion (`lesion_id`)**: một tổn thương da cụ thể trên cơ thể bệnh nhân (ví dụ: một nốt ruồi, một vết đổi màu, một khối u da). Mỗi lesion chỉ có **1 chẩn đoán duy nhất** (melanoma hoặc nevus) vì nó là cùng một tổn thương vật lý.
- **Image (`isic_id`)**: một ảnh dermoscopy cụ thể, chụp tại 1 thời điểm/góc độ/độ zoom. Một lesion có thể được chụp nhiều lần (theo dõi qua thời gian, chụp lại ở nhiều góc...), nên 1 `lesion_id` có thể ứng với nhiều `isic_id`.

## Sơ đồ ví dụ

```
patient_id (1 bệnh nhân)
 ├── lesion_id A (nốt ruồi ở tay)
 │     ├── isic_id 1 (ảnh chụp lần 1)
 │     ├── isic_id 2 (ảnh chụp lần 2, theo dõi sau 6 tháng)
 │     └── isic_id 3 (ảnh zoom gần)
 ├── lesion_id B (nốt ruồi ở lưng)
 │     └── isic_id 4
 └── lesion_id C (khối u nghi ngờ ở chân → sinh thiết → melanoma)
       ├── isic_id 5
       └── isic_id 6
```

## Thống kê thực tế trong dataset

- Tổng số ảnh: 18,133
- Số ảnh có ghi `lesion_id`: 14,047 (77.5%)
- Số lesion duy nhất: 7,369
- Trung bình ảnh/lesion: ~1.91
- Lesion có nhiều ảnh nhất: 31 ảnh

| Số ảnh/lesion | Số lesion |
|---|---|
| 1 | 4,311 |
| 2 | 1,661 |
| 3 | 771 |
| 4 | 207 |
| 5 | 123 |
| ... | ... |
| 31 (max) | 1 |

## Vì sao cấu trúc này quan trọng cho machine learning

Vì nhiều ảnh có thể thuộc cùng 1 lesion (và cùng 1 bệnh nhân), **không được split train/val/test ngẫu nhiên theo từng ảnh**. Nếu làm vậy, model có thể học "nhận mặt" một lesion cụ thể từ các ảnh của nó trong tập train, rồi dễ dàng đoán đúng ảnh khác của cùng lesion đó ở tập test — gây **data leakage**, khiến độ chính xác đánh giá bị thổi phồng giả tạo, không phản ánh đúng khả năng tổng quát hóa của model.

Rò rỉ còn đến từ hai nguồn khác ngoài lesion:

- **Cùng bệnh nhân:** ảnh các lesion khác nhau của một người thường giống nhau về màu da, máy chụp, ánh sáng. Chỉ 27,6% ảnh có `patient_id`, nhưng một lesion có `patient_id` ở một ảnh thì các ảnh khác của lesion đó cũng thuộc bệnh nhân này.
- **Ảnh đăng lại:** cùng một ảnh có thể xuất hiện ở hai collection mà không chung id nào.

**Giải pháp** (`splitting/resplit.py`): hai ảnh được nối nếu **chung `lesion_id`, hoặc chung `patient_id`, hoặc là ảnh gần trùng**. Ảnh gần trùng được dò bằng perceptual hash để lọc ứng viên, rồi xác nhận bằng tương quan pixel ở vùng giữa ảnh, chuẩn hoá từng kênh màu. Mỗi thành phần liên thông là một nhóm, toàn bộ ảnh của một nhóm nằm trọn trong một tập. Các nhóm được chia train/val/test 70/15/15 và phân tầng theo nhãn. Ảnh trùng nhau nhưng mang nhãn khác nhau là lỗi nhãn ở nguồn và bị loại (14 ảnh, 6 cụm).
