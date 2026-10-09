# TCGA Melanoma — Melanoma & Nevus Classification

Bài toán phân loại nhị phân melanoma vs. nevus từ ảnh dermoscopy, dựa trên bộ dữ liệu ISIC (International Skin Imaging Collaboration) có xác nhận mô bệnh học.

Repo: https://github.com/thiennguyen37-qn/TCGA-melanoma

## Nguồn dữ liệu

Ảnh và metadata lấy từ [ISIC Archive](https://www.isic-archive.com/). Mỗi ảnh có thể thuộc 1 trong 3 giấy phép khác nhau (`CC-0`, `CC-BY`, `CC-BY-NC`) — xem cột `copyright_license` trong `data/raw/metadata.csv` và chi tiết từng giấy phép trong `docs/`. Khi sử dụng ảnh CC-BY/CC-BY-NC cần ghi công theo `docs/attribution.txt`.

## Cấu trúc project

```
.
├── 01_metadata_preprocessing.ipynb   # load raw metadata, loại cột, group-split train/val/test
├── 02_EDA.ipynb                      # phân tích khám phá, kiểm định MCAR/MAR
├── 03_imputation.ipynb               # impute missing values (stratified random hot-deck)
├── data/
│   ├── raw/metadata.csv              # metadata gốc (18,133 ảnh)
│   └── splitted/                     # train/val/test sau group-split
├── docs/
│   ├── attribution.txt               # nguồn đóng góp ảnh
│   ├── CC-0.txt / CC-BY.txt / CC-BY-NC.txt
│   ├── data_hierarchy.md             # giải thích cấu trúc patient/lesion/image
│   └── metadata_description.md       # giải thích các cột trong metadata.csv
├── images/                           # ảnh dermoscopy — KHÔNG có trong Git, xem bên dưới
├── requirements.txt
└── .gitignore
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Tải ảnh (`images/`)

Thư mục `images/` chứa 18,133 ảnh (~5.4GB) nên **không được đưa lên GitHub** (xem `.gitignore`). Để tái tạo:

1. Tải ảnh từ [ISIC Archive](https://www.isic-archive.com/) theo danh sách `isic_id` trong `data/raw/metadata.csv`.
2. Đặt toàn bộ ảnh (`ISIC_XXXXXXX.jpg`) vào thư mục `images/` ở root của repo.

## Thứ tự chạy notebook

1. **`01_metadata_preprocessing.ipynb`** — load `data/raw/metadata.csv`, loại các cột không cần thiết (hằng số, hành chính, rò rỉ nhãn, quá thưa), group-split train/val/test theo `lesion_id`/`patient_id` (tránh data leakage), lưu kết quả vào `data/splitted/`.
2. **`02_EDA.ipynb`** — khám phá dữ liệu (phân phối tuổi/giới/vị trí tổn thương theo nhãn), kiểm định cơ chế missing (MCAR vs. MAR) cho `age_approx`, `sex`, `anatom_site_general`.
3. **`03_imputation.ipynb`** — impute các cột còn thiếu bằng stratified random hot-deck (sample có hoàn lại từ dữ liệu thật, điều kiện theo `diagnosis`/`attribution_grouped`), fit trên train, áp dụng xuống val/test để tránh leakage.
