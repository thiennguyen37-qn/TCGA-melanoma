# TCGA Melanoma — Melanoma & Nevus Classification

Bài toán phân loại nhị phân melanoma vs. nevus từ ảnh dermoscopy, dựa trên bộ dữ liệu ISIC (International Skin Imaging Collaboration) có xác nhận mô bệnh học.

Repo: https://github.com/thiennguyen37-qn/TCGA-melanoma

## Nguồn dữ liệu

Ảnh và metadata lấy từ bộ dữ liệu [Melanoma and Nevus Dermoscopy Images with Confirmed Histopathological Diagnosis](https://api.isic-archive.com/doi/melanoma-and-nevus-dermoscopy-images-with-confirmed-histopathological-diagnosis/) trên ISIC Archive. Mỗi ảnh có thể thuộc 1 trong 3 giấy phép khác nhau (`CC-0`, `CC-BY`, `CC-BY-NC`) — xem cột `copyright_license` trong `data/raw/metadata.csv` và chi tiết từng giấy phép trong `docs/`. Khi sử dụng ảnh CC-BY/CC-BY-NC cần ghi công theo `docs/attribution.txt`.

## Kết quả

Model tốt nhất: `notebooks/07_train_improved.ipynb` (multimodal — ảnh + metadata, backbone EfficientNet-B0, Focal Loss, Test-Time Augmentation), đánh giá trên test set:

| Metric | 05 Baseline (image-only) | 06 Multimodal | **07 Improved** |
|---|---|---|---|
| AUC | 0.8692 | 0.8784 | **0.9011** |
| Accuracy | 0.7933 | 0.8071 | **0.8313** |
| Sensitivity | 0.7886 | 0.7905 | **0.8180** |

## Cấu trúc project

```
.
├── notebooks/
│   ├── 01_metadata_preprocessing.ipynb   # load raw metadata, loại cột, group-split train/val/test
│   ├── 02_EDA.ipynb                      # phân tích khám phá, kiểm định MCAR/MAR
│   ├── 03_imputation.ipynb               # impute missing values, feature engineering, encoding, scaling
│   ├── 04_image_pipeline.ipynb           # Dataset/DataLoader PyTorch, chạy được local/Colab/Kaggle
│   ├── 05_train_baseline.ipynb           # baseline image-only (ResNet18)
│   ├── 06_train_multimodal.ipynb         # ảnh + metadata (ResNet18 + MLP, late fusion)
│   └── 07_train_improved.ipynb           # EfficientNet-B0 + Focal Loss + TTA, có resume-từ-checkpoint
├── data/
│   ├── raw/metadata.csv              # metadata gốc (18,133 ảnh)
│   ├── splitted/                     # train/val/test sau group-split (01)
│   └── processed/                    # train/val/test đã impute + feature engineer (03)
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

1. Tải ảnh từ [bộ dữ liệu trên ISIC Archive](https://api.isic-archive.com/doi/melanoma-and-nevus-dermoscopy-images-with-confirmed-histopathological-diagnosis/) theo danh sách `isic_id` trong `data/raw/metadata.csv`.
2. Đặt toàn bộ ảnh (`ISIC_XXXXXXX.jpg`) vào thư mục `images/` ở root của repo.

## Thứ tự chạy notebook

1. **`notebooks/01_metadata_preprocessing.ipynb`** — load `data/raw/metadata.csv`, loại các cột không cần thiết (hằng số, hành chính, rò rỉ nhãn, quá thưa), group-split train/val/test theo `lesion_id`/`patient_id` (tránh data leakage), lưu kết quả vào `data/splitted/`.
2. **`notebooks/02_EDA.ipynb`** — khám phá dữ liệu (phân phối tuổi/giới/vị trí tổn thương theo nhãn), kiểm định cơ chế missing (MCAR vs. MAR) cho `age_approx`, `sex`, `anatom_site_general`.
3. **`notebooks/03_imputation.ipynb`** — impute các cột còn thiếu bằng stratified random hot-deck (sample có hoàn lại từ dữ liệu thật, điều kiện theo `attribution_grouped`, fit trên train áp dụng xuống val/test tránh leakage); feature engineering (`U50`/`U60`/`U70`), encoding (binary/one-hot), scaling (`StandardScaler`); lưu vào `data/processed/`.
4. **`notebooks/04_image_pipeline.ipynb`** — join ảnh với metadata, định nghĩa PyTorch `Dataset`/`DataLoader`, augmentation (chỉ train). Tự nhận diện môi trường chạy (local/Colab/Kaggle) và điều chỉnh đường dẫn phù hợp.
5. **`notebooks/05_train_baseline.ipynb`** — fine-tune ResNet18 trên ảnh (bỏ qua metadata), làm baseline.
6. **`notebooks/06_train_multimodal.ipynb`** — ResNet18 (ảnh, 512-dim) + MLP (metadata, 32-dim) nối lại (late fusion), so sánh với baseline để xác nhận metadata có đóng góp thật.
7. **`notebooks/07_train_improved.ipynb`** — nâng cấp từ `06`: backbone EfficientNet-B0, Focal Loss (thay BCE+pos_weight, nhắm vào sensitivity), Test-Time Augmentation khi đánh giá, có khả năng resume từ checkpoint nếu training bị ngắt giữa chừng.

**Lưu ý:** `05`–`07` cần GPU để train trong thời gian hợp lý (CPU khả thi nhưng rất chậm — ước tính vài giờ/epoch). Khuyến nghị chạy trên Google Colab hoặc Kaggle Notebook (đều có GPU miễn phí); `notebooks/04_image_pipeline.ipynb` đã hỗ trợ sẵn cả 2 nền tảng.
