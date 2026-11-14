# Melanoma — Melanoma & Nevus Classification

Bài toán phân loại nhị phân melanoma vs. nevus từ ảnh dermoscopy, dựa trên bộ dữ liệu ISIC (International Skin Imaging Collaboration) có xác nhận mô bệnh học.

Repo: https://github.com/thiennguyen37-qn/Melanoma

Bản lưu trữ (code tại thời điểm nộp bài, dự đoán từng ảnh, kết quả phân tích, trọng số 25 model): [doi:10.5281/zenodo.24217148](https://doi.org/10.5281/zenodo.24217148)

## Nguồn dữ liệu

Ảnh và metadata lấy từ bộ dữ liệu [Melanoma and Nevus Dermoscopy Images with Confirmed Histopathological Diagnosis](https://api.isic-archive.com/doi/melanoma-and-nevus-dermoscopy-images-with-confirmed-histopathological-diagnosis/) trên ISIC Archive. Mỗi ảnh có thể thuộc 1 trong 3 giấy phép khác nhau (`CC-0`, `CC-BY`, `CC-BY-NC`) — xem cột `copyright_license` trong `data/raw/metadata.csv` và chi tiết từng giấy phép trong `docs/`. Khi sử dụng ảnh CC-BY/CC-BY-NC cần ghi công theo `docs/attribution.txt`.

## Trạng thái

- Dữ liệu được chia train/val/test theo cụm lesion, bệnh nhân và ảnh gần trùng (`splitting/resplit.py`): 12.684 / 2.718 / 2.717 ảnh, melanoma 39,7% ở mỗi tập.
- Thí nghiệm chính nằm trong [`experiments/`](experiments/): protocol đánh giá khoá trước bằng tag `protocol-v1`, 5 biến thể (baseline, giám sát attention bằng mask tổn thương, mask ngẫu nhiên, augmentation chống chuyển miền, augmentation + mask), chỉ dùng ảnh, EfficientNet-B0, 5 seed, external là BALD và PAD-UFES-20.
- Đã chạy xong 25 run. Giám sát attention bằng mask kéo attention vào tổn thương (57% → 85% năng lượng Grad-CAM++ trong mask), nhưng AUC ngoài chỉ tăng +0,012 trên BALD và không tăng có ý nghĩa trên PAD-UFES-20; mask ngẫu nhiên và augmentation đạt tương đương. Chi tiết ở [experiments/README.md](experiments/README.md).

## Cấu trúc project

```
.
├── notebooks/
│   ├── 01_metadata_preprocessing.ipynb   # load raw metadata, loại cột
│   ├── 02_EDA.ipynb                      # phân tích khám phá, kiểm định MCAR/MAR
│   ├── 03_imputation.ipynb               # impute missing values, feature engineering, encoding, scaling
│   ├── 04_image_pipeline.ipynb           # Dataset/DataLoader PyTorch, chạy được local/Colab/Kaggle
│   ├── 05_train_baseline.ipynb           # baseline image-only (ResNet18)
│   ├── 06_train_multimodal.ipynb         # ảnh + metadata (ResNet18 + MLP, late fusion)
│   ├── 07_train_improved.ipynb           # EfficientNet-B0 + Focal Loss + TTA, có resume-từ-checkpoint
│   ├── 08_external_validation.ipynb      # đánh giá ngoài trên PAD-UFES-20 và BALD
│   ├── 09_gradcam.ipynb                  # Grad-CAM trên tập external
│   └── 10_train_b3_color.ipynb           # EfficientNet-B3 + chuẩn hoá màu Shades of Gray
├── experiments/                      # protocol đánh giá, train 5 biến thể, thống kê, phân tích attention (RQ4)
├── splitting/resplit.py              # chia train/val/test theo cụm lesion, patient, ảnh gần trùng
├── segmentation/                     # U-Net sinh mask tổn thương và kiểm tra chất lượng mask
├── data/
│   ├── raw/metadata.csv              # metadata gốc (18,133 ảnh)
│   ├── resplit/                      # kết quả chia lại: split.csv, perceptual hash, ảnh trùng khác nhãn, trùng với external
│   ├── splitted/                     # train/val/test do splitting/resplit.py export ghi ra
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

1. **`notebooks/01_metadata_preprocessing.ipynb`** — load `data/raw/metadata.csv`, loại các cột không cần thiết (hằng số, hành chính, rò rỉ nhãn, quá thưa).
   - **Chia dữ liệu: `splitting/resplit.py`** — chạy lần lượt `hash`, `thumbs`, `pairs`, `split --keep-indeterminate`, `export`. Ảnh được nhóm theo chung `lesion_id`, chung `patient_id` hoặc ảnh gần trùng, chia 70/15/15 theo nhóm và phân tầng theo nhãn, ghi vào `data/splitted/`. Chi tiết ở `docs/data_hierarchy.md`.
2. **`notebooks/02_EDA.ipynb`** — khám phá dữ liệu (phân phối tuổi/giới/vị trí tổn thương theo nhãn), kiểm định cơ chế missing (MCAR vs. MAR) cho `age_approx`, `sex`, `anatom_site_general`.
3. **`notebooks/03_imputation.ipynb`** — impute các cột còn thiếu bằng stratified random hot-deck (sample có hoàn lại từ dữ liệu thật, điều kiện theo `attribution_grouped`, fit trên train áp dụng xuống val/test tránh leakage); feature engineering (`U50`/`U60`/`U70`), encoding (binary/one-hot), scaling (`StandardScaler`); lưu vào `data/processed/`.
4. **`notebooks/04_image_pipeline.ipynb`** — join ảnh với metadata, định nghĩa PyTorch `Dataset`/`DataLoader`, augmentation (chỉ train). Tự nhận diện môi trường chạy (local/Colab/Kaggle) và điều chỉnh đường dẫn phù hợp.
5. **`notebooks/05_train_baseline.ipynb`** — fine-tune ResNet18 trên ảnh (bỏ qua metadata), làm baseline.
6. **`notebooks/06_train_multimodal.ipynb`** — ResNet18 (ảnh, 512-dim) + MLP (metadata, 32-dim) nối lại (late fusion), so sánh với baseline để xác nhận metadata có đóng góp thật.
7. **`notebooks/07_train_improved.ipynb`** — nâng cấp từ `06`: backbone EfficientNet-B0, Focal Loss (thay BCE+pos_weight, nhắm vào sensitivity), Test-Time Augmentation khi đánh giá, có khả năng resume từ checkpoint nếu training bị ngắt giữa chừng.

8. **Thí nghiệm của paper: `experiments/`** — `prepare_cache.py` đóng gói ảnh 320×320 và mask, `train.py` chạy 25 run trên Colab theo `PROTOCOL.md`, rồi `stats.py`, `rq4_attention.py`, `mask_overlap.py` chạy trên máy local. Chi tiết ở `experiments/README.md`.

**Lưu ý:** `05`–`07` cần GPU để train trong thời gian hợp lý (CPU khả thi nhưng rất chậm — ước tính vài giờ/epoch). Khuyến nghị chạy trên Google Colab hoặc Kaggle Notebook (đều có GPU miễn phí); `notebooks/04_image_pipeline.ipynb` đã hỗ trợ sẵn cả 2 nền tảng.
