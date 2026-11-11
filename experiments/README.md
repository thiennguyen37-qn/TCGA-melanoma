# Experiments

Code cho các thí nghiệm của paper, chạy đúng theo [PROTOCOL.md](PROTOCOL.md) (đã khoá bằng tag `protocol-v1`).

| File | Chức năng |
|---|---|
| `PROTOCOL.md` | Protocol đánh giá: dữ liệu, 5 biến thể, chọn λ, ngưỡng, endpoint, thống kê |
| `external_ids.csv`, `lock_external.py` | Danh sách ảnh external đã khoá (BALD 660, PAD-UFES-20 296) |
| `prepare_cache.py` | Gói ảnh 320×320, mask, split và danh sách external thành `data/bundle320.zip` |
| `augment.py` | Augmentation mặc định (notebook 04) và policy `mix` của Kozachok 2026a |
| `train.py` | Train và đánh giá một biến thể × seed |
| `select_lambda.py` | Chọn λ theo quy tắc ở mục 4 của protocol |

## Chuẩn bị (máy local, một lần)

```bash
python experiments/prepare_cache.py --zip      # tạo data/bundle320.zip
```

Upload `data/bundle320.zip` lên Google Drive, ví dụ `MyDrive/Melanoma/bundle320.zip`.

## Chạy trên Colab (GPU)

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -b add-gradcam-and-b3-notebooks https://github.com/thiennguyen37-qn/TCGA-melanoma.git /content/repo
!pip install -q albumentations==2.0.8
!cp /content/drive/MyDrive/Melanoma/bundle320.zip /content/ && unzip -q /content/bundle320.zip -d /content/data
RUNS = '/content/drive/MyDrive/Melanoma/runs'   # lưu trên Drive để không mất khi bị ngắt phiên
```

Mỗi lần chạy:

```python
!python /content/repo/experiments/train.py --data /content/data --runs {RUNS} --variant V0 --seed 0
```

Bị ngắt phiên thì chạy lại đúng lệnh đó: script tự resume từ `last.pt`. Kết quả trên external chỉ được tính một lần cho mỗi lần chạy.

Trên Kaggle: thay `--data` bằng thư mục dataset đã giải nén và `--runs` bằng `/kaggle/working/runs`.

## Thứ tự chạy

1. **Baseline seed 0:** `--variant V0 --seed 0`.
2. **Chọn λ** (không chấm external):
   ```
   --variant V1 --seed 0 --lam 0.1 --no-external --tag lamsearch
   --variant V1 --seed 0 --lam 0.3 --no-external --tag lamsearch
   --variant V1 --seed 0 --lam 1   --no-external --tag lamsearch
   --variant V1 --seed 0 --lam 3   --no-external --tag lamsearch
   ```
   rồi `python experiments/select_lambda.py --runs {RUNS}`.
3. **Các lần chạy chính**, với λ đã chọn (25 lần, kể cả V0 seed 0 ở bước 1):
   - V0: seed 0–4
   - V1, V2, V4: `--lam <λ>`, seed 0–4
   - V3: seed 0–4

## Kiểm tra nhanh trên CPU

```bash
python experiments/train.py --data data/cache320 --runs runs_smoke --variant V4 --lam 1 --seed 0 \
    --limit 64 --epochs 1 --max-steps 2 --device cpu --workers 0
```
