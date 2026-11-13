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
| `stats.py` | Bootstrap theo cụm có ghép cặp, Holm cho endpoint chính, so với reader trên BALD (mục 6–7) |
| `rq4_attention.py` | RQ4: tỉ lệ năng lượng Grad-CAM++ trong mask (334 ảnh test có mask chuẩn, ảnh BALD có qc_flag = 0) và sanity check ngẫu nhiên hoá trọng số theo tầng; chạy được trên CPU |
| `mask_overlap.py` | Số liệu mô tả: độ trùng giữa mask thật và mask hoán vị của V2, so với một hình tròn giữa ảnh |
| `centre_prior.py` | RQ4, thăm dò: tỉ lệ năng lượng trong mask của bản đồ cố định "nhìn vào giữa ảnh", so với Grad-CAM++ của từng biến thể |
| `subgroup_bald.py` | Thăm dò: AUC, sensitivity, specificity trên BALD theo vị trí acral và mức xâm lấn của melanoma, so với reader |

## Trạng thái

- λ = 3, chọn trên val theo mục 4 của protocol (cả 4 giá trị trong lưới đều qua ngưỡng).
- Đã chạy xong 25 run chính (5 biến thể × seed 0–4) trên Colab T4, cùng commit `f5ac3d8`. Kết quả nằm trong `runs/` trên máy và trên Drive; `runs/` không có trong git.
- Các sai lệch so với protocol được ghi ở mục 9 của [PROTOCOL.md](PROTOCOL.md).

Kết quả chính (`stats.py`, `rq4_attention.py`; trung bình 5 seed, bootstrap theo cụm 2.000 lần):

| | BALD | PAD-UFES-20 |
|---|---|---|
| V1 − V0 (endpoint chính, Holm) | +0,012, 97,5% CI [+0,001; +0,023], p Holm 0,028 | +0,006, 95% CI [−0,017; +0,030], p Holm 0,60 |
| V1 − V2 (mask thật − mask ngẫu nhiên) | −0,007 [−0,016; +0,001] | −0,026 [−0,052; 0,000] |
| V1 − V3, V4 − V3 | CI chứa 0 | CI chứa 0 |
| AUC so với reader (0,973) | mọi biến thể thấp hơn 0,035–0,055 | – |

- Tỉ lệ năng lượng Grad-CAM++ trong mask: V0 0,573 → V1 0,847 trên 334 ảnh test có mask chuẩn; 0,691 → 0,948 trên BALD.
- Mức cải thiện của V1 trên BALD có ý nghĩa thống kê nhưng nhỏ hơn mức hiệu ứng đáng quan tâm định trước (+0,02). Mask ngẫu nhiên và augmentation đạt tương đương.
- Mask hoán vị của V2 trùng tổn thương thật 62% (Dice 0,44), nên V2 đóng vai đối chứng "nhìn vào vùng trung tâm".
- Mốc "nhìn vào giữa ảnh" (hình tròn cố định, diện tích bằng mask trung vị) đạt 0,561 năng lượng trong mask ở test và 0,731 ở BALD, ngang V0 (0,573; 0,691). V1 và V2 vượt mốc này rõ (+0,29 và +0,24 ở test): có giám sát thì attention bám vào chính tổn thương, nhưng AUC ngoài vẫn gần như không đổi.
- Trên BALD, melanoma in situ là nhóm khó nhất (AUC 0,87–0,89 so với reader 0,94); chênh lệch giữa các biến thể nhỏ ở mọi nhóm (kết quả thăm dò).

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

## Ô chạy toàn bộ các lần chạy chính trên Colab

Chạy các seed lần lượt, mỗi seed đủ 5 biến thể. Run nào đã có `summary.json` thì bỏ qua, run đang dở thì resume, nên bị ngắt chỉ cần chạy lại ô setup rồi chạy lại ô này.

```python
import os
LAM, SEEDS = 3, [0, 1, 2, 3, 4]
plan = [(v, s) for s in SEEDS for v in ('V0', 'V1', 'V2', 'V3', 'V4')]
for v, s in plan:
    lam = LAM if v in ('V1', 'V2', 'V4') else 0
    name = f"{v}_s{s}" + (f"_lam{lam:g}" if lam else "")
    if os.path.exists(f"{RUNS}/{name}/summary.json"):
        print("skip", name); continue
    print("=== run", name)
    lam_arg = f"--lam {lam}" if lam else ""
    !python /content/repo/experiments/train.py --data /content/data --runs {RUNS} --variant {v} --seed {s} {lam_arg}
```

Colab chỉ chạy khi máy local bật và tab còn mở: tắt chế độ sleep của máy trong lúc chạy.

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

## Thống kê (máy local)

Tải thư mục `runs/` từ Drive về `D:\Melanoma\runs\`. Chỉ cần `preds_*.csv`, `summary.json` và `lambda_choice.json`, không cần các file `.pt`. Sau đó chạy:

```bash
python experiments/stats.py --runs runs
```

Kết quả in ra màn hình và ghi vào `runs/stats/comparisons.csv`, `runs/stats/per_variant.csv`. Script chạy được khi mới có một phần seed: mỗi phép so sánh dùng các seed mà cả hai biến thể đều đã có.

## RQ4 (máy local, CPU)

Cần `best.pt` của 25 run chính trong `runs/<run>/`. Sau đó chạy:

```bash
python experiments/rq4_attention.py --data data/cache320 --runs runs
```

Kết quả nằm trong `runs/rq4/`. Mất khoảng 40 phút trên CPU; nếu bị ngắt, chạy lại sẽ bỏ qua các run đã tính.

## Độ trùng mask của V2 (máy local)

```bash
python experiments/mask_overlap.py --data data/cache320
```

Dùng đúng hoán vị của `train.py` (`default_rng(seed)` trên các dòng train của `split.csv`), không dùng đầu ra của model. Kết quả ghi vào `runs/mask_overlap.csv`.

## Phân tích thăm dò (máy local)

```bash
python experiments/centre_prior.py --data data/cache320 --runs runs   # cần runs/rq4/energy_per_image.csv
python experiments/subgroup_bald.py --runs runs
```

Kết quả ghi vào `runs/rq4/centre_prior_energy.csv`, `runs/rq4/centre_prior_comparisons.csv` và `runs/stats/subgroup_bald.csv`.

## Kiểm tra nhanh trên CPU

```bash
python experiments/train.py --data data/cache320 --runs runs_smoke --variant V4 --lam 1 --seed 0 \
    --limit 64 --epochs 1 --max-steps 2 --device cpu --workers 0
```
