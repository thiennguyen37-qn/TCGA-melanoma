# Melanoma Classifier — Desktop App

Desktop app demo phục vụ model đã train trong `../notebooks/07_train_improved.ipynb` (multimodal: EfficientNet-B0 + metadata, Focal Loss, TTA). Upload ảnh dermoscopy + nhập tuổi/giới tính/vị trí tổn thương (không bắt buộc — thiếu thì dùng giá trị mặc định từ tập train), trả về dự đoán melanoma/nevus kèm độ tin cậy.

Bên trong là FastAPI backend + giao diện HTML, `desktop.py` mở giao diện đó trong 1 cửa sổ riêng (pywebview). Khi cần debug vẫn chạy được trong browser.

⚠️ **Đây là demo học thuật, không phải công cụ chẩn đoán y tế.**

## Cấu trúc

```
desktop_app/
├── app/
│   ├── main.py            # FastAPI app, routes (/, /health, /predict)
│   ├── model.py            # Kiến trúc MultimodalModel (khớp 07_train_improved.ipynb) + load checkpoint
│   ├── preprocessing.py    # Transform ảnh + metadata giống lúc train (scaler/encoder tính từ data/processed/train.csv)
│   ├── schemas.py          # Pydantic response schema
│   ├── templates/index.html
│   └── static/{style.css,app.js}
├── models/                 # Đặt checkpoint .pt vào đây (gitignored, xem bên dưới)
├── tests/                  # pytest, dùng checkpoint giả (random weight) nên không cần model thật để test
├── desktop.py              # Chạy app dạng desktop (pywebview)
└── requirements.txt
```

## Lấy checkpoint model

File `best_model_improved.pt` được train trên Colab (xem `../notebooks/07_train_improved.ipynb`), lưu trên Google Drive — **không có trong Git** vì quá nặng.

1. Tải `best_model_improved.pt` từ Google Drive (`Melanoma/best_model_improved.pt`)
2. Đặt vào `desktop_app/models/best_model_improved.pt`

## Chạy desktop app

```bash
cd desktop_app
pip install -r requirements.txt
python desktop.py
```

Mở 1 cửa sổ riêng chứa giao diện app (FastAPI chạy ngầm bên trong), đóng cửa sổ là tắt app.

## Chạy trong browser (khi cần debug)

```bash
cd desktop_app
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Mở `http://127.0.0.1:8000`.

## Chạy test

```bash
pip install pytest httpx2   # httpx2: TestClient của FastAPI/Starlette cần gói này
pytest tests/ -v
```

Test dùng checkpoint giả (trọng số random, tạo tự động trong `tests/conftest.py`) nên **không cần** file model thật — chỉ kiểm tra logic API/preprocessing đúng, không kiểm tra độ chính xác dự đoán.
