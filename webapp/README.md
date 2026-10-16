# Melanoma Classifier — Web App

Demo web app phục vụ model đã train trong `../07_train_improved.ipynb` (multimodal: EfficientNet-B0 + metadata, Focal Loss, TTA). Upload ảnh dermoscopy + nhập tuổi/giới tính/vị trí tổn thương, trả về dự đoán melanoma/nevus kèm độ tin cậy.

⚠️ **Đây là demo học thuật, không phải công cụ chẩn đoán y tế.**

## Cấu trúc

```
webapp/
├── app/
│   ├── main.py            # FastAPI app, routes (/, /health, /predict)
│   ├── model.py            # Kiến trúc MultimodalModel (khớp 07_train_improved.ipynb) + load checkpoint
│   ├── preprocessing.py    # Transform ảnh + metadata giống lúc train (scaler/encoder tính từ data/processed/train.csv)
│   ├── schemas.py          # Pydantic response schema
│   ├── templates/index.html
│   └── static/{style.css,app.js}
├── models/                 # Đặt checkpoint .pt vào đây (gitignored, xem bên dưới)
├── tests/                  # pytest, dùng checkpoint giả (random weight) nên không cần model thật để test
├── Dockerfile
└── requirements.txt
```

## Lấy checkpoint model

File `best_model_improved.pt` được train trên Colab (xem `../07_train_improved.ipynb`), lưu trên Google Drive — **không có trong Git** vì quá nặng.

1. Tải `best_model_improved.pt` từ Google Drive (`Melanoma/best_model_improved.pt`)
2. Đặt vào `webapp/models/best_model_improved.pt`

## Chạy local (không Docker)

```bash
cd webapp
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Mở `http://127.0.0.1:8000`.

## Chạy test

```bash
pytest tests/ -v
```

Test dùng checkpoint giả (trọng số random, tạo tự động trong `tests/conftest.py`) nên **không cần** file model thật — chỉ kiểm tra logic API/preprocessing đúng, không kiểm tra độ chính xác dự đoán.

## Chạy bằng Docker

```bash
docker build -t melanoma-webapp .
docker run -p 8000:8000 melanoma-webapp
```

(Cần đặt checkpoint vào `models/` trước khi build, vì Dockerfile copy thư mục này vào image.)

## Deploy

Khuyến nghị: [Render](https://render.com) (free tier, nhận deploy thẳng từ Dockerfile) hoặc tương tự (Railway, Fly.io).

**Lưu ý:** checkpoint không có trong Git nên cần cấu hình riêng để model có mặt lúc deploy — ví dụ build arg tải file từ 1 URL lưu trữ ổn định (Hugging Face Hub, GitHub Releases...) trong lúc build/khởi động container, thay vì mong đợi nó có sẵn trong source code.
