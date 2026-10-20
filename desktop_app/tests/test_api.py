import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # chạy trong context manager để lifespan (load model) được kích hoạt
        yield c


def _make_test_image_bytes():
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Melanoma" in response.text


def test_predict_valid_input(client):
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", _make_test_image_bytes(), "image/jpeg")},
        data={"age": 45, "sex": "male", "anatom_site": "head/neck"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("melanoma", "nevus")
    assert 0.0 <= body["probability_melanoma"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    assert "disclaimer" in body


def test_predict_invalid_age(client):
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", _make_test_image_bytes(), "image/jpeg")},
        data={"age": 200, "sex": "male", "anatom_site": "head/neck"},
    )
    assert response.status_code == 422


def test_predict_invalid_anatom_site(client):
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", _make_test_image_bytes(), "image/jpeg")},
        data={"age": 45, "sex": "male", "anatom_site": "khong_ton_tai"},
    )
    assert response.status_code == 422


def test_predict_invalid_image(client):
    response = client.post(
        "/predict",
        files={"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        data={"age": 45, "sex": "male", "anatom_site": "head/neck"},
    )
    assert response.status_code == 422


def test_predict_missing_optional_fields(client):
    # age/sex/anatom_site không bắt buộc - thiếu hết vẫn phải trả về 200 (dùng fallback)
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", _make_test_image_bytes(), "image/jpeg")},
        data={},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("melanoma", "nevus")


def test_predict_missing_image(client):
    # image là field bắt buộc duy nhất
    response = client.post(
        "/predict",
        data={"age": 45, "sex": "male", "anatom_site": "head/neck"},
    )
    assert response.status_code == 422
