import io

import pytest
from PIL import Image

from app.preprocessing import AGE_MEAN, ANATOM_CATEGORIES, SEX_MALE_RATIO, preprocess_image, preprocess_metadata


def test_preprocess_image_shape():
    img = Image.new("RGB", (100, 150), color=(255, 0, 0))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_image_handles_grayscale():
    img = Image.new("L", (100, 100))  # ảnh grayscale, phải tự convert sang RGB
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_metadata_shape():
    tensor = preprocess_metadata(age=45, sex="male", anatom_site=ANATOM_CATEGORIES[0])
    assert tensor.shape == (1, 13)


def test_preprocess_metadata_age_thresholds():
    # age=55: U50=1, U60=0, U70=0
    tensor = preprocess_metadata(age=55, sex="female", anatom_site=ANATOM_CATEGORIES[0])
    u50, u60, u70 = tensor[0, 0].item(), tensor[0, 1].item(), tensor[0, 2].item()
    assert u50 == 1.0
    assert u60 == 0.0
    assert u70 == 0.0


def test_preprocess_metadata_sex_encoding():
    male_tensor = preprocess_metadata(age=40, sex="male", anatom_site=ANATOM_CATEGORIES[0])
    female_tensor = preprocess_metadata(age=40, sex="female", anatom_site=ANATOM_CATEGORIES[0])
    assert male_tensor[0, 3].item() == 1.0
    assert female_tensor[0, 3].item() == 0.0


def test_preprocess_metadata_onehot_exactly_one_hot():
    tensor = preprocess_metadata(age=40, sex="male", anatom_site="head/neck")
    onehot_part = tensor[0, 5:]
    assert onehot_part.sum().item() == 1.0


def test_preprocess_metadata_invalid_anatom_site():
    with pytest.raises(ValueError):
        preprocess_metadata(age=40, sex="male", anatom_site="khong_hop_le")


def test_preprocess_metadata_invalid_sex():
    with pytest.raises(ValueError):
        preprocess_metadata(age=40, sex="other", anatom_site=ANATOM_CATEGORIES[0])


def test_preprocess_metadata_all_none_fallback():
    # Không có field nào (vd: dataset external thiếu metadata) - vẫn phải ra tensor hợp lệ
    tensor = preprocess_metadata(age=None, sex=None, anatom_site=None)
    assert tensor.shape == (1, 13)
    onehot_part = tensor[0, 5:]
    assert onehot_part.sum().item() == 0.0  # không có site nào được chọn


def test_preprocess_metadata_age_none_uses_train_mean():
    tensor = preprocess_metadata(age=None, sex="male", anatom_site=ANATOM_CATEGORIES[0])
    age_scaled = tensor[0, 4].item()
    assert age_scaled == pytest.approx(0.0, abs=1e-6)  # (AGE_MEAN - AGE_MEAN) / AGE_STD = 0
    u50 = tensor[0, 0].item()
    assert u50 == (1.0 if AGE_MEAN > 50 else 0.0)


def test_preprocess_metadata_sex_none_uses_train_ratio():
    tensor = preprocess_metadata(age=40, sex=None, anatom_site=ANATOM_CATEGORIES[0])
    assert tensor[0, 3].item() == pytest.approx(SEX_MALE_RATIO)
