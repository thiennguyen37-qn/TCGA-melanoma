"""Tiền xử lý input từ web (ảnh thô + form) thành đúng định dạng model cần.

Hằng số AGE_MEAN/AGE_STD và thứ tự ANATOM_CATEGORIES được tính từ
data/processed/train.csv (03_imputation.ipynb) để khớp chính xác với
StandardScaler/OneHotEncoder đã dùng lúc train - không fit lại, vì val/test
(và giờ là dữ liệu người dùng thật) phải dùng đúng thống kê từ train, tránh lệch
phân phối so với lúc model học.
"""
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

AGE_MEAN = 52.01913988657845
AGE_STD = 17.761047728089167

# Tỷ lệ nam trong tập train (data/processed/train.csv) - dùng làm fallback cho
# sex_encoded khi không có thông tin giới tính (vd: dataset external thiếu cột này).
SEX_MALE_RATIO = 0.5172495274102079

# Thứ tự alphabet - khớp với OneHotEncoder.categories_ đã dùng lúc train
ANATOM_CATEGORIES = [
    "anterior torso", "head/neck", "lateral torso", "lower extremity",
    "oral/genital", "palms/soles", "posterior torso", "upper extremity",
]

# Nhãn tiếng Việt hiển thị trên form - value gửi lên server vẫn là ANATOM_CATEGORIES (tiếng Anh)
ANATOM_LABELS_VI = {
    "anterior torso": "Thân trước",
    "head/neck": "Đầu/cổ",
    "lateral torso": "Thân bên",
    "lower extremity": "Chi dưới",
    "oral/genital": "Niêm mạc miệng/sinh dục",
    "palms/soles": "Lòng bàn tay/lòng bàn chân",
    "posterior torso": "Thân sau",
    "upper extremity": "Chi trên",
}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """PIL Image -> tensor (1, 3, 224, 224), sẵn sàng đưa vào model."""
    rgb_image = pil_image.convert("RGB")
    tensor = image_transform(rgb_image)
    return tensor.unsqueeze(0)


def preprocess_metadata(
    age: Optional[int], sex: Optional[str], anatom_site: Optional[str]
) -> torch.Tensor:
    """age (5-85, số nguyên hoặc None), sex ('male'/'female' hoặc None),
    anatom_site (1 trong 8 category hoặc None) -> tensor (1, 13) đúng thứ tự
    METADATA_COLS.

    Fallback khi thiếu field (vd: dataset external không có cột tương ứng):
    - age=None -> dùng AGE_MEAN của tập train (age_scaled = 0, trung tâm phân phối)
    - sex=None -> dùng SEX_MALE_RATIO của tập train (thay vì ép cứng 0/1)
    - anatom_site=None -> one-hot toàn 0 (giống hành vi OneHotEncoder(handle_unknown='ignore')
      lúc train khi gặp category chưa biết)
    """
    if anatom_site is not None and anatom_site not in ANATOM_CATEGORIES:
        raise ValueError(f"anatom_site phải là 1 trong {ANATOM_CATEGORIES}, nhận được: {anatom_site!r}")
    if sex is not None and sex not in ("male", "female"):
        raise ValueError(f"sex phải là 'male' hoặc 'female', nhận được: {sex!r}")

    age_for_threshold = age if age is not None else AGE_MEAN
    u50 = 1.0 if age_for_threshold > 50 else 0.0
    u60 = 1.0 if age_for_threshold > 60 else 0.0
    u70 = 1.0 if age_for_threshold > 70 else 0.0
    age_scaled = (age - AGE_MEAN) / AGE_STD if age is not None else 0.0

    sex_encoded = SEX_MALE_RATIO if sex is None else (1.0 if sex == "male" else 0.0)

    anatom_onehot = [1.0 if cat == anatom_site else 0.0 for cat in ANATOM_CATEGORIES]

    values = [u50, u60, u70, sex_encoded, age_scaled] + anatom_onehot
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)
