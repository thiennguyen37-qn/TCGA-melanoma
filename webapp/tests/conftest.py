"""Tạo checkpoint giả (trọng số random, chưa train) để test logic API/pipeline
mà không cần file checkpoint thật đã train (quá nặng để đưa vào CI).

Phải set biến môi trường CHECKPOINT_PATH ở cấp module (không phải trong fixture),
vì app.main đọc os.environ ngay lúc import module - fixture chạy quá muộn so với
thời điểm các file test import `from app.main import app`.
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.model import METADATA_COLS, MultimodalModel  # noqa: E402

_tmp_dir = tempfile.mkdtemp()
_fake_checkpoint_path = os.path.join(_tmp_dir, "fake_checkpoint.pt")

_model = MultimodalModel(n_metadata_features=len(METADATA_COLS))
torch.save(_model.state_dict(), _fake_checkpoint_path)

os.environ["CHECKPOINT_PATH"] = _fake_checkpoint_path
