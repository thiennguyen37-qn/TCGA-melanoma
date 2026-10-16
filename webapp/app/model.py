"""Kiến trúc model - phải khớp chính xác với 07_train_improved.ipynb để load đúng checkpoint."""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

METADATA_COLS = [
    "U50", "U60", "U70", "sex_encoded", "age_approx_scaled",
    "anatom_anterior torso", "anatom_head/neck", "anatom_lateral torso",
    "anatom_lower extremity", "anatom_oral/genital", "anatom_palms/soles",
    "anatom_posterior torso", "anatom_upper extremity",
]

LABEL_NAMES = {0: "nevus", 1: "melanoma"}


class MultimodalModel(nn.Module):
    def __init__(self, n_metadata_features, image_feature_dim=1280, metadata_hidden_dim=32):
        super().__init__()

        backbone = efficientnet_b0(weights=None)  # load trọng số từ checkpoint, không cần tải pretrained
        backbone.classifier = nn.Identity()
        self.image_branch = backbone

        self.metadata_branch = nn.Sequential(
            nn.Linear(n_metadata_features, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(image_feature_dim + metadata_hidden_dim, 1)

    def forward(self, image, metadata):
        image_features = self.image_branch(image)
        metadata_features = self.metadata_branch(metadata)
        combined = torch.cat([image_features, metadata_features], dim=1)
        return self.classifier(combined)


def load_model(checkpoint_path, device):
    model = MultimodalModel(n_metadata_features=len(METADATA_COLS))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
