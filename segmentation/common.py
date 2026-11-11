"""Shared pieces for the lesion-segmentation U-Net (ResNet18 encoder, 256x256 input)."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIZE = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def gt_mask_path(isic_id):
    """Ground-truth ISIC 2018 Task 1 mask path, or None if there is none."""
    for split in ("training", "validation", "test"):
        p = os.path.join(ROOT, "data", "masks_isic2018_task1", split, f"{isic_id}_segmentation.png")
        if os.path.exists(p):
            return p
    return None


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2, 64
        self.pool = r.maxpool                              # /4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64,128,256,512
        self.up3 = _Up(512, 256, 256)
        self.up2 = _Up(256, 128, 128)
        self.up1 = _Up(128, 64, 64)
        self.up0 = _Up(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.up3(s4, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, s0)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(d)  # logits
