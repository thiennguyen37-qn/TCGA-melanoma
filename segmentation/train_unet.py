"""Train a lesion-segmentation U-Net on the ISIC 2018 Task 1 masks that have a local image.

Only images already in images/ are used (about 2.3k of the 3.7k masks). A random 12% is held out
for validation (Dice / IoU). Caveat: the split is by image, not by lesion, so the validation
Dice is slightly optimistic if the same lesion was photographed more than once.

Usage: python segmentation/train_unet.py --epochs 12
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from common import MEAN, ROOT, SIZE, STD, UNetResNet18, gt_mask_path


def load_pairs():
    ids = []
    for split in ("training", "validation", "test"):
        d = os.path.join(ROOT, "data", "masks_isic2018_task1", split)
        ids += [f.replace("_segmentation.png", "") for f in os.listdir(d)]
    pairs = []
    for i in sorted(ids):
        img = os.path.join(ROOT, "images", f"{i}.jpg")
        if os.path.exists(img):
            pairs.append((i, img, gt_mask_path(i)))
    return pairs


def to_tensor(img, mask):
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    y = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None]
    return x, y


class DS(torch.utils.data.Dataset):
    def __init__(self, pairs, train):
        self.train = train
        # Decode + resize once and keep in RAM (2.3k x 256x256x3 = ~450 MB).
        self.items = []
        for _, ip, mp in pairs:
            im = Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            mk = Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)
            self.items.append((im, mk))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = to_tensor(*self.items[i])
        if self.train:
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = random.randint(0, 3)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)).clamp(0, 1)
        return (x - MEAN) / STD, y


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    dices, ious = [], []
    for x, y in loader:
        pr = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (pr * y).sum((1, 2, 3))
        union = pr.sum((1, 2, 3)) + y.sum((1, 2, 3)) - inter
        dices += ((2 * inter + 1e-6) / (pr.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-6)).tolist()
        ious += ((inter + 1e-6) / (union + 1e-6)).tolist()
    return float(np.mean(dices)), float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N pairs")
    ap.add_argument("--out", default=os.path.join(ROOT, "segmentation", "unet_resnet18.pt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(os.cpu_count())

    pairs = load_pairs()
    if a.limit:
        pairs = pairs[: a.limit]
    random.shuffle(pairs)
    n_val = max(8, int(0.12 * len(pairs)))
    val_p, tr_p = pairs[:n_val], pairs[n_val:]
    print(f"pairs={len(pairs)} train={len(tr_p)} val={len(val_p)}", flush=True)

    tr = DS(tr_p, True); va = DS(val_p, False)
    tl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=32)

    model = UNetResNet18(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    bce = torch.nn.BCEWithLogitsLoss()
    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            lg = model(x)
            loss = bce(lg, y) + dice_loss(lg, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        d, j = evaluate(model, vl)
        print(f"epoch {ep+1}/{a.epochs} loss={tot/len(tl):.4f} val_dice={d:.4f} val_iou={j:.4f} "
              f"time={time.time()-t0:.0f}s", flush=True)
        if d > best:
            best = d
            torch.save(model.state_dict(), a.out)
    print(f"best val dice {best:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
