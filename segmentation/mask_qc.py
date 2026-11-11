"""Label-free quality features for U-Net lesion masks, used to decide which training masks to trust.

Two sets:
  unet_val    the 12% hold-out of train_unet.py (same seed, same shuffle), which has ground truth,
              so each row also gets the Dice against the ground-truth mask. Used to calibrate the rule.
  pred        every image in data/masks_unet/index.csv with source=pred in the chosen datasets
              (default train,val). Used to apply the rule.

Features per image (all from the U-Net probability map, no label needed):
  area_frac       mask area / image area
  border_frac     share of the image border pixels covered by the mask
  mean_conf       mean probability inside the mask
  uncertain_frac  pixels with 0.2 < p < 0.8, divided by mask area
  largest_share   share of the raw (thresholded) mask kept by the largest component
  tta_dice        mean Dice between the mask and the masks from horizontal and vertical flips

Usage:
  python segmentation/mask_qc.py --set unet_val --out <csv>
  python segmentation/mask_qc.py --set pred --datasets train,val --out <csv>
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from common import MEAN, ROOT, SIZE, STD, UNetResNet18
from predict_masks import clean


def unet_val_items():
    from train_unet import load_pairs
    random.seed(0)
    pairs = load_pairs()
    random.shuffle(pairs)
    n_val = max(8, int(0.12 * len(pairs)))
    return [(i, ip, mp) for i, ip, mp in pairs[:n_val]]


def pred_items(datasets):
    with open(os.path.join(ROOT, "data", "masks_unet", "index.csv"), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "pred" and r["dataset"] in datasets]
    out = []
    for r in rows:
        if r["dataset"] in ("train", "val", "test"):
            ip = os.path.join(ROOT, "images", r["isic_id"] + ".jpg")
        else:
            d = "bald" if r["dataset"] == "bald" else "pad-ufes-20"
            ip = os.path.join(ROOT, "test", d, "images", r["isic_id"] + ".jpg")
        out.append((r["isic_id"], ip, None))
    return out


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def features(p, p_h, p_v):
    raw = p > 0.5
    m = clean(raw)
    area = m.sum()
    border = np.concatenate([m[0], m[-1], m[1:-1, 0], m[1:-1, -1]])
    lab, n = ndimage.label(raw)
    largest = ndimage.sum(raw, lab, range(1, n + 1)).max() / raw.sum() if n else 0.0
    return m, {
        "area_frac": float(m.mean()),
        "border_frac": float(border.mean()),
        "mean_conf": float(p[m].mean()) if area else 0.0,
        "uncertain_frac": float(((p > 0.2) & (p < 0.8)).sum() / max(area, 1)),
        "largest_share": float(largest),
        "tta_dice": (dice(m, clean(p_h > 0.5)) + dice(m, clean(p_v > 0.5))) / 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["unet_val", "pred"], required=True)
    ap.add_argument("--datasets", default="train,val")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "segmentation", "unet_resnet18.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    model = UNetResNet18(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.eval()

    items = unet_val_items() if a.set == "unet_val" else pred_items(set(a.datasets.split(",")))
    print(f"{len(items)} images", flush=True)
    ex = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(6)
    load = lambda it: Image.open(it[1]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    rows, t0 = [], time.time()
    for s in range(0, len(items), a.bs):
        chunk = items[s : s + a.bs]
        x = torch.stack([torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
                         for im in ex.map(load, chunk)])
        x = (x - MEAN) / STD
        with torch.no_grad():
            p = torch.sigmoid(model(x))[:, 0]
            p_h = torch.sigmoid(model(x.flip(-1)))[:, 0].flip(-1)
            p_v = torch.sigmoid(model(x.flip(-2)))[:, 0].flip(-2)
        for k, (iid, _, mp) in enumerate(chunk):
            m, f = features(p[k].numpy(), p_h[k].numpy(), p_v[k].numpy())
            row = {"isic_id": iid, **{n: round(v, 4) for n, v in f.items()}}
            if mp:
                g = np.asarray(Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127
                row["dice_gt"] = round(dice(m, g), 4)
                # recall: share of the true lesion inside the mask (low = the loss would push attention off the lesion)
                row["recall_gt"] = round(float((m & g).sum() / max(g.sum(), 1)), 4)
                row["precision_gt"] = round(float((m & g).sum() / max(m.sum(), 1)), 4)
            rows.append(row)
        if (s // a.bs) % 20 == 0:
            print(f"{s+len(chunk)}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
