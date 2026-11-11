"""Write a 256x256 lesion mask for every image in the project (train/val/test + BALD + PAD-UFES-20).

Images that have an ISIC 2018 ground-truth mask get that mask (resized, source=gt); the rest get
the U-Net prediction after keeping the largest component and filling holes (source=pred).
Output: data/masks_unet/<isic_id>.png (0/255) and data/masks_unet/index.csv with QC columns.

Usage: python segmentation/predict_masks.py
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from common import MEAN, ROOT, SIZE, STD, UNetResNet18, gt_mask_path


def collect():
    items = []  # (dataset, isic_id, image_path)
    # every internal image, whatever its split (the split lives in data/resplit/split.csv)
    with open(os.path.join(ROOT, "data", "raw", "metadata.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            items.append(("internal", r["isic_id"], os.path.join(ROOT, "images", r["isic_id"] + ".jpg")))
    for name, d in (("bald", "bald"), ("pad", "pad-ufes-20")):
        with open(os.path.join(ROOT, "test", d, "metadata.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                items.append((name, r["isic_id"], os.path.join(ROOT, "test", d, "images", r["isic_id"] + ".jpg")))
    return items


def clean(m):
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "segmentation", "unet_resnet18.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "masks_unet"))
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    model = UNetResNet18(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.eval()

    items = collect()
    if a.limit:
        items = items[: a.limit]
    rows, t0 = [], time.time()
    ex = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(6)

    def load(it):
        return Image.open(it[2]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)

    for s in range(0, len(items), a.bs):
        chunk = items[s : s + a.bs]
        ims = list(ex.map(load, chunk))
        x = torch.stack([torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1) for im in ims])
        with torch.no_grad():
            p = torch.sigmoid(model((x - MEAN) / STD))[:, 0].numpy()
        for (ds, iid, _), pr in zip(chunk, p):
            gp = gt_mask_path(iid)
            if gp:
                m = np.asarray(Image.open(gp).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127
                src = "gt"
            else:
                m = clean(pr > 0.5)
                src = "pred"
            Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(a.out, f"{iid}.png"))
            border = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
            rows.append({"isic_id": iid, "dataset": ds, "source": src, "area_frac": round(float(m.mean()), 4),
                         "touches_border": int(border), "mean_conf": round(float(pr[m].mean()) if m.any() else 0.0, 4)})
        if (s // a.bs) % 20 == 0:
            print(f"{s+len(chunk)}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(a.out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
