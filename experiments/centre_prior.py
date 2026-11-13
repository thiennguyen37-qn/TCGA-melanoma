"""Centre-prior reference for RQ4 (exploratory, PROTOCOL.md section 9).

How much of the attention-in-mask energy reported by rq4_attention.py could a map that ignores the image and
simply looks at the image centre achieve? For every evaluation image, the share of a fixed prior map's energy that
falls inside the lesion mask, for three image-independent maps:
  uniform    a constant map (its in-mask share equals the mask area)
  disc       a centred disc whose area is the median training-mask area
  mean_mask  the average of all training lesion masks (the empirical centre prior)
Compared with the per-image Grad-CAM++ energy of each variant (mean over seeds) from runs/rq4/energy_per_image.csv,
with paired cluster-bootstrap CIs (clusters as in rq4_attention.py).

Usage: python experiments/centre_prior.py --data data/cache320 --runs runs
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import SIZE  # noqa: E402
from rq4_attention import BOOT, BOOT_SEED, boot_weights, eval_sets  # noqa: E402


def load_mask(data_dir, isic_id):
    """Lesion mask at the evaluation resolution, resized as LesionDS + eval_tf do (nearest, via 320 px)."""
    m = cv2.imread(os.path.join(data_dir, "masks", isic_id + ".png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(os.path.join(data_dir, "images", isic_id + ".jpg"), cv2.IMREAD_GRAYSCALE)
    m = (cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cache320")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    out = os.path.join(a.runs, "rq4")
    per_image = pd.read_csv(os.path.join(out, "energy_per_image.csv"))

    sp = pd.read_csv(os.path.join(a.data, "split.csv"))
    train_ids = sp[sp.split == "train"].isic_id.tolist()
    mean_mask = np.zeros((SIZE, SIZE), np.float64)
    areas = []
    for k in train_ids:
        m = load_mask(a.data, k)
        mean_mask += m
        areas.append(m.mean())
    mean_mask /= len(train_ids)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.sqrt(np.median(areas) * SIZE * SIZE / np.pi)
    disc = (((yy - (SIZE - 1) / 2) ** 2 + (xx - (SIZE - 1) / 2) ** 2) <= r * r).astype(np.float64)
    print(f"training masks: {len(train_ids)}; median area {np.median(areas):.3f}; disc radius {r:.1f} px of {SIZE}")

    rows, comps = [], []
    for set_name, df in eval_sets(a.data).items():
        masks = np.stack([load_mask(a.data, k) for k in df.isic_id])
        prior = {"uniform": masks.mean((1, 2)),
                 "disc": (masks * disc).sum((1, 2)) / disc.sum(),
                 "mean_mask": (masks * mean_mask).sum((1, 2)) / mean_mask.sum()}
        sub = per_image[per_image.set == set_name]
        model = {v: g.groupby("isic_id").energy.mean().loc[df.isic_id].to_numpy() for v, g in sub.groupby("variant")}
        W = boot_weights(df.cluster.to_numpy(), BOOT, BOOT_SEED)
        wmean = lambda x: (W * x).sum(1) / W.sum(1)
        for name, e in list(prior.items()) + [(f"{v} (Grad-CAM++)", e) for v, e in sorted(model.items())]:
            rows.append({"set": set_name, "map": name, "energy_in_mask": float(np.nanmean(e)), "n": int(np.isfinite(e).sum())})
        for v, e in sorted(model.items()):
            for pname in ("disc", "mean_mask"):
                d = e - prior[pname]
                ok = np.isfinite(d)
                reps = (W[:, ok] * d[ok]).sum(1) / W[:, ok].sum(1)
                comps.append({"set": set_name, "variant": v, "minus": pname, "delta": float(d[ok].mean()),
                              "ci_lo": float(np.percentile(reps, 2.5)), "ci_hi": float(np.percentile(reps, 97.5)),
                              "share_images_above": float((d[ok] > 0).mean())})
    rows, comps = pd.DataFrame(rows), pd.DataFrame(comps)
    rows.to_csv(os.path.join(out, "centre_prior_energy.csv"), index=False)
    comps.to_csv(os.path.join(out, "centre_prior_comparisons.csv"), index=False)
    print("\n== Energy inside the lesion mask: image-independent maps vs trained models (mean over images) ==")
    print(rows.round(4).to_string(index=False))
    print("\n== Model minus centre prior, paired cluster bootstrap 95% CI; share of images where the model is higher ==")
    print(comps.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
