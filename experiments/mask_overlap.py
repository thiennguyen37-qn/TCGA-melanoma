"""How different is the V2 control from V1? Overlap between each training image's own lesion mask and the
mask it receives under the V2 permutation (same permutation as train.py: default_rng(seed) over the
training rows in split.csv order). Descriptive only; it does not use any model output.

Reports, per seed and over the images that actually receive L_att (received mask has qc_flag = 0):
  dice        Dice(own mask, received mask)
  own_cov     share of the own lesion covered by the received mask
  recv_in     share of the received mask that lies on the own lesion
plus the same values for the centred-disc reference (a disc with the median mask area at the image centre).

Usage: python experiments/mask_overlap.py --data data/cache320
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

RES = 64


def load_masks(data_dir, ids):
    out = np.zeros((len(ids), RES, RES), bool)
    for i, k in enumerate(ids):
        m = cv2.imread(os.path.join(data_dir, "masks", k + ".png"), cv2.IMREAD_GRAYSCALE)
        out[i] = cv2.resize(m, (RES, RES), interpolation=cv2.INTER_AREA) > 127
    return out


def overlap(a, b):
    inter = (a & b).sum((1, 2)).astype(float)
    sa, sb = a.sum((1, 2)).astype(float), b.sum((1, 2)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * inter / (sa + sb), inter / sa, inter / sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cache320")
    ap.add_argument("--out", default="runs/mask_overlap.csv")
    a = ap.parse_args()
    sp = pd.read_csv(os.path.join(a.data, "split.csv"))
    sp = sp[sp.split.isin(["train", "val", "test"])]
    mi = pd.read_csv(os.path.join(a.data, "mask_index.csv"), usecols=["isic_id", "qc_flag"])
    sp = sp.merge(mi, on="isic_id", how="left")
    sp["qc_flag"] = sp.qc_flag.fillna(1).astype(int)
    tr = sp[sp.split == "train"].reset_index(drop=True)  # same row order as train.py
    masks = load_masks(a.data, tr.isic_id)
    flags = tr.qc_flag.to_numpy()

    area = masks.mean((1, 2))
    yy, xx = np.mgrid[:RES, :RES]
    r = np.sqrt(np.median(area) * RES * RES / np.pi)
    disc = ((yy - (RES - 1) / 2) ** 2 + (xx - (RES - 1) / 2) ** 2 <= r * r)[None].repeat(len(tr), 0)

    rows = []
    use_own = flags == 0
    d, c, ri = overlap(masks, disc)
    rows.append({"comparison": "own vs centred disc", "seed": -1, "n": int(use_own.sum()),
                 "dice": np.nanmean(d[use_own]), "own_cov": np.nanmean(c[use_own]), "recv_in": np.nanmean(ri[use_own])})
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(tr))
        use = flags[perm] == 0  # train.py: the received mask's qc_flag decides whether L_att is applied
        d, c, ri = overlap(masks, masks[perm])
        rows.append({"comparison": "own vs V2 received", "seed": seed, "n": int(use.sum()),
                     "dice": np.nanmean(d[use]), "own_cov": np.nanmean(c[use]), "recv_in": np.nanmean(ri[use])})
    out = pd.DataFrame(rows)
    print(f"training images: {len(tr)}; median mask area {np.median(area):.3f}, mean {area.mean():.3f}")
    print(out.round(3).to_string(index=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
