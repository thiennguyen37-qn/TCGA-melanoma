"""Build the training bundle: every image downscaled to 320x320 plus masks and the locked lists.

The originals are ~5.4 GB of large JPEGs, too slow to read from Google Drive. All variants read the
same 320x320 copies (squashed like notebook 04's Resize((224, 224))); the final resize to 224 and the
augmentations happen in the training script, so the mix policy is still applied before the final resize.

Output (default data/cache320/, gitignored):
  images/<isic_id>.jpg   internal and external images, 320x320, JPEG quality 95
  masks/<isic_id>.png    lesion masks (copied from data/masks_unet, 256x256, 0/255)
  split.csv              data/resplit/split.csv
  mask_index.csv         data/masks_unet/index.csv (source, qc_flag)
  external_ids.csv       experiments/external_ids.csv
and bundle320.zip next to it when --zip is given.

Usage: python experiments/prepare_cache.py --zip
"""
import argparse
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = 320


def sources():
    out = []
    sp = pd.read_csv(os.path.join(ROOT, "data", "resplit", "split.csv"), usecols=["isic_id"])
    out += [(i, os.path.join(ROOT, "images", i + ".jpg")) for i in sp.isic_id]
    ex = pd.read_csv(os.path.join(ROOT, "experiments", "external_ids.csv"))
    sub = {"bald": "bald", "pad": "pad-ufes-20"}
    out += [(i, os.path.join(ROOT, "test", sub[d], "images", i + ".jpg")) for i, d in zip(ex.isic_id, ex.dataset)]
    return out


def shrink(item, out_dir):
    iid, src = item
    dst = os.path.join(out_dir, "images", iid + ".jpg")
    if os.path.exists(dst):
        return
    im = Image.open(src)
    im.draft("RGB", (2 * SIZE, 2 * SIZE))  # fast JPEG downscale, still >= 2x the target
    im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache320"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)

    items = sources()
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(lambda it: shrink(it, a.out), items), 1):
            if k % 2000 == 0:
                print(f"{k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    for iid, _ in items:
        src = os.path.join(ROOT, "data", "masks_unet", iid + ".png")
        dst = os.path.join(a.out, "masks", iid + ".png")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    shutil.copyfile(os.path.join(ROOT, "data", "resplit", "split.csv"), os.path.join(a.out, "split.csv"))
    shutil.copyfile(os.path.join(ROOT, "data", "masks_unet", "index.csv"), os.path.join(a.out, "mask_index.csv"))
    shutil.copyfile(os.path.join(ROOT, "experiments", "external_ids.csv"), os.path.join(a.out, "external_ids.csv"))
    print(f"images {len(items)} in {time.time()-t0:.0f}s", flush=True)

    if a.zip:
        zp = os.path.join(os.path.dirname(a.out), "bundle320.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:  # JPEG/PNG are already compressed
            for dp, _, fs in os.walk(a.out):
                for f in fs:
                    full = os.path.join(dp, f)
                    z.write(full, os.path.relpath(full, a.out))
        print(f"wrote {zp} ({os.path.getsize(zp) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
