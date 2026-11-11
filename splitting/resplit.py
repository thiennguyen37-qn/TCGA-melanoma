"""Leakage-controlled train/val/test split for the melanoma-vs-nevus data.

Why: notebook 01 grouped each image by ONE key (lesion_id, else patient_id, else isic_id), so the
different lesions of one patient could land in train and test, and re-uploaded copies of the same
image (no shared id) were never linked.

How: two images are linked when they share a lesion_id, OR share a patient_id, OR are near-duplicates
by perceptual hash. Each connected component is one group; groups are assigned to train/val/test
(70/15/15) with a greedy rule that balances both image count and melanoma count. Near-duplicate
pairs whose labels disagree are dropped (label conflict in the source).

Steps:
  python splitting/resplit.py hash                 # perceptual hash of every internal + external image
  python splitting/resplit.py thumbs               # cache 64x64 thumbnails (data/resplit/cache/, gitignored)
  python splitting/resplit.py pairs --max-dist 12  # hash candidates, each verified by centre pixel correlation
  python splitting/resplit.py split --keep-indeterminate  # build groups and write the split
  python splitting/resplit.py export               # write data/splitted/*.csv, then re-run notebook 03

Indeterminate lesions (80 images) are kept and labelled nevus, as in the raw `diagnosis` column
(decision of 2026-11-11); pass no flag to drop them instead.

The perceptual hash alone is not enough for dermoscopy: images share the same layout (dark blob on
skin, dark vignette), so many different lesions get near-identical hashes. It only proposes
candidates; the centre-crop pixel correlation decides. Thresholds were set by visual review.

Outputs go to data/resplit/.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "resplit")
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---------- perceptual hash ----------

def phash(path):
    """64-bit DCT perceptual hash (same idea as imagehash.phash, written out to avoid a dependency)."""
    im = Image.open(path)
    im.draft("L", (128, 128))  # JPEG DCT downscaling: decodes large images several times faster
    a = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dctn(a, norm="ortho")[:8, :8].ravel()
    bits = d > np.median(d[1:])  # skip the DC term when taking the median
    return int(np.packbits(bits).view(">u8")[0])


def image_list():
    raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "metadata.csv"), usecols=["isic_id"])
    items = [(i, "internal", os.path.join(ROOT, "images", i + ".jpg")) for i in raw.isic_id]
    for ds, d in (("bald", "bald"), ("pad", "pad-ufes-20")):
        m = pd.read_csv(os.path.join(ROOT, "test", d, "metadata.csv"), usecols=["isic_id"])
        items += [(i, ds, os.path.join(ROOT, "test", d, "images", i + ".jpg")) for i in m.isic_id]
    return items


def cmd_hash(a):
    items = image_list()
    if a.limit:
        items = items[: a.limit]
    t0 = time.time()

    def one(it):
        try:
            return it[0], it[1], f"{phash(it[2]):016x}"
        except Exception as e:  # unreadable file: record it, do not crash the run
            print("fail", it[0], e, flush=True)
            return it[0], it[1], ""

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(one, items), 1):
            rows.append(r)
            if k % 2000 == 0:
                print(f"{k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows, columns=["isic_id", "dataset", "phash"]).to_csv(os.path.join(OUT, "phash.csv"), index=False)
    print(f"done {len(rows)} in {time.time()-t0:.0f}s", flush=True)


# ---------- near-duplicate pairs ----------

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_rows(block, allh):
    x = (block[:, None] ^ allh[None, :]).view(np.uint8).reshape(len(block), len(allh), 8)
    return _POP[x].sum(-1, dtype=np.uint16)


def near_pairs(h, max_dist, bs=512):
    """All pairs (i < j) with Hamming distance <= max_dist."""
    out = []
    for s in range(0, len(h), bs):
        d = hamming_rows(h[s : s + bs], h)
        i, j = np.nonzero(d <= max_dist)
        i = i + s
        keep = i < j
        out.append(np.stack([i[keep], j[keep], d[i[keep] - s, j[keep]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), int)


def load_hashes():
    ph = pd.read_csv(os.path.join(OUT, "phash.csv"), dtype={"phash": str})
    ph = ph[ph.phash.notna() & (ph.phash != "")].reset_index(drop=True)
    h = np.array([int(x, 16) for x in ph.phash], dtype=np.uint64)
    return ph, h


THUMB = 64
CACHE = os.path.join(OUT, "cache")


def thumb(path):
    im = Image.open(path)
    im.draft("RGB", (2 * THUMB, 2 * THUMB))
    return np.asarray(im.convert("RGB").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8)


def cmd_thumbs(a):
    """Cache 64x64 RGB thumbnails, in the same row order as phash.csv (used to verify hash candidates)."""
    ph, _ = load_hashes()
    paths = dict((i, p) for i, _, p in image_list())
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        arr = np.stack(list(ex.map(lambda i: thumb(paths[i]), ph.isic_id)))
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, "thumbs64.npy"), arr)
    print(f"done {arr.shape} in {time.time()-t0:.0f}s", flush=True)


def pixel_corr(t, i, j, bs=4096):
    """Pearson correlation between the central half (32x32 RGB) of the thumbnails of pairs (i[k], j[k]).
    The centre is used because the black circular vignette of some dermoscopes (e.g. Barcelona)
    makes whole-image correlation high even for different lesions. Each colour channel is
    standardised on its own, so a re-upload with a colour cast (BALD is colour-adjusted) still matches."""
    q = THUMB // 4
    t = t[:, q : THUMB - q, q : THUMB - q]

    def z(v):
        v = v.reshape(len(v), -1, 3).astype(np.float32)
        return (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-6)

    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), bs):
        out[s : s + bs] = (z(t[i[s : s + bs]]) * z(t[j[s : s + bs]])).mean((1, 2))
    return out


def cmd_pairs(a):
    """Hash candidates (Hamming <= max_dist), each verified by thumbnail pixel correlation."""
    ph, h = load_hashes()
    p = near_pairs(h, a.max_dist)
    t = np.load(os.path.join(CACHE, "thumbs64.npy"))
    df = pd.DataFrame({"a": ph.isic_id.values[p[:, 0]], "a_set": ph.dataset.values[p[:, 0]],
                       "b": ph.isic_id.values[p[:, 1]], "b_set": ph.dataset.values[p[:, 1]], "dist": p[:, 2],
                       "corr": pixel_corr(t, p[:, 0], p[:, 1]).round(4)})
    df.sort_values("corr", ascending=False).to_csv(os.path.join(OUT, "near_dup_pairs.csv"), index=False)
    bins = [-1, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.01]
    df["corr_bin"] = pd.cut(df["corr"], bins)
    print(f"{len(df)} candidate pairs (hash dist <= {a.max_dist})")
    print(pd.crosstab(df.corr_bin, df.a_set + "-" + df.b_set).to_string())


# ---------- grouping and split ----------

class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def label_table(drop_indeterminate):
    raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "metadata.csv"), low_memory=False,
                      usecols=["isic_id", "lesion_id", "patient_id", "diagnosis_1", "attribution"])
    raw["label"] = (raw.diagnosis_1 == "Malignant").astype(int)
    raw["drop_reason"] = ""
    if drop_indeterminate:
        raw.loc[raw.diagnosis_1 == "Indeterminate", "drop_reason"] = "indeterminate"
    return raw


def assign(groups, seed):
    """Greedy stratified group assignment. Groups are shuffled, then taken largest first; each goes
    to the split that keeps image counts AND melanoma counts closest to the 70/15/15 proportions
    (minimises the spread of count/target across splits for both quantities)."""
    g = groups.sample(frac=1, random_state=seed).sort_values("n", ascending=False, kind="stable")
    names = list(FRACS)
    f = np.array([FRACS[s] for s in names])
    tot = np.array([g.n.sum(), g.m.sum()], dtype=float)
    cur = np.zeros((len(names), 2))
    out = {}
    for gid, n, m in zip(g.index, g.n, g.m):
        add = np.array([n, m], dtype=float)
        costs = []
        for k in range(len(names)):
            c = cur.copy(); c[k] += add
            r = c / f[:, None] / tot  # fill ratio per split, per quantity
            costs.append(r[:, 0].var() + r[:, 1].var())
        k = int(np.argmin(costs))
        cur[k] += add
        out[gid] = names[k]
    return out


def cmd_split(a):
    meta = label_table(a.drop_indeterminate)
    idx ={k: i for i, k in enumerate(meta.isic_id)}
    dsu = DSU(len(meta))

    # 1) shared lesion_id or patient_id
    n_links = {"lesion": 0, "patient": 0, "near_dup": 0}
    for col, key in (("lesion_id", "lesion"), ("patient_id", "patient")):
        for _, ids in meta.dropna(subset=[col]).groupby(col).isic_id:
            ids = [idx[i] for i in ids]
            for j in ids[1:]:
                dsu.union(ids[0], j); n_links[key] += 1

    # 2) near-duplicates among internal images. Two tiers (chosen by visual review):
    #    link  (loose):  hash dist <= link_dist and centre corr >= link_corr -> same group. A false link only
    #                    makes a group bigger, which is harmless; a missed link is a leak, so this errs wide.
    #    dup   (strict): hash dist <= dup_dist and centre corr >= dup_corr -> the same picture. Only these
    #                    pairs count as label conflicts when their labels disagree.
    cand = pd.read_csv(os.path.join(OUT, "near_dup_pairs.csv"))
    cand = cand[(cand.a_set == "internal") & (cand.b_set == "internal")]
    link = cand[(cand.dist <= a.link_dist) & (cand["corr"] >= a.link_corr)]
    pairs = list(zip(link.a, link.b, link["corr"], link.dist))
    lab = meta.set_index("isic_id").label
    pre_drop = set(meta.isic_id[meta.drop_reason != ""])  # e.g. indeterminate: not a real label conflict
    conflicts = []
    for x, y, c, d in pairs:
        dsu.union(idx[x], idx[y]); n_links["near_dup"] += 1
        if d <= a.dup_dist and c >= a.dup_corr and lab[x] != lab[y] and not ({x, y} & pre_drop):
            conflicts.append((x, y, float(c)))
    for x, y, _ in conflicts:
        meta.loc[meta.isic_id.isin([x, y]) & (meta.drop_reason == ""), "drop_reason"] = "label_conflict_dup"

    meta["group_id"] = [dsu.find(i) for i in range(len(meta))]
    keep = meta[meta.drop_reason == ""]
    groups = keep.groupby("group_id").agg(n=("label", "size"), m=("label", "sum"))
    split = assign(groups, a.seed)
    meta["split"] = meta.group_id.map(split).fillna("dropped")
    meta.loc[meta.drop_reason != "", "split"] = "dropped"

    os.makedirs(OUT, exist_ok=True)
    cols = ["isic_id", "lesion_id", "patient_id", "attribution", "label", "group_id", "split", "drop_reason"]
    meta[cols].to_csv(os.path.join(OUT, "split.csv"), index=False)
    pd.DataFrame(conflicts, columns=["a", "b", "corr"]).to_csv(os.path.join(OUT, "label_conflicts.csv"), index=False)

    # ---------- report ----------
    print(f"links: {n_links}; near-dup links (dist<={a.link_dist}, corr>={a.link_corr}): {len(pairs)}")
    print(f"groups: {meta.group_id.nunique()}  largest: {groups.n.max()} images  "
          f"groups >50 images: {(groups.n > 50).sum()}")
    print("dropped:", meta.drop_reason.value_counts().to_dict())
    print("label conflicts:", conflicts)
    rep = meta[meta.split != "dropped"].groupby("split").label.agg(n="size", melanoma="sum")
    rep["mel_frac"] = (rep.melanoma / rep.n).round(4)
    rep["share"] = (rep.n / rep.n.sum()).round(4)
    print(rep.to_string())
    # leakage checks: no lesion, patient or near-dup group may span two splits
    k = meta[meta.split != "dropped"]
    for col in ("lesion_id", "patient_id", "group_id"):
        span = (k.dropna(subset=[col]).groupby(col).split.nunique() > 1).sum()
        print(f"{col} spanning >1 split: {span}")
    sp =meta.set_index("isic_id").split
    cross = sum(sp[x] != sp[y] and "dropped" not in (sp[x], sp[y]) for x, y, _, _ in pairs)
    print(f"near-dup pairs spanning >1 split: {cross}")
    # external sets: any internal image that is the same picture as an external one
    ext = pd.read_csv(os.path.join(OUT, "near_dup_pairs.csv"))
    # loose (link) rule on purpose: missing an overlap with an external set is the costly error
    ext = ext[(ext.a_set == "internal") & (ext.b_set != "internal") & (ext.dist <= a.link_dist) & (ext["corr"] >= a.link_corr)]
    ext = ext.assign(split_a=ext.a.map(sp))
    ext.to_csv(os.path.join(OUT, "external_overlap.csv"), index=False)
    print(f"internal-external candidate duplicates (link rule): {len(ext)}")
    if len(ext):
        print(ext[["a", "b", "b_set", "dist", "corr", "split_a"]].to_string(index=False))


SPLITTED_COLS = ["isic_id", "attribution", "age_approx", "anatom_site_general", "concomitant_biopsy",
                 "diagnosis", "lesion_id", "melanocytic", "patient_id", "sex"]  # columns kept by notebook 01


def cmd_export(a):
    """Write data/splitted/{train,val,test}.csv (the input of notebook 03) from data/resplit/split.csv.
    Replaces the GroupShuffleSplit output of notebook 01; the old files stay in git history."""
    raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "metadata.csv"), low_memory=False)[SPLITTED_COLS]
    sp = pd.read_csv(os.path.join(OUT, "split.csv"), usecols=["isic_id", "split"]).set_index("isic_id").split
    raw["split"] = raw.isic_id.map(sp)
    for s in FRACS:
        part = raw[raw.split == s].drop(columns="split")
        part.to_csv(os.path.join(ROOT, "data", "splitted", f"{s}.csv"), index=False)
        print(s, len(part), part.diagnosis.value_counts().to_dict())
    print("not exported (dropped):", int((raw.split == "dropped").sum()))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hash"); p.add_argument("--limit", type=int, default=0); p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("thumbs"); p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("pairs"); p.add_argument("--max-dist", type=int, default=12)
    p = sub.add_parser("split")
    p.add_argument("--link-dist", type=int, default=8); p.add_argument("--link-corr", type=float, default=0.93)
    p.add_argument("--dup-dist", type=int, default=2); p.add_argument("--dup-corr", type=float, default=0.97)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-indeterminate", dest="drop_indeterminate", action="store_false")
    sub.add_parser("export")
    a = ap.parse_args()
    {"hash": cmd_hash, "thumbs": cmd_thumbs, "pairs": cmd_pairs, "split": cmd_split, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
