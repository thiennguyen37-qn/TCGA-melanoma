"""Exploratory subgroup analysis on BALD (PROTOCOL.md section 9; not a pre-registered endpoint).

Subgroups: acral site (sole, palm; images without a site are left out of this split) and melanoma
invasiveness (in situ, invasive; each compared with all nevi). For every variant: AUC (mean over seeds), and
sensitivity / specificity at each run's validation threshold (mean over seeds). The reader vote fraction
(answer_mel_count / num_rated) gives the reader AUC in the same subgroups. 95% CIs from 2,000 bootstrap
resamples of images (one image per lesion on BALD), shared by every model, as in stats.py.

Usage: python experiments/subgroup_bald.py --runs runs
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import BOOT_SEED, ROOT, find_runs, weighted_auc  # noqa: E402

N_BOOT = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--external", default=os.path.join(ROOT, "experiments", "external_ids.csv"))
    ap.add_argument("--reader", default=os.path.join(ROOT, "test", "bald", "extra", "BALD_metadata_with_annotations_isic.csv"))
    a = ap.parse_args()
    runs, _ = find_runs(a.runs)
    ex = pd.read_csv(a.external)
    bald = ex[ex.dataset == "bald"][["isic_id", "label"]]
    rd = pd.read_csv(a.reader, sep=";")
    bald = bald.merge(rd[["isic_id", "site", "invasive", "answer_mel_count", "num_rated"]], on="isic_id", how="left")
    bald["reader"] = bald.answer_mel_count / bald.num_rated
    ids = bald.isic_id.to_numpy()

    probs, thr = {}, {}
    for v in sorted(runs):
        for s, d in sorted(runs[v].items()):
            p = pd.read_csv(os.path.join(d, "preds_bald.csv")).set_index("isic_id").loc[ids, "prob"].to_numpy()
            probs[(v, s)] = p
            with open(os.path.join(d, "summary.json")) as f:
                thr[(v, s)] = json.load(f)["threshold"]

    nev = bald.label.to_numpy() == 0
    groups = {
        "all": np.ones(len(bald), bool),
        "site=sole": (bald.site == "sole").to_numpy(),
        "site=palm": (bald.site == "palm").to_numpy(),
        "melanoma in situ vs all nevi": nev | (bald.invasive == "Insitu").to_numpy(),
        "invasive melanoma vs all nevi": nev | (bald.invasive == "Invasive").to_numpy(),
    }
    rng = np.random.default_rng(BOOT_SEED + 7)
    rows = []
    for gname, sel in groups.items():
        y = bald.label.to_numpy()[sel]
        n = int(sel.sum())
        counts = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float64)
        reader = bald.reader.to_numpy()[sel]
        r_pt = weighted_auc(y, reader, np.ones((1, n)))[0]
        r_bs = weighted_auc(y, reader, counts)
        rows.append({"subgroup": gname, "model": "reader", "n": n, "n_mel": int(y.sum()), "auc": r_pt,
                     "auc_lo": np.percentile(r_bs, 2.5), "auc_hi": np.percentile(r_bs, 97.5), "sens": np.nan, "spec": np.nan})
        for v in sorted(runs):
            seeds = sorted(runs[v])
            pts = [weighted_auc(y, probs[(v, s)][sel], np.ones((1, n)))[0] for s in seeds]
            bs = np.mean([weighted_auc(y, probs[(v, s)][sel], counts) for s in seeds], axis=0)
            sens = np.mean([(probs[(v, s)][sel][y == 1] >= thr[(v, s)]).mean() for s in seeds]) if y.sum() else np.nan
            spec = np.mean([(probs[(v, s)][sel][y == 0] < thr[(v, s)]).mean() for s in seeds]) if (y == 0).sum() else np.nan
            rows.append({"subgroup": gname, "model": v, "n": n, "n_mel": int(y.sum()), "auc": float(np.mean(pts)),
                         "auc_lo": np.percentile(bs, 2.5), "auc_hi": np.percentile(bs, 97.5), "sens": sens, "spec": spec})
    out = pd.DataFrame(rows)
    path = os.path.join(a.runs, "stats", "subgroup_bald.csv")
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200):
        print(out.round(3).to_string(index=False))
    print(f"\nimages without a site label: {int(bald.site.isna().sum())}; wrote {path}")


if __name__ == "__main__":
    main()
