"""Paired cluster-bootstrap comparisons between variants (PROTOCOL.md sections 6 and 7).

For each comparison A - B on one evaluation set, every bootstrap replicate resamples clusters with
replacement (patient-level groups from data/resplit/split.csv for the internal test set, PAD-UFES-20
patients, BALD images), computes AUC(A, seed s) - AUC(B, seed s) on that replicate for every seed the
two variants share, and averages over seeds. The same replicates are used for every model on a set, so
all comparisons are paired. CI = 2.5/97.5 percentiles; two-sided p from the bootstrap distribution.
Primary endpoint (V1 - V0 on BALD and on PAD-UFES-20) is Holm-corrected; the others are exploratory.
RQ5 compares each variant's seed-averaged AUC with the reader vote fraction on BALD.

Only preds_*.csv and summary.json are read, so the checkpoints do not need to be downloaded.

Usage: python experiments/stats.py --runs runs [--boot 2000]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ("test", "bald", "pad")
PRIMARY = [("V1", "V0", "bald"), ("V1", "V0", "pad")]
SECONDARY = [("V1", "V0", "test"),
             ("V1", "V2", "bald"), ("V1", "V2", "pad"),
             ("V1", "V3", "bald"), ("V1", "V3", "pad"),
             ("V4", "V3", "bald"), ("V4", "V3", "pad")]
MASK_VARIANTS = ("V1", "V2", "V4")
BOOT_SEED = 20261111
ALPHA = 0.05


def find_runs(runs_dir):
    """{variant: {seed: run_dir}} for the main runs (lambda-search runs carry a tag and are skipped)."""
    lam = None
    choice = os.path.join(runs_dir, "lambda_choice.json")
    if os.path.exists(choice):
        with open(choice) as f:
            lam = json.load(f)["chosen"]
    out = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        if s["args"].get("tag"):
            continue
        if s["variant"] in MASK_VARIANTS and lam is not None and abs(s["lam"] - lam) > 1e-12:
            continue
        if s["seed"] in out.setdefault(s["variant"], {}):
            raise SystemExit(f"two main runs for {s['variant']} seed {s['seed']}")
        out[s["variant"]][s["seed"]] = os.path.dirname(path)
    return out, lam


def clusters(set_name, split_path, ext_path):
    if set_name == "test":
        sp = pd.read_csv(split_path)
        return dict(zip(sp.isic_id, sp.group_id.astype(str)))
    ex = pd.read_csv(ext_path)
    ex = ex[ex.dataset == set_name]
    return dict(zip(ex.isic_id, ex.cluster_id.astype(str)))


def weighted_auc(y, p, w):
    """AUC for every row of the weight matrix w (B, n); ties count one half, as sklearn."""
    o = np.argsort(p, kind="mergesort")
    y, p, w = y[o], p[o], w[:, o]
    start = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    pos = np.add.reduceat(w * (y == 1), start, axis=1)
    neg = np.add.reduceat(w * (y == 0), start, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


class SetData:
    """Bootstrap replicates and per-model AUC replicates for one evaluation set."""

    def __init__(self, name, ids, y, cl, n_boot, chunk=250):
        self.name, self.ids, self.y = name, ids, y
        codes, uniq = pd.factorize(pd.Series(cl))
        self.n_clusters = len(uniq)
        rng = np.random.default_rng(BOOT_SEED + SETS.index(name) if name in SETS else BOOT_SEED)
        counts = rng.multinomial(self.n_clusters, np.full(self.n_clusters, 1 / self.n_clusters), size=n_boot)
        self.w = counts[:, codes].astype(np.float64)
        self.chunk = chunk
        self.cache = {}

    def auc(self, key, p):
        """(point AUC, replicate AUCs) for one model's probabilities aligned to self.ids."""
        if key not in self.cache:
            point = weighted_auc(self.y, p, np.ones((1, len(p))))[0]
            reps = np.concatenate([weighted_auc(self.y, p, self.w[i:i + self.chunk])
                                   for i in range(0, len(self.w), self.chunk)])
            self.cache[key] = (point, reps)
        return self.cache[key]


def load_preds(run_dir, set_name, ref_ids):
    f = os.path.join(run_dir, f"preds_{set_name}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).set_index("isic_id")
    if set(d.index) != set(ref_ids):
        raise SystemExit(f"{f}: image ids differ from the locked list")
    return d.loc[ref_ids, "prob"].to_numpy(np.float64)


def boot_p(reps):
    """Two-sided bootstrap p-value for H0: delta = 0."""
    b = len(reps)
    lo = (np.sum(reps <= 0) + 1) / (b + 1)
    hi = (np.sum(reps >= 0) + 1) / (b + 1)
    return min(1.0, 2 * min(lo, hi))


def compare(sd, runs, a, b):
    seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    pts, reps = [], []
    for s in seeds:
        pa = load_preds(runs[a][s], sd.name, sd.ids)
        pb = load_preds(runs[b][s], sd.name, sd.ids)
        if pa is None or pb is None:
            continue
        (xa, ra), (xb, rb) = sd.auc((a, s), pa), sd.auc((b, s), pb)
        pts.append(xa - xb); reps.append(ra - rb)
    if not pts:
        return None
    reps = np.mean(reps, axis=0)
    return {"a": a, "b": b, "set": sd.name, "n_seeds": len(pts), "delta": float(np.mean(pts)),
            "sd_seeds": float(np.std(pts, ddof=1)) if len(pts) > 1 else float("nan"),
            "ci_lo": float(np.percentile(reps, 2.5)), "ci_hi": float(np.percentile(reps, 97.5)),
            "p": boot_p(reps), "_reps": reps}


def holm(rows):
    """Holm-adjusted p and the matching CI level (1 - alpha / (m - rank)) for each primary row."""
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        r = rows[i]
        running = max(running, min(1.0, (m - rank) * r["p"]))
        r["p_holm"] = running
        level = ALPHA / (m - rank)
        r["ci_level"] = 1 - level
        r["ci_lo_holm"] = float(np.percentile(r["_reps"], 100 * level / 2))
        r["ci_hi_holm"] = float(np.percentile(r["_reps"], 100 * (1 - level / 2)))
        r["improved"] = bool(r["p_holm"] < ALPHA and r["delta"] > 0)


def reader_rows(sd, runs, reader_path):
    rd = pd.read_csv(reader_path, sep=";").set_index("isic_id")
    r = (rd.loc[sd.ids, "answer_mel_count"] / rd.loc[sd.ids, "num_rated"]).to_numpy(np.float64)
    xr, rr = sd.auc(("reader", 0), r)
    out = [{"a": "reader", "b": "-", "set": "bald", "n_seeds": 0, "delta": float(xr), "sd_seeds": float("nan"),
            "ci_lo": float(np.percentile(rr, 2.5)), "ci_hi": float(np.percentile(rr, 97.5)), "p": float("nan")}]
    for v in sorted(runs):
        pts, reps = [], []
        for s in sorted(runs[v]):
            p = load_preds(runs[v][s], "bald", sd.ids)
            if p is None:
                continue
            x, rep = sd.auc((v, s), p)
            pts.append(x); reps.append(rep)
        if not pts:
            continue
        reps = np.mean(reps, axis=0) - rr
        out.append({"a": v, "b": "reader", "set": "bald", "n_seeds": len(pts), "delta": float(np.mean(pts) - xr),
                    "sd_seeds": float(np.std(pts, ddof=1)) if len(pts) > 1 else float("nan"),
                    "ci_lo": float(np.percentile(reps, 2.5)), "ci_hi": float(np.percentile(reps, 97.5)),
                    "p": boot_p(reps)})
    return out


def variant_table(runs):
    rows = []
    for v in sorted(runs):
        for s, d in sorted(runs[v].items()):
            with open(os.path.join(d, "summary.json")) as f:
                res = json.load(f)["results"]
            for set_name, m in res.items():
                rows.append({"variant": v, "seed": s, "set": set_name, **{k: m[k] for k in ("auc", "sens", "spec")}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["variant", "set"])
    tab = g[["auc", "sens", "spec"]].agg(["mean", "std"])
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab["n_seeds"] = g.size()
    return tab.reset_index()


def fmt(r):
    s = f"{r['a']:>6} - {r['b']:<6} {r['set']:<5} seeds={r['n_seeds']}  delta={r['delta']:+.4f}"
    s += f"  95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    if not np.isnan(r["p"]):
        s += f"  p={r['p']:.4f}"
    if "p_holm" in r:
        s += (f"  p_holm={r['p_holm']:.4f}  {100 * r['ci_level']:.1f}% CI "
              f"[{r['ci_lo_holm']:+.4f}, {r['ci_hi_holm']:+.4f}]  {'IMPROVED' if r['improved'] else 'not shown'}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default=os.path.join(ROOT, "data", "resplit", "split.csv"))
    ap.add_argument("--external", default=os.path.join(ROOT, "experiments", "external_ids.csv"))
    ap.add_argument("--reader", default=os.path.join(ROOT, "test", "bald", "extra", "BALD_metadata_with_annotations_isic.csv"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="output folder (default: <runs>/stats)")
    a = ap.parse_args()

    runs, lam = find_runs(a.runs)
    print(f"chosen lambda: {lam}; main runs found: " + ", ".join(f"{v}:{sorted(s)}" for v, s in sorted(runs.items())))

    sets = {}
    for name in SETS:
        cl = clusters(name, a.split, a.external)
        ref = next((pd.read_csv(os.path.join(d, f"preds_{name}.csv")) for v in runs for d in runs[v].values()
                    if os.path.exists(os.path.join(d, f"preds_{name}.csv"))), None)
        if ref is None:
            continue
        missing = set(ref.isic_id) - set(cl)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} images without a cluster id")
        ids = ref.isic_id.to_numpy()
        sets[name] = SetData(name, ids, ref.label.to_numpy(), [cl[i] for i in ids], a.boot)
        print(f"{name}: {len(ids)} images, {int(ref.label.sum())} melanoma, {sets[name].n_clusters} clusters")

    primary = [r for a_, b_, s in PRIMARY if s in sets and (r := compare(sets[s], runs, a_, b_))]
    if len(primary) == len(PRIMARY):
        holm(primary)
    secondary = [r for a_, b_, s in SECONDARY if s in sets and (r := compare(sets[s], runs, a_, b_))]
    rq5 = reader_rows(sets["bald"], runs, a.reader) if "bald" in sets and os.path.exists(a.reader) else []

    print("\n== Primary (RQ1), Holm over BALD and PAD ==")
    for r in primary:
        print(fmt(r))
    if primary and len(primary) < len(PRIMARY):
        print("   (Holm not applied: one primary comparison is missing)")
    print("\n== Secondary (exploratory, no correction) ==")
    for r in secondary:
        print(fmt(r))
    print("\n== RQ5: model AUC - reader AUC on BALD (first row: reader AUC itself) ==")
    for r in rq5:
        print(fmt(r))

    tab = variant_table(runs)
    print("\n== Per variant, mean (SD) over seeds; sens/spec at the val threshold ==")
    if not tab.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(tab.round(4).to_string(index=False))

    out = a.out or os.path.join(a.runs, "stats")
    os.makedirs(out, exist_ok=True)
    rows = [{k: v for k, v in r.items() if k != "_reps"} | {"group": g}
            for g, rs in (("primary", primary), ("secondary", secondary), ("rq5", rq5)) for r in rs]
    pd.DataFrame(rows).to_csv(os.path.join(out, "comparisons.csv"), index=False)
    tab.to_csv(os.path.join(out, "per_variant.csv"), index=False)
    print(f"\nwrote {out}/comparisons.csv and per_variant.csv")


if __name__ == "__main__":
    main()
