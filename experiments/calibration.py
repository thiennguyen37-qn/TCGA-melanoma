"""Calibration of every main run (exploratory, PROTOCOL.md section 9; requested by JAMA guidance for AI studies).

For each run and each evaluation set (val, internal test, BALD, PAD-UFES-20), from the TTA probabilities in
preds_<set>.csv:
  prevalence / mean_prob   observed melanoma fraction and mean predicted probability
  brier                    mean squared error of the probability
  cal_intercept            calibration-in-the-large: a in logit P(y=1) = a + logit(p) (ideal 0; < 0 = over-predicts)
  cal_slope                b in logit P(y=1) = c + b * logit(p) (ideal 1; < 1 = too extreme, > 1 = too modest)
  ece                      expected calibration error over 10 equal-width bins
Per variant: mean (SD) over seeds. Binned reliability points (mean over seeds) are written for plotting.

Usage: python experiments/calibration.py --runs runs
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import find_runs  # noqa: E402

SETS = ("val", "test", "bald", "pad")
BINS = np.linspace(0, 1, 11)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_intercept(y, p):
    lp = logit(p)
    f = lambda a: (1 / (1 + np.exp(-(a + lp)))).mean() - y.mean()
    return brentq(f, -20, 20)


def cal_slope(y, p):
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)  # unpenalised
    m.fit(logit(p)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def ece(y, p):
    idx = np.clip(np.digitize(p, BINS[1:-1]), 0, len(BINS) - 2)
    total = 0.0
    for b in range(len(BINS) - 1):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return total


def reliability(y, p):
    idx = np.clip(np.digitize(p, BINS[1:-1]), 0, len(BINS) - 2)
    rows = []
    for b in range(len(BINS) - 1):
        sel = idx == b
        rows.append((b, int(sel.sum()), p[sel].mean() if sel.any() else np.nan, y[sel].mean() if sel.any() else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    runs, _ = find_runs(a.runs)
    per_run, curve = [], []
    for v in sorted(runs):
        for s, d in sorted(runs[v].items()):
            for set_name in SETS:
                f = os.path.join(d, f"preds_{set_name}.csv")
                if not os.path.exists(f):
                    continue
                df = pd.read_csv(f)
                y, p = df.label.to_numpy().astype(float), df.prob.to_numpy().astype(float)
                slope, _ = cal_slope(y, p)
                per_run.append({"variant": v, "seed": s, "set": set_name, "n": len(y), "prevalence": y.mean(),
                                "mean_prob": p.mean(), "brier": float(((p - y) ** 2).mean()),
                                "cal_intercept": cal_intercept(y, p), "cal_slope": slope, "ece": ece(y, p)})
                for b, n, mp, obs in reliability(y, p):
                    curve.append({"variant": v, "seed": s, "set": set_name, "bin": b, "n": n, "mean_prob": mp, "observed": obs})
    per_run = pd.DataFrame(per_run)
    metrics = ["prevalence", "mean_prob", "brier", "cal_intercept", "cal_slope", "ece"]
    g = per_run.groupby(["set", "variant"])[metrics]
    tab = g.mean().round(4).astype(str) + " (" + g.std().round(4).astype(str) + ")"
    tab = tab.reset_index()
    tab["set"] = pd.Categorical(tab["set"], SETS, ordered=True)
    tab = tab.sort_values(["set", "variant"])
    curve = pd.DataFrame(curve).groupby(["set", "variant", "bin"]).agg(
        n=("n", "mean"), mean_prob=("mean_prob", "mean"), observed=("observed", "mean")).reset_index()
    out = os.path.join(a.runs, "stats")
    os.makedirs(out, exist_ok=True)
    per_run.to_csv(os.path.join(out, "calibration_per_run.csv"), index=False)
    tab.to_csv(os.path.join(out, "calibration_per_variant.csv"), index=False)
    curve.to_csv(os.path.join(out, "calibration_curves.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("== Calibration, mean (SD) over 5 seeds ==")
        print(tab.to_string(index=False))
    print(f"\nwrote {out}/calibration_per_run.csv, calibration_per_variant.csv, calibration_curves.csv")


if __name__ == "__main__":
    main()
