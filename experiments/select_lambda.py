"""Pick lambda by the rule in PROTOCOL.md section 4.

Reads runs/V0_s0/summary.json and runs/V1_s0_lam<x>_lamsearch/summary.json for the grid
{0.1, 0.3, 1, 3}. Rule: the largest lambda whose validation AUC (TTA) is >= V0 seed-0 validation AUC
- 0.005; if none qualifies, the lambda with the highest validation AUC. External results are never read.

Usage: python experiments/select_lambda.py --runs runs
"""
import argparse
import json
import os

GRID = [0.1, 0.3, 1.0, 3.0]
MARGIN = 0.005


def val_auc(path):
    with open(path) as f:
        return json.load(f)["results"]["val"]["auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    ref = val_auc(os.path.join(a.runs, "V0_s0", "summary.json"))
    rows = []
    for lam in GRID:
        p = os.path.join(a.runs, f"V1_s0_lam{lam:g}_lamsearch", "summary.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        rows.append((lam, val_auc(p)))
    ok = [lam for lam, auc in rows if auc >= ref - MARGIN]
    chosen = max(ok) if ok else max(rows, key=lambda r: r[1])[0]
    print(f"V0 seed 0 val AUC {ref:.4f}; threshold {ref - MARGIN:.4f}")
    for lam, auc in rows:
        print(f"  lambda {lam:g}: val AUC {auc:.4f} {'ok' if auc >= ref - MARGIN else ''}")
    print(f"chosen lambda = {chosen:g}" + ("" if ok else " (no lambda within margin; highest val AUC)"))
    with open(os.path.join(a.runs, "lambda_choice.json"), "w") as f:
        json.dump({"reference_val_auc": ref, "grid": dict(rows), "chosen": chosen}, f, indent=2)


if __name__ == "__main__":
    main()
