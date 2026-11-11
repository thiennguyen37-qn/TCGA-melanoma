"""Write the locked external test lists to experiments/external_ids.csv.

Same selection as notebooks/08_external_validation.ipynb:
  BALD         diagnosis_3 in {Nevus, Melanoma Invasive, Melanoma in situ}, minus the 6 images that
               duplicate internal images (confirmed visually, see data/resplit/external_overlap.csv).
  PAD-UFES-20  diagnosis_1 != Indeterminate and diagnosis_3 in {Nevus, Melanoma, NOS}.
cluster_id is the unit for bootstrap resampling: patient_id for PAD; BALD has no patient id in the
metadata, so each image is its own cluster (666 images come from 643 patients, so this is nearly exact).

Usage: python experiments/lock_external.py   (prints the SHA-256 that goes into PROTOCOL.md)
"""
import hashlib
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BALD_DUPLICATES = ["ISIC_0253805", "ISIC_6177828", "ISIC_6846234", "ISIC_0558386", "ISIC_4464596", "ISIC_8082578"]


def main():
    b = pd.read_csv(os.path.join(ROOT, "test", "bald", "metadata.csv"), low_memory=False)
    b = b[b.diagnosis_3.isin(["Nevus", "Melanoma Invasive", "Melanoma in situ"]) & ~b.isic_id.isin(BALD_DUPLICATES)]
    b = pd.DataFrame({"isic_id": b.isic_id, "dataset": "bald", "label": (b.diagnosis_3 != "Nevus").astype(int),
                      "cluster_id": b.isic_id})
    p = pd.read_csv(os.path.join(ROOT, "test", "pad-ufes-20", "metadata.csv"), low_memory=False)
    p = p[(p.diagnosis_1 != "Indeterminate") & p.diagnosis_3.isin(["Nevus", "Melanoma, NOS"])]
    p = pd.DataFrame({"isic_id": p.isic_id, "dataset": "pad", "label": (p.diagnosis_3 != "Nevus").astype(int),
                      "cluster_id": p.patient_id.fillna(p.isic_id)})
    out = pd.concat([b, p]).sort_values(["dataset", "isic_id"])
    path = os.path.join(ROOT, "experiments", "external_ids.csv")
    out.to_csv(path, index=False, lineterminator="\n")
    print(out.groupby("dataset").agg(n=("label", "size"), melanoma=("label", "sum"), clusters=("cluster_id", "nunique")))
    print("sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
