"""Table 1 (TRIPOD+AI items 20b, 20c): characteristics of the development and evaluation data, and the flow counts
for the participant-flow diagram (item 20a).

Rows per set (train, val, internal test, BALD, PAD-UFES-20): images, lesions, patients with an id, melanoma n (%),
melanoma subtype, reference standard, age mean (SD) and % missing, sex, anatomical site (harmonised), Fitzpatrick
type where available.

Usage: python experiments/table1.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SITE = {  # harmonised anatomical site
    "anterior torso": "Trunk", "posterior torso": "Trunk", "lateral torso": "Trunk", "Trunk": "Trunk",
    "head/neck": "Head and neck", "Head and neck": "Head and neck",
    "upper extremity": "Upper extremity", "Upper extremity": "Upper extremity",
    "lower extremity": "Lower extremity", "Lower extremity": "Lower extremity",
    "palms/soles": "Palms or soles",
}


def pct(n, d):
    return f"{n} ({100 * n / d:.1f}%)" if d else "0"


def describe(df, name):
    n = len(df)
    mel = int(df.label.sum())
    row = {"set": name, "images": n,
           "lesions": df.lesion_id.nunique() if "lesion_id" in df and df.lesion_id.notna().any() else "NA",
           "images with patient id": pct(int(df.patient_id.notna().sum()), n) if "patient_id" in df else "NA",
           "melanoma": pct(mel, n)}
    sub = df[df.label == 1].diagnosis_3.fillna("Missing").value_counts()
    row["melanoma subtype"] = "; ".join(f"{k} {v}" for k, v in sub.items())
    ref = df.ref_standard.value_counts()
    row["reference standard"] = "; ".join(f"{k} {pct(v, n)}" for k, v in ref.items())
    age = df.age_approx
    row["age, mean (SD)"] = f"{age.mean():.1f} ({age.std():.1f})"
    row["age missing"] = pct(int(age.isna().sum()), n)
    for s in ("female", "male"):
        row[f"sex {s}"] = pct(int((df.sex == s).sum()), n)
    row["sex missing"] = pct(int(df.sex.isna().sum()), n)
    site = df.site_h.fillna("Missing")
    for s in ("Head and neck", "Trunk", "Upper extremity", "Lower extremity", "Palms or soles", "Missing"):
        row[f"site: {s}"] = pct(int((site == s).sum()), n)
    if "fitzpatrick_skin_type" in df:
        fz = df.fitzpatrick_skin_type.fillna("Missing").value_counts()
        row["Fitzpatrick"] = "; ".join(f"{k} {v}" for k, v in sorted(fz.items()))
    return row


def main():
    meta = pd.read_csv(os.path.join(ROOT, "data", "raw", "metadata.csv"), low_memory=False)
    sp = pd.read_csv(os.path.join(ROOT, "data", "resplit", "split.csv"))
    ext = pd.read_csv(os.path.join(ROOT, "experiments", "external_ids.csv"))
    rows, flow = [], []

    isic = sp.merge(meta[["isic_id", "age_approx", "sex", "anatom_site_general", "diagnosis_3",
                          "diagnosis_confirm_type", "fitzpatrick_skin_type"]], on="isic_id", how="left")
    isic["site_h"] = isic.anatom_site_general.map(SITE)
    isic["ref_standard"] = isic.diagnosis_confirm_type.fillna("Missing")
    flow.append(("ISIC collection, images", len(meta)))
    flow.append(("dropped (duplicate images with conflicting labels)", int((~isic.split.isin(["train", "val", "test"])).sum())))
    for s in ("train", "val", "test"):
        d = isic[isic.split == s]
        flow.append((f"{s}: images / melanoma", f"{len(d)} / {int(d.label.sum())}"))
        rows.append(describe(d, {"test": "internal test"}.get(s, s)))

    bald_meta = pd.read_csv(os.path.join(ROOT, "test", "bald", "metadata.csv"), low_memory=False)
    bald = ext[ext.dataset == "bald"].merge(bald_meta[["isic_id", "age_approx", "sex", "anatom_site_special",
                                                        "diagnosis_3"]], on="isic_id", how="left")
    bald["site_h"] = bald.anatom_site_special.map({"acral palms or soles": "Palms or soles"})
    # Muller 2025: all melanomas excised and histopathologically examined; of 522 nevi, 193 excised, the rest typical
    # morphology or monitored for at least 1 year. Per-image method is not released, so it is reported at set level.
    bald["ref_standard"] = bald.label.map({1: "melanoma: histopathology", 0: "nevus: histopathology or follow-up/morphology (per-image not released)"})
    flow.append(("BALD, images", len(bald_meta)))
    flow.append(("BALD after removing 6 duplicates of training images: images / melanoma",
                 f"{len(bald)} / {int(bald.label.sum())}"))
    rows.append(describe(bald, "BALD"))

    pad_meta = pd.read_csv(os.path.join(ROOT, "test", "pad-ufes-20", "metadata.csv"), low_memory=False)
    pad = ext[ext.dataset == "pad"].merge(pad_meta[["isic_id", "age_approx", "sex", "anatom_site_1", "diagnosis_3",
                                                    "diagnosis_confirm_type", "fitzpatrick_skin_type", "lesion_id",
                                                    "patient_id"]], on="isic_id", how="left")
    pad["site_h"] = pad.anatom_site_1.map(SITE)
    pad["ref_standard"] = pad.diagnosis_confirm_type.fillna("Missing")
    flow.append(("PAD-UFES-20, images", len(pad_meta)))
    flow.append(("PAD-UFES-20 melanoma and nevus: images / melanoma / patients",
                 f"{len(pad)} / {int(pad.label.sum())} / {pad.patient_id.nunique()}"))
    rows.append(describe(pad, "PAD-UFES-20"))
    ref_by_label = pd.crosstab(pad.diagnosis_confirm_type, pad.label)

    tab = pd.DataFrame(rows).set_index("set").T
    out = os.path.join(ROOT, "runs", "stats")
    os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, "table1.csv"))
    pd.DataFrame(flow, columns=["step", "count"]).to_csv(os.path.join(out, "flow_counts.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(tab.to_string())
        print("\nPAD-UFES-20 reference standard by label (0 = nevus, 1 = melanoma):")
        print(ref_by_label.to_string())
        print("\nFlow:")
        for k, v in flow:
            print(f"  {k}: {v}")
    print(f"\nwrote {out}/table1.csv and flow_counts.csv")


if __name__ == "__main__":
    main()
