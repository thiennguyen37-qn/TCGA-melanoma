"""Figures for the manuscript (static, for print). Reads the analysis outputs in runs/ and writes PNG (300 dpi) and
PDF files to runs/figures/.

  fig1_forest           paired differences in external AUC (primary and exploratory comparisons)
  fig2_attention        share of the attention map inside the lesion mask vs image-independent references
  fig3_reader           BALD AUC of each variant vs the reader votes, all lesions and melanoma in situ
  efig_calibration      reliability curves per evaluation set
  efig_sanity           cascading weight-randomisation sanity check
  efig_flow             participant flow (counts computed from the split, mask index and external lists)

Colours follow a validated categorical order (blue, orange, aqua, yellow, magenta); every series also has its own
marker shape and a label, so identity never relies on colour alone.

Usage: python experiments/figures.py --runs runs
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]
NAMES = {"V0": "Standard training", "V1": "Mask supervision", "V2": "Mismatched-mask control",
         "V3": "Augmentation", "V4": "Augmentation + mask supervision"}
SETS = {"test": "Internal test", "bald": "BALD (acral)", "pad": "PAD-UFES-20 (smartphone)"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def grid_x(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def grid_y(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1(runs, out):
    c = pd.read_csv(os.path.join(runs, "stats", "comparisons.csv"))
    rows = [("primary", "V1", "V0", "bald"), ("primary", "V1", "V0", "pad"), ("secondary", "V1", "V0", "test"),
            ("secondary", "V1", "V2", "bald"), ("secondary", "V1", "V2", "pad"),
            ("secondary", "V1", "V3", "bald"), ("secondary", "V1", "V3", "pad"),
            ("secondary", "V4", "V3", "bald"), ("secondary", "V4", "V3", "pad")]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = []
    for i, (grp, a, b, s) in enumerate(rows):
        r = c[(c.group == grp) & (c.a == a) & (c.b == b) & (c.set == s)].iloc[0]
        lo, hi = (r.ci_lo_holm, r.ci_hi_holm) if grp == "primary" else (r.ci_lo, r.ci_hi)
        y = -i
        ax.plot([lo, hi], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot(r.delta, y, marker="o", markersize=8, color=SERIES[0],
                markerfacecolor=SERIES[0] if grp == "primary" else SURFACE, markeredgewidth=2)
        level = f"{100 * r.ci_level:.1f}%" if grp == "primary" else "95%"
        ax.text(0.085, y, f"{r.delta:+.3f} ({lo:+.3f} to {hi:+.3f})", va="center", fontsize=8, color=INK2)
        labels.append(f"{NAMES[a]} − {NAMES[b]}\n{SETS[s]}" + ("  [primary]" if grp == "primary" else ""))
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvline(0.02, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(0.0205, 0.6, "minimum effect\nof interest (+0.02)", fontsize=7.5, color=MUTED, va="bottom")
    ax.axhline(-1.5, color=GRID, linewidth=0.8)
    ax.set_yticks([-i for i in range(len(rows))], labels, fontsize=7.5, color=INK2)
    ax.set_xlim(-0.065, 0.08)
    ax.set_ylim(-len(rows) + 0.4, 1.3)
    ax.set_xlabel("Difference in AUC (mean of 5 training runs)")
    ax.text(0.085, 0.9, "Δ AUC (CI)", fontsize=8, color=INK, fontweight="bold")
    grid_x(ax)
    ax.set_title("Paired differences in AUC", loc="left")
    fig.text(0.01, -0.04, "Filled markers: primary comparisons, CI adjusted for 2 comparisons (Holm) (97.5% for BALD, 95% for "
             "PAD-UFES-20). Hollow markers: secondary comparisons, 95% CI. Bootstrap, 2,000 resamples.",
             fontsize=7, color=MUTED, wrap=True)
    save(fig, out, "fig1_forest")


def fig2(runs, out):
    per = pd.read_csv(os.path.join(runs, "rq4", "energy_per_run.csv"))
    pri = pd.read_csv(os.path.join(runs, "rq4", "centre_prior_energy.csv"))
    panels = [("test_gt", "Internal test, ground-truth masks (n = 334)"), ("bald_qc0", "BALD, predicted masks passing quality check (n = 613)")]
    short = {"V0": "Standard", "V1": "Mask\nsupervision", "V2": "Mismatched\nmask", "V3": "Augmen-\ntation", "V4": "Augmen-\ntation\n+ mask"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (s, title) in zip(axes, panels):
        d = per[(per.set == s) & (per.variant.isin(VARIANTS))].groupby("variant").energy_mean.agg(["mean", "std"])
        x = np.arange(len(VARIANTS))
        for i, v in enumerate(VARIANTS):
            ax.errorbar(i, d.loc[v, "mean"], yerr=d.loc[v, "std"], fmt=MARKERS[i], color=SERIES[0], markersize=8,
                        markerfacecolor=SERIES[0], elinewidth=2, capsize=0)
            ax.text(i + 0.18, d.loc[v, "mean"], f"{d.loc[v, 'mean']:.2f}", ha="left", va="center", fontsize=7.5, color=INK2)
        disc = pri[(pri["set"] == s) & (pri["map"] == "disc")].energy_in_mask.iloc[0]
        uni = pri[(pri["set"] == s) & (pri["map"] == "uniform")].energy_in_mask.iloc[0]
        ax.axhline(disc, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
        ax.text(1.5, disc - 0.025, f"centred disc {disc:.2f}", fontsize=7, color=INK2, va="top", ha="center")
        ax.axhline(uni, color=MUTED, linewidth=1, linestyle=(0, (1, 2)), zorder=0)
        ax.text(1.5, uni - 0.025, f"uniform map {uni:.2f}", fontsize=7, color=MUTED, va="top", ha="center")
        ax.set_xticks(x, [short[v] for v in VARIANTS], fontsize=7, color=INK2)
        ax.set_xlim(-0.5, 4.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontsize=8.5)
        grid_y(ax)
    axes[0].set_ylabel("Share of attention map inside the lesion mask")
    fig.suptitle("Where the models look", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK, y=1.03)
    fig.text(0.01, -0.07, "Markers: mean over 5 training runs; bars: ±1 SD across runs. Reference lines are image-independent maps: "
             "a centred disc with the median training-mask area, and a uniform map (equal to the mask area).",
             fontsize=7, color=MUTED, wrap=True)
    save(fig, out, "fig2_attention")


def fig3(runs, out):
    sg = pd.read_csv(os.path.join(runs, "stats", "subgroup_bald.csv"))
    panels = [("all", "All BALD lesions (140 melanoma, 520 nevi)"), ("melanoma in situ vs all nevi", "Melanoma in situ vs all nevi (46 vs 520)")]
    order = ["reader"] + VARIANTS
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (g, title) in zip(axes, panels):
        d = sg[sg.subgroup == g].set_index("model")
        for i, m in enumerate(order):
            y = -i
            r = d.loc[m]
            col = INK if m == "reader" else SERIES[0]
            ax.plot([r.auc_lo, r.auc_hi], [y, y], color=col, linewidth=2, solid_capstyle="round")
            ax.plot(r.auc, y, marker="D" if m == "reader" else "o", markersize=8, color=col)
            ax.text(r.auc_hi + 0.004, y, f"{r.auc:.3f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks([-i for i in range(len(order))], ["Readers (vote fraction, n = 409)"] + [NAMES[v] for v in VARIANTS],
                      fontsize=7.5, color=INK2)
        ax.set_xlim(0.78, 1.0)
        ax.set_xlabel("AUC (95% CI)")
        ax.set_title(title, loc="left", fontsize=8.5)
        grid_x(ax)
    fig.suptitle("Models and readers on BALD", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, -0.08, "Model AUC: mean over 5 training runs; CI from 2,000 bootstrap resamples of images shared by all models. "
             "Exploratory subgroup analysis.", fontsize=7, color=MUTED, wrap=True)
    save(fig, out, "fig3_reader")


def efig_calibration(runs, out):
    cur = pd.read_csv(os.path.join(runs, "stats", "calibration_curves.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, s in zip(axes, ["test", "bald", "pad"]):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        for i, v in enumerate(VARIANTS):
            d = cur[(cur.set == s) & (cur.variant == v) & (cur.n >= 5)].sort_values("bin")
            ax.plot(d.mean_prob, d.observed, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=5,
                    label=NAMES[v])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(SETS[s], loc="left", fontsize=8.5)
        ax.set_xlabel("Predicted probability")
        grid_y(ax)
    axes[0].set_ylabel("Observed melanoma fraction")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=7.5)
    fig.suptitle("Calibration (10 equal-width bins, mean over 5 training runs; bins with < 5 images omitted)", x=0.01,
                 ha="left", fontsize=9.5, fontweight="bold", color=INK, y=1.04)
    save(fig, out, "efig_calibration")


def efig_sanity(runs, out):
    sc = pd.read_csv(os.path.join(runs, "rq4", "sanity_check.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    stages = sc[sc.variant == "V0"].sort_values("stage").randomised_down_to.tolist()
    tick = ["head" if s == "fc" else s.replace("features.", "block ") for s in stages]
    sc = sc.copy()
    sc.loc[sc.n_constant_map > sc.n / 2, "spearman_mean"] = np.nan  # nearly all maps are zero: correlation undefined
    for i, v in enumerate(["V0", "V1", "V2"]):
        d = sc[sc.variant == v].sort_values("stage")
        ax.plot(d.stage, d.spearman_mean, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=6, label=NAMES[v])
    ax.legend(loc="upper right", fontsize=7.5)
    const = sc[(sc.variant == "V0") & (sc.n_constant_map > sc.n / 2)]
    for _, r in const.iterrows():
        ax.text(r.stage, 0.05, "maps all zero", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(sorted(sc.stage.unique()), tick, fontsize=7.5, color=INK2)
    ax.set_xlim(0.6, len(stages) + 0.4)
    ax.set_ylim(-0.4, 1.05)
    ax.set_xlabel("Weights re-initialised cumulatively from the head down to")
    ax.set_ylabel("Spearman ρ with original map")
    ax.set_title("Cascading weight-randomisation sanity check (first training run, 334 internal test images)", loc="left", fontsize=9.5)
    grid_y(ax)
    save(fig, out, "efig_sanity")


def flow_counts(root):
    """Every number in the flow diagram, computed from the split, the mask index and the external lists."""
    sp = pd.read_csv(os.path.join(root, "data", "resplit", "split.csv"))
    mi = pd.read_csv(os.path.join(root, "data", "cache320", "mask_index.csv")).merge(sp[["isic_id", "split"]], on="isic_id")
    ext = pd.read_csv(os.path.join(root, "experiments", "external_ids.csv"))
    bald = pd.read_csv(os.path.join(root, "test", "bald", "metadata.csv"), low_memory=False)
    pad = pd.read_csv(os.path.join(root, "test", "pad-ufes-20", "metadata.csv"), low_memory=False)
    bald_mask = pd.read_csv(os.path.join(root, "data", "cache320", "mask_index.csv"))
    c = {"isic": len(sp), "dropped": int((sp.split == "dropped").sum())}
    for s in ("train", "val", "test"):
        d = sp[sp.split == s]
        c[s], c[s + "_mel"] = len(d), int(d.label.sum())
    tr = mi[mi.split == "train"]
    c["train_gt"] = int((tr.source == "gt").sum())
    c["train_pred"] = int((tr.source == "pred").sum())
    c["train_flag"] = int((tr.qc_flag == 1).sum())
    c["test_gt"] = int(((mi.split == "test") & (mi.source == "gt")).sum())
    b = ext[ext.dataset == "bald"]
    c["bald_raw"] = len(bald)
    c["bald"], c["bald_mel"] = len(b), int(b.label.sum())
    c["bald_qc"] = int(bald_mask[bald_mask.isic_id.isin(b.isic_id) & (bald_mask.qc_flag == 0)].shape[0])
    p = ext[ext.dataset == "pad"]
    c["pad_raw"] = len(pad)
    c["pad_other"] = pad.diagnosis_3.value_counts().drop(["Nevus", "Melanoma, NOS"]).to_dict()
    c["pad"], c["pad_mel"], c["pad_pat"] = len(p), int(p.label.sum()), p.cluster_id.nunique()
    return c


def efig_flow(root, out):
    c = flow_counts(root)
    n = lambda v: f"{v:,}".replace(",", " ")  # JAMA style: space as thousands separator
    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(11, 100)
    ax.axis("off")

    def box(x, y, w, h, title=None, body="", side=False, fs=6.9):
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", linewidth=0.9,
            edgecolor=AXIS, facecolor="#f4f3ef" if side else SURFACE))
        if title:
            ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=fs + 0.3, color=INK,
                    fontweight="bold")
            ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.35)
        else:
            ax.text(x + w / 2, y + h / 2, body, ha="center", va="center", fontsize=fs, color=INK2 if side else INK,
                    linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=0.9, shrinkA=0, shrinkB=0))

    def branch(x0, y, x1):
        ax.plot([x0, x1], [y, y], color=INK2, linewidth=0.9)

    ax.text(0, 100, "Development data", fontsize=9, fontweight="bold", color=INK, va="top")
    ax.text(52, 100, "External test data", fontsize=9, fontweight="bold", color=INK, va="top")

    # development
    box(2, 85, 42.5, 10, body=f"ISIC archive: melanoma and nevus with\nhistopathologic diagnosis\n{n(c['isic'])} images")
    arrow(23.25, 85, 23.25, 66)
    branch(23.25, 75.5, 27)
    box(27, 71.5, 19.5, 8, body=f"Excluded: {c['dropped']} images in\nduplicate clusters with\nconflicting labels", side=True)
    box(2, 57, 42.5, 9, body=f"{n(c['isic'] - c['dropped'])} images linked into clusters by lesion,\n"
                           "patient, and near-duplicate image;\neach cluster assigned whole to 1 set")
    xs = (0, 16, 32)
    for x in xs:
        arrow(x + 7.25, 57, x + 7.25, 50)
    for (s, label), x in zip([("train", "Training"), ("val", "Validation"), ("test", "Internal test")], xs):
        box(x, 39, 14.5, 11, label, f"{n(c[s])} images\n{n(c[s + '_mel'])} melanoma")
    box(0, 14, 14.5, 22, "Lesion masks", f"{n(c['train_gt'])} ground truth\n{n(c['train_pred'])} predicted\n\n"
                                           f"{n(c['train_flag'])} flagged by\nquality check and\nleft out of the\nattention penalty",
        side=True, fs=6.6)
    box(32, 22, 14.5, 14, "Attention", f"{n(c['test_gt'])} images with\nground-truth\nmasks", side=True, fs=6.6)

    # external
    box(52, 85, 22, 10, "BALD, acral dermoscopy", f"7 centers\n{n(c['bald_raw'])} images", fs=6.6)
    box(77, 85, 22, 10, "PAD-UFES-20, smartphone", f"{n(c['pad_raw'])} images", fs=6.6)
    arrow(56, 85, 56, 50)
    branch(56, 75, 59)
    box(59, 69.5, 15.5, 11, body=f"Excluded: {c['bald_raw'] - c['bald']}\ncopies of\ndevelopment\nimages", side=True)
    arrow(81, 85, 81, 50)
    branch(81, 75, 84)
    box(84, 70.5, 15.5, 9, body=f"Excluded: {n(sum(c['pad_other'].values()))}\nimages with other\ndiagnoses*", side=True)
    box(52, 39, 22, 11, "BALD test set", f"{n(c['bald'])} images\n{c['bald_mel']} melanoma")
    box(77, 39, 22, 11, "PAD-UFES-20 test set", f"{n(c['pad'])} images, {c['pad_pat']} patients\n{c['pad_mel']} melanoma")
    box(52, 14, 22, 22, "Analyses", f"AUC: all {n(c['bald'])} images\n\nReaders: all {n(c['bald'])} images\n(409 clinicians)\n\n"
                                        f"Attention: {n(c['bald_qc'])} images\nwith predicted masks\npassing quality check",
        side=True, fs=6.6)
    box(77, 28, 22, 8, "Analyses", f"AUC: all {n(c['pad'])} images", side=True, fs=6.6)

    for x0, y0, y1 in ((7.25, 39, 36), (39.25, 39, 36), (63, 39, 36), (88, 39, 36)):
        ax.plot([x0, x0], [y0, y1], color=AXIS, linewidth=0.9)
    names = {"Basal cell carcinoma": "basal cell carcinoma", "Solar or actinic keratosis": "actinic keratosis",
             "Seborrheic keratosis": "seborrheic keratosis", "Squamous cell carcinoma, NOS": "squamous cell carcinoma"}
    parts = [f"{n(v)} {names.get(k, k)}" for k, v in c["pad_other"].items()]
    fig.text(0.02, 0.0, "*" + ", ".join(parts[:-1]) + f", and {parts[-1]}. Training and validation data were used for all "
             "model choices; external test sets were used only for final evaluation.",
             fontsize=6.8, color=MUTED, wrap=True)
    save(fig, out, "efig_flow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    out = os.path.join(a.runs, "figures")
    os.makedirs(out, exist_ok=True)
    efig_flow(a.root, out)
    fig1(a.runs, out)
    fig2(a.runs, out)
    fig3(a.runs, out)
    efig_calibration(a.runs, out)
    efig_sanity(a.runs, out)
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
