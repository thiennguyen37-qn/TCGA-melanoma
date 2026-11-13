"""eFigure: Grad-CAM++ examples for standard training, mask supervision and the mismatched-mask control.

Images are drawn at random with a fixed seed (not chosen by eye): 1 melanoma and 1 nevus from BALD images whose
predicted mask passed quality check, and 1 melanoma and 1 nevus from PAD-UFES-20. Maps come from the first training
run (seed 0) of each variant, as in the attention analysis (best.pt, no flips, melanoma logit, last feature map).
The BALD lesion mask is drawn as a white outline; PAD-UFES-20 masks were not used in any analysis and are not shown.

Usage: python experiments/cam_examples.py --data data/cache320 --runs runs
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import eval_tf  # noqa: E402
from figures import INK, INK2, MUTED, SURFACE, save  # noqa: E402
from rq4_attention import eval_sets, gradcampp, load_model  # noqa: E402
from stats import find_runs  # noqa: E402
from train import MEAN, STD, LesionDS  # noqa: E402

SEED = 20261113
VARIANTS = [("V0", "Standard training"), ("V1", "Mask supervision"), ("V2", "Mismatched-mask control")]
# single-hue sequential ramp (blue), transparent at zero attention
CMAP = LinearSegmentedColormap.from_list("att", [(0.725, 0.835, 0.965, 0.0), (0.333, 0.596, 0.906, 0.55),
                                                 (0.110, 0.361, 0.671, 0.8), (0.051, 0.212, 0.420, 0.9)])


def pick(data_dir):
    rng = np.random.default_rng(SEED)
    bald = eval_sets(data_dir)["bald_qc0"]
    ex = pd.read_csv(os.path.join(data_dir, "external_ids.csv"))
    pad = ex[ex.dataset == "pad"]
    rows = []
    for name, df, has_mask in (("BALD", bald, True), ("PAD-UFES-20", pad, False)):
        for lab, lab_name in ((1, "melanoma"), (0, "nevus")):
            ids = sorted(df[df.label == lab].isic_id)
            rows.append((name, lab_name, ids[rng.integers(len(ids))], lab, has_mask))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cache320")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    runs, _ = find_runs(a.runs)
    models = {v: load_model(runs[v][0], "cpu") for v, _ in VARIANTS}
    rows = pick(a.data)

    fig, axes = plt.subplots(len(rows), 1 + len(VARIANTS), figsize=(7.2, 7.6),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.32})
    for r, (setname, lab_name, iid, lab, has_mask) in enumerate(rows):
        ds = LesionDS(a.data, [iid], [lab], eval_tf(), mask_ids=[iid] if has_mask else None,
                      flags=np.zeros(1) if has_mask else None)
        x, _, m, _ = ds[0]
        img = np.clip(x.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
        ax = axes[r, 0]
        ax.imshow(img)
        if has_mask:
            ax.contour(m.numpy(), levels=[0.5], colors="white", linewidths=1.0)
        ax.set_ylabel(f"{setname}\n{lab_name}", fontsize=7.5, color=INK, rotation=0, ha="right", va="center",
                      labelpad=6)
        for c, (v, title) in enumerate(VARIANTS, start=1):
            model = models[v]
            cam, _ = gradcampp(model, x[None])
            with torch.no_grad():
                p = torch.sigmoid(model(x[None])).item()
            cam = cam[0].numpy()
            cam = cam / cam.max() if cam.max() > 0 else cam
            ax = axes[r, c]
            ax.imshow(img)
            ax.imshow(cam, cmap=CMAP, vmin=0, vmax=1)
            if has_mask:
                ax.contour(m.numpy(), levels=[0.5], colors="white", linewidths=1.0)
                share = float((cam * m.numpy()).sum() / cam.sum()) if cam.sum() > 0 else float("nan")
                note = f"P(melanoma) {p:.2f}" + chr(10) + f"inside mask {share:.0%}"
            else:
                note = f"P(melanoma) {p:.2f}"
            ax.set_xlabel(note, fontsize=6.5, color=INK2, labelpad=2)
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK, fontweight="bold")
        if r == 0:
            axes[r, 0].set_title("Image", fontsize=8, color=INK, fontweight="bold")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.02, 0.035, "Images drawn at random (fixed seed) from each set and class. Darker blue marks a larger share "
             "of the Grad-CAM++ map of the melanoma output (first training run of each variant; each map scaled to its "
             "maximum). White outline: predicted lesion mask (BALD only). P(melanoma): predicted probability of melanoma from a single pass without flips.",
             fontsize=6.8, color=MUTED, wrap=True)
    out = os.path.join(a.runs, "figures")
    os.makedirs(out, exist_ok=True)
    save(fig, out, "efig_cam_examples")
    print("wrote efig_cam_examples;", [(s, l, i) for s, l, i, _, _ in rows])


if __name__ == "__main__":
    main()
