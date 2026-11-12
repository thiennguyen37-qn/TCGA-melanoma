"""RQ4 (PROTOCOL.md section 6): where does the trained model look?

1. Energy in mask. For every main run (best.pt, no TTA), Grad-CAM++ of the melanoma logit at the last
   feature map, upsampled bilinearly to 224, and share of its energy inside the lesion mask:
     - primary:   the 334 internal test images with an ISIC 2018 ground-truth mask
     - secondary: BALD images whose U-Net mask passed QC (qc_flag = 0)
   Grad-CAM++ is used instead of the plain CAM that L_att optimises, so the measure is not the training
   objective itself. The plain CAM energy is also written (column energy_cam) as a supplementary measure
   outside the protocol: with a GAP + linear head, Grad-CAM++ weights reduce to about 1 / mean(A_k) for
   channels with w_k > 0, so they depend little on the head weights. Per-variant values are means over images, then mean (SD) over seeds; paired
   differences (V1 - V0, V1 - V2, V4 - V3) get a cluster bootstrap CI as in stats.py.
2. Sanity check (Adebayo et al. 2018), seed 0 of V0, V1, V2: cascading re-initialisation from the head
   down to the stem; Spearman rank correlation between the original and the randomised map per image.

Only the head is differentiated (features are computed under no_grad), so this runs on CPU.

Usage: python experiments/rq4_attention.py --data data/cache320 --runs runs
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import SIZE, eval_tf  # noqa: E402
from stats import find_runs  # noqa: E402
from train import LesionDS, Net  # noqa: E402

COMPARISONS = [("V1", "V0"), ("V1", "V2"), ("V4", "V3")]
SANITY_VARIANTS = ("V0", "V1", "V2")
BOOT, BOOT_SEED = 2000, 20261111


def eval_sets(data_dir):
    sp = pd.read_csv(os.path.join(data_dir, "split.csv"))
    mi = pd.read_csv(os.path.join(data_dir, "mask_index.csv"))
    ex = pd.read_csv(os.path.join(data_dir, "external_ids.csv"))
    te = sp[sp.split == "test"].merge(mi[["isic_id", "source"]], on="isic_id")
    te = te[te.source == "gt"][["isic_id", "label", "group_id"]].rename(columns={"group_id": "cluster"})
    bald = ex[ex.dataset == "bald"].merge(mi[["isic_id", "qc_flag"]], on="isic_id", how="left")
    bald = bald[bald.qc_flag == 0][["isic_id", "label", "cluster_id"]].rename(columns={"cluster_id": "cluster"})
    return {"test_gt": te.reset_index(drop=True), "bald_qc0": bald.reset_index(drop=True)}


def loader(data_dir, df, bs):
    ds = LesionDS(data_dir, df.isic_id, df.label, eval_tf(), mask_ids=df.isic_id, flags=np.zeros(len(df)))
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


def gradcampp(model, x):
    """Grad-CAM++ (Chattopadhay et al. 2018) of the melanoma logit, upsampled to the input size."""
    with torch.no_grad():
        f = model.features(x)
    f = f.detach().requires_grad_(True)
    logit = model.fc(model.pool(f).flatten(1)).squeeze(1)  # eval mode: dropout is the identity
    (g,) = torch.autograd.grad(logit.sum(), f)
    with torch.no_grad():
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + f.sum((2, 3), keepdim=True) * g3
        alpha = torch.where(g != 0, g2 / torch.where(denom != 0, denom, torch.ones_like(denom)), torch.zeros_like(g))
        w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
        up = lambda c: F.interpolate(c[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam_pp = F.relu((w * f).sum(1))
        cam = F.relu(torch.einsum("bchw,c->bhw", f, model.fc.weight[0]))
        return up(cam_pp), up(cam)


def share_inside(cam, m):
    tot = cam.sum((1, 2))
    return torch.where(tot > 0, (cam * m).sum((1, 2)) / tot.clamp(min=1e-12), torch.full_like(tot, float("nan")))


def maps_and_energy(model, dl, device, keep_maps=False):
    """(Grad-CAM++ maps or None, energy-in-mask Grad-CAM++, energy-in-mask CAM, mask area) per image."""
    model.eval()
    maps, energy, energy_cam, area = [], [], [], []
    for x, _, m, _ in dl:
        x, m = x.to(device), m.to(device)
        cam_pp, cam = gradcampp(model, x)
        energy.append(share_inside(cam_pp, m).cpu().numpy()); energy_cam.append(share_inside(cam, m).cpu().numpy())
        area.append(m.mean((1, 2)).cpu().numpy())
        if keep_maps:
            maps.append(cam_pp.cpu().numpy().astype(np.float32))
    return (np.concatenate(maps) if keep_maps else None), np.concatenate(energy), np.concatenate(energy_cam), np.concatenate(area)


def load_model(run_dir, device):
    model = Net(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    return model.to(device).eval()


def reinit(module, gen_seed):
    torch.manual_seed(gen_seed)
    for mod in module.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()


def spearman_rows(a, b):
    """Spearman correlation per image between two stacks of maps (n, H, W); NaN for constant maps."""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        ra, rb = rankdata(a[i].ravel()), rankdata(b[i].ravel())
        if ra.std() > 0 and rb.std() > 0:
            out[i] = np.corrcoef(ra, rb)[0, 1]
    return out


def boot_weights(clusters, n_boot, seed):
    codes, uniq = pd.factorize(pd.Series(clusters))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)
    return counts[:, codes].astype(np.float64)


def summarise(per_image, sets, out_dir):
    rows, comps = [], []
    for set_name, df in sets.items():
        sub = per_image[per_image.set == set_name]
        W = boot_weights(df.cluster.to_numpy(), BOOT, BOOT_SEED)
        piv = {}
        for (v, s), g in sub.groupby(["variant", "seed"]):
            e = g.set_index("isic_id").loc[df.isic_id, "energy"].to_numpy()
            ok = ~np.isnan(e)
            piv[(v, s)] = (e, ok)
            ec = g.set_index("isic_id").loc[df.isic_id, "energy_cam"].to_numpy()
            rows.append({"set": set_name, "variant": v, "seed": s, "energy_mean": float(np.nanmean(e)),
                         "energy_cam_mean": float(np.nanmean(ec)), "n_zero_map": int((~ok).sum())})
        area = sub.drop_duplicates("isic_id").set_index("isic_id").loc[df.isic_id, "area"].to_numpy()
        rows.append({"set": set_name, "variant": "mask_area", "seed": -1, "energy_mean": float(area.mean()),
                     "energy_cam_mean": float(area.mean()), "n_zero_map": 0})
        for a, b in COMPARISONS:
            seeds = sorted({s for v, s in piv if v == a} & {s for v, s in piv if v == b})
            pts, reps = [], []
            for s in seeds:
                (ea, oa), (eb, ob) = piv[(a, s)], piv[(b, s)]
                ok = oa & ob
                pts.append(ea[ok].mean() - eb[ok].mean())
                w = W[:, ok]
                reps.append((w * ea[ok]).sum(1) / w.sum(1) - (w * eb[ok]).sum(1) / w.sum(1))
            if not pts:
                continue
            reps = np.mean(reps, axis=0)
            comps.append({"set": set_name, "a": a, "b": b, "n_seeds": len(pts), "delta": float(np.mean(pts)),
                          "sd_seeds": float(np.std(pts, ddof=1)) if len(pts) > 1 else float("nan"),
                          "ci_lo": float(np.percentile(reps, 2.5)), "ci_hi": float(np.percentile(reps, 97.5))})
    per_run = pd.DataFrame(rows)
    grp = per_run[per_run.variant != "mask_area"].groupby(["set", "variant"])
    tab = grp.energy_mean.agg(["mean", "std", "count"]).reset_index()
    tab = tab.merge(grp.energy_cam_mean.agg(cam_mean="mean", cam_std="std").reset_index(), on=["set", "variant"])
    area_rows = per_run[per_run.variant == "mask_area"][["set", "energy_mean"]].rename(columns={"energy_mean": "mask_area"})
    tab = tab.merge(area_rows, on="set")
    comps = pd.DataFrame(comps)
    per_run.to_csv(os.path.join(out_dir, "energy_per_run.csv"), index=False)
    tab.to_csv(os.path.join(out_dir, "energy_per_variant.csv"), index=False)
    comps.to_csv(os.path.join(out_dir, "energy_comparisons.csv"), index=False)
    print("\n== Energy inside the lesion mask: mean over images, mean (SD) over seeds ==")
    print("   mean/std = Grad-CAM++ (protocol); cam_mean/cam_std = plain CAM (supplementary, outside the protocol)")
    print("   mask_area = share of the image covered by the mask, i.e. the value for a uniform map")
    print(tab.round(4).to_string(index=False))
    print("\n== Paired differences in energy-in-mask, cluster bootstrap 95% CI ==")
    for _, r in comps.iterrows():
        print(f"  {r.set:<9} {r.a} - {r.b}: {r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  seeds={r.n_seeds}")


def sanity(runs, sets, data_dir, device, bs, out_dir, limit):
    df = sets["test_gt"] if not limit else sets["test_gt"].head(limit)
    dl = loader(data_dir, df, bs)
    rows = []
    for v in SANITY_VARIANTS:
        if 0 not in runs.get(v, {}):
            continue
        model = load_model(runs[v][0], device)
        base = maps_and_energy(model, dl, device, keep_maps=True)[0]
        stages = ["fc"] + [f"features.{i}" for i in range(len(model.features) - 1, -1, -1)]
        for k, name in enumerate(stages):
            reinit(model.get_submodule(name), gen_seed=1000 + k)
            rnd = maps_and_energy(model, dl, device, keep_maps=True)[0]
            r = spearman_rows(base, rnd)
            ok = ~np.isnan(r)  # NaN = the randomised map is constant (all zero): the explanation is destroyed
            rows.append({"variant": v, "stage": k + 1, "randomised_down_to": name,
                         "spearman_mean": float(r[ok].mean()) if ok.any() else float("nan"),
                         "spearman_sd": float(r[ok].std()) if ok.any() else float("nan"),
                         "n_constant_map": int((~ok).sum()), "n": len(r)})
            print(f"  sanity {v} {name:<11} spearman {rows[-1]['spearman_mean']:+.3f} (constant maps: {rows[-1]['n_constant_map']})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "sanity_check.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cache320")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None, help="default: <runs>/rq4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N images per set, first runs only")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    out = a.out or os.path.join(a.runs, "rq4")
    os.makedirs(out, exist_ok=True)
    device = torch.device(a.device)
    runs, lam = find_runs(a.runs)
    sets = eval_sets(a.data)
    if a.limit:
        sets = {k: v.head(a.limit) for k, v in sets.items()}
    print("sets: " + ", ".join(f"{k} {len(v)} images ({int(v.label.sum())} melanoma)" for k, v in sets.items()))

    cache = os.path.join(out, "energy_per_image.csv")
    per_image = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame()
    done = set(zip(per_image.get("variant", []), per_image.get("seed", [])))
    loaders = {k: loader(a.data, v, a.bs) for k, v in sets.items()}
    for v in sorted(runs):
        for s, d in sorted(runs[v].items()):
            if (v, s) in done:
                continue
            model = load_model(d, device)
            parts = []
            for set_name, df in sets.items():
                _, e, ec, ar = maps_and_energy(model, loaders[set_name], device)
                parts.append(pd.DataFrame({"variant": v, "seed": s, "set": set_name, "isic_id": df.isic_id,
                                           "label": df.label, "energy": e, "energy_cam": ec, "area": ar}))
            per_image = pd.concat([per_image] + parts, ignore_index=True)
            per_image.to_csv(cache, index=False)  # checkpoint after every run
            print(f"  {v} seed {s}: " + ", ".join(f"{p.set.iloc[0]} {np.nanmean(p.energy):.3f}" for p in parts), flush=True)
    summarise(per_image, sets, out)
    if not a.skip_sanity:
        print("\n== Sanity check: cascading randomisation, Spearman(original, randomised) ==")
        sanity(runs, sets, a.data, device, a.bs, out, a.limit)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"lambda": lam, "sets": {k: len(v) for k, v in sets.items()}, "boot": BOOT}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
