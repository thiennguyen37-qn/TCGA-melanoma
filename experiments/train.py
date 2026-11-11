"""Train and evaluate one variant x seed of the protocol (experiments/PROTOCOL.md).

Variants
  V0  baseline                    focal loss, default augmentation
  V1  mask-guided                 + lambda * L_att with the image's own lesion mask
  V2  random-mask control         + lambda * L_att with the mask of another training image (fixed per seed)
  V3  anti-shift augmentation     Kozachok 2026a "mix" policy, no L_att
  V4  augmentation + mask         V3 + lambda * L_att with the image's own mask

L_att = 1 - sum(A * M) / sum(A), A = class activation map of the melanoma logit (ReLU, upsampled to
224). With EfficientNet's GAP + linear head, Grad-CAM of the logit equals this CAM up to a constant
factor (the gradient wrt each feature map is w_k / HW), so it is computed directly from the head
weights; L_att is scale-invariant, so the constant does not matter. Images with qc_flag = 1 get no L_att.

Outputs in <runs>/<run_name>/: best.pt, last.pt (for resuming), history.csv, preds_<set>.csv
(isic_id, label, prob with TTA) and summary.json. External sets are scored only once per run: if
preds_bald.csv / preds_pad.csv already exist they are never recomputed (protocol section 8).

Usage
  python experiments/train.py --data data/cache320 --variant V1 --lam 1.0 --seed 0
  python experiments/train.py --data data/cache320 --variant V1 --lam 0.3 --seed 0 --no-external --tag lamsearch
  smoke test: --limit 64 --epochs 1 --max-steps 2 --device cpu
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.ops import sigmoid_focal_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import SIZE, default_train, eval_tf, mix_train  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
USES_MASK = {"V0": False, "V1": True, "V2": True, "V3": False, "V4": True}
USES_MIX = {"V0": False, "V1": False, "V2": False, "V3": True, "V4": True}
SENS_TARGET = 0.90


# ---------------- data ----------------

class LesionDS(Dataset):
    def __init__(self, data_dir, ids, labels, tf, mask_ids=None, flags=None):
        self.d, self.ids, self.labels, self.tf = data_dir, list(ids), np.asarray(labels, np.float32), tf
        self.mask_ids = list(mask_ids) if mask_ids is not None else None
        self.flags = np.asarray(flags, np.float32) if flags is not None else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.d, "images", self.ids[i] + ".jpg")), cv2.COLOR_BGR2RGB)
        if self.mask_ids is not None:
            m = cv2.imread(os.path.join(self.d, "masks", self.mask_ids[i] + ".png"), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            o = self.tf(image=img, mask=m.astype(np.uint8))
            mask, w = torch.from_numpy(o["mask"].astype(np.float32)), 1.0 - self.flags[i]
        else:
            o = self.tf(image=img)
            mask, w = torch.zeros(SIZE, SIZE), 0.0
        x = torch.from_numpy(((o["image"].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
        return x, torch.tensor(self.labels[i]), mask, torch.tensor(w, dtype=torch.float32)


def load_tables(data_dir, limit, seed):
    sp = pd.read_csv(os.path.join(data_dir, "split.csv"))
    sp = sp[sp.split.isin(["train", "val", "test"])]
    mi = pd.read_csv(os.path.join(data_dir, "mask_index.csv"), usecols=["isic_id", "qc_flag"])
    sp = sp.merge(mi, on="isic_id", how="left")
    sp["qc_flag"] = sp.qc_flag.fillna(1).astype(int)  # no QC information -> do not trust the mask
    ext = pd.read_csv(os.path.join(data_dir, "external_ids.csv"))
    if limit:  # smoke tests only
        sub = lambda df, col: pd.concat([g.sample(min(limit, len(g)), random_state=seed) for _, g in df.groupby(col)])
        sp, ext = sub(sp, "split"), sub(ext, "dataset")
    return sp.reset_index(drop=True), ext.reset_index(drop=True)


# ---------------- model ----------------

class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        b = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = b.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x, cam=False):
        f = self.features(x)
        logit = self.fc(self.drop(self.pool(f).flatten(1))).squeeze(1)
        if not cam:
            return logit
        with torch.autocast(f.device.type, enabled=False):  # CAM and L_att in fp32
            a = F.relu(torch.einsum("bchw,c->bhw", f.float(), self.fc.weight[0].float()))
        return logit, a


def attention_loss(a, mask, w):
    """1 - share of CAM energy inside the mask, averaged over images with w = 1."""
    a = F.interpolate(a[:, None], size=mask.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    inside = (a * mask).sum((1, 2)) / (a.sum((1, 2)) + 1e-6)
    return ((1.0 - inside) * w).sum() / w.sum().clamp(min=1.0)


# ---------------- evaluation ----------------

@torch.no_grad()
def predict(model, loader, device, tta=True):
    model.eval()
    out = []
    for x, *_ in loader:
        x = x.to(device, non_blocking=True)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.stack([torch.sigmoid(model(v).float()) for v in views]).mean(0)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def threshold_at_sensitivity(y, p, target=SENS_TARGET):
    """Largest threshold t with sensitivity(p >= t) >= target."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos)))
    return float(pos[k - 1])


def metrics(y, p, thr):
    pred = p >= thr
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return {"auc": float(auc), "sens": sens, "spec": spec, "n": int(len(y)), "n_mel": int(y.sum())}


def git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------- main ----------------

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="bundle dir from prepare_cache.py")
    ap.add_argument("--variant", choices=list(USES_MASK), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lam", type=float, default=0.0, help="attention-loss weight (V1, V2, V4)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. lamsearch")
    ap.add_argument("--no-external", action="store_true", help="skip external scoring (lambda search)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N images per split")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: batches per epoch")
    a = ap.parse_args()
    if USES_MASK[a.variant] and a.lam <= 0:
        ap.error(f"{a.variant} needs --lam > 0")
    if not USES_MASK[a.variant]:
        a.lam = 0.0

    name = f"{a.variant}_s{a.seed}" + (f"_lam{a.lam:g}" if USES_MASK[a.variant] else "") + (f"_{a.tag}" if a.tag else "")
    out = os.path.join(a.runs, name)
    os.makedirs(out, exist_ok=True)
    device = torch.device(a.device)
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True

    sp, ext = load_tables(a.data, a.limit, a.seed)
    tr, va, te = (sp[sp.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    mask_ids, flags = tr.isic_id.values, tr.qc_flag.values
    if a.variant == "V2":  # fixed random reassignment of masks for this seed
        perm = np.random.default_rng(a.seed).permutation(len(tr))
        mask_ids, flags = mask_ids[perm], flags[perm]
    train_tf = mix_train() if USES_MIX[a.variant] else default_train()
    tr_ds = LesionDS(a.data, tr.isic_id, tr.label, train_tf,
                     mask_ids if USES_MASK[a.variant] else None, flags if USES_MASK[a.variant] else None)
    g = torch.Generator(); g.manual_seed(a.seed)
    wi = lambda k: np.random.seed((a.seed * 1000 + k + torch.initial_seed()) % 2**32)
    tr_dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                       pin_memory=device.type == "cuda", generator=g, worker_init_fn=wi, persistent_workers=a.workers > 0)
    mk = lambda df: DataLoader(LesionDS(a.data, df.isic_id, df.label, eval_tf()), batch_size=64,
                               num_workers=a.workers, pin_memory=device.type == "cuda")
    va_dl, te_dl = mk(va), mk(te)

    model = Net(pretrained=True).to(device)
    alpha = float((tr.label == 0).mean())  # focal alpha = nevus fraction, as notebook 07
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start, best_auc, bad, hist = 1, -1.0, 0, []
    last = os.path.join(out, "last.pt")
    if os.path.exists(last):  # resume after a Colab/Kaggle disconnect
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best_auc, bad, hist = ck["epoch"] + 1, ck["best_auc"], ck["bad"], ck["hist"]
        print(f"resume from epoch {start}, best val AUC {best_auc:.4f}", flush=True)

    for ep in range(start, a.epochs + 1):
        if bad >= a.patience:
            break
        model.train(); t0 = time.time(); sums = np.zeros(3); n = 0
        for step, (x, y, m, w) in enumerate(tr_dl, 1):
            x, y, m, w = (t.to(device, non_blocking=True) for t in (x, y, m, w))
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                if a.lam > 0:
                    logit, cam = model(x, cam=True)
                else:
                    logit = model(x)
                lcls = sigmoid_focal_loss(logit.float(), y, alpha=alpha, gamma=2.0, reduction="mean")
            latt = attention_loss(cam, m, w) if a.lam > 0 else torch.zeros((), device=device)
            loss = lcls + a.lam * latt
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sums += [loss.item(), lcls.item(), latt.item()]; n += 1
            if a.max_steps and step >= a.max_steps:
                break
        pv = predict(model, va_dl, device, tta=False)
        vauc = roc_auc_score(va.label.values, pv)
        sched.step(vauc)
        improved = vauc > best_auc
        if improved:
            best_auc, bad = vauc, 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
        row = {"epoch": ep, "loss": sums[0] / n, "loss_cls": sums[1] / n, "loss_att": sums[2] / n,
               "val_auc": vauc, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        hist.append(row)
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best_auc": best_auc, "bad": bad, "hist": hist}, last)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)

    # ---- final evaluation with the best checkpoint, TTA ----
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    res = {}
    pv = predict(model, va_dl, device)
    thr = threshold_at_sensitivity(va.label.values, pv)
    for nm, df, p in (("val", va, pv), ("test", te, predict(model, te_dl, device))):
        pd.DataFrame({"isic_id": df.isic_id, "label": df.label, "prob": p}).to_csv(os.path.join(out, f"preds_{nm}.csv"), index=False)
        res[nm] = metrics(df.label.values, p, thr)
    if not a.no_external:
        for ds in ("bald", "pad"):
            f = os.path.join(out, f"preds_{ds}.csv")
            if os.path.exists(f):  # protocol: each run is scored on external data only once
                print(f"{ds}: already scored, kept as is", flush=True)
                d = pd.read_csv(f)
            else:
                e = ext[ext.dataset == ds].reset_index(drop=True)
                d = pd.DataFrame({"isic_id": e.isic_id, "label": e.label, "prob": predict(model, mk(e), device)})
                d.to_csv(f, index=False)
            res[ds] = metrics(d.label.values, d.prob.values, thr)
    summary = {"run": name, "variant": a.variant, "seed": a.seed, "lam": a.lam, "threshold": thr,
               "best_val_auc_no_tta": best_auc, "epochs_run": len(hist), "results": res,
               "git_commit": git_commit(), "args": vars(a)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
