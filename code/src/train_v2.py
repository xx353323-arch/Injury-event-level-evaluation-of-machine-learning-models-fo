import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import GroupKFold

from tca_gnn import TCAGNN

torch.manual_seed(42)
np.random.seed(42)


def focal_loss(logit, target, alpha=0.75, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(target == 1, p, 1 - p)
    at = torch.where(target == 1, torch.full_like(target, alpha), torch.full_like(target, 1 - alpha))
    loss = at * (1 - pt).pow(gamma) * ce
    return loss.mean()


def normalize_fit(X):
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    return mu, sd


def apply_norm(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def best_threshold(y_true, y_prob):
    thrs = np.linspace(0.05, 0.98, 94)
    best_t, best_f = 0.5, -1.0
    for t in thrs:
        yh = (y_prob >= t).astype(int)
        f = f1_score(y_true, yh, zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t
    return best_t, best_f


def temperature_scale(logits, labels):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_T = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=80)

    def closure():
        opt.zero_grad()
        T = log_T.exp()
        loss = F.binary_cross_entropy_with_logits(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_T.exp().detach().item())
    if not np.isfinite(T) or T < 0.1 or T > 20.0:
        T = 1.0
    return T


def train_epoch(model, loader, opt, device, loss_fn):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).float()
        opt.zero_grad()
        logit = model(xb)
        loss = loss_fn(logit, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ys, logits = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        lg = model(xb).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        logits.extend(lg.tolist())
    return np.array(ys), np.array(logits)


def make_loader(X, y, bs=64, shuffle=False):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)


def run_fold(Xtr, ytr, Xva, yva, Xte, yte, loss_fn, device, epochs=10):
    mu, sd = normalize_fit(Xtr)
    Xtr = apply_norm(Xtr, mu, sd)
    Xva = apply_norm(Xva, mu, sd)
    Xte = apply_norm(Xte, mu, sd)
    tr = make_loader(Xtr, ytr, bs=64, shuffle=True)
    va = make_loader(Xva, yva, bs=128)
    te = make_loader(Xte, yte, bs=128)
    model = TCAGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_auc, best_state = -1, None
    for ep in range(1, epochs + 1):
        tl = train_epoch(model, tr, opt, device, loss_fn)
        yv, lv = infer(model, va, device)
        pv = 1 / (1 + np.exp(-lv))
        val_auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    yv, lv = infer(model, va, device)
    T = temperature_scale(lv, yv)
    lv_cal = lv / T
    pv_cal = 1 / (1 + np.exp(-lv_cal))
    tau, _ = best_threshold(yv, pv_cal)
    yt, lt = infer(model, te, device)
    lt_cal = lt / T
    pt_cal = 1 / (1 + np.exp(-lt_cal))
    yh = (pt_cal >= tau).astype(int)
    auc = roc_auc_score(yt, pt_cal) if len(np.unique(yt)) > 1 else 0.5
    ap = average_precision_score(yt, pt_cal) if yt.sum() > 0 else 0.0
    f1 = f1_score(yt, yh, zero_division=0)
    pr = precision_score(yt, yh, zero_division=0)
    rc = recall_score(yt, yh, zero_division=0)
    return dict(auc=auc, ap=ap, f1=f1, precision=pr, recall=rc, T=T, tau=tau)


def main():
    here = os.path.dirname(__file__)
    data = np.load(os.path.join(here, "windows.npz"), allow_pickle=True)
    X, y, players = data["X"], data["y"], data["players"]
    print(f"total N={len(y)}  pos={int(y.sum())}")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    n_use = len(y) // 3
    sub = idx[:n_use]
    Xs, ys, ps = X[sub], y[sub], players[sub]
    print(f"1/3 subset N={len(ys)}  pos={int(ys.sum())}  rate={ys.mean():.4f}")
    print(f"unique players in subset: {len(np.unique(ps))}")
    device = torch.device("cpu")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(Xs, ys, groups=ps))
    results_bce = []
    results_focal = []
    for fold, (tr_idx, te_idx) in enumerate(splits):
        tr_players = np.unique(ps[tr_idx])
        rng2 = np.random.default_rng(fold)
        val_players = rng2.choice(tr_players, size=max(len(tr_players) // 5, 2), replace=False)
        va_mask = np.isin(ps[tr_idx], val_players)
        va_idx = tr_idx[va_mask]
        tr2_idx = tr_idx[~va_mask]
        print(f"\n==== Fold {fold+1}  train={len(tr2_idx)}  val={len(va_idx)}  test={len(te_idx)} ====")
        print(f"    pos: tr={int(ys[tr2_idx].sum())} va={int(ys[va_idx].sum())} te={int(ys[te_idx].sum())}")
        if ys[te_idx].sum() < 2 or ys[va_idx].sum() < 2:
            print("    skipped (too few positives)")
            continue
        pw = (ys[tr2_idx] == 0).sum() / max((ys[tr2_idx] == 1).sum(), 1)
        pw_t = torch.tensor([pw], dtype=torch.float32)

        def bce_loss(logit, tgt, w=pw_t):
            return F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=w)

        r1 = run_fold(Xs[tr2_idx], ys[tr2_idx], Xs[va_idx], ys[va_idx], Xs[te_idx], ys[te_idx], bce_loss, device, epochs=15)
        print(f"  [BCE+pw]   AUC={r1['auc']:.4f} AP={r1['ap']:.4f} F1={r1['f1']:.4f} P={r1['precision']:.4f} R={r1['recall']:.4f} T={r1['T']:.2f} tau={r1['tau']:.2f}")
        results_bce.append(r1)

        r2 = run_fold(Xs[tr2_idx], ys[tr2_idx], Xs[va_idx], ys[va_idx], Xs[te_idx], ys[te_idx], focal_loss, device, epochs=15)
        print(f"  [Focal]    AUC={r2['auc']:.4f} AP={r2['ap']:.4f} F1={r2['f1']:.4f} P={r2['precision']:.4f} R={r2['recall']:.4f} T={r2['T']:.2f} tau={r2['tau']:.2f}")
        results_focal.append(r2)

    def summarize(rs, name):
        mean = {k: float(np.mean([r[k] for r in rs])) for k in ["auc", "ap", "f1", "precision", "recall"]}
        std = {k: float(np.std([r[k] for r in rs])) for k in ["auc", "ap", "f1", "precision", "recall"]}
        print(f"\n==== {name}  (mean±std over {len(rs)} folds) ====")
        for k in ["auc", "ap", "f1", "precision", "recall"]:
            print(f"   {k:9s} {mean[k]:.4f} ± {std[k]:.4f}")

    summarize(results_bce, "BCE+pos_weight+GroupKFold+ThrSearch+TempScale")
    summarize(results_focal, "FocalLoss+GroupKFold+ThrSearch+TempScale")


if __name__ == "__main__":
    main()
