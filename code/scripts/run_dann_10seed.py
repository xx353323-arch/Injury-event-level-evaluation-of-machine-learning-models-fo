import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines_comparison import (
    ALL_TASKS,
    ABL_TARGETS,
    normalize_fit,
    apply_norm,
    eval_metrics,
    focal_loss,
)

SEEDS = [42, 123, 2026, 7, 17, 31, 71, 113, 211, 419]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DANNModel(nn.Module):
    def __init__(self, input_dim=10, hidden=64, n_layers=2, dropout=0.3, n_domains=2):
        super().__init__()
        if input_dim <= 0 or hidden <= 0:
            raise ValueError("DANN dims must be positive")
        self.encoder = nn.LSTM(input_dim, hidden, n_layers,
                               batch_first=True, dropout=dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.dom_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_domains),
        )

    def encode(self, x):
        out, _ = self.encoder(x)
        return out[:, -1, :]

    def forward(self, x, lambd=1.0):
        z = self.encode(x)
        y_logit = self.cls_head(z).squeeze(-1)
        z_rev = grad_reverse(z, lambd)
        d_logit = self.dom_head(z_rev)
        return y_logit, d_logit


def _to_tensor(X, y=None, d=None):
    X_t = torch.from_numpy(X.astype(np.float32))
    if y is None:
        return X_t
    y_t = torch.from_numpy(y.astype(np.float32))
    if d is None:
        return X_t, y_t
    d_t = torch.from_numpy(d.astype(np.int64))
    return X_t, y_t, d_t


def train_dann(model, X_src, y_src, X_tgt_unlab, X_sup, y_sup,
               epochs=12, lr=1e-3, bs=64, weight_decay=1e-5, max_lambda=1.0):
    if len(X_src) == 0 or len(X_tgt_unlab) == 0:
        raise ValueError("DANN requires non-empty source and target streams")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_src_t, y_src_t = _to_tensor(X_src, y_src)
    X_tgt_t = _to_tensor(X_tgt_unlab)
    X_sup_t, y_sup_t = _to_tensor(X_sup, y_sup)

    n_src = X_src_t.shape[0]
    n_tgt = X_tgt_t.shape[0]
    n_sup = X_sup_t.shape[0]
    steps_per_epoch = max(1, (n_src + n_sup) // bs)

    rng = np.random.default_rng(0)
    for ep in range(epochs):
        p = float(ep) / max(epochs - 1, 1)
        lambd = max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        for _ in range(steps_per_epoch):
            idx_src = rng.integers(0, n_src, size=min(bs, n_src))
            idx_tgt = rng.integers(0, n_tgt, size=min(bs, n_tgt))
            xb_src = X_src_t[idx_src]
            yb_src = y_src_t[idx_src]
            xb_tgt = X_tgt_t[idx_tgt]

            if n_sup > 0:
                idx_sup = rng.integers(0, n_sup, size=min(bs // 4, n_sup))
                xb_sup = X_sup_t[idx_sup]
                yb_sup = y_sup_t[idx_sup]
                xb_lab = torch.cat([xb_src, xb_sup], dim=0)
                yb_lab = torch.cat([yb_src, yb_sup], dim=0)
            else:
                xb_lab, yb_lab = xb_src, yb_src

            y_logit_lab, d_logit_lab = model(xb_lab, lambd=lambd)
            _, d_logit_tgt = model(xb_tgt, lambd=lambd)

            cls_loss = focal_loss(y_logit_lab, yb_lab)
            d_lab = torch.zeros(xb_lab.size(0), dtype=torch.long)
            d_tgt = torch.ones(xb_tgt.size(0), dtype=torch.long)
            dom_logit = torch.cat([d_logit_lab, d_logit_tgt], dim=0)
            dom_label = torch.cat([d_lab, d_tgt], dim=0)
            dom_loss = F.cross_entropy(dom_logit, dom_label)

            loss = cls_loss + dom_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def run_dann(X_src, y_src, X_sup, y_sup, X_te, y_te, seed):
    if len(np.unique(y_te)) < 2:
        return None
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DANNModel(input_dim=X_src.shape[-1])
    model = train_dann(model, X_src, y_src, X_te, X_sup, y_sup,
                       epochs=12, lr=1e-3, bs=64)
    model.eval()
    with torch.no_grad():
        logit, _ = model(torch.from_numpy(X_te.astype(np.float32)))
        prob = torch.sigmoid(logit).numpy()
    return eval_metrics(y_te, prob)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "data", "windows.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    d = np.load(data_path, allow_pickle=True)
    X, y, tasks = d["X"], d["y"], d["tasks"]
    print(f"Dataset: N={len(y)} pos={int(y.sum())}")
    print(f"Seeds: {SEEDS}")
    print(f"Targets: {ABL_TARGETS}")
    print("=" * 70)

    out_path = os.path.join(OUT_DIR, "exp_dann_10seed_results.json")
    existing = []
    existing_keys = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            for r in existing:
                if "auc" in r:
                    existing_keys.add((r["target"], r["seed"]))
        except (json.JSONDecodeError, KeyError):
            existing = []

    all_results = list(existing)
    t0 = time.time()

    for tgt_task in ABL_TARGETS:
        tgt_mask = tasks == tgt_task
        src_mask = ~tgt_mask & np.isin(tasks, [t for t in ALL_TASKS if t != tgt_task])
        X_src = X[src_mask]
        y_src = y[src_mask]
        X_tgt = X[tgt_mask]
        y_tgt = y[tgt_mask]

        if len(y_src) == 0 or len(y_tgt) == 0:
            print(f"  SKIP {tgt_task}: empty split")
            continue

        mu, sd = normalize_fit(X_src)
        X_src_n = apply_norm(X_src, mu, sd)

        for seed in SEEDS:
            if (tgt_task, seed) in existing_keys:
                print(f"--- {tgt_task} seed={seed} [REUSE] ---")
                continue

            rng = np.random.default_rng(seed)
            pos_idx = np.where(y_tgt == 1)[0]
            neg_idx = np.where(y_tgt == 0)[0]
            if len(pos_idx) < 2:
                continue
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            k = 10
            sup_pos = pos_idx[:min(k, len(pos_idx))]
            sup_neg = neg_idx[:min(k * 5, len(neg_idx))]
            sup_idx = np.concatenate([sup_pos, sup_neg])
            sup_set = set(sup_idx.tolist())
            te_idx = np.array([i for i in range(len(y_tgt)) if i not in sup_set])
            if len(te_idx) == 0:
                continue

            X_sup_n = apply_norm(X_tgt[sup_idx], mu, sd)
            X_te_n = apply_norm(X_tgt[te_idx], mu, sd)
            y_sup = y_tgt[sup_idx]
            y_te = y_tgt[te_idx]

            print(f"--- {tgt_task} seed={seed} (sup={len(y_sup)} sup_pos={int(y_sup.sum())} "
                  f"te={len(y_te)} te_pos={int(y_te.sum())}) "
                  f"[{(time.time()-t0)/60:.1f}min] ---")

            try:
                r = run_dann(X_src_n, y_src, X_sup_n, y_sup, X_te_n, y_te, seed)
            except (ValueError, RuntimeError) as exc:
                print(f"  [DANN] FAILED: {exc}")
                all_results.append(dict(method="DANN", target=tgt_task, seed=seed,
                                        status="fail", error=str(exc)))
                continue

            if r is None:
                print(f"  [DANN] degenerate test split (single class)")
                all_results.append(dict(method="DANN", target=tgt_task, seed=seed, status="degenerate"))
                continue

            print(f"  [DANN] AUC={r['auc']:.4f} AP={r['ap']:.4f} F1={r['f1']:.4f}")
            all_results.append(dict(method="DANN", target=tgt_task, seed=seed, **r))

            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("DANN AGGREGATE (10 seeds)")
    print("=" * 70)
    summary = {}
    for tgt in ABL_TARGETS:
        runs = [r for r in all_results if r.get("target") == tgt and "auc" in r]
        if not runs:
            print(f"  {tgt}: no valid runs")
            continue
        aucs = np.array([r["auc"] for r in runs])
        aps = np.array([r["ap"] for r in runs])
        f1s = np.array([r["f1"] for r in runs])
        print(f"  {tgt} (n={len(runs)}): "
              f"AUC={aucs.mean():.4f}±{aucs.std():.4f} "
              f"AP={aps.mean():.4f}±{aps.std():.4f} "
              f"F1={f1s.mean():.4f}±{f1s.std():.4f}")
        summary[tgt] = dict(n=len(runs),
                            auc_mean=float(aucs.mean()), auc_sd=float(aucs.std()),
                            ap_mean=float(aps.mean()), ap_sd=float(aps.std()),
                            f1_mean=float(f1s.mean()), f1_sd=float(f1s.std()))

    summary_path = os.path.join(OUT_DIR, "exp_dann_10seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_path}")
    print(f"Summary: {summary_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
