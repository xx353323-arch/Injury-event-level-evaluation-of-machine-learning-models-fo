import os
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

ALL_TASKS = ["TeamA-2020", "TeamA-2021", "TeamB-2020", "TeamB-2021"]
ABL_TARGETS = ["TeamA-2020", "TeamA-2021", "TeamB-2020"]
SEEDS = [42, 123, 2026]


def focal_loss(logit, target, alpha=0.75, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(target == 1, p, 1 - p)
    at = torch.where(target == 1, torch.full_like(target, alpha), torch.full_like(target, 1 - alpha))
    return (at * (1 - pt).pow(gamma) * ce).mean()


def normalize_fit(X):
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    return mu, sd


def apply_norm(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def eval_metrics(y_true, y_prob):
    if y_true.sum() < 2:
        return None
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    thrs = np.linspace(0.05, 0.95, 91)
    thr = max(thrs, key=lambda t: f1_score(y_true, (y_prob >= t).astype(int), zero_division=0))
    yh = (y_prob >= thr).astype(int)
    return dict(
        auc=float(auc), ap=float(ap),
        f1=float(f1_score(y_true, yh, zero_division=0)),
        precision=float(precision_score(y_true, yh, zero_division=0)),
        recall=float(recall_score(y_true, yh, zero_division=0)),
    )


class LSTMModel(nn.Module):
    def __init__(self, input_dim=10, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim)
        self.a_l = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_r = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))

    def forward(self, h, adj=None):
        B, T, _ = h.shape
        H = self.n_heads
        dk = self.head_dim
        h_proj = self.W(h).view(B, T, H, dk)
        el = (h_proj * self.a_l).sum(-1)
        er = (h_proj * self.a_r).sum(-1)
        e = self.leaky(el.unsqueeze(3) + er.unsqueeze(2))
        alpha = torch.softmax(e, dim=3)
        out = torch.einsum("bihj,bjhd->bihd", alpha, h_proj)
        return out.reshape(B, T, H * dk)


class StandardGAT(nn.Module):
    def __init__(self, n_ch=10, d_model=32, n_heads=2, T=14):
        super().__init__()
        self.n_ch = n_ch
        self.proj = nn.Linear(1, d_model)
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        B, T, C = x.shape
        ch_embeds = []
        for c in range(C):
            h = self.proj(x[:, :, c:c+1])
            out1, _ = self.attn1(h, h, h)
            h = self.norm1(h + out1)
            out2, _ = self.attn2(h, h, h)
            h = self.norm2(h + out2)
            ch_embeds.append(h.mean(dim=1, keepdim=True))
        pooled = torch.cat(ch_embeds, dim=1).mean(dim=1)
        return self.head(pooled).squeeze(-1)


def train_nn(model, X_tr, y_tr, epochs=10, lr=1e-3, bs=64):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = focal_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def run_xgboost(X_src, y_src, X_sup, y_sup, X_te, y_te, seed):
    B_src, T, C = X_src.shape
    X_src_flat = X_src.reshape(B_src, -1)
    X_sup_flat = X_sup.reshape(len(X_sup), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)
    X_train = np.concatenate([X_src_flat, X_sup_flat], axis=0)
    y_train = np.concatenate([y_src, y_sup], axis=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_te_flat = scaler.transform(X_te_flat)
    sw = np.where(y_train == 1, (y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    clf.fit(X_train, y_train, sample_weight=sw)
    prob = clf.predict_proba(X_te_flat)[:, 1]
    return eval_metrics(y_te, prob)


def run_lstm(X_src, y_src, X_sup, y_sup, X_te, y_te, seed):
    torch.manual_seed(seed)
    X_train = np.concatenate([X_src, X_sup], axis=0).astype(np.float32)
    y_train = np.concatenate([y_src, y_sup], axis=0)
    model = LSTMModel()
    model = train_nn(model, X_train, y_train, epochs=10, lr=1e-3)
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(X_te.astype(np.float32))).numpy()
    prob = 1 / (1 + np.exp(-logit))
    return eval_metrics(y_te, prob)


def run_gat(X_src, y_src, X_sup, y_sup, X_te, y_te, seed):
    torch.manual_seed(seed)
    X_train = np.concatenate([X_src, X_sup], axis=0).astype(np.float32)
    y_train = np.concatenate([y_src, y_sup], axis=0)
    model = StandardGAT()
    model = train_nn(model, X_train, y_train, epochs=10, lr=1e-3)
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(X_te.astype(np.float32))).numpy()
    prob = 1 / (1 + np.exp(-logit))
    return eval_metrics(y_te, prob)


def run_source_only_nn(model_class, X_src, y_src, X_te, y_te, seed):
    torch.manual_seed(seed)
    model = model_class()
    model = train_nn(model, X_src, y_src, epochs=30, lr=1e-3)
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(X_te.astype(np.float32))).numpy()
    prob = 1 / (1 + np.exp(-logit))
    return eval_metrics(y_te, prob)


def main():
    here = os.path.dirname(__file__)
    d = np.load(os.path.join(here, "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    print(f"Dataset: N={len(y)} pos={int(y.sum())}")

    all_results = []
    t0 = time.time()

    for tgt_task in ABL_TARGETS:
        for seed in SEEDS:
            print(f"\n===== target={tgt_task} seed={seed} [{(time.time()-t0)/60:.1f}min] =====")
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)

            src_tasks = [t for t in ALL_TASKS if t != tgt_task]
            tgt_mask = tasks == tgt_task
            X_tgt, y_tgt = X[tgt_mask], y[tgt_mask]

            pos_idx = np.where(y_tgt == 1)[0]
            neg_idx = np.where(y_tgt == 0)[0]
            if len(pos_idx) < 4:
                print(f"  SKIP: pos={len(pos_idx)}")
                continue
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            n_sup_pos = max(len(pos_idx) // 3, 2)
            sup_p, te_p = pos_idx[:n_sup_pos], pos_idx[n_sup_pos:]
            n_sup_neg = min(4 * n_sup_pos, len(neg_idx) // 3)
            sup_n, te_n = neg_idx[:n_sup_neg], neg_idx[n_sup_neg:]
            sup_idx = np.concatenate([sup_p, sup_n])
            te_idx = np.concatenate([te_p, te_n])
            X_sup, y_sup = X_tgt[sup_idx], y_tgt[sup_idx]
            X_te, y_te = X_tgt[te_idx], y_tgt[te_idx]

            X_src = np.concatenate([X[tasks == t] for t in src_tasks], axis=0)
            y_src = np.concatenate([y[tasks == t] for t in src_tasks], axis=0)

            mu, sd = normalize_fit(X_src)
            X_src_n = apply_norm(X_src, mu, sd)
            X_sup_n = apply_norm(X_sup, mu, sd)
            X_te_n = apply_norm(X_te, mu, sd)

            print(f"  src={len(y_src)}(pos={int(y_src.sum())}) sup={len(y_sup)}(pos={int(y_sup.sum())}) te={len(y_te)}(pos={int(y_te.sum())})")

            r_xgb = run_xgboost(X_src_n, y_src, X_sup_n, y_sup, X_te_n, y_te, seed)
            if r_xgb:
                print(f"  [XGBoost]     AUC={r_xgb['auc']:.4f} AP={r_xgb['ap']:.4f} F1={r_xgb['f1']:.4f}")
            all_results.append(dict(method="XGBoost", target=tgt_task, seed=seed, **r_xgb) if r_xgb else dict(method="XGBoost", target=tgt_task, seed=seed, status="fail"))

            r_lstm = run_lstm(X_src_n, y_src, X_sup_n, y_sup, X_te_n, y_te, seed)
            if r_lstm:
                print(f"  [LSTM]        AUC={r_lstm['auc']:.4f} AP={r_lstm['ap']:.4f} F1={r_lstm['f1']:.4f}")
            all_results.append(dict(method="LSTM", target=tgt_task, seed=seed, **r_lstm) if r_lstm else dict(method="LSTM", target=tgt_task, seed=seed, status="fail"))

            r_gat = run_gat(X_src_n, y_src, X_sup_n, y_sup, X_te_n, y_te, seed)
            if r_gat:
                print(f"  [GAT]         AUC={r_gat['auc']:.4f} AP={r_gat['ap']:.4f} F1={r_gat['f1']:.4f}")
            all_results.append(dict(method="GAT", target=tgt_task, seed=seed, **r_gat) if r_gat else dict(method="GAT", target=tgt_task, seed=seed, status="fail"))

    with open(os.path.join(here, "baselines_comparison_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\n============ COMPARISON AGGREGATE ============")
    methods = ["XGBoost", "LSTM", "GAT"]
    for tgt in ABL_TARGETS:
        print(f"\n  {tgt}:")
        for method in methods:
            runs = [r for r in all_results if r.get("target") == tgt and r.get("method") == method and "auc" in r]
            if not runs:
                print(f"    [{method:18s}] no valid runs")
                continue
            aucs = [r["auc"] for r in runs]
            aps = [r["ap"] for r in runs]
            f1s = [r["f1"] for r in runs]
            print(f"    [{method:18s}] AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f} AP={np.mean(aps):.4f}±{np.std(aps):.4f} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} (n={len(runs)})")

    print("\n\n===== REFERENCE: Our PAML results (from paml_full.py) =====")
    print("  [TeamA-2020][PAML] AUC=0.9011±0.0084 AP=0.2669±0.0313 F1=0.3668±0.0145")
    print("  [TeamA-2021][PAML] AUC=0.9039±0.0101 AP=0.0334±0.0041 F1=0.0915±0.0243")
    print("  [TeamB-2020][PAML] AUC=0.9862±0.0060 AP=0.3284±0.1809 F1=0.3653±0.1466")


if __name__ == "__main__":
    main()
