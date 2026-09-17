import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score


def focal_loss(logit, target, alpha=0.75, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(target == 1, p, 1 - p)
    at = torch.where(target == 1, torch.full_like(target, alpha), torch.full_like(target, 1 - alpha))
    return (at * (1 - pt).pow(gamma) * ce).mean()


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


class TransformerEncoder(nn.Module):
    def __init__(self, n_channels=10, T=14, d_model=32, n_heads=2, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


class ProtoNet(nn.Module):
    def __init__(self, n_channels=10, T=14, d_model=32, n_heads=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj_head = nn.Linear(d_model, d_model)

    def embed(self, x):
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.proj_head(h)

    def forward(self, x):
        return self.embed(x)


def train_transformer(X_src, y_src, X_sup, y_sup, X_te, y_te, seed, epochs=10, lr=1e-3, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train = np.concatenate([X_src, X_sup], axis=0).astype(np.float32)
    y_train = np.concatenate([y_src, y_sup], axis=0)
    model = TransformerEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = focal_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(X_te.astype(np.float32))).numpy()
    prob = 1 / (1 + np.exp(-logit))
    return eval_metrics(y_te, prob)


def train_protonet(X_src, y_src, X_sup, y_sup, X_te, y_te, seed, episodes=50, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ProtoNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_src_t = torch.from_numpy(X_src.astype(np.float32))
    y_src_t = torch.from_numpy(y_src.astype(np.float32))
    pos_idx = np.where(y_src == 1)[0]
    neg_idx = np.where(y_src == 0)[0]

    model.train()
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        k_pos = min(10, len(pos_idx))
        k_neg = k_pos * 4
        q_pos = min(10, len(pos_idx))
        q_neg = q_pos * 4

        sp = rng.choice(pos_idx, k_pos, replace=False)
        sn = rng.choice(neg_idx, k_neg, replace=False)
        qp = rng.choice(pos_idx, q_pos, replace=False)
        qn = rng.choice(neg_idx, q_neg, replace=False)

        sup_x = X_src_t[np.concatenate([sp, sn])]
        sup_y = y_src_t[np.concatenate([sp, sn])]
        qry_x = X_src_t[np.concatenate([qp, qn])]
        qry_y = y_src_t[np.concatenate([qp, qn])]

        sup_emb = model(sup_x)
        proto_pos = sup_emb[sup_y == 1].mean(dim=0)
        proto_neg = sup_emb[sup_y == 0].mean(dim=0)

        qry_emb = model(qry_x)
        dist_pos = torch.sum((qry_emb - proto_pos) ** 2, dim=1)
        dist_neg = torch.sum((qry_emb - proto_neg) ** 2, dim=1)
        logits = dist_neg - dist_pos
        loss = F.binary_cross_entropy_with_logits(logits, qry_y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        X_sup_t = torch.from_numpy(X_sup.astype(np.float32))
        y_sup_np = y_sup
        sup_emb = model(X_sup_t)
        proto_pos = sup_emb[y_sup_np == 1].mean(dim=0)
        proto_neg = sup_emb[y_sup_np == 0].mean(dim=0)

        X_te_t = torch.from_numpy(X_te.astype(np.float32))
        te_emb = model(X_te_t)
        dist_pos = torch.sum((te_emb - proto_pos) ** 2, dim=1)
        dist_neg = torch.sum((te_emb - proto_neg) ** 2, dim=1)
        logits = (dist_neg - dist_pos).numpy()

    prob = 1 / (1 + np.exp(-logits))
    return eval_metrics(y_te, prob)
