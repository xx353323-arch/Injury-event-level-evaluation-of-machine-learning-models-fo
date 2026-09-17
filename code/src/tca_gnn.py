import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

WIN = 14
N_CH = 10
D_MODEL = 32
N_HEAD = 2


def build_nvg_batch(x_btc):
    B, T, C = x_btc.shape
    device = x_btc.device
    y = x_btc.transpose(1, 2)
    t_i = torch.arange(T, device=device).view(T, 1, 1).float()
    t_j = torch.arange(T, device=device).view(1, T, 1).float()
    t_k = torch.arange(T, device=device).view(1, 1, T).float()
    y_i = y.unsqueeze(-1).unsqueeze(-1)
    y_j = y.unsqueeze(-2).unsqueeze(-1)
    y_k = y.unsqueeze(-2).unsqueeze(-2)
    diff = (t_j - t_i)
    slope = (y_j - y_i) / torch.clamp(diff, min=1e-6)
    line_k = y_i + slope * (t_k - t_i)
    between = ((t_k > t_i) & (t_k < t_j)).float()
    block = ((y_k >= line_k).float() * between).sum(dim=-1)
    any_block = (block > 0).float()
    causal = (t_j.squeeze(-1) > t_i.squeeze(-1)).float().unsqueeze(0).unsqueeze(0)
    visible = (1 - any_block) * causal
    eye = torch.eye(T, device=device).view(1, 1, T, T)
    visible = visible * (1 - eye) + eye
    return visible


class CausalIntraChannelAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.scale = (d_model // n_head) ** 0.5

    def forward(self, h_btd, adj_btt):
        B, T, D = h_btd.shape
        H = self.n_head
        dk = D // H
        Q = self.Wq(h_btd).view(B, T, H, dk).transpose(1, 2)
        K = self.Wk(h_btd).view(B, T, H, dk).transpose(1, 2)
        V = self.Wv(h_btd).view(B, T, H, dk).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        mask_val = -10.0 * (1 - adj_btt).unsqueeze(1)
        scores = scores + mask_val
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return self.norm(h_btd + out)


class InterChannelAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_bcd):
        out, _ = self.attn(h_bcd, h_bcd, h_bcd, need_weights=False)
        return self.norm(h_bcd + out)


class PhysioPositionalEncoding(nn.Module):
    def __init__(self, d_model, T):
        super().__init__()
        pe = torch.zeros(T, d_model)
        pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TCAGNN(nn.Module):
    def __init__(self, n_ch=N_CH, d_model=D_MODEL, n_head=N_HEAD, T=WIN):
        super().__init__()
        self.n_ch = n_ch
        self.d_model = d_model
        self.proj = nn.Linear(1, d_model)
        self.pos = PhysioPositionalEncoding(d_model, T)
        self.intra = nn.ModuleList([CausalIntraChannelAttention(d_model, n_head) for _ in range(n_ch)])
        self.inter = InterChannelAttention(d_model, n_head)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_btc):
        B, T, C = x_btc.shape
        adj = build_nvg_batch(x_btc)
        ch_embeds = []
        for c in range(C):
            h = self.proj(x_btc[:, :, c : c + 1])
            h = self.pos(h)
            h = self.intra[c](h, adj[:, c])
            ch_embeds.append(h.mean(dim=1, keepdim=True))
        h_bcd = torch.cat(ch_embeds, dim=1)
        h_bcd = self.inter(h_bcd)
        pooled = h_bcd.mean(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return logit


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X = (X - mean) / std
    return X.astype(np.float32), mean, std


def train_one(model, loader, opt, device, pos_weight):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).float()
        opt.zero_grad()
        logit = model(xb)
        loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit = model(xb)
        prob = torch.sigmoid(logit).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        ps.extend(prob.tolist())
    ys = np.array(ys)
    ps = np.array(ps)
    auc = roc_auc_score(ys, ps)
    ap = average_precision_score(ys, ps)
    thr = 0.5
    yh = (ps >= thr).astype(int)
    f1 = f1_score(ys, yh, zero_division=0)
    pr = precision_score(ys, yh, zero_division=0)
    rc = recall_score(ys, yh, zero_division=0)
    return dict(auc=auc, ap=ap, f1=f1, precision=pr, recall=rc)


def main():
    here = os.path.dirname(__file__)
    data = np.load(os.path.join(here, "windows.npz"))
    X, y = data["X"], data["y"]
    print(f"total N={len(y)}  pos={int(y.sum())}")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    n_use = len(y) // 5
    use_idx = idx[:n_use]
    Xu, yu = X[use_idx], y[use_idx]
    print(f"using 1/5 subset: N={len(yu)}  pos={int(yu.sum())}  rate={yu.mean():.4f}")
    Xtr, Xte, ytr, yte = train_test_split(Xu, yu, test_size=0.2, stratify=yu, random_state=42)
    Xtr, mu, sd = normalize(Xtr)
    Xte, _, _ = normalize(Xte, mu, sd)
    device = torch.device("cpu")
    bs = 64
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    te_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)
    model = TCAGNN().to(device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32, device=device)
    print(f"pos_weight={pos_weight.item():.2f}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_auc = 0.0
    best_log = None
    EPOCHS = 10
    for ep in range(1, EPOCHS + 1):
        tl = train_one(model, tr_loader, opt, device, pos_weight)
        m = eval_one(model, te_loader, device)
        print(f"ep{ep:02d} loss={tl:.4f} AUC={m['auc']:.4f} AP={m['ap']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_log = m
    print("\nBEST:", best_log)
    return best_log


if __name__ == "__main__":
    main()
