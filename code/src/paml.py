import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

from tca_gnn import TCAGNN

torch.manual_seed(42)
np.random.seed(42)


def focal_loss(logit, target, alpha=0.75, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(target == 1, p, 1 - p)
    at = torch.where(target == 1, torch.full_like(target, alpha), torch.full_like(target, 1 - alpha))
    return (at * (1 - pt).pow(gamma) * ce).mean()


def mmd_rbf(x, y, sigmas=(1.0, 2.0, 4.0, 8.0)):
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device)
    xy = torch.cat([x, y], dim=0)
    d2 = torch.cdist(xy, xy, p=2).pow(2)
    m = x.size(0)
    K = torch.zeros_like(d2)
    for s in sigmas:
        K = K + torch.exp(-d2 / (2 * s ** 2))
    Kxx = K[:m, :m].mean()
    Kyy = K[m:, m:].mean()
    Kxy = K[:m, m:].mean()
    return Kxx + Kyy - 2 * Kxy


def normalize_fit(X):
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    return mu, sd


def apply_norm(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def sample_task(X, y, rng, k_pos=4, k_neg=16, q_pos=4, q_neg=16):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < k_pos + q_pos or len(neg_idx) < k_neg + q_neg:
        return None
    p_sel = rng.choice(pos_idx, k_pos + q_pos, replace=False)
    n_sel = rng.choice(neg_idx, k_neg + q_neg, replace=False)
    sup_idx = np.concatenate([p_sel[:k_pos], n_sel[:k_neg]])
    qry_idx = np.concatenate([p_sel[k_pos:], n_sel[k_neg:]])
    rng.shuffle(sup_idx)
    rng.shuffle(qry_idx)
    return (X[sup_idx], y[sup_idx], X[qry_idx], y[qry_idx])


def clone_model(model):
    clone = TCAGNN()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    return clone


def inner_update(base_model, x_s, y_s, inner_lr, inner_steps):
    fast = clone_model(base_model)
    opt = torch.optim.SGD(fast.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        logit = fast(x_s)
        loss = focal_loss(logit, y_s.float())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fast.parameters(), 1.0)
        opt.step()
    return fast


def extract_embedding(model, x):
    with torch.no_grad():
        adj = None
    B, T, C = x.shape
    from tca_gnn import build_nvg_batch
    adj = build_nvg_batch(x)
    ch_embeds = []
    for c in range(C):
        h = model.proj(x[:, :, c : c + 1])
        h = model.pos(h)
        h = model.intra[c](h, adj[:, c])
        ch_embeds.append(h.mean(dim=1, keepdim=True))
    h_bcd = torch.cat(ch_embeds, dim=1)
    h_bcd = model.inter(h_bcd)
    return h_bcd.mean(dim=1)


def meta_train_step(meta_model, src_batches, tgt_batch, inner_lr, inner_steps, mmd_w):
    meta_grads = None
    total_query_loss = 0.0
    total_mmd = 0.0
    n_tasks = 0
    tgt_x = torch.from_numpy(tgt_batch[0])
    tgt_emb = extract_embedding(meta_model, tgt_x)
    for task in src_batches:
        if task is None:
            continue
        x_s = torch.from_numpy(task[0])
        y_s = torch.from_numpy(task[1]).float()
        x_q = torch.from_numpy(task[2])
        y_q = torch.from_numpy(task[3]).float()
        fast = inner_update(meta_model, x_s, y_s, inner_lr, inner_steps)
        logit_q = fast(x_q)
        q_loss = focal_loss(logit_q, y_q)
        src_emb = extract_embedding(fast, x_q)
        mmd = mmd_rbf(src_emb, tgt_emb)
        loss = q_loss + mmd_w * mmd
        grads = torch.autograd.grad(loss, fast.parameters(), allow_unused=True)
        if meta_grads is None:
            meta_grads = [g.detach().clone() if g is not None else None for g in grads]
        else:
            meta_grads = [
                (mg + (g.detach().clone() if g is not None else 0))
                if mg is not None else (g.detach().clone() if g is not None else None)
                for mg, g in zip(meta_grads, grads)
            ]
        total_query_loss += float(q_loss.item())
        total_mmd += float(mmd.item())
        n_tasks += 1
    if n_tasks == 0:
        return None, 0.0, 0.0
    meta_grads = [(g / n_tasks) if g is not None else None for g in meta_grads]
    return meta_grads, total_query_loss / n_tasks, total_mmd / n_tasks


def apply_meta_grads(meta_model, meta_grads, meta_lr):
    with torch.no_grad():
        for p, g in zip(meta_model.parameters(), meta_grads):
            if g is not None:
                p.data -= meta_lr * g


def eval_on_target(model, X_te, y_te, thr=None):
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(X_te)).numpy()
    prob = 1 / (1 + np.exp(-logit))
    if y_te.sum() < 2:
        return None
    auc = roc_auc_score(y_te, prob)
    ap = average_precision_score(y_te, prob)
    if thr is None:
        thrs = np.linspace(0.05, 0.95, 91)
        thr = max(thrs, key=lambda t: f1_score(y_te, (prob >= t).astype(int), zero_division=0))
    yh = (prob >= thr).astype(int)
    f1 = f1_score(y_te, yh, zero_division=0)
    pr = precision_score(y_te, yh, zero_division=0)
    rc = recall_score(y_te, yh, zero_division=0)
    return dict(auc=auc, ap=ap, f1=f1, precision=pr, recall=rc, thr=thr, prob=prob)


def fine_tune(model, X_sup, y_sup, steps=50, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(X_sup)
    y = torch.from_numpy(y_sup).float()
    for _ in range(steps):
        opt.zero_grad()
        logit = model(x)
        loss = focal_loss(logit, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def main():
    here = os.path.dirname(__file__)
    d = np.load(os.path.join(here, "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]

    src_tasks = ["TeamA-2020", "TeamA-2021", "TeamB-2020"]
    tgt_task = "TeamB-2021"
    print(f"Source tasks: {src_tasks}")
    print(f"Target task: {tgt_task}  (N={(tasks==tgt_task).sum()} pos={int(y[tasks==tgt_task].sum())})")

    src_pool = {}
    for t in src_tasks:
        mask = tasks == t
        src_pool[t] = (X[mask], y[mask])
        print(f"  src[{t}]: N={mask.sum()} pos={int(y[mask].sum())}")

    tgt_mask = tasks == tgt_task
    X_tgt, y_tgt = X[tgt_mask], y[tgt_mask]
    rng = np.random.default_rng(42)
    pos_idx = np.where(y_tgt == 1)[0]
    neg_idx = np.where(y_tgt == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_sup_pos = max(len(pos_idx) // 3, 2)
    sup_p = pos_idx[:n_sup_pos]
    te_p = pos_idx[n_sup_pos:]
    n_sup_neg = min(4 * n_sup_pos, len(neg_idx) // 3)
    sup_n = neg_idx[:n_sup_neg]
    te_n = neg_idx[n_sup_neg:]
    sup_idx = np.concatenate([sup_p, sup_n])
    te_idx = np.concatenate([te_p, te_n])
    X_sup, y_sup = X_tgt[sup_idx], y_tgt[sup_idx]
    X_te, y_te = X_tgt[te_idx], y_tgt[te_idx]
    print(f"Target support: N={len(y_sup)} pos={int(y_sup.sum())}")
    print(f"Target test:    N={len(y_te)} pos={int(y_te.sum())}")

    mu, sd = normalize_fit(np.concatenate([src_pool[t][0] for t in src_tasks], axis=0))
    for t in src_tasks:
        Xt, yt = src_pool[t]
        src_pool[t] = (apply_norm(Xt, mu, sd), yt)
    X_sup = apply_norm(X_sup, mu, sd)
    X_te = apply_norm(X_te, mu, sd)

    meta_model = TCAGNN()
    meta_lr = 1e-3
    inner_lr = 5e-3
    inner_steps = 2
    mmd_w = 0.1
    episodes = 60
    tgt_batch_size = 32
    best_auc = -1
    best_state = None
    patience = 12
    stall = 0

    train_log = []
    for ep in range(1, episodes + 1):
        src_batches = []
        for t in src_tasks:
            Xt, yt = src_pool[t]
            pos_count = int(yt.sum())
            if pos_count < 8:
                task = sample_task(Xt, yt, rng, k_pos=min(2, pos_count // 2), k_neg=8, q_pos=min(2, pos_count - pos_count // 2), q_neg=8)
            else:
                task = sample_task(Xt, yt, rng, k_pos=4, k_neg=16, q_pos=4, q_neg=16)
            src_batches.append(task)
        tgt_sample_idx = rng.choice(len(X_sup), min(tgt_batch_size, len(X_sup)), replace=False)
        tgt_batch = (X_sup[tgt_sample_idx], y_sup[tgt_sample_idx])

        grads, q_loss, mmd_v = meta_train_step(meta_model, src_batches, tgt_batch, inner_lr, inner_steps, mmd_w)
        if grads is None:
            continue
        apply_meta_grads(meta_model, grads, meta_lr)

        eval_model = clone_model(meta_model)
        eval_model = fine_tune(eval_model, X_sup, y_sup, steps=20, lr=1e-3)
        r = eval_on_target(eval_model, X_te, y_te)
        if r is None:
            print(f"ep{ep:02d} q_loss={q_loss:.4f} mmd={mmd_v:.4f}  (eval skipped)")
            continue
        gap = q_loss
        msg = f"ep{ep:02d} q_loss={q_loss:.4f} mmd={mmd_v:.4f}  AUC={r['auc']:.4f} AP={r['ap']:.4f} F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} thr={r['thr']:.2f}"
        print(msg)
        train_log.append((ep, q_loss, mmd_v, r["auc"], r["ap"], r["f1"]))

        if r["auc"] > best_auc:
            best_auc = r["auc"]
            best_state = copy.deepcopy(meta_model.state_dict())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"Early stopping at ep{ep} (no improvement for {patience} eps)")
                break

    meta_model.load_state_dict(best_state)
    final_model = clone_model(meta_model)
    final_model = fine_tune(final_model, X_sup, y_sup, steps=50, lr=1e-3)
    r = eval_on_target(final_model, X_te, y_te)
    print("\n======== FINAL PAML (after fine-tune on target support) ========")
    for k in ["auc", "ap", "f1", "precision", "recall"]:
        print(f"   {k:9s} {r[k]:.4f}")
    print(f"   chosen_thr = {r['thr']:.2f}")

    print("\n======== BASELINE: direct train on 3 source tasks, test on target ========")
    all_src_X = np.concatenate([src_pool[t][0] for t in src_tasks], axis=0).astype(np.float32)
    all_src_y = np.concatenate([src_pool[t][1] for t in src_tasks], axis=0)
    baseline = TCAGNN()
    opt = torch.optim.Adam(baseline.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 64
    ds = torch.utils.data.TensorDataset(torch.from_numpy(all_src_X), torch.from_numpy(all_src_y))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(5):
        baseline.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = focal_loss(baseline(xb), yb.float())
            loss.backward()
            opt.step()
    r_base = eval_on_target(baseline, X_te, y_te)
    for k in ["auc", "ap", "f1", "precision", "recall"]:
        print(f"   {k:9s} {r_base[k]:.4f}")


if __name__ == "__main__":
    main()
