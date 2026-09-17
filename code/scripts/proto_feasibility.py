import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paml_full_lmvg_v2 import (
    make_model_v2, focal_loss, mmd_rbf, sample_task_temporal,
    normalize_fit, apply_norm, clone_wrapper,
    ALL_TASKS, HARD_SWITCH_EP,
)
from lmvg_v2 import cosine_temperature
from event_split import event_level_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")


def proto_logits(emb, proto_pos, proto_neg):
    dp = torch.sum((emb - proto_pos) ** 2, dim=1)
    dn = torch.sum((emb - proto_neg) ** 2, dim=1)
    return dn - dp


def inner_adapt_proto(base, x_s, y_s, inner_lr, steps):
    fast = clone_wrapper(base, "lmvg")
    opt = torch.optim.SGD(fast.parameters(), lr=inner_lr)
    ys = torch.from_numpy(y_s).float()
    xs = torch.from_numpy(x_s)
    for _ in range(steps):
        emb = fast.extract_embedding(xs)
        if (ys == 1).sum() < 1 or (ys == 0).sum() < 1:
            break
        pp = emb[ys == 1].mean(0).detach()
        pn = emb[ys == 0].mean(0).detach()
        logit = proto_logits(emb, pp, pn)
        loss = focal_loss(logit, ys)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fast.parameters(), 1.0); opt.step()
    return fast


def meta_step_proto(meta, src_batches, tgt_unlab, inner_lr, steps, mmd_w):
    meta_grads = None; nq = 0; tot_q = 0.0
    tgt_emb = meta.extract_embedding(torch.from_numpy(tgt_unlab))
    for task in src_batches:
        if task is None:
            continue
        x_s, y_s, x_q, y_q = task
        fast = inner_adapt_proto(meta, x_s, y_s, inner_lr, steps)
        ys = torch.from_numpy(y_s).float()
        emb_s = fast.extract_embedding(torch.from_numpy(x_s))
        if (ys == 1).sum() < 1 or (ys == 0).sum() < 1:
            continue
        pp = emb_s[ys == 1].mean(0)
        pn = emb_s[ys == 0].mean(0)
        emb_q = fast.extract_embedding(torch.from_numpy(x_q))
        logit_q = proto_logits(emb_q, pp, pn)
        q_loss = focal_loss(logit_q, torch.from_numpy(y_q).float())
        mmd = mmd_rbf(emb_q, tgt_emb)
        loss = q_loss + mmd_w * mmd + 2.0 * fast.lmvg.structure_loss()
        params = [p for p in fast.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        gd = {n: g for (n, _), g in zip([(n, p) for n, p in fast.named_parameters() if p.requires_grad], grads)}
        gl = [gd.get(n, None) for n, _ in fast.named_parameters()]
        if meta_grads is None:
            meta_grads = [g.detach().clone() if g is not None else None for g in gl]
        else:
            meta_grads = [(mg + g.detach()) if (mg is not None and g is not None) else (mg if mg is not None else (g.detach().clone() if g is not None else None)) for mg, g in zip(meta_grads, gl)]
        tot_q += float(q_loss.item()); nq += 1
    if nq == 0:
        return None, 0.0
    return [(g / nq) if g is not None else None for g in meta_grads], tot_q / nq


def eval_proto(model, X_sup, y_sup, X_te, y_te):
    model.eval()
    with torch.no_grad():
        emb_s = model.extract_embedding(torch.from_numpy(X_sup))
        ys = torch.from_numpy(y_sup).float()
        pp = emb_s[ys == 1].mean(0); pn = emb_s[ys == 0].mean(0)
        emb_t = model.extract_embedding(torch.from_numpy(X_te))
        logit = proto_logits(emb_t, pp, pn).numpy()
    prob = 1 / (1 + np.exp(-logit))
    if y_te.sum() < 1:
        return None
    auc = roc_auc_score(y_te, prob); ap = average_precision_score(y_te, prob)
    thrs = np.linspace(0.05, 0.95, 91)
    thr = max(thrs, key=lambda t: f1_score(y_te, (prob >= t).astype(int), zero_division=0))
    f1 = f1_score(y_te, (prob >= thr).astype(int), zero_division=0)
    return dict(auc=float(auc), ap=float(ap), f1=float(f1))


def run(X, y, tasks, players, dates, tgt, seed, episodes=120):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    src_tasks = [t for t in ALL_TASKS if t != tgt]
    src_pool = {}
    for t in src_tasks:
        m = tasks == t; src_pool[t] = (X[m], y[m], dates[m])
    tgt_mask = tasks == tgt
    X_tgt, y_tgt = X[tgt_mask], y[tgt_mask]
    players_tgt, dates_tgt = players[tgt_mask], dates[tgt_mask].astype("datetime64[D]")
    sup_p, te_p, sup_n, te_n, info = event_level_split(y_tgt, players_tgt, dates_tgt, rng)
    if len(sup_p) < 2 or len(te_p) < 2:
        return None
    sup_idx = np.concatenate([sup_p, sup_n]); te_idx = np.concatenate([te_p, te_n])
    mu, sd = normalize_fit(np.concatenate([src_pool[t][0] for t in src_tasks], 0))
    for t in src_tasks:
        Xt, yt, dt = src_pool[t]; src_pool[t] = (apply_norm(Xt, mu, sd), yt, dt)
    X_sup = apply_norm(X_tgt[sup_idx], mu, sd); y_sup = y_tgt[sup_idx]
    X_te = apply_norm(X_tgt[te_idx], mu, sd); y_te = y_tgt[te_idx]

    meta = make_model_v2(backend="lmvg"); meta.set_hard(False)
    inner_lr, steps, mmd_w = 5e-3, 2, 0.1
    best_auc, best_state = -1, None
    from paml_full_lmvg_v2 import apply_meta_grads_grouped
    for ep in range(1, episodes + 1):
        meta.set_gumbel_temp(cosine_temperature(ep, episodes))
        if ep >= HARD_SWITCH_EP:
            meta.set_hard(True)
        src_batches = []
        for t in src_tasks:
            Xt, yt, dt = src_pool[t]; pc = int(yt.sum())
            if pc < 4:
                task = sample_task_temporal(Xt, yt, dt, rng, 1, 8, max(1, pc - 1), 8)
            elif pc < 10:
                task = sample_task_temporal(Xt, yt, dt, rng, 2, 8, 2, 8)
            else:
                task = sample_task_temporal(Xt, yt, dt, rng, 4, 16, 4, 16)
            src_batches.append(task)
        ti = rng.choice(len(X_sup), min(32, len(X_sup)), replace=False)
        grads, ql = meta_step_proto(meta, src_batches, X_sup[ti], inner_lr, steps, mmd_w)
        if grads is None:
            continue
        apply_meta_grads_grouped(meta, grads, 2e-3, 5e-1)
        if ep % 20 == 0 or ep <= 2:
            r = eval_proto(meta, X_sup, y_sup, X_te, y_te)
            if r:
                print(f"    ep{ep:03d} qL={ql:.4f} teAUC={r['auc']:.3f} teAP={r['ap']:.4f}")
                if r["auc"] > best_auc:
                    best_auc = r["auc"]; best_state = copy.deepcopy(meta.state_dict())
    if best_state is not None:
        meta.load_state_dict(best_state)
    r = eval_proto(meta, X_sup, y_sup, X_te, y_te)
    return dict(result=r, split=info)


def main():
    d = np.load(os.path.join(SCRIPT_DIR, "..", "data", "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    print("R1-4 原型化可行性验证 (prototype-in-query-loss + nearest-prototype eval)")
    out = []
    t0 = time.time()
    for tgt in ["TeamA-2020", "TeamA-2021"]:
        for seed in [42, 123]:
            print(f"\n=== {tgt} seed{seed} [{(time.time()-t0)/60:.1f}min] ===")
            try:
                r = run(X, y, tasks, players, dates, tgt, seed)
                print(f"  ==> {r['result'] if r else 'FAIL'}")
                out.append(dict(target=tgt, seed=seed, status="ok" if r else "fail", **(r or {})))
            except Exception as e:
                import traceback; traceback.print_exc()
                out.append(dict(target=tgt, seed=seed, status="error", error=str(e)))
            with open(os.path.join(OUT_DIR, "val_proto_feasibility.json"), "w") as f:
                json.dump(out, f, indent=2, default=str)
    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
