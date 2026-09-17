import os
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

from graph_backend import make_model
from lmvg import hamming_distance

ABL_TARGETS = ["TeamA-2020", "TeamA-2021", "TeamB-2020"]
SEEDS = [42, 123, 2026]
ALL_TASKS = ["TeamA-2020", "TeamA-2021", "TeamB-2020", "TeamB-2021"]


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
    return K[:m, :m].mean() + K[m:, m:].mean() - 2 * K[:m, m:].mean()


def normalize_fit(X):
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    return mu, sd


def apply_norm(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def sample_task_temporal(X, y, dates, rng, k_pos, k_neg, q_pos, q_neg, window_days=60):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    need_pos = k_pos + q_pos
    need_neg = k_neg + q_neg
    if len(pos_idx) < need_pos:
        k_pos = min(k_pos, max(len(pos_idx) // 2, 1))
        q_pos = max(len(pos_idx) - k_pos, 1)
        need_pos = k_pos + q_pos
    if len(pos_idx) < need_pos or len(neg_idx) < need_neg:
        return None
    anchor = rng.choice(pos_idx)
    anchor_date = dates[anchor]
    window = np.timedelta64(window_days, "D")
    time_mask = np.abs(dates - anchor_date) <= window
    local_pos = np.intersect1d(pos_idx, np.where(time_mask)[0])
    local_neg = np.intersect1d(neg_idx, np.where(time_mask)[0])
    if len(local_pos) < need_pos:
        local_pos = pos_idx
    if len(local_neg) < need_neg:
        local_neg = neg_idx
    p_sel = rng.choice(local_pos, need_pos, replace=False)
    n_sel = rng.choice(local_neg, need_neg, replace=False)
    sup_idx = np.concatenate([p_sel[:k_pos], n_sel[:k_neg]])
    qry_idx = np.concatenate([p_sel[k_pos:], n_sel[k_neg:]])
    rng.shuffle(sup_idx)
    rng.shuffle(qry_idx)
    return (X[sup_idx], y[sup_idx], X[qry_idx], y[qry_idx])


def task_weights_by_inverse_sqrt(src_pool):
    counts = {t: max(int(y.sum()), 1) for t, (X, y, d) in src_pool.items()}
    raw = {t: 1.0 / np.sqrt(c) for t, c in counts.items()}
    s = sum(raw.values())
    return {t: v / s for t, v in raw.items()}


def clone_wrapper(model, backend):
    clone = make_model(backend=backend)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    return clone


def inner_update(base_model, backend, x_s, y_s, inner_lr, inner_steps):
    fast = clone_wrapper(base_model, backend)
    opt = torch.optim.SGD(fast.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        logit = fast(x_s)
        loss = focal_loss(logit, y_s.float())
        if backend == "lmvg" and fast.lmvg is not None:
            loss = loss + 2.0 * fast.lmvg.structure_loss()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fast.parameters(), 1.0)
        opt.step()
    return fast


def meta_train_step(meta_model, backend, src_batches, tgt_batch, inner_lr, inner_steps, mmd_w):
    meta_grads = None
    total_q, total_m, n = 0.0, 0.0, 0
    tgt_emb = meta_model.extract_embedding(torch.from_numpy(tgt_batch[0]))
    for task in src_batches:
        if task is None:
            continue
        x_s = torch.from_numpy(task[0])
        y_s = torch.from_numpy(task[1]).float()
        x_q = torch.from_numpy(task[2])
        y_q = torch.from_numpy(task[3]).float()
        fast = inner_update(meta_model, backend, x_s, y_s, inner_lr, inner_steps)
        logit_q = fast(x_q)
        q_loss = focal_loss(logit_q, y_q)
        src_emb = fast.extract_embedding(x_q)
        mmd = mmd_rbf(src_emb, tgt_emb)
        loss = q_loss + mmd_w * mmd
        if backend == "lmvg" and fast.lmvg is not None:
            loss = loss + 2.0 * fast.lmvg.structure_loss()
        grads = torch.autograd.grad(loss, fast.parameters(), allow_unused=True)
        if meta_grads is None:
            meta_grads = [g.detach().clone() if g is not None else None for g in grads]
        else:
            meta_grads = [
                (mg + (g.detach().clone() if g is not None else 0)) if mg is not None else (g.detach().clone() if g is not None else None)
                for mg, g in zip(meta_grads, grads)
            ]
        total_q += float(q_loss.item())
        total_m += float(mmd.item())
        n += 1
    if n == 0:
        return None, 0.0, 0.0
    return [(g / n) if g is not None else None for g in meta_grads], total_q / n, total_m / n


def apply_meta_grads(meta_model, meta_grads, meta_lr, lmvg_lr=5e-1):
    lmvg_names = set()
    if meta_model.lmvg is not None:
        for n, _ in meta_model.lmvg.named_parameters():
            lmvg_names.add(f"lmvg.{n}")
    with torch.no_grad():
        for (name, p), g in zip(meta_model.named_parameters(), meta_grads):
            if g is not None:
                lr = lmvg_lr if name in lmvg_names else meta_lr
                p.data -= lr * g


def eval_on_target(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(X_te)).numpy()
    prob = 1 / (1 + np.exp(-logit))
    if y_te.sum() < 2:
        return None
    auc = roc_auc_score(y_te, prob)
    ap = average_precision_score(y_te, prob)
    thrs = np.linspace(0.05, 0.95, 91)
    thr = max(thrs, key=lambda t: f1_score(y_te, (prob >= t).astype(int), zero_division=0))
    yh = (prob >= thr).astype(int)
    return dict(
        auc=float(auc),
        ap=float(ap),
        f1=float(f1_score(y_te, yh, zero_division=0)),
        precision=float(precision_score(y_te, yh, zero_division=0)),
        recall=float(recall_score(y_te, yh, zero_division=0)),
        thr=float(thr),
    )


def fine_tune(model, X_sup, y_sup, steps=50, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(X_sup)
    y = torch.from_numpy(y_sup).float()
    for _ in range(steps):
        opt.zero_grad()
        loss = focal_loss(model(x), y)
        if model.lmvg is not None:
            loss = loss + 2.0 * model.lmvg.structure_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def run_one(backend, X, y, tasks, players, dates_all, tgt_task, seed, log_rows):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    src_tasks = [t for t in ALL_TASKS if t != tgt_task]
    src_pool = {}
    for t in src_tasks:
        m = tasks == t
        src_pool[t] = (X[m], y[m], dates_all[m])

    tgt_mask = tasks == tgt_task
    X_tgt, y_tgt, d_tgt = X[tgt_mask], y[tgt_mask], dates_all[tgt_mask]
    pos_idx = np.where(y_tgt == 1)[0]
    neg_idx = np.where(y_tgt == 0)[0]
    if len(pos_idx) < 4:
        return dict(status="skip_few_pos", target=tgt_task, seed=seed, backend=backend, n_pos=len(pos_idx))
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

    mu, sd = normalize_fit(np.concatenate([src_pool[t][0] for t in src_tasks], axis=0))
    for t in src_tasks:
        Xt, yt, dt = src_pool[t]
        src_pool[t] = (apply_norm(Xt, mu, sd), yt, dt)
    X_sup = apply_norm(X_sup, mu, sd)
    X_te = apply_norm(X_te, mu, sd)

    weights = task_weights_by_inverse_sqrt(src_pool)
    print(f"  [GUARD1 task weights] {weights}")

    n_val_pos = max(len(sup_p) // 3, 1)
    val_p = sup_p[:n_val_pos]
    tr_p = sup_p[n_val_pos:]
    val_n = sup_n[:max(len(sup_n) // 3, 4)]
    tr_n = sup_n[max(len(sup_n) // 3, 4):]
    val_idx = np.concatenate([val_p, val_n])
    tr_idx = np.concatenate([tr_p, tr_n])
    X_val = apply_norm(X_tgt[val_idx], mu, sd)
    y_val = y_tgt[val_idx]
    X_tr = apply_norm(X_tgt[tr_idx], mu, sd)
    y_tr = y_tgt[tr_idx]
    print(f"  [support split] train={len(y_tr)}(pos={int(y_tr.sum())}) val={len(y_val)}(pos={int(y_val.sum())})")
    if y_val.sum() < 1:
        X_val = X_sup
        y_val = y_sup

    meta_model = make_model(backend=backend)
    meta_lr, inner_lr, inner_steps, mmd_w = 1e-3, 5e-3, 2, 0.1
    lmvg_lr = 5e-1
    episodes, patience = 150, 20
    best_val_auc, best_state, stall = -1, None, 0
    prev_q_loss = None
    q_stuck = 0
    prev_hard_adj = None

    epoch_log = []
    for ep in range(1, episodes + 1):
        src_batches = []
        for t in src_tasks:
            Xt, yt, dt = src_pool[t]
            pos_count = int(yt.sum())
            if pos_count < 4:
                task = sample_task_temporal(Xt, yt, dt, rng, k_pos=1, k_neg=8, q_pos=max(1, pos_count - 1), q_neg=8)
            elif pos_count < 10:
                task = sample_task_temporal(Xt, yt, dt, rng, k_pos=2, k_neg=8, q_pos=2, q_neg=8)
            else:
                task = sample_task_temporal(Xt, yt, dt, rng, k_pos=4, k_neg=16, q_pos=4, q_neg=16)
            src_batches.append(task)

        t_idx = rng.choice(len(X_tr), min(32, len(X_tr)), replace=False)
        tgt_batch = (X_tr[t_idx], y_tr[t_idx])
        grads, q_loss, mmd_v = meta_train_step(meta_model, backend, src_batches, tgt_batch, inner_lr, inner_steps, mmd_w)
        if grads is None:
            continue
        apply_meta_grads(meta_model, grads, meta_lr, lmvg_lr)

        grad_tau_norm = 0.0
        if backend == "lmvg" and meta_model.lmvg is not None:
            lmvg_names = {f"lmvg.{n}" for n, _ in meta_model.lmvg.named_parameters()}
            for (name, _p), g in zip(meta_model.named_parameters(), grads):
                if g is not None and "lmvg.tau" in name:
                    grad_tau_norm = float(g.norm().item())

        with torch.no_grad():
            probe_x = torch.from_numpy(X_tr[:min(8, len(X_tr))])
            _ = meta_model(probe_x)
            stats = meta_model._last_stats
            curr_hard = meta_model._last_hard_adj.clone() if meta_model._last_hard_adj is not None else None
        ham = hamming_distance(prev_hard_adj, curr_hard)
        prev_hard_adj = curr_hard

        eval_model = clone_wrapper(meta_model, backend)
        eval_model = fine_tune(eval_model, X_tr, y_tr, steps=15, lr=1e-3)
        r_val = eval_on_target(eval_model, X_val, y_val)
        r_te = eval_on_target(eval_model, X_te, y_te)
        if r_val is None or r_te is None:
            continue

        train_eval = eval_on_target(eval_model, X_tr, y_tr) or {"auc": 0, "f1": 0}
        gap = train_eval["auc"] - r_val["auc"]

        flag_overfit = "OK"
        if gap > 0.20:
            flag_overfit = "OVERFIT"
        elif gap > 0.10:
            flag_overfit = "warn"

        flag_qloss = "OK"
        if prev_q_loss is not None and abs(q_loss - prev_q_loss) < 1e-4:
            q_stuck += 1
            if q_stuck >= 5:
                flag_qloss = "STUCK"
        else:
            q_stuck = 0
        prev_q_loss = q_loss

        flag_mmd = "OK" if mmd_v < 1.0 else "HIGH_MMD"

        flag_drift = "OK"
        if ham > 0.5:
            flag_drift = "DRIFT_HALT"
        elif ham > 0.3:
            flag_drift = "drift_warn"

        vw = stats["view_w"]
        row = dict(
            backend=backend, target=tgt_task, seed=seed, epoch=ep,
            q_loss=round(q_loss, 4), mmd=round(mmd_v, 4),
            train_auc=round(train_eval["auc"], 4),
            val_auc=round(r_val["auc"], 4), val_f1=round(r_val["f1"], 4),
            test_auc=round(r_te["auc"], 4), test_ap=round(r_te["ap"], 4),
            test_f1=round(r_te["f1"], 4), gap=round(gap, 4),
            tau_mean=round(stats["tau_mean"], 4), tau_std=round(stats["tau_std"], 4),
            grad_tau_norm=round(grad_tau_norm, 6),
            vw_causal=round(vw[0], 4), vw_corr=round(vw[1], 4), vw_temp=round(vw[2], 4),
            sparsity=round(stats["sparsity"], 4), hamming=round(ham, 4),
            flag_overfit=flag_overfit, flag_qloss=flag_qloss, flag_mmd=flag_mmd, flag_drift=flag_drift,
        )
        epoch_log.append(row)
        log_rows.append(row)
        print(
            f"  ep{ep:02d}[{backend}] qL={q_loss:.4f} mmd={mmd_v:.3f} "
            f"teAUC={r_te['auc']:.3f} teF1={r_te['f1']:.3f} "
            f"∇τ={grad_tau_norm:.5f} "
            f"tau={stats['tau_mean']:+.2f}±{stats['tau_std']:.2f} "
            f"w=[{vw[0]:.2f},{vw[1]:.2f},{vw[2]:.2f}] "
            f"sp={stats['sparsity']:.3f} ham={ham:.3f} "
            f"[{flag_overfit}|{flag_qloss}|{flag_mmd}|{flag_drift}]"
        )

        if flag_overfit == "OVERFIT" and ep >= 5:
            print(f"  [GUARD-HALT] overfit at ep{ep}")
            break
        if flag_qloss == "STUCK":
            print(f"  [GUARD-HALT] q_loss stuck at ep{ep}")
            break
        if flag_drift == "DRIFT_HALT":
            print(f"  [GUARD-HALT-DRIFT] hamming={ham:.3f} > 0.5 at ep{ep}")
            break

        if r_val["auc"] > best_val_auc:
            best_val_auc = r_val["auc"]
            best_state = copy.deepcopy(meta_model.state_dict())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"  [GUARD-HALT] val stall at ep{ep}")
                break

    if best_state is None:
        return dict(status="no_improvement", backend=backend, target=tgt_task, seed=seed, log=epoch_log)

    meta_model.load_state_dict(best_state)
    final_model = clone_wrapper(meta_model, backend)
    final_model = fine_tune(final_model, X_sup, y_sup, steps=50, lr=1e-3)
    r_final = eval_on_target(final_model, X_te, y_te)

    return dict(
        status="ok", backend=backend, target=tgt_task, seed=seed,
        n_sup=int(len(y_sup)), n_sup_pos=int(y_sup.sum()),
        n_te=int(len(y_te)), n_te_pos=int(y_te.sum()),
        paml=r_final, log=epoch_log,
    )


def main():
    here = os.path.dirname(__file__)
    d = np.load(os.path.join(here, "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    print(f"FULL dataset: N={len(y)} pos={int(y.sum())}")

    all_results = []
    log_rows = []
    t0 = time.time()
    for backend in ["lmvg", "nvg"]:
        for tgt in ABL_TARGETS:
            for seed in SEEDS:
                print(f"\n===== backend={backend} target={tgt} seed={seed} [{(time.time()-t0)/60:.1f}min] =====")
                res = run_one(backend, X, y, tasks, players, dates, tgt, seed, log_rows)
                all_results.append(res)
                if res["status"] == "ok":
                    p = res["paml"]
                    print(f"  ==> [{backend}] AUC={p['auc']:.4f} AP={p['ap']:.4f} F1={p['f1']:.4f}")
                else:
                    print(f"  ==> STATUS: {res['status']}")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(os.path.join(here, "lmvg_ablation_results.json"), "w") as f:
        json.dump([clean(r) for r in all_results], f, indent=2, default=str)
    pd.DataFrame(log_rows).to_csv(os.path.join(here, "lmvg_ablation_epoch_log.csv"), index=False)

    print("\n============ ABLATION AGGREGATE ============")
    for tgt in ABL_TARGETS:
        for backend in ["lmvg", "nvg"]:
            runs = [r for r in all_results if r["target"] == tgt and r["backend"] == backend and r["status"] == "ok"]
            if not runs:
                print(f"  {tgt} [{backend}]: no valid runs")
                continue
            aucs = [r["paml"]["auc"] for r in runs]
            aps = [r["paml"]["ap"] for r in runs]
            f1s = [r["paml"]["f1"] for r in runs]
            print(f"  [{tgt}][{backend:5s}] AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                  f"AP={np.mean(aps):.4f}±{np.std(aps):.4f} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} (n={len(runs)})")


if __name__ == "__main__":
    main()
