import os, sys, json, time, copy
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paml_full_lmvg_v2 as P
from paml_full_lmvg_v2 import (
    make_model_v2, normalize_fit, apply_norm, eval_on_target, clone_wrapper,
    apply_meta_grads_grouped, ALL_TASKS, HARD_SWITCH_EP,
)
from lmvg_v2 import hamming_distance, cosine_temperature
from event_split import event_level_split

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
SEEDS = [42, 123, 2026, 0, 1, 2, 3, 4, 5, 6]
TARGETS = ["TeamA-2020", "TeamA-2021", "TeamB-2020"]
VARIANTS = {
    "full": dict(backend="lmvg", mmd_w=0.1, window_days=60, loss="focal"),
    "no_mmd": dict(backend="lmvg", mmd_w=0.0, window_days=60, loss="focal"),
    "no_focal": dict(backend="lmvg", mmd_w=0.1, window_days=60, loss="bce"),
    "no_temporal": dict(backend="lmvg", mmd_w=0.1, window_days=100000, loss="focal"),
    "nvg": dict(backend="nvg", mmd_w=0.1, window_days=60, loss="focal"),
}
_focal = P.focal_loss


def bce_weighted(logit, target, pos_weight_val=10.0):
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=torch.tensor([pos_weight_val]))


def run_one(X, y, tasks, players, dates, tgt, seed, cfg, episodes=120, patience=20):
    P.focal_loss = bce_weighted if cfg["loss"] == "bce" else _focal
    backend, mmd_w, wd = cfg["backend"], cfg["mmd_w"], cfg["window_days"]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    src_tasks = [t for t in ALL_TASKS if t != tgt]
    src_pool = {t: (X[tasks == t], y[tasks == t], dates[tasks == t]) for t in src_tasks}
    m = tasks == tgt
    X_tgt, y_tgt, p_tgt, d_tgt = X[m], y[m], players[m], dates[m].astype("datetime64[D]")
    sup_p, te_p, sup_n, te_n, info = event_level_split(y_tgt, p_tgt, d_tgt, rng)
    if len(sup_p) < 2 or len(te_p) < 2 or len(sup_n) < 4 or len(te_n) < 4:
        return None, info
    sup_idx = np.concatenate([sup_p, sup_n]); te_idx = np.concatenate([te_p, te_n])
    mu, sd = normalize_fit(np.concatenate([src_pool[t][0] for t in src_tasks], 0))
    for t in src_tasks:
        Xs, ys, ds = src_pool[t]; src_pool[t] = (apply_norm(Xs, mu, sd), ys, ds)
    X_sup = apply_norm(X_tgt[sup_idx], mu, sd); y_sup = y_tgt[sup_idx]
    X_te = apply_norm(X_tgt[te_idx], mu, sd); y_te = y_tgt[te_idx]
    nvp = max(len(sup_p) // 3, 1); lp = np.arange(len(sup_p)); rng.shuffle(lp)
    val_p, tr_p = sup_p[lp[:nvp]], sup_p[lp[nvp:]]
    ln = np.arange(len(sup_n)); rng.shuffle(ln); nvn = max(len(sup_n) // 3, min(4, len(sup_n)))
    val_n, tr_n = sup_n[ln[:nvn]], sup_n[ln[nvn:]]
    tr_l = np.concatenate([tr_p, tr_n]) if len(tr_p) else np.concatenate([sup_p, tr_n])
    X_tr = apply_norm(X_tgt[tr_l], mu, sd); y_tr = y_tgt[tr_l]
    X_val = apply_norm(X_tgt[np.concatenate([val_p, val_n])], mu, sd); y_val = y_tgt[np.concatenate([val_p, val_n])]
    if y_tr.sum() < 1: X_tr, y_tr = X_sup, y_sup
    if y_val.sum() < 1: X_val, y_val = X_sup, y_sup

    meta = make_model_v2(backend=backend); meta.set_hard(False)
    best_val, best_state, stall, prev_hard = -1, None, 0, None
    for ep in range(1, episodes + 1):
        meta.set_gumbel_temp(cosine_temperature(ep, episodes))
        if ep >= HARD_SWITCH_EP: meta.set_hard(True)
        batches = []
        for t in src_tasks:
            Xs, ys, ds = src_pool[t]; pc = int(ys.sum())
            k = (4, 16, 4, 16) if pc >= 10 else ((2, 8, 2, 8) if pc >= 4 else (1, 8, max(1, pc - 1), 8))
            batches.append(P.sample_task_temporal(Xs, ys, ds, rng, *k, window_days=wd))
        ti = rng.choice(len(X_tr), min(32, len(X_tr)), replace=False)
        grads, ql, mv = P.meta_train_step(meta, backend, batches, (X_tr[ti], y_tr[ti]), 5e-3, 2, mmd_w)
        if grads is None: continue
        apply_meta_grads_grouped(meta, grads, 2e-3, 5e-1)
        with torch.no_grad():
            _ = meta(torch.from_numpy(X_tr[:min(8, len(X_tr))]))
            curr = meta._last_hard_adj.clone() if meta._last_hard_adj is not None else None
        ham = hamming_distance(prev_hard, curr); prev_hard = curr
        em = P.fine_tune(clone_wrapper(meta, backend), X_tr, y_tr, steps=15, lr=1e-3)
        rv = eval_on_target(em, X_val, y_val)
        if rv is None: continue
        if ham > 0.5 and backend == "lmvg": break
        if rv["auc"] > best_val: best_val, best_state, stall = rv["auc"], copy.deepcopy(meta.state_dict()), 0
        else:
            stall += 1
            if stall >= patience: break
    if best_state is None: return None, info
    meta.load_state_dict(best_state)
    fm = P.fine_tune(clone_wrapper(meta, backend), X_sup, y_sup, steps=50, lr=1e-3)
    return eval_on_target(fm, X_te, y_te), info


def main():
    variant = sys.argv[1]; cfg = VARIANTS[variant]
    out_path = os.path.join(OUT_DIR, f"val_ablation_eventsplit_{variant}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["target"], r["seed"]) for r in out if r.get("status") == "ok"}
    d = np.load(os.path.join(OUT_DIR, "..", "data", "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    print(f"variant={variant} cfg={cfg} resume={len(done)}", flush=True)
    t0 = time.time()
    for tgt in TARGETS:
        for seed in SEEDS:
            if (tgt, seed) in done: continue
            print(f"[{variant}] {tgt} seed{seed} [{(time.time()-t0)/60:.1f}min]", flush=True)
            try:
                r, info = run_one(X, y, tasks, players, dates, tgt, seed, cfg)
                print(f"    -> {r}", flush=True)
                out.append(dict(variant=variant, target=tgt, seed=seed, status="ok" if r else "fail", result=r, split=info))
            except Exception as e:
                import traceback; traceback.print_exc()
                out.append(dict(variant=variant, target=tgt, seed=seed, status="error", error=str(e)[:200]))
            json.dump(out, open(out_path, "w"), indent=2, default=str)
    print(f"DONE {variant} {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
