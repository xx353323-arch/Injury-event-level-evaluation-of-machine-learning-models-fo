import os, sys, copy, json, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paml_full_lmvg_v2 import (
    make_model_v2, focal_loss, mmd_rbf, sample_task_temporal,
    task_weights_by_inverse_sqrt, inner_update, meta_train_step,
    apply_meta_grads_grouped, compute_tau_grad_norm,
    normalize_fit, apply_norm, eval_on_target, fine_tune, clone_wrapper,
    ABL_TARGETS, ALL_TASKS, HARD_SWITCH_EP, NVG_BASELINE_AUC,
)
from lmvg_v2 import hamming_distance, cosine_temperature
from event_split import event_level_split

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
SEEDS_ALL = [42, 123, 2026, 0, 1, 2, 3, 4, 5, 6]


def run_one_eventsplit(X, y, tasks, players, dates, tgt_task, seed, episodes=200, patience=20):
    meta_lr_base, meta_lr_lmvg = 2e-3, 5e-1
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    src_tasks = [t for t in ALL_TASKS if t != tgt_task]
    src_pool = {}
    for t in src_tasks:
        m = tasks == t
        src_pool[t] = (X[m], y[m], dates[m])

    tgt_mask = tasks == tgt_task
    X_tgt, y_tgt = X[tgt_mask], y[tgt_mask]
    players_tgt = players[tgt_mask]
    dates_tgt = dates[tgt_mask].astype("datetime64[D]")

    sup_p, te_p, sup_n, te_n, split_stats = event_level_split(y_tgt, players_tgt, dates_tgt, rng)
    print(f"  [EVENT-SPLIT] {split_stats}")
    if len(sup_p) < 2 or len(te_p) < 2 or len(sup_n) < 4 or len(te_n) < 4:
        return None, split_stats

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

    n_val_pos = max(len(sup_p) // 3, 1)
    val_p_local = np.arange(len(sup_p))
    rng.shuffle(val_p_local)
    val_p = sup_p[val_p_local[:n_val_pos]]
    tr_p = sup_p[val_p_local[n_val_pos:]]
    val_n_local = np.arange(len(sup_n))
    rng.shuffle(val_n_local)
    n_val_neg = max(len(sup_n) // 3, min(4, len(sup_n)))
    val_n = sup_n[val_n_local[:n_val_neg]]
    tr_n = sup_n[val_n_local[n_val_neg:]]
    val_idx_local = np.concatenate([val_p, val_n])
    tr_idx_local = np.concatenate([tr_p, tr_n]) if len(tr_p) > 0 else np.concatenate([sup_p, tr_n])
    X_val = apply_norm(X_tgt[val_idx_local], mu, sd)
    y_val = y_tgt[val_idx_local]
    X_tr = apply_norm(X_tgt[tr_idx_local], mu, sd)
    y_tr = y_tgt[tr_idx_local]
    if y_tr.sum() < 1:
        X_tr, y_tr = X_sup, y_sup
    if y_val.sum() < 1:
        X_val, y_val = X_sup, y_sup
    print(f"  [support split] train={len(y_tr)}(pos={int(y_tr.sum())}) val={len(y_val)}(pos={int(y_val.sum())})")

    backend = "lmvg"
    meta_model = make_model_v2(backend=backend)
    meta_model.set_hard(False)
    inner_lr, inner_steps, mmd_w = 5e-3, 2, 0.1
    best_val_auc, best_state, stall = -1, None, 0
    prev_hard_adj = None
    nvg_ref = NVG_BASELINE_AUC.get(tgt_task, 0.90)
    hard_switched = False

    for ep in range(1, episodes + 1):
        t_g = cosine_temperature(ep, episodes, t_start=1.0, t_end=0.1)
        meta_model.set_gumbel_temp(t_g)
        if not hard_switched and ep >= HARD_SWITCH_EP:
            meta_model.set_hard(True)
            hard_switched = True

        src_batches = []
        for t in src_tasks:
            Xt, yt, dt = src_pool[t]
            pc = int(yt.sum())
            if pc < 4:
                task = sample_task_temporal(Xt, yt, dt, rng, 1, 8, max(1, pc - 1), 8)
            elif pc < 10:
                task = sample_task_temporal(Xt, yt, dt, rng, 2, 8, 2, 8)
            else:
                task = sample_task_temporal(Xt, yt, dt, rng, 4, 16, 4, 16)
            src_batches.append(task)

        t_idx = rng.choice(len(X_tr), min(32, len(X_tr)), replace=False)
        tgt_batch = (X_tr[t_idx], y_tr[t_idx])
        grads, q_loss, mmd_v = meta_train_step(meta_model, backend, src_batches, tgt_batch, inner_lr, inner_steps, mmd_w)
        if grads is None:
            continue
        apply_meta_grads_grouped(meta_model, grads, meta_lr_base, meta_lr_lmvg)

        with torch.no_grad():
            probe_x = torch.from_numpy(X_tr[:min(8, len(X_tr))])
            _ = meta_model(probe_x)
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
        flag_overfit = gap > 0.20
        flag_drift = ham > 0.5

        if ep % 20 == 1 or ep <= 3:
            print(f"  ep{ep:03d} qL={q_loss:.4f} valAUC={r_val['auc']:.3f} teAUC={r_te['auc']:.3f} teAP={r_te['ap']:.4f}")

        if flag_overfit and ep >= 20:
            print(f"  [HALT overfit] ep{ep}")
            break
        if flag_drift:
            print(f"  [HALT drift] ep{ep}")
            break

        if r_val["auc"] > best_val_auc:
            best_val_auc = r_val["auc"]
            best_state = copy.deepcopy(meta_model.state_dict())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"  [HALT] val stall at ep{ep}")
                break

    if best_state is None:
        return None, split_stats

    meta_model.load_state_dict(best_state)
    final_model = clone_wrapper(meta_model, backend)
    final_model = fine_tune(final_model, X_sup, y_sup, steps=50, lr=1e-3)
    r_final = eval_on_target(final_model, X_te, y_te)
    return r_final, split_stats


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


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "windows.npz")
    d = np.load(data_path, allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    print(f"Dataset: N={len(y)} pos={int(y.sum())}")
    print(f"Targets: {ABL_TARGETS}")
    print(f"Seeds: {SEEDS_ALL}")
    print("=" * 70)

    all_results = []
    t0 = time.time()
    for tgt in ABL_TARGETS:
        for seed in SEEDS_ALL:
            print(f"\n===== [EVENT-SPLIT] target={tgt} seed={seed} [{(time.time()-t0)/60:.1f}min] =====")
            r_final, split_stats = run_one_eventsplit(X, y, tasks, players, dates, tgt, seed)
            if r_final is not None:
                print(f"  ==> AUC={r_final['auc']:.4f} AP={r_final['ap']:.4f} F1={r_final['f1']:.4f}")
                all_results.append(dict(status="ok", target=tgt, seed=seed, paml=r_final, split=split_stats))
            else:
                print(f"  ==> FAILED/SKIPPED {split_stats}")
                all_results.append(dict(status="fail", target=tgt, seed=seed, split=split_stats))
            out_path = os.path.join(OUT_DIR, "exp1_main_eventsplit_results.json")
            with open(out_path, "w") as f:
                json.dump(clean(all_results), f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("AGGREGATE (event-level split)")
    print("=" * 70)
    for tgt in ABL_TARGETS:
        runs = [r for r in all_results if r["target"] == tgt and r["status"] == "ok"]
        if not runs:
            print(f"  {tgt}: no valid runs")
            continue
        aucs = np.array([r["paml"]["auc"] for r in runs])
        aps = np.array([r["paml"]["ap"] for r in runs])
        f1s = np.array([r["paml"]["f1"] for r in runs])
        print(f"  {tgt}: n={len(runs)} AUC={aucs.mean():.4f}+-{aucs.std():.4f} "
              f"AP={aps.mean():.4f}+-{aps.std():.4f} F1={f1s.mean():.4f}+-{f1s.std():.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
