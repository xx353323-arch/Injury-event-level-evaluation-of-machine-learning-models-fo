import os, sys, json, time, copy
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paml_full_lmvg_v2 import (
    make_model_v2, sample_task_temporal, normalize_fit, apply_norm, eval_on_target,
    fine_tune, clone_wrapper, meta_train_step, apply_meta_grads_grouped, ALL_TASKS, HARD_SWITCH_EP,
)
from lmvg_v2 import hamming_distance, cosine_temperature
from event_split import event_level_split

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
SEEDS = [42, 123, 2026]
THR = 0.5


def train_and_predict(X, y, tasks, players, dates, tgt, seed, episodes=120, patience=20):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    src_tasks = [t for t in ALL_TASKS if t != tgt]
    src_pool = {t: (X[tasks == t], y[tasks == t], dates[tasks == t]) for t in src_tasks}
    m = tasks == tgt
    X_tgt, y_tgt, p_tgt, d_tgt = X[m], y[m], players[m], dates[m].astype("datetime64[D]")
    sup_p, te_p, sup_n, te_n, info = event_level_split(y_tgt, p_tgt, d_tgt, rng)
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
    meta = make_model_v2(backend="lmvg"); meta.set_hard(False)
    best_val, best_state, stall, prev_hard = -1, None, 0, None
    for ep in range(1, episodes + 1):
        meta.set_gumbel_temp(cosine_temperature(ep, episodes))
        if ep >= HARD_SWITCH_EP: meta.set_hard(True)
        batches = []
        for t in src_tasks:
            Xs, ys, ds = src_pool[t]; pc = int(ys.sum())
            k = (4, 16, 4, 16) if pc >= 10 else ((2, 8, 2, 8) if pc >= 4 else (1, 8, max(1, pc - 1), 8))
            batches.append(sample_task_temporal(Xs, ys, ds, rng, *k))
        ti = rng.choice(len(X_tr), min(32, len(X_tr)), replace=False)
        grads, ql, mv = meta_train_step(meta, "lmvg", batches, (X_tr[ti], y_tr[ti]), 5e-3, 2, 0.1)
        if grads is None: continue
        apply_meta_grads_grouped(meta, grads, 2e-3, 5e-1)
        with torch.no_grad():
            _ = meta(torch.from_numpy(X_tr[:min(8, len(X_tr))]))
            curr = meta._last_hard_adj.clone() if meta._last_hard_adj is not None else None
        ham = hamming_distance(prev_hard, curr); prev_hard = curr
        em = fine_tune(clone_wrapper(meta, "lmvg"), X_tr, y_tr, steps=15, lr=1e-3)
        rv = eval_on_target(em, X_val, y_val)
        if rv is None: continue
        if ham > 0.5: break
        if rv["auc"] > best_val: best_val, best_state, stall = rv["auc"], copy.deepcopy(meta.state_dict()), 0
        else:
            stall += 1
            if stall >= patience: break
    meta.load_state_dict(best_state)
    fm = fine_tune(clone_wrapper(meta, "lmvg"), X_sup, y_sup, steps=50, lr=1e-3); fm.eval()
    with torch.no_grad():
        prob = 1 / (1 + np.exp(-fm(torch.from_numpy(X_te)).numpy()))
    return dict(prob=prob, y=y_te, player=p_tgt[te_idx], date=d_tgt[te_idx], metrics=eval_on_target(fm, X_te, y_te), split=info)


def weekly_metrics(prob, y, player, date, thr=THR):
    df = pd.DataFrame(dict(prob=prob, y=y, player=player, date=pd.to_datetime(date)))
    iso = df["date"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    wk = df.groupby(["player", "week"]).agg(score=("prob", "max"), inj=("y", "max"), n_days=("y", "size")).reset_index()
    flagged = wk["score"] >= thr
    healthy = wk["inj"] == 0; injured = wk["inj"] == 1
    fpr_w = float((flagged & healthy).sum() / max(healthy.sum(), 1))
    sens_w = float((flagged & injured).sum() / max(injured.sum(), 1))
    ppv_w = float((flagged & injured).sum() / max(flagged.sum(), 1))
    fpr_d = float(((df["prob"] >= thr) & (df["y"] == 0)).sum() / max((df["y"] == 0).sum(), 1))
    sens_d = float(((df["prob"] >= thr) & (df["y"] == 1)).sum() / max((df["y"] == 1).sum(), 1))
    return dict(n_athlete_weeks=int(len(wk)), n_healthy_weeks=int(healthy.sum()), n_injury_weeks=int(injured.sum()),
                weekly_fpr=fpr_w, weekly_sensitivity=sens_w, weekly_ppv=ppv_w,
                daily_fpr=fpr_d, daily_sensitivity=sens_d,
                false_alerts_per_25_athletes_per_week=25 * fpr_w, naive_daily_extrapolation=25 * fpr_d)


def main():
    d = np.load(os.path.join(OUT_DIR, "..", "data", "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    out = []; t0 = time.time()
    for tgt in ["TeamA-2020", "TeamA-2021", "TeamB-2020"]:
        for seed in SEEDS:
            print(f"=== {tgt} seed{seed} [{(time.time()-t0)/60:.1f}min]", flush=True)
            r = train_and_predict(X, y, tasks, players, dates, tgt, seed)
            wm = weekly_metrics(r["prob"], r["y"], r["player"], r["date"])
            print(f"    window AUC={r['metrics']['auc']:.3f}  daily FPR={wm['daily_fpr']:.3f}  weekly FPR={wm['weekly_fpr']:.3f}  weekly sens={wm['weekly_sensitivity']:.3f}  weekly PPV={wm['weekly_ppv']:.3f}", flush=True)
            out.append(dict(target=tgt, seed=seed, window_metrics=r["metrics"], weekly=wm, split=r["split"]))
            json.dump(out, open(os.path.join(OUT_DIR, "val_weekly_fpr.json"), "w"), indent=2, default=str)
    print(f"DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
