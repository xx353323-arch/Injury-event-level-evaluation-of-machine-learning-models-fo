import os, sys, json, time, copy
import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paml_full_lmvg_v2 import (
    make_model_v2, focal_loss, mmd_rbf, sample_task_temporal,
    normalize_fit, apply_norm, eval_on_target, fine_tune, clone_wrapper,
    meta_train_step, apply_meta_grads_grouped, ALL_TASKS, HARD_SWITCH_EP,
)
from lmvg_v2 import hamming_distance, cosine_temperature
from event_split import event_level_split, chronological_window_split
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")

PROTOCOLS = {
    "chrono_window": ("chronological", None),
    "event_emb0": ("event", 0),
    "event_emb7": ("event", 7),
    "event_emb21": ("event", 21),
}


def do_split(protocol, y_tgt, players_tgt, dates_tgt, rng):
    kind, emb = PROTOCOLS[protocol]
    if kind == "chronological":
        return chronological_window_split(y_tgt, dates_tgt, rng)
    return event_level_split(y_tgt, players_tgt, dates_tgt, rng, embargo_days=emb)


def run_one(X, y, tasks, players, dates, tgt, seed, protocol, episodes=120, patience=20):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    src_tasks = [t for t in ALL_TASKS if t != tgt]
    src_pool = {}
    for t in src_tasks:
        m = tasks == t; src_pool[t] = (X[m], y[m], dates[m])
    tgt_mask = tasks == tgt
    X_tgt, y_tgt = X[tgt_mask], y[tgt_mask]
    players_tgt, dates_tgt = players[tgt_mask], dates[tgt_mask].astype("datetime64[D]")

    sup_p, te_p, sup_n, te_n, info = do_split(protocol, y_tgt, players_tgt, dates_tgt, rng)
    if len(sup_p) < 2 or len(te_p) < 2 or len(sup_n) < 4 or len(te_n) < 4:
        return None, info
    sup_idx = np.concatenate([sup_p, sup_n]); te_idx = np.concatenate([te_p, te_n])
    X_sup, y_sup = X_tgt[sup_idx], y_tgt[sup_idx]
    X_te, y_te = X_tgt[te_idx], y_tgt[te_idx]

    mu, sd = normalize_fit(np.concatenate([src_pool[t][0] for t in src_tasks], 0))
    for t in src_tasks:
        Xt, yt, dt = src_pool[t]; src_pool[t] = (apply_norm(Xt, mu, sd), yt, dt)
    X_sup = apply_norm(X_sup, mu, sd); X_te = apply_norm(X_te, mu, sd)

    n_val_pos = max(len(sup_p) // 3, 1)
    lp = np.arange(len(sup_p)); rng.shuffle(lp)
    val_p = sup_p[lp[:n_val_pos]]; tr_p = sup_p[lp[n_val_pos:]]
    ln = np.arange(len(sup_n)); rng.shuffle(ln)
    nvn = max(len(sup_n) // 3, min(4, len(sup_n)))
    val_n = sup_n[ln[:nvn]]; tr_n = sup_n[ln[nvn:]]
    tr_local = np.concatenate([tr_p, tr_n]) if len(tr_p) else np.concatenate([sup_p, tr_n])
    val_local = np.concatenate([val_p, val_n])
    X_tr = apply_norm(X_tgt[tr_local], mu, sd); y_tr = y_tgt[tr_local]
    X_val = apply_norm(X_tgt[val_local], mu, sd); y_val = y_tgt[val_local]
    if y_tr.sum() < 1:
        X_tr, y_tr = X_sup, y_sup
    if y_val.sum() < 1:
        X_val, y_val = X_sup, y_sup

    meta = make_model_v2(backend="lmvg"); meta.set_hard(False)
    inner_lr, steps, mmd_w = 5e-3, 2, 0.1
    best_val, best_state, stall, prev_hard = -1, None, 0, None
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
        ti = rng.choice(len(X_tr), min(32, len(X_tr)), replace=False)
        grads, ql, mv = meta_train_step(meta, "lmvg", src_batches, (X_tr[ti], y_tr[ti]), inner_lr, steps, mmd_w)
        if grads is None:
            continue
        apply_meta_grads_grouped(meta, grads, 2e-3, 5e-1)
        with torch.no_grad():
            _ = meta(torch.from_numpy(X_tr[:min(8, len(X_tr))]))
            curr = meta._last_hard_adj.clone() if meta._last_hard_adj is not None else None
        ham = hamming_distance(prev_hard, curr); prev_hard = curr
        em = clone_wrapper(meta, "lmvg"); em = fine_tune(em, X_tr, y_tr, steps=15, lr=1e-3)
        rv = eval_on_target(em, X_val, y_val)
        if rv is None:
            continue
        if ham > 0.5:
            break
        if rv["auc"] > best_val:
            best_val = rv["auc"]; best_state = copy.deepcopy(meta.state_dict()); stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state is None:
        return None, info
    meta.load_state_dict(best_state)
    fm = clone_wrapper(meta, "lmvg"); fm = fine_tune(fm, X_sup, y_sup, steps=50, lr=1e-3)
    r = eval_on_target(fm, X_te, y_te)
    return r, info


def main():
    d = np.load(os.path.join(SCRIPT_DIR, "..", "data", "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    print("补救协议敏感性: LMVG-TCA-PAML across split protocols (3 seed)")
    print(f"protocols: {list(PROTOCOLS.keys())}")
    out = []
    t0 = time.time()
    for protocol in PROTOCOLS:
        for tgt in ["TeamA-2020", "TeamA-2021", "TeamB-2020"]:
            for seed in [42, 123, 2026]:
                print(f"\n=== {protocol} | {tgt} seed{seed} [{(time.time()-t0)/60:.1f}min] ===")
                try:
                    r, info = run_one(X, y, tasks, players, dates, tgt, seed, protocol)
                    print(f"  ==> {r if r else 'FAIL'}  split={info}")
                    out.append(dict(protocol=protocol, target=tgt, seed=seed,
                                    status="ok" if r else "fail", result=r, split=info))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    out.append(dict(protocol=protocol, target=tgt, seed=seed, status="error", error=str(e)))
                with open(os.path.join(OUT_DIR, "val_protocol_sensitivity.json"), "w") as f:
                    json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 70 + "\nAGGREGATE by protocol x target\n" + "=" * 70)
    for protocol in PROTOCOLS:
        print(f"\n[{protocol}]")
        for tgt in ["TeamA-2020", "TeamA-2021", "TeamB-2020"]:
            runs = [o for o in out if o["protocol"] == protocol and o["target"] == tgt and o["status"] == "ok" and o["result"]]
            if not runs:
                print(f"  {tgt}: no valid"); continue
            auc = np.array([o["result"]["auc"] for o in runs]); ap = np.array([o["result"]["ap"] for o in runs])
            print(f"  {tgt}: AUC={auc.mean():.3f}+-{auc.std():.3f} AP={ap.mean():.3f}+-{ap.std():.3f} n={len(runs)}")
    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
