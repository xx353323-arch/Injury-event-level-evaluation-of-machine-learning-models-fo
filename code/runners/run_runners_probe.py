import os, sys, json, time, copy
import numpy as np
import torch

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(CODE, "src"))
sys.path.insert(0, os.path.join(CODE, "scripts"))
import paml_full_lmvg_v2 as P
from paml_full_lmvg_v2 import (
    focal_loss, sample_task_temporal, normalize_fit, apply_norm, eval_on_target,
    fine_tune, meta_train_step, apply_meta_grads_grouped, HARD_SWITCH_EP,
)
from graph_backend_v2 import make_model_v2
from lmvg_v2 import hamming_distance, cosine_temperature
from baselines_comparison import run_xgboost, run_lstm
from new_baselines import train_transformer, train_protonet

T = 7
GROUPS = ["G1", "G2", "G3"]
SEEDS = [42, 123, 2026]


def clone_T(model, backend):
    c = make_model_v2(backend=backend, T=T)
    c.load_state_dict(copy.deepcopy(model.state_dict()))
    c.set_gumbel_temp(model._gumbel_temp)
    if backend == "lmvg":
        c.set_hard(model.lmvg.hard)
    return c


P.clone_wrapper = clone_T


def split_target(y_t, rng):
    pos = np.where(y_t == 1)[0]; neg = np.where(y_t == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_sp = max(len(pos) // 3, 2)
    if os.environ.get('RUNNERS_KSUP'):
        n_sp = min(int(os.environ['RUNNERS_KSUP']), n_sp)
    sup_p, te_p = pos[:n_sp], pos[n_sp:]
    n_sn = min(4 * n_sp, len(neg) // 3)
    sup_n, te_n = neg[:n_sn], neg[n_sn:]
    return sup_p, te_p, sup_n, te_n


def run_ours(X, y, tasks, dates, tgt, seed, episodes=120, patience=20):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    src_tasks = [g for g in GROUPS if g != tgt]
    src_pool = {g: (X[tasks == g], y[tasks == g], dates[tasks == g]) for g in src_tasks}
    X_tgt, y_tgt = X[tasks == tgt], y[tasks == tgt]
    sup_p, te_p, sup_n, te_n = split_target(y_tgt, rng)
    sup_idx = np.concatenate([sup_p, sup_n]); te_idx = np.concatenate([te_p, te_n])
    mu, sd = normalize_fit(np.concatenate([src_pool[g][0] for g in src_tasks], 0))
    for g in src_tasks:
        Xs, ys, ds = src_pool[g]; src_pool[g] = (apply_norm(Xs, mu, sd), ys, ds)
    X_sup = apply_norm(X_tgt[sup_idx], mu, sd); y_sup = y_tgt[sup_idx]
    X_te = apply_norm(X_tgt[te_idx], mu, sd); y_te = y_tgt[te_idx]
    nvp = max(len(sup_p) // 3, 1)
    lp = np.arange(len(sup_p)); rng.shuffle(lp)
    val_p = sup_p[lp[:nvp]]; tr_p = sup_p[lp[nvp:]]
    ln = np.arange(len(sup_n)); rng.shuffle(ln)
    nvn = max(len(sup_n) // 3, 4)
    val_n = sup_n[ln[:nvn]]; tr_n = sup_n[ln[nvn:]]
    X_tr = apply_norm(X_tgt[np.concatenate([tr_p, tr_n])], mu, sd); y_tr = y_tgt[np.concatenate([tr_p, tr_n])]
    X_val = apply_norm(X_tgt[np.concatenate([val_p, val_n])], mu, sd); y_val = y_tgt[np.concatenate([val_p, val_n])]

    meta = make_model_v2(backend="lmvg", T=T); meta.set_hard(False)
    best_val, best_state, stall, prev_hard = -1, None, 0, None
    for ep in range(1, episodes + 1):
        meta.set_gumbel_temp(cosine_temperature(ep, episodes))
        if ep >= HARD_SWITCH_EP:
            meta.set_hard(True)
        batches = []
        for g in src_tasks:
            Xs, ys, ds = src_pool[g]; pc = int(ys.sum())
            k = (4, 16, 4, 16) if pc >= 10 else ((2, 8, 2, 8) if pc >= 4 else (1, 8, max(1, pc - 1), 8))
            batches.append(sample_task_temporal(Xs, ys, ds, rng, *k))
        ti = rng.choice(len(X_tr), min(32, len(X_tr)), replace=False)
        grads, ql, mv = meta_train_step(meta, "lmvg", batches, (X_tr[ti], y_tr[ti]), 5e-3, 2, 0.1)
        if grads is None:
            continue
        apply_meta_grads_grouped(meta, grads, 2e-3, 5e-1)
        with torch.no_grad():
            _ = meta(torch.from_numpy(X_tr[:8]))
            curr = meta._last_hard_adj.clone() if meta._last_hard_adj is not None else None
        ham = hamming_distance(prev_hard, curr); prev_hard = curr
        em = fine_tune(clone_T(meta, "lmvg"), X_tr, y_tr, steps=15, lr=1e-3)
        rv = eval_on_target(em, X_val, y_val)
        if rv is None or ham > 0.5:
            if ham > 0.5: break
            continue
        if ep % 20 == 0 or ep <= 2:
            rt = eval_on_target(em, X_te, y_te)
            print(f"    ep{ep:03d} qL={ql:.4f} valAUC={rv['auc']:.3f} teAUC={rt['auc']:.3f} teAP={rt['ap']:.4f}", flush=True)
        if rv["auc"] > best_val:
            best_val = rv["auc"]; best_state = copy.deepcopy(meta.state_dict()); stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state is None:
        return None
    meta.load_state_dict(best_state)
    fm = fine_tune(clone_T(meta, "lmvg"), X_sup, y_sup, steps=50, lr=1e-3)
    return eval_on_target(fm, X_te, y_te)


def run_baselines(X, y, tasks, tgt, seed):
    rng = np.random.default_rng(seed)
    src_mask = np.isin(tasks, [g for g in GROUPS if g != tgt])
    X_src, y_src = X[src_mask], y[src_mask]
    X_tgt, y_tgt = X[tasks == tgt], y[tasks == tgt]
    sup_p, te_p, sup_n, te_n = split_target(y_tgt, rng)
    sup_idx = np.concatenate([sup_p, sup_n]); te_idx = np.concatenate([te_p, te_n])
    mu, sd = normalize_fit(X_src)
    Xs = apply_norm(X_src, mu, sd); Xp = apply_norm(X_tgt[sup_idx], mu, sd); Xt = apply_norm(X_tgt[te_idx], mu, sd)
    yp, yt = y_tgt[sup_idx], y_tgt[te_idx]
    out = {}
    for name, fn in [("XGBoost", run_xgboost), ("LSTM", run_lstm), ("Transformer", train_transformer), ("ProtoNet", train_protonet)]:
        try:
            r = fn(Xs, y_src, Xp, yp, Xt, yt, seed)
            out[name] = r
            print(f"    [{name:12s}] AUC={r['auc']:.4f} AP={r['ap']:.4f}", flush=True)
        except Exception as e:
            out[name] = dict(error=str(e)[:200])
            print(f"    [{name:12s}] ERROR {str(e)[:120]}", flush=True)
    return out


def main():
    d = np.load(os.path.join(HERE, "windows_runners.npz"), allow_pickle=True)
    X, y, tasks = d["X"], d["y"], d["tasks"]
    dates = (np.datetime64("2012-01-01") + d["dates"].astype("timedelta64[D]")).astype("datetime64[D]")
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    targets = sys.argv[2].split(",") if len(sys.argv) > 2 else GROUPS
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else SEEDS
    print(f"runners probe: N={len(y)} pos={int(y.sum())} T={T} episodes={episodes} targets={targets} seeds={seeds}", flush=True)
    out_path = os.path.join(HERE, "val_runners_probe_k%s.json" % os.environ["RUNNERS_KSUP"] if os.environ.get("RUNNERS_KSUP") else "val_runners_probe.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) and len(sys.argv) <= 1 else []
    t0 = time.time()
    for tgt in targets:
        for seed in seeds:
            print(f"\n=== target={tgt} seed={seed} [{(time.time()-t0)/60:.1f}min] ===", flush=True)
            rec = dict(target=tgt, seed=seed)
            try:
                rec["ours"] = run_ours(X, y, tasks, dates, tgt, seed, episodes=episodes)
                print(f"    [ours        ] {rec['ours']}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc(); rec["ours"] = dict(error=str(e)[:200])
            rec["baselines"] = run_baselines(X, y, tasks, tgt, seed)
            out.append(rec)
            json.dump(out, open(out_path, "w"), indent=2, default=str)
    print(f"\nTotal {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
