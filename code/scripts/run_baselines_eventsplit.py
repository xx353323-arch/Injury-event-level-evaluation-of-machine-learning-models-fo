import os, sys, json, time
import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from baselines_comparison import (
    ALL_TASKS, ABL_TARGETS, normalize_fit, apply_norm, eval_metrics,
    run_xgboost, run_lstm, run_gat,
)
from new_baselines import train_transformer, train_protonet
from run_dann_10seed import train_dann, DANNModel
from event_split import event_level_split
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")
SEEDS_ALL = [42, 123, 2026, 0, 1, 2, 3, 4, 5, 6]
METHODS = ["XGBoost", "LSTM", "GAT", "Transformer", "ProtoNet", "DANN"]


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


def dann_clean(Xs, ys, Xp, yp, Xt, yt, s, X_unlab):
    if len(np.unique(yt)) < 2:
        return None
    torch.manual_seed(s); np.random.seed(s)
    model = DANNModel(input_dim=Xs.shape[-1])
    model = train_dann(model, Xs, ys, X_unlab, Xp, yp, epochs=12, lr=1e-3, bs=64)
    model.eval()
    with torch.no_grad():
        logit, _ = model(torch.from_numpy(Xt.astype(np.float32)))
        prob = torch.sigmoid(logit).numpy()
    return eval_metrics(yt, prob)


def run_method(name, Xs, ys, Xp, yp, Xt, yt, seed):
    if name == "XGBoost":
        return run_xgboost(Xs, ys, Xp, yp, Xt, yt, seed)
    if name == "LSTM":
        return run_lstm(Xs, ys, Xp, yp, Xt, yt, seed)
    if name == "GAT":
        return run_gat(Xs, ys, Xp, yp, Xt, yt, seed)
    if name == "Transformer":
        return train_transformer(Xs, ys, Xp, yp, Xt, yt, seed)
    if name == "ProtoNet":
        return train_protonet(Xs, ys, Xp, yp, Xt, yt, seed)
    if name == "DANN":
        return dann_clean(Xs, ys, Xp, yp, Xt, yt, seed, Xp)
    raise ValueError(name)


def load_done():
    done = {}
    for fn in os.listdir(OUT_DIR):
        if fn.startswith("val_baselines_eventsplit") and fn.endswith(".json"):
            try:
                for r in json.load(open(os.path.join(OUT_DIR, fn))):
                    if r.get("status") == "ok":
                        done[(r["method"], r["target"], r["seed"])] = r
            except Exception:
                pass
    return done


def main():
    targets = [a for a in sys.argv[1:] if a in ABL_TARGETS] or ABL_TARGETS
    tag = "_".join(targets) if len(targets) < len(ABL_TARGETS) else "all"
    out_path = os.path.join(OUT_DIR, f"val_baselines_eventsplit_{tag}.json")
    d = np.load(os.path.join(SCRIPT_DIR, "..", "data", "windows.npz"), allow_pickle=True)
    X, y, tasks, players = d["X"], d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values.astype("datetime64[D]")
    done = load_done()
    print(f"targets={targets} resume: {len(done)} ok runs already on disk", flush=True)

    results = list(done.values()) if tag == "all" else [r for r in done.values() if r["target"] in targets]
    t0 = time.time()
    for tgt in targets:
        tgt_mask = tasks == tgt
        src_mask = np.isin(tasks, [t for t in ALL_TASKS if t != tgt])
        X_src, y_src = X[src_mask], y[src_mask]
        X_tgt, y_tgt = X[tgt_mask], y[tgt_mask]
        players_tgt, dates_tgt = players[tgt_mask], dates[tgt_mask]
        mu, sd = normalize_fit(X_src)
        X_src_n = apply_norm(X_src, mu, sd)
        for seed in SEEDS_ALL:
            todo = [m for m in METHODS if (m, tgt, seed) not in done]
            if not todo:
                print(f"  {tgt} seed{seed}: all done, skip", flush=True)
                continue
            rng = np.random.default_rng(seed)
            sup_p, te_p, sup_n, te_n, info = event_level_split(y_tgt, players_tgt, dates_tgt, rng)
            if len(sup_p) < 2 or len(te_p) < 2:
                print(f"  {tgt} seed{seed} skip {info}", flush=True)
                continue
            sup_idx = np.concatenate([sup_p, sup_n]); te_idx = np.concatenate([te_p, te_n])
            X_sup_n = apply_norm(X_tgt[sup_idx], mu, sd); y_sup = y_tgt[sup_idx]
            X_te_n = apply_norm(X_tgt[te_idx], mu, sd); y_te = y_tgt[te_idx]
            print(f"\n--- {tgt} seed{seed} sup={len(y_sup)}(p{int(y_sup.sum())}) te={len(y_te)}(p{int(y_te.sum())}) todo={todo} [{(time.time()-t0)/60:.1f}min] ---", flush=True)
            for name in todo:
                try:
                    r = run_method(name, X_src_n, y_src, X_sup_n, y_sup, X_te_n, y_te, seed)
                    if r:
                        print(f"  [{name:12s}] AUC={r['auc']:.4f} AP={r['ap']:.4f} F1={r['f1']:.4f}", flush=True)
                        rec = dict(method=name, target=tgt, seed=seed, status="ok", **r)
                    else:
                        rec = dict(method=name, target=tgt, seed=seed, status="fail")
                except Exception as e:
                    print(f"  [{name:12s}] ERROR {e}", flush=True)
                    rec = dict(method=name, target=tgt, seed=seed, status="error", error=str(e))
                results.append(rec)
                if rec["status"] == "ok":
                    done[(name, tgt, seed)] = rec
                with open(out_path, "w") as f:
                    json.dump(clean(results), f, indent=2, default=str)

    print("\n" + "=" * 70 + f"\nAGGREGATE baselines event-split targets={targets}\n" + "=" * 70, flush=True)
    for tgt in targets:
        print(f"\n  {tgt}:", flush=True)
        for name in METHODS:
            runs = [r for r in results if r["target"] == tgt and r["method"] == name and r.get("status") == "ok"]
            if not runs:
                print(f"    [{name:12s}] no valid runs", flush=True); continue
            auc = np.array([r["auc"] for r in runs]); ap = np.array([r["ap"] for r in runs]); f1 = np.array([r["f1"] for r in runs])
            print(f"    [{name:12s}] AUC={auc.mean():.4f}+-{auc.std():.4f} AP={ap.mean():.4f}+-{ap.std():.4f} F1={f1.mean():.4f}+-{f1.std():.4f} n={len(runs)}", flush=True)
    print(f"\nTotal time {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
