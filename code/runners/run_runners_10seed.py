import os, sys, json, time
import numpy as np, torch
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(CODE, "src")); sys.path.insert(0, os.path.join(CODE, "scripts")); sys.path.insert(0, HERE)
import run_runners_probe as RP
import new_baselines as NB
from baselines_comparison import run_xgboost, run_lstm, run_gat, normalize_fit, apply_norm, eval_metrics
from run_dann_10seed import DANNModel, train_dann

T = 7
for cls in (NB.TransformerEncoder, NB.ProtoNet):
    dd = list(cls.__init__.__defaults__); dd[1] = T; cls.__init__.__defaults__ = tuple(dd)

GROUPS = ["G1", "G2", "G3"]
SEEDS = [42, 123, 2026, 0, 1, 2, 3, 4, 5, 6]
REGIMES = {"full": None, "k10": "10"}
BASELINES = ["XGBoost", "LSTM", "GAT", "Transformer", "ProtoNet", "DANN"]


def load_done():
    done = {}
    def put(regime, target, seed, method, r):
        if isinstance(r, dict) and "auc" in r:
            done[(regime, target, seed, method)] = r
    for fn, regime in [("val_runners_probe.json", "full"), ("val_runners_probe_k10.json", "k10")]:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            for rec in json.load(open(p)):
                put(regime, rec["target"], rec["seed"], "ours", rec.get("ours"))
                for m, r in (rec.get("baselines") or {}).items():
                    put(regime, rec["target"], rec["seed"], m, r)
    for fn, regime in [("val_runners_probe_tp.json", "full"), ("val_runners_probe_tp_k10.json", "k10")]:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            for rec in json.load(open(p)):
                for m in ("Transformer", "ProtoNet"):
                    put(regime, rec["target"], rec["seed"], m, rec.get(m))
    for fn in os.listdir(HERE):
        if fn.startswith("val_runners_10seed_") and fn.endswith(".json"):
            for rec in json.load(open(os.path.join(HERE, fn))):
                put(rec["regime"], rec["target"], rec["seed"], rec["method"], rec.get("result"))
    return done


def run_baseline(name, X, y, tasks, tgt, seed):
    rng = np.random.default_rng(seed)
    src_mask = np.isin(tasks, [g for g in GROUPS if g != tgt])
    X_src, y_src = X[src_mask], y[src_mask]; X_tgt, y_tgt = X[tasks == tgt], y[tasks == tgt]
    sup_p, te_p, sup_n, te_n = RP.split_target(y_tgt, rng)
    si = np.concatenate([sup_p, sup_n]); ti = np.concatenate([te_p, te_n])
    mu, sd = normalize_fit(X_src)
    Xs = apply_norm(X_src, mu, sd); Xp = apply_norm(X_tgt[si], mu, sd); Xt = apply_norm(X_tgt[ti], mu, sd)
    yp, yt = y_tgt[si], y_tgt[ti]
    if name == "XGBoost": return run_xgboost(Xs, y_src, Xp, yp, Xt, yt, seed)
    if name == "LSTM": return run_lstm(Xs, y_src, Xp, yp, Xt, yt, seed)
    if name == "GAT": return run_gat(Xs, y_src, Xp, yp, Xt, yt, seed)
    if name == "Transformer": return NB.train_transformer(Xs, y_src, Xp, yp, Xt, yt, seed)
    if name == "ProtoNet": return NB.train_protonet(Xs, y_src, Xp, yp, Xt, yt, seed)
    if name == "DANN":
        torch.manual_seed(seed); np.random.seed(seed)
        m = DANNModel(input_dim=Xs.shape[-1]); m = train_dann(m, Xs, y_src, Xp, Xp, yp, epochs=12, lr=1e-3, bs=64); m.eval()
        with torch.no_grad():
            lg, _ = m(torch.from_numpy(Xt.astype(np.float32)))
        return eval_metrics(yt, torch.sigmoid(lg).numpy())
    raise ValueError(name)


def main():
    targets = [a for a in sys.argv[1:] if a in GROUPS] or GROUPS
    tag = "_".join(targets)
    out_path = os.path.join(HERE, f"val_runners_10seed_{tag}.json")
    d = np.load(os.path.join(HERE, "windows_runners.npz"), allow_pickle=True)
    X, y, tasks = d["X"], d["y"], d["tasks"]
    dates = (np.datetime64("2012-01-01") + d["dates"].astype("timedelta64[D]")).astype("datetime64[D]")
    done = load_done()
    print(f"targets={targets} resume: {len(done)} runs on disk", flush=True)
    out = []; t0 = time.time()
    for regime, ksup in REGIMES.items():
        if ksup: os.environ["RUNNERS_KSUP"] = ksup
        else: os.environ.pop("RUNNERS_KSUP", None)
        for tgt in targets:
            for seed in SEEDS:
                for method in ["ours"] + BASELINES:
                    key = (regime, tgt, seed, method)
                    if key in done:
                        continue
                    print(f"[{regime}] {tgt} seed{seed} {method} [{(time.time()-t0)/60:.1f}min]", flush=True)
                    try:
                        r = RP.run_ours(X, y, tasks, dates, tgt, seed, episodes=120) if method == "ours" else run_baseline(method, X, y, tasks, tgt, seed)
                        print(f"    -> AUC={r['auc']:.4f} AP={r['ap']:.4f}", flush=True)
                        rec = dict(regime=regime, target=tgt, seed=seed, method=method, result=r)
                    except Exception as e:
                        print(f"    -> ERROR {str(e)[:150]}", flush=True)
                        rec = dict(regime=regime, target=tgt, seed=seed, method=method, result=dict(error=str(e)[:200]))
                    out.append(rec); done[key] = rec["result"]
                    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print(f"DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
