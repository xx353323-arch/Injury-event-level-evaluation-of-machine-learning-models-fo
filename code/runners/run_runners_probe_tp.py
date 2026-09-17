import os, sys, json, time
import numpy as np, torch
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(CODE, "src")); sys.path.insert(0, os.path.join(CODE, "scripts"))
import new_baselines as NB
from baselines_comparison import normalize_fit, apply_norm

T = 7
for cls in (NB.TransformerEncoder, NB.ProtoNet):
    d = list(cls.__init__.__defaults__); d[1] = T; cls.__init__.__defaults__ = tuple(d)

GROUPS = ["G1", "G2", "G3"]; SEEDS = [42, 123, 2026]


def split_target(y_t, rng):
    pos = np.where(y_t == 1)[0]; neg = np.where(y_t == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_sp = max(len(pos) // 3, 2)
    if os.environ.get('RUNNERS_KSUP'):
        n_sp = min(int(os.environ['RUNNERS_KSUP']), n_sp); sup_p, te_p = pos[:n_sp], pos[n_sp:]
    n_sn = min(4 * n_sp, len(neg) // 3); sup_n, te_n = neg[:n_sn], neg[n_sn:]
    return sup_p, te_p, sup_n, te_n


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sanity":
        x = torch.zeros(2, T, 10)
        print("Transformer out", NB.TransformerEncoder()(x).shape, " ProtoNet out", NB.ProtoNet()(x).shape); return
    d = np.load(os.path.join(HERE, "windows_runners.npz"), allow_pickle=True)
    X, y, tasks = d["X"], d["y"], d["tasks"]
    out = []; t0 = time.time()
    for tgt in GROUPS:
        src_mask = np.isin(tasks, [g for g in GROUPS if g != tgt])
        X_src, y_src = X[src_mask], y[src_mask]; X_tgt, y_tgt = X[tasks == tgt], y[tasks == tgt]
        mu, sd = normalize_fit(X_src); Xs = apply_norm(X_src, mu, sd)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            sup_p, te_p, sup_n, te_n = split_target(y_tgt, rng)
            si = np.concatenate([sup_p, sup_n]); ti = np.concatenate([te_p, te_n])
            Xp = apply_norm(X_tgt[si], mu, sd); Xt = apply_norm(X_tgt[ti], mu, sd); yp, yt = y_tgt[si], y_tgt[ti]
            rec = dict(target=tgt, seed=seed)
            for name, fn in [("Transformer", NB.train_transformer), ("ProtoNet", NB.train_protonet)]:
                try:
                    r = fn(Xs, y_src, Xp, yp, Xt, yt, seed); rec[name] = r
                    print(f"{tgt} seed{seed} [{name:12s}] AUC={r['auc']:.4f} AP={r['ap']:.4f} [{(time.time()-t0)/60:.1f}min]", flush=True)
                except Exception as e:
                    rec[name] = dict(error=str(e)[:200]); print(f"{tgt} seed{seed} [{name}] ERROR {e}", flush=True)
            out.append(rec); json.dump(out, open(os.path.join(HERE, "val_runners_probe_tp_k%s.json" % os.environ["RUNNERS_KSUP"] if os.environ.get("RUNNERS_KSUP") else "val_runners_probe_tp.json"), "w"), indent=2, default=str)
    print(f"Total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
