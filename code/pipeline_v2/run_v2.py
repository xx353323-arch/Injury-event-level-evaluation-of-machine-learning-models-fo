import os, sys, json, time, copy, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, HERE)
from sklearn.metrics import roc_auc_score, average_precision_score
from events import load_injury_table, reconstruct_events
from splits import event_level_split_v2, athlete_grouped_folds, _blocks
import paml_full_lmvg_v2 as P
from graph_backend_v2 import make_model_v2
from lmvg_v2 import hamming_distance, cosine_temperature
import baselines_comparison as BC
import new_baselines as NB
import run_dann_10seed as RD

CAPTURE = {}
def _capturing_eval(y_true, y_prob):
    CAPTURE["prob"] = np.asarray(y_prob, dtype=float).copy()
    if y_true.sum() < 1 or len(np.unique(y_true)) < 2:
        return dict(auc=float("nan"), ap=float("nan"))
    return dict(auc=float(roc_auc_score(y_true, y_prob)), ap=float(average_precision_score(y_true, y_prob)))
BC.eval_metrics = _capturing_eval; NB.eval_metrics = _capturing_eval; RD.eval_metrics = _capturing_eval

OUT = os.path.join(ROOT, "output_v2")
FT_STEPS = 30


def load_cohort(name):
    if name == "soccermon":
        d = np.load(os.path.join(ROOT, "data", "windows.npz"), allow_pickle=True)
        dates = pd.to_datetime(d["dates"]).values.astype("datetime64[D]")
        ev = reconstruct_events(load_injury_table(os.path.join(ROOT, "..", "data", "soccermon", "injury", "injury.csv")), gap_days=7)
        return dict(X=d["X"], y=d["y"], players=d["players"].astype(str), dates=dates, tasks=d["tasks"].astype(str), events=ev, T=14, horizon=7, tasks_all=["TeamA-2020", "TeamA-2021", "TeamB-2020", "TeamB-2021"], targets=["TeamA-2020", "TeamA-2021", "TeamB-2020"])
    d = np.load(os.path.join(ROOT, "data", "windows_runners.npz"), allow_pickle=True)
    dates = (np.datetime64("2012-01-01") + d["dates"].astype("timedelta64[D]")).astype("datetime64[D]")
    players = d["athletes"].astype(str); y = d["y"]
    ev = {}
    for i in np.where(y == 1)[0]:
        ev.setdefault(players[i], []).append(dict(start=dates[i] + np.timedelta64(1, "D"), end=dates[i] + np.timedelta64(1, "D"), parts=(), n_reports=1))
    for p in ev:
        ev[p].sort(key=lambda e: e["start"])
    return dict(X=d["X"], y=y, players=players, dates=dates, tasks=d["tasks"].astype(str), events=ev, T=7, horizon=1, tasks_all=["G1", "G2", "G3"], targets=["G1", "G2", "G3"])


def build_task(C, protocol, target, split_seed, embargo=20):
    rng = np.random.default_rng(split_seed)
    y, players, dates, tasks = C["y"], C["players"], C["dates"], C["tasks"]
    if protocol.startswith("event_v2"):
        tgt = target; m = tasks == tgt; g = np.where(m)[0]
        assign = "chronological" if protocol.endswith("chrono") else "random"
        sp, tp, sn, tn, info = event_level_split_v2(y[m], players[m], dates[m], C["events"], rng, T=C["T"], horizon=C["horizon"], embargo=embargo, assign=assign)
        sup_idx, te_idx = g[np.concatenate([sp, sn])], g[np.concatenate([tp, tn])]
        sup_p_g = g[sp]
        src_mask = np.isin(tasks, [t for t in C["tasks_all"] if t != tgt])
    else:
        tgt, fold = target.split(":"); fold = int(fold); m = tasks == tgt; g = np.where(m)[0]
        folds = athlete_grouped_folds(players[m], y[m], n_folds=5, rng=np.random.default_rng(split_seed))
        held = set(folds[fold]); te_mask = np.array([players[i] in held for i in g])
        te_idx = g[te_mask]; sup_pool = g[~te_mask]
        sp = sup_pool[y[sup_pool] == 1]; sn_pool = sup_pool[y[sup_pool] == 0]
        byp = {}
        for i in sn_pool: byp.setdefault(players[i], []).append(i)
        blocks = []
        for p, idxs in byp.items():
            idxs = sorted(idxs, key=lambda i: dates[i]); blocks += _blocks(idxs, dates, 14)
        rng.shuffle(blocks); sn = []
        for b in blocks:
            if len(sn) >= 4 * max(len(sp), 1): break
            sn += b
        sn = np.array(sn, dtype=int)
        sup_idx, te_idx = np.concatenate([sp, sn]), te_idx; sup_p_g = sp
        src_mask = np.isin(tasks, [t for t in C["tasks_all"] if t != tgt]) & ~np.isin(players, list(held))
        info = dict(protocol="athlete_grouped5", fold=fold, held_out_athletes=sorted(held), n_sup_pos=int(len(sp)), n_sup_neg=int(len(sn)), n_te_pos=int(y[te_idx].sum()), n_te_neg=int((y[te_idx] == 0).sum()), n_te_events=int(y[te_idx].sum()) if C["horizon"] == 1 else None, frac_te_pos_same_athlete_as_sup=0.0)
    ev_of = lambda i: __import__("events").window_event_id(C["events"], players[i], dates[i], C["horizon"]) if y[i] == 1 else ""
    sup_events = sorted(set(ev_of(i) for i in sup_p_g if ev_of(i)))
    return dict(sup_idx=sup_idx, te_idx=te_idx, src_mask=src_mask, info=info, sup_events=sup_events, ev_of=ev_of)


def norm_fit_apply(C, task):
    Xs = C["X"][task["src_mask"]]
    mu, sd = P.normalize_fit(Xs)
    return mu, sd, P.apply_norm(Xs, mu, sd), C["y"][task["src_mask"]]


def athlete_rate_probs(C, task):
    sup, te = task["sup_idx"], task["te_idx"]
    df = pd.DataFrame(dict(p=C["players"][sup], y=C["y"][sup])); r = df.groupby("p")["y"].mean(); prior = C["y"][sup].mean()
    return np.array([r.get(p, prior) for p in C["players"][te]])


def train_ours(C, task, train_seed, episodes=120, patience=20, T=None):
    T = T or C["T"]; torch.manual_seed(train_seed); ep_rng = np.random.default_rng(train_seed + 1000)
    y, players, dates, X = C["y"], C["players"], C["dates"], C["X"]
    mu, sd, _, _ = norm_fit_apply(C, task)
    src_tasks = [t for t in C["tasks_all"] if t not in task["info"].get("protocol", "")]
    src_pool = {}
    for t in C["tasks_all"]:
        m = (C["tasks"] == t) & task["src_mask"]
        if m.sum() > 0: src_pool[t] = (P.apply_norm(X[m], mu, sd), y[m], dates[m])
    sup, te = task["sup_idx"], task["te_idx"]
    X_sup, y_sup = P.apply_norm(X[sup], mu, sd), y[sup]; X_te = P.apply_norm(X[te], mu, sd)
    sup_events = task["sup_events"]
    if len(sup_events) >= 2:
        val_ev = ep_rng.choice(sup_events)
        is_val_pos = np.array([task["ev_of"](i) == val_ev for i in sup])
        sup_neg_local = np.where(y[sup] == 0)[0]; ep_rng.shuffle(sup_neg_local)
        n_val_neg = max(4 * int(is_val_pos.sum()), 4)
        val_mask = is_val_pos.copy(); val_mask[sup_neg_local[:n_val_neg]] = True
        X_val, y_val = X_sup[val_mask], y_sup[val_mask]; X_tr, y_tr = X_sup[~val_mask], y_sup[~val_mask]
        early = True
    else:
        X_tr, y_tr, X_val, y_val, early = X_sup, y_sup, X_sup, y_sup, False
    def clone(model):
        c = make_model_v2(backend="lmvg", T=T); c.load_state_dict(copy.deepcopy(model.state_dict())); c.set_gumbel_temp(model._gumbel_temp); c.set_hard(model.lmvg.hard); return c
    P.clone_wrapper = lambda model, backend: clone(model)
    meta = make_model_v2(backend="lmvg", T=T); meta.set_hard(False)
    best_val, best_state, stall, prev_hard = -1, None, 0, None
    for ep in range(1, episodes + 1):
        meta.set_gumbel_temp(cosine_temperature(ep, episodes))
        if ep >= P.HARD_SWITCH_EP: meta.set_hard(True)
        batches = []
        for t, (Xs_, ys_, ds_) in src_pool.items():
            pc = int(ys_.sum()); k = (4, 16, 4, 16) if pc >= 10 else ((2, 8, 2, 8) if pc >= 4 else (1, 8, max(1, pc - 1), 8))
            batches.append(P.sample_task_temporal(Xs_, ys_, ds_, ep_rng, *k))
        ti = ep_rng.choice(len(X_tr), min(32, len(X_tr)), replace=False)
        grads, ql, mv = P.meta_train_step(meta, "lmvg", batches, (X_tr[ti], y_tr[ti]), 5e-3, 2, 0.1)
        if grads is None: continue
        P.apply_meta_grads_grouped(meta, grads, 2e-3, 5e-1)
        with torch.no_grad():
            _ = meta(torch.from_numpy(X_tr[:min(8, len(X_tr))])); curr = meta._last_hard_adj.clone() if meta._last_hard_adj is not None else None
        ham = hamming_distance(prev_hard, curr); prev_hard = curr
        if ham > 0.5: break
        if not early: continue
        em = P.fine_tune(clone(meta), X_tr, y_tr, steps=FT_STEPS, lr=1e-3); em.eval()
        with torch.no_grad(): pv = 1 / (1 + np.exp(-em(torch.from_numpy(X_val)).numpy()))
        if y_val.sum() < 1 or (y_val == 0).sum() < 1: continue
        va = roc_auc_score(y_val, pv)
        if va > best_val: best_val, best_state, stall = va, copy.deepcopy(meta.state_dict()), 0
        else:
            stall += 1
            if stall >= patience: break
    if best_state is not None: meta.load_state_dict(best_state)
    fm = P.fine_tune(clone(meta), X_sup, y_sup, steps=FT_STEPS, lr=1e-3); fm.eval()
    with torch.no_grad(): prob = 1 / (1 + np.exp(-fm(torch.from_numpy(X_te)).numpy()))
    return prob, dict(best_val_auc=float(best_val), early_stopping=early, episodes_run=ep)


def run_baseline(name, C, task, train_seed):
    mu, sd, Xs, ys = norm_fit_apply(C, task)
    sup, te = task["sup_idx"], task["te_idx"]
    Xp, yp = P.apply_norm(C["X"][sup], mu, sd), C["y"][sup]; Xt, yt = P.apply_norm(C["X"][te], mu, sd), C["y"][te]
    T = C["T"]
    for cls in (NB.TransformerEncoder, NB.ProtoNet):
        dd = list(cls.__init__.__defaults__); dd[1] = T; cls.__init__.__defaults__ = tuple(dd)
    CAPTURE.clear()
    if name == "XGBoost": BC.run_xgboost(Xs, ys, Xp, yp, Xt, yt, train_seed)
    elif name == "LSTM": BC.run_lstm(Xs, ys, Xp, yp, Xt, yt, train_seed)
    elif name == "GAT": BC.run_gat(Xs, ys, Xp, yp, Xt, yt, train_seed)
    elif name == "Transformer": NB.train_transformer(Xs, ys, Xp, yp, Xt, yt, train_seed)
    elif name == "ProtoNet": NB.train_protonet(Xs, ys, Xp, yp, Xt, yt, train_seed)
    elif name == "DANN":
        torch.manual_seed(train_seed); np.random.seed(train_seed)
        m = RD.DANNModel(input_dim=Xs.shape[-1]); m = RD.train_dann(m, Xs, ys, Xp, Xp, yp, epochs=12, lr=1e-3, bs=64); m.eval()
        with torch.no_grad(): lg, _ = m(torch.from_numpy(Xt.astype(np.float32)))
        CAPTURE["prob"] = torch.sigmoid(lg).numpy()
    else: raise ValueError(name)
    return CAPTURE["prob"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--targets", default=""); ap.add_argument("--methods", default="ours,LSTM,XGBoost,GAT,Transformer,ProtoNet,DANN,AthleteRate")
    ap.add_argument("--seeds", default="42,123,2026,0,1,2,3,4,5,6"); ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--embargo", type=int, default=20)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    C = load_cohort(a.cohort)
    targets = a.targets.split(",") if a.targets else (C["targets"] if a.protocol.startswith("event_v2") else [f"{t}:{k}" for t in C["targets"] for k in range(5)])
    methods = a.methods.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    dump_dir = os.path.join(OUT, "dumps", a.cohort, a.protocol); os.makedirs(dump_dir, exist_ok=True)
    summ_path = os.path.join(OUT, f"summary_{a.cohort}_{a.protocol}_{targets[0].replace(':','_')}.jsonl")
    t0 = time.time()
    for target in targets:
        for seed in seeds:
            task = build_task(C, a.protocol, target, seed, embargo=a.embargo)
            te = task["te_idx"]; y_te = C["y"][te]
            if y_te.sum() < 1:
                print(f"skip {target} seed{seed}: no test positives", flush=True); continue
            for method in methods:
                fn = os.path.join(dump_dir, f"{target.replace(':','_')}__{method}__s{seed}.npz")
                if os.path.exists(fn): continue
                print(f"[{a.cohort}/{a.protocol}] {target} seed{seed} {method} [{(time.time()-t0)/60:.1f}min]", flush=True)
                try:
                    extra = {}
                    if method == "ours": prob, extra = train_ours(C, task, seed, episodes=a.episodes)
                    elif method == "AthleteRate": prob = athlete_rate_probs(C, task)
                    else: prob = run_baseline(method, C, task, seed)
                    auc = float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) > 1 else float("nan"); apv = float(average_precision_score(y_te, prob))
                    np.savez_compressed(fn, te_idx=te, te_player=C["players"][te], te_date=C["dates"][te].astype(str), te_event_id=np.array([task["ev_of"](i) for i in te]), te_y=y_te, te_prob=prob,
                                        sup_idx=task["sup_idx"], sup_y=C["y"][task["sup_idx"]], sup_player=C["players"][task["sup_idx"]], sup_date=C["dates"][task["sup_idx"]].astype(str),
                                        info=json.dumps(dict(task["info"], **extra), default=str))
                    rec = dict(cohort=a.cohort, protocol=a.protocol, target=target, seed=seed, method=method, auc=auc, ap=apv, n_te_pos=int(y_te.sum()), n_te=int(len(te)), split_id=task["info"].get("split_id", ""), **{k: v for k, v in extra.items()})
                    print(f"    -> AUC={auc:.4f} AP={apv:.4f}", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    rec = dict(cohort=a.cohort, protocol=a.protocol, target=target, seed=seed, method=method, error=str(e)[:200])
                with open(summ_path, "a") as f: f.write(json.dumps(rec, default=str) + "\n")
    print(f"DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
