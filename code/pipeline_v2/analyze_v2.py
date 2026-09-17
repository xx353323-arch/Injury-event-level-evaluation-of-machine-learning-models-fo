import os, sys, json, glob, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, brier_score_loss
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "..", "output_v2", "dumps")
METHODS = ["ours", "LSTM", "Transformer", "GAT", "ProtoNet", "DANN", "XGBoost", "AthleteRate"]


def load(cohort, protocol):
    recs = []
    for f in glob.glob(os.path.join(DUMPS, cohort, protocol, "*.npz")):
        base = os.path.basename(f)[:-4]
        target, method, seed = base.split("__"); seed = int(seed[1:])
        d = np.load(f, allow_pickle=True)
        recs.append(dict(target=target, method=method, seed=seed, y=d["te_y"], prob=d["te_prob"],
                         player=d["te_player"], event=d["te_event_id"], date=d["te_date"],
                         sup_y=d["sup_y"], sup_player=d["sup_player"], info=json.loads(str(d["info"]))))
    return recs


def calibrated_threshold(sup_y, prob_te):
    q = float(np.mean(sup_y))
    if q <= 0 or q >= 1:
        return 0.5
    return float(np.quantile(prob_te, 1 - q))


def metrics(r):
    y, p = r["y"], r["prob"]
    out = dict(n_te=len(y), n_pos=int(y.sum()), n_events=len(set(e for e in r["event"] if e)))
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p)); out["ap"] = float(average_precision_score(y, p))
        out["prevalence"] = float(y.mean()); out["lift"] = out["ap"] / max(out["prevalence"], 1e-9)
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out["brier"] = float(brier_score_loss(y, pc))
        out["brier_skill"] = 1 - out["brier"] / max(brier_score_loss(y, np.full_like(pc, y.mean())), 1e-12)
    wa, n_wa = within_athlete_auc(r); ba, n_ba = between_athlete_auc(r)
    out["auc_within_athlete"] = wa; out["n_athletes_with_both"] = n_wa
    out["auc_between_athlete"] = ba; out["n_athletes_test"] = n_ba
    thr_cal = calibrated_threshold(r["sup_y"], p)
    yh = (p >= thr_cal).astype(int)
    out["thr_calibrated"] = thr_cal
    out["f1_calibrated"] = float(f1_score(y, yh, zero_division=0))
    out["precision_calibrated"] = float(precision_score(y, yh, zero_division=0))
    out["recall_calibrated"] = float(recall_score(y, yh, zero_division=0))
    grid = np.linspace(0.05, 0.95, 91)
    out["f1_oracle"] = float(max(f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid))
    if out["n_events"] > 0:
        det = []
        for e in set(x for x in r["event"] if x):
            m = r["event"] == e
            det.append(int((p[m] >= thr_cal).any()))
        k = int(sum(det)); n = len(det)
        out["event_detection_rate"] = k / n
        lo, hi = proportion_ci(k, n)
        out["event_detection_ci"] = (lo, hi)
    return out


def within_athlete_auc(r):
    """按运动员分层的 AUC：只在同一运动员内部比较正负窗口，剔除个体间差异。
    回答「能否预测该运动员何时受伤」，区别于整体 AUC 可能来自「区分哪个运动员会受伤」。"""
    y, p, players = r["y"], r["prob"], r["player"]
    aucs, weights, n_used = [], [], 0
    for a in np.unique(players):
        m = players == a
        if y[m].sum() > 0 and (y[m] == 0).sum() > 0:
            aucs.append(roc_auc_score(y[m], p[m])); weights.append(int(y[m].sum())); n_used += 1
    if not aucs:
        return float("nan"), 0
    return float(np.average(aucs, weights=weights)), n_used


def between_athlete_auc(r):
    """个体间 AUC：把每名运动员聚合为一个点（平均预测分 vs 是否有伤病），
    衡量模型多大程度只是在区分「哪个运动员会受伤」。"""
    y, p, players = r["y"], r["prob"], r["player"]
    uniq = np.unique(players)
    ay = np.array([1 if y[players == a].sum() > 0 else 0 for a in uniq])
    ap_ = np.array([p[players == a].mean() for a in uniq])
    if len(np.unique(ay)) < 2:
        return float("nan"), len(uniq)
    return float(roc_auc_score(ay, ap_)), len(uniq)


def proportion_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_bootstrap_auc(r, B=2000, unit="player", seed=0):
    rng = np.random.default_rng(seed)
    units = r[unit]; uniq = np.array(sorted(set(units.tolist())))
    idx_by = {u: np.where(units == u)[0] for u in uniq}
    vals = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        y, p = r["y"][idx], r["prob"][idx]
        if len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, p))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def tost(d, margin=0.02):
    n = len(d)
    if n < 3:
        return float("nan")
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    t1 = (d.mean() + margin) / se; t2 = (d.mean() - margin) / se
    return float(max(1 - stats.t.cdf(t1, n - 1), stats.t.cdf(t2, n - 1)))


def hedges_g(d):
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    g = d.mean() / d.std(ddof=1)
    return float(g * (1 - 3 / (4 * n - 5)))


def pooled_report(recs, cohort):
    """athlete_grouped5 专用：把同一 cohort 同一 seed 的 5 折测试预测拼接后统一计算，
    每名运动员恰好在测试侧出现一次，等价于分层的留一运动员交叉验证。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in recs:
        base = r["target"].split(":")[0] if ":" in r["target"] else r["target"].rsplit("_", 1)[0]
        groups[(base, r["method"], r["seed"])].append(r)
    rows = []
    for (base, method, seed), rs in groups.items():
        y = np.concatenate([r["y"] for r in rs]); p = np.concatenate([r["prob"] for r in rs])
        pl = np.concatenate([r["player"] for r in rs]); ev = np.concatenate([r["event"] for r in rs])
        sy = np.concatenate([r["sup_y"] for r in rs])
        if len(np.unique(y)) < 2: continue
        merged = dict(y=y, prob=p, player=pl, event=ev, sup_y=sy)
        wa, nwa = within_athlete_auc(merged); ba, nba = between_athlete_auc(merged)
        thr = calibrated_threshold(sy, p)
        det = []
        for e in set(x for x in ev if x):
            m = ev == e; det.append(int((p[m] >= thr).any()))
        rows.append(dict(cohort=cohort, target=base, method=method, seed=seed, n_folds=len(rs),
                         n_te=len(y), n_pos=int(y.sum()), n_events=len(set(x for x in ev if x)),
                         n_athletes=len(np.unique(pl)),
                         auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                         auc_within_athlete=wa, n_athletes_with_both=nwa, auc_between_athlete=ba,
                         f1_calibrated=float(f1_score(y, (p >= thr).astype(int), zero_division=0)),
                         event_detection_rate=(sum(det) / len(det)) if det else float("nan")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--bootstrap", type=int, default=0)
    a = ap.parse_args()
    recs = load(a.cohort, a.protocol)
    if not recs:
        print("no dumps"); return
    rows = []
    for r in recs:
        m = metrics(r)
        if a.bootstrap:
            lo, hi = cluster_bootstrap_auc(r, B=a.bootstrap, unit="player", seed=r["seed"])
            m["auc_ci_lo"], m["auc_ci_hi"] = lo, hi
        rows.append(dict(target=r["target"], method=r["method"], seed=r["seed"], split_id=r["info"].get("split_id", ""), **m))
    df = pd.DataFrame(rows)
    if a.protocol == "athlete_grouped5":
        pdf = pooled_report(recs, a.cohort)
        pdf.to_csv(os.path.join(HERE, "..", "output_v2", f"pooled_{a.cohort}_{a.protocol}.csv"), index=False)
        print("\n" + "=" * 112)
        print("拼接 5 折后的统一估计（每名运动员在测试侧恰好出现一次）")
        print("=" * 112)
        for t in sorted(pdf.target.unique()):
            sub = pdf[pdf.target == t]
            print(f"\n### {t}  折数={int(sub.n_folds.median())}  测试事件={int(sub.n_events.median())}  运动员={int(sub.n_athletes.median())}  seed数={sub.seed.nunique()}")
            print(f"{'方法':<13}{'AUC(整体)':<16}{'AUC(个体内)':<14}{'AUC(个体间)':<14}{'AP':<16}{'F1(标定)':<10}{'事件检出率'}")
            for m in METHODS:
                ss = sub[sub.method == m]
                if not len(ss): continue
                print(f"{m:<13}{ss.auc.mean():.3f}±{ss.auc.std():.3f}    {ss.auc_within_athlete.mean():<14.3f}{ss.auc_between_athlete.mean():<14.3f}{ss.ap.mean():.4f}±{ss.ap.std():.4f}  {ss.f1_calibrated.mean():<10.3f}{ss.event_detection_rate.mean():.3f}")
            ref = sub[sub.method == "LSTM"].set_index("seed")
            print(f"{'':13}{'对 LSTM 配对（整体 / 个体内）':<44}{'Δ整体':<10}{'p':<9}{'Δ个体内':<10}{'p'}")
            for m in METHODS:
                if m == "LSTM": continue
                ss = sub[sub.method == m].set_index("seed")
                k = ss.index.intersection(ref.index)
                if len(k) < 2: continue
                d1 = (ss.loc[k].auc - ref.loc[k].auc).values
                d2 = (ss.loc[k].auc_within_athlete - ref.loc[k].auc_within_athlete).values
                p1 = stats.ttest_1samp(d1, 0)[1] if len(d1) > 2 else float("nan")
                p2 = stats.ttest_1samp(d2, 0)[1] if len(d2) > 2 else float("nan")
                print(f"{m:<13}{'':44}{d1.mean():+.4f}   {p1:<9.4f}{d2.mean():+.4f}   {p2:.4f}")
    out_csv = os.path.join(HERE, "..", "output_v2", f"metrics_{a.cohort}_{a.protocol}.csv")
    df.to_csv(out_csv, index=False)
    print(f"{a.cohort} / {a.protocol}   n_runs={len(df)}   -> {os.path.basename(out_csv)}")
    print("=" * 112)
    for t in sorted(df.target.unique()):
        sub = df[df.target == t]
        nev = int(sub.n_events.median()); npos = int(sub.n_pos.median()); nsplit = sub.split_id.nunique()
        tag = "  [事件数<5，标记 not evaluable]" if nev < 5 else ""
        print(f"\n### {t}   测试事件数={nev}  测试正窗={npos}  独立划分={nsplit}{tag}")
        print(f"{'方法':<13}{'AUC(整体)':<16}{'AUC(个体内)':<13}{'AUC(个体间)':<13}{'AP':<14}{'lift':<7}{'F1(标定)':<10}{'事件检出率':<12}{'BrierSkill'}")
        for m in METHODS:
            s = sub[sub.method == m]
            if not len(s): continue
            det = f"{s.event_detection_rate.mean():.3f}" if "event_detection_rate" in s else "—"
            print(f"{m:<13}{s.auc.mean():.3f}±{s.auc.std():.3f}    {s.auc_within_athlete.mean():<13.3f}{s.auc_between_athlete.mean():<13.3f}{s.ap.mean():.4f}±{s.ap.std():.4f}  {s.lift.mean():<7.1f}{s.f1_calibrated.mean():<10.3f}{det:<12}{s.brier_skill.mean():.4f}")
        ref = sub[sub.method == "LSTM"].set_index("seed").auc
        flo = sub[sub.method == "AthleteRate"].set_index("seed").auc
        print(f"{'':13}{'--- 对 LSTM ---':<30}{'Δ':<10}{'p':<9}{'TOST p':<9}{'Hedges g':<10}{'--- 对 AthleteRate 地板 ---':<24}{'Δ':<9}{'p'}")
        for m in METHODS:
            if m == "LSTM": continue
            s = sub[sub.method == m].set_index("seed").auc
            k = s.index.intersection(ref.index)
            if len(k) < 3: continue
            d = (s.loc[k] - ref.loc[k]).values
            k2 = s.index.intersection(flo.index)
            d2 = (s.loc[k2] - flo.loc[k2]).values if len(k2) >= 3 else np.array([np.nan])
            p2 = stats.ttest_1samp(d2, 0)[1] if len(d2) > 2 and not np.isnan(d2).any() else float("nan")
            print(f"{m:<13}{'':30}{d.mean():+.4f}   {stats.ttest_1samp(d,0)[1]:<9.4f}{tost(d):<9.4f}{hedges_g(d):<10.2f}{'':24}{np.nanmean(d2):+.4f}  {p2:.4f}")


if __name__ == "__main__":
    main()
