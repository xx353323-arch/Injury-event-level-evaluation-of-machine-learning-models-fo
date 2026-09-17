import hashlib
import numpy as np
import pandas as pd
from events import window_event_id


def _obs_interval(obs_end, T):
    return obs_end - np.timedelta64(T - 1, "D"), obs_end


def _blocks(dates_sorted_idx, dates, max_len):
    blocks, cur = [], [dates_sorted_idx[0]]
    for i in dates_sorted_idx[1:]:
        if (dates[i] - dates[cur[-1]]).astype(int) == 1 and len(cur) < max_len:
            cur.append(i)
        else:
            blocks.append(cur); cur = [i]
    blocks.append(cur)
    return blocks


def event_level_split_v2(y, players, dates, events, rng, T=14, horizon=7, embargo=20,
                         sup_frac=1.0 / 3.0, assign="random", neg_ratio=4, neg_buffer=13, block_len=14):
    dates = dates.astype("datetime64[D]")
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    eid = np.array([window_event_id(events, players[i], dates[i], horizon) for i in pos_idx], dtype=object)
    keep = np.array([e is not None for e in eid])
    pos_idx, eid = pos_idx[keep], eid[keep]
    uniq = sorted(set(eid.tolist()))
    first_date = {e: dates[pos_idx[eid == e]].min() for e in uniq}
    if assign == "chronological":
        order = sorted(uniq, key=lambda e: first_date[e])
        n_sup_ev = max(int(len(order) * sup_frac), 1)
        sup_events, te_events = set(order[:n_sup_ev]), set(order[n_sup_ev:])
    else:
        order = list(uniq); rng.shuffle(order)
        n_te_ev = len(order) - max(int(len(order) * sup_frac), 1)
        te_events, sup_events = set(order[:n_te_ev]), set(order[n_te_ev:])
    te_p = pos_idx[np.isin(eid, list(te_events))]
    sup_p_raw = pos_idx[np.isin(eid, list(sup_events))]

    te_regions = {}
    for i in te_p:
        p = players[i]; a, b = _obs_interval(dates[i], T)
        ev_end = dates[i] + np.timedelta64(horizon, "D")
        lo, hi = te_regions.get(p, (a, ev_end))
        te_regions[p] = (min(lo, a), max(hi, ev_end))
    te_regions_list = {}
    for i in te_p:
        p = players[i]; a, _ = _obs_interval(dates[i], T)
        te_regions_list.setdefault(p, []).append((a, dates[i] + np.timedelta64(horizon, "D")))

    def overlaps_any(p, a, b, regions):
        for lo, hi in regions.get(p, []):
            if not (b < lo or a > hi):
                return True
        return False

    sup_p = np.array([i for i in sup_p_raw if not overlaps_any(players[i], *_obs_interval(dates[i], T), te_regions_list)], dtype=int)
    n_purged_pos_sup = len(sup_p_raw) - len(sup_p)

    sup_ev_regions = {}
    for i in sup_p:
        p = players[i]; a, _ = _obs_interval(dates[i], T)
        sup_ev_regions.setdefault(p, []).append((a - np.timedelta64(embargo, "D"), dates[i] + np.timedelta64(horizon + embargo, "D")))
    te_ev_regions = {}
    for i in te_p:
        p = players[i]; a, _ = _obs_interval(dates[i], T)
        te_ev_regions.setdefault(p, []).append((a - np.timedelta64(embargo, "D"), dates[i] + np.timedelta64(horizon + embargo, "D")))

    neg_ok_sup = np.array([i for i in neg_idx if not overlaps_any(players[i], dates[i], dates[i], te_ev_regions)], dtype=int)
    neg_ok_te = np.array([i for i in neg_idx if not overlaps_any(players[i], dates[i], dates[i], sup_ev_regions)], dtype=int)

    by_player = {}
    for i in neg_ok_sup:
        by_player.setdefault(players[i], []).append(i)
    blocks = []
    for p, idxs in by_player.items():
        idxs = sorted(idxs, key=lambda i: dates[i])
        blocks += _blocks(idxs, dates, block_len)
    rng.shuffle(blocks)
    target_n = neg_ratio * max(len(sup_p), 1)
    sup_n = []
    for b in blocks:
        if len(sup_n) >= target_n:
            break
        sup_n += b
    sup_n = np.array(sup_n, dtype=int)

    sup_n_regions = {}
    for i in sup_n:
        p = players[i]
        sup_n_regions.setdefault(p, []).append((dates[i] - np.timedelta64(neg_buffer, "D"), dates[i] + np.timedelta64(neg_buffer, "D")))
    sup_n_set = set(sup_n.tolist())
    te_n_candidates = [i for i in neg_ok_te if i not in sup_n_set]
    te_n = np.array([i for i in te_n_candidates if not overlaps_any(players[i], dates[i], dates[i], sup_n_regions)], dtype=int)
    n_purged_neg_te = len(te_n_candidates) - len(te_n)

    sup_p_regions = {}
    for i in sup_p:
        p = players[i]; a, b = _obs_interval(dates[i], T)
        sup_p_regions.setdefault(p, []).append((a, b))
    assert not any(overlaps_any(players[i], *_obs_interval(dates[i], T), sup_p_regions) for i in te_p)
    sup_eids = set(eid[np.isin(pos_idx, sup_p)].tolist()); te_eids = set(eid[np.isin(pos_idx, te_p)].tolist())
    assert not (sup_eids & te_eids)
    sup_n_obs = {}
    for j in sup_n:
        sup_n_obs.setdefault(players[j], []).append(_obs_interval(dates[j], T))
    resid = float(np.mean([overlaps_any(players[i], *_obs_interval(dates[i], T), sup_n_obs) for i in te_n])) if len(te_n) else 0.0

    win_per_event = pd.Series(eid).value_counts()
    info = dict(
        protocol="event_v2_" + assign, T=T, horizon=horizon, embargo=embargo, gap_symbol="g", horizon_symbol="h", embargo_symbol="e",
        n_athletes=int(len(set(players.tolist()))), n_injured_athletes=int(len(set(players[pos_idx].tolist()))),
        n_events=len(uniq), n_sup_events=len(sup_eids), n_te_events=len(te_eids),
        windows_per_event=(int(win_per_event.min()), int(win_per_event.median()), int(win_per_event.max())),
        n_sup_pos=int(len(sup_p)), n_te_pos=int(len(te_p)), n_purged_pos_sup=int(n_purged_pos_sup),
        n_sup_neg=int(len(sup_n)), n_te_neg=int(len(te_n)), n_purged_neg_te=int(n_purged_neg_te),
        residual_te_neg_obs_overlap_frac=float(resid),
        frac_te_pos_same_athlete_as_sup=float(np.mean([players[i] in set(players[sup_p].tolist()) for i in te_p])) if len(te_p) else 0.0,
        split_id=hashlib.md5(",".join(sorted(te_eids)).encode()).hexdigest()[:10],
    )
    return sup_p, te_p, sup_n, te_n, info


def athlete_grouped_folds(players, y, n_folds=5, rng=None, leave_one_out=False):
    uniq = np.array(sorted(set(players.tolist())))
    injured = np.array([y[players == p].sum() > 0 for p in uniq])
    if leave_one_out:
        return [[p] for p in uniq]
    rng = rng or np.random.default_rng(0)
    inj_players = list(uniq[injured]); healthy = list(uniq[~injured])
    rng.shuffle(inj_players); rng.shuffle(healthy)
    folds = [[] for _ in range(n_folds)]
    for k, p in enumerate(inj_players):
        folds[k % n_folds].append(p)
    for k, p in enumerate(healthy):
        folds[k % n_folds].append(p)
    return folds
