import numpy as np
import pandas as pd
import os

INJURY_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "soccermon", "injury", "injury.csv"
)
WIN, HORIZON = 14, 7
EMBARGO = WIN + HORIZON - 1


def build_events(gap_days=7):
    inj = pd.read_csv(INJURY_CSV)
    inj["date"] = pd.to_datetime(inj["timestamp"], format="%d.%m.%Y", errors="coerce")
    inj = inj.dropna(subset=["date"])
    ev = {}
    for p, g in inj.groupby("player_name"):
        ds = np.unique(np.sort(g["date"].values.astype("datetime64[D]")))
        if len(ds) == 0:
            continue
        groups = []
        cur = [ds[0]]
        for x in ds[1:]:
            if (x - cur[-1]).astype(int) <= gap_days:
                cur.append(x)
            else:
                groups.append((cur[0], cur[-1]))
                cur = [x]
        groups.append((cur[0], cur[-1]))
        ev[p] = groups
    return ev


_EVENTS = build_events(gap_days=7)


def _win_event_id(player, obs_end_date):
    hs = obs_end_date + np.timedelta64(1, "D")
    he = obs_end_date + np.timedelta64(HORIZON, "D")
    if player not in _EVENTS:
        return None
    for k, (a, b) in enumerate(_EVENTS[player]):
        if not (he < a or hs > b):
            return f"{player}|{k}"
    return None


def chronological_window_split(y_tgt, dates_tgt, rng, sup_frac=1.0/3.0):
    order = np.argsort(dates_tgt)
    pos_ord = order[y_tgt[order] == 1]
    neg_ord = order[y_tgt[order] == 0]
    n_sup_pos = max(int(len(pos_ord) * sup_frac), 2)
    sup_p = pos_ord[:n_sup_pos]
    te_p = pos_ord[n_sup_pos:]
    n_sup_neg = min(4 * n_sup_pos, len(neg_ord) // 3)
    sup_n = neg_ord[:n_sup_neg]
    te_n = neg_ord[n_sup_neg:]
    return sup_p, te_p, sup_n, te_n, dict(
        protocol="chronological_window", n_sup_pos=len(sup_p), n_te_pos=len(te_p),
        n_sup_neg=len(sup_n), n_te_neg=len(te_n),
    )


def event_level_split(y_tgt, players_tgt, dates_tgt, rng, embargo_days=None):
    embargo = EMBARGO if embargo_days is None else int(embargo_days)
    """
    Injury-event-level support/test split with temporal purge+embargo.
    Positive windows are grouped by the injury event they predict; whole
    events (never individual windows) are assigned to support or test, so no
    event's overlapping windows can straddle both sides. An embargo of
    WIN+HORIZON-1 days around every event date range purges same-player
    windows (positive or negative) that fall inside the opposite side's
    embargo zone, removing residual overlap leakage from negatives too.
    """
    pos_idx = np.where(y_tgt == 1)[0]
    neg_idx = np.where(y_tgt == 0)[0]

    eids = np.array([_win_event_id(players_tgt[i], dates_tgt[i]) for i in pos_idx])
    valid = eids != None  # noqa: E711
    pos_idx = pos_idx[valid]
    eids = eids[valid]

    uniq_events = sorted(set(eids.tolist()))
    rng.shuffle(uniq_events)
    n_sup_ev = max(len(uniq_events) // 3, 1)
    sup_events = set(uniq_events[:n_sup_ev])

    sup_p = pos_idx[np.isin(eids, list(sup_events))]
    te_p = pos_idx[~np.isin(eids, list(sup_events))]

    sup_event_players = set()
    sup_event_ranges = []
    te_event_ranges = []
    for eid in uniq_events:
        player, k = eid.rsplit("|", 1)
        a, b = _EVENTS[player][int(k)]
        lo = a - np.timedelta64(embargo, "D")
        hi = b + np.timedelta64(embargo, "D")
        if eid in sup_events:
            sup_event_ranges.append((player, lo, hi))
            sup_event_players.add(player)
        else:
            te_event_ranges.append((player, lo, hi))

    def in_any_range(player, date, ranges):
        for p, lo, hi in ranges:
            if p == player and lo <= date <= hi:
                return True
        return False

    neg_ok_for_sup, neg_ok_for_te = [], []
    for i in neg_idx:
        p, d = players_tgt[i], dates_tgt[i]
        blocked_by_te = in_any_range(p, d, te_event_ranges)
        blocked_by_sup = in_any_range(p, d, sup_event_ranges)
        if not blocked_by_te:
            neg_ok_for_sup.append(i)
        if not blocked_by_sup:
            neg_ok_for_te.append(i)

    neg_ok_for_sup = np.array(neg_ok_for_sup)
    neg_ok_for_te = np.array(neg_ok_for_te)
    rng.shuffle(neg_ok_for_sup)
    rng.shuffle(neg_ok_for_te)

    n_sup_neg = min(4 * max(len(sup_p), 1), len(neg_ok_for_sup))
    sup_n = neg_ok_for_sup[:n_sup_neg]
    te_n_pool = np.setdiff1d(neg_ok_for_te, sup_n, assume_unique=False)
    te_n = te_n_pool

    return sup_p, te_p, sup_n, te_n, dict(
        n_events=len(uniq_events), n_sup_events=len(sup_events),
        n_te_events=len(uniq_events) - len(sup_events),
        n_sup_pos=len(sup_p), n_te_pos=len(te_p),
        n_sup_neg=len(sup_n), n_te_neg=len(te_n),
    )
