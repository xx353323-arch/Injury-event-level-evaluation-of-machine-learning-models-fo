import json
import numpy as np
import pandas as pd


def load_injury_table(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    def parts(s):
        try:
            return tuple(sorted(json.loads(s).keys()))
        except Exception:
            return tuple()
    df["parts"] = df["type"].apply(parts)
    return df[["player_name", "date", "parts"]].rename(columns={"player_name": "player"})


def reconstruct_events(injury_df, gap_days=7, split_by_bodypart=False):
    events = {}
    for player, g in injury_df.groupby("player"):
        rows = g.sort_values("date")
        recs = []
        for _, r in rows.iterrows():
            d = np.datetime64(r["date"].date())
            recs.append((d, r["parts"]))
        merged = []
        for d, p in recs:
            if merged:
                last = merged[-1]
                same_part = (not split_by_bodypart) or (set(p) & set(last["parts"])) or (not p) or (not last["parts"])
                if (d - last["end"]).astype(int) <= gap_days and same_part:
                    last["end"] = max(last["end"], d)
                    last["parts"] = tuple(sorted(set(last["parts"]) | set(p)))
                    last["n_reports"] += 1
                    continue
            merged.append(dict(start=d, end=d, parts=p, n_reports=1))
        events[player] = merged
    return events


def window_event_id(events, player, obs_end, horizon):
    hs = obs_end + np.timedelta64(1, "D")
    he = obs_end + np.timedelta64(horizon, "D")
    for k, ev in enumerate(events.get(player, [])):
        if not (he < ev["start"] or hs > ev["end"]):
            return f"{player}|{k}"
    return None


def event_summary(events):
    n_ev = sum(len(v) for v in events.values())
    n_inj_athletes = sum(1 for v in events.values() if v)
    gaps = []
    for v in events.values():
        for a, b in zip(v[:-1], v[1:]):
            gaps.append((b["start"] - a["end"]).astype(int))
    return dict(n_events=n_ev, n_injured_athletes=n_inj_athletes,
                inter_event_gap_days=(int(np.min(gaps)) if gaps else None, int(np.median(gaps)) if gaps else None, int(np.max(gaps)) if gaps else None))
