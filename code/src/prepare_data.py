import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "soccermon"
)

WELLNESS = ["fatigue", "mood", "readiness", "sleep_duration", "soreness", "stress"]
LOAD = ["daily_load", "atl", "ctl28", "acwr"]
CHANNELS = LOAD + WELLNESS

WIN = 14
HORIZON = 7


def read_wide(path, tag):
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    long = df.melt(id_vars=["date"], var_name="player", value_name=tag)
    return long


def load_channels():
    frames = []
    for c in LOAD:
        frames.append(read_wide(os.path.join(BASE, "training-load", f"{c}.csv"), c))
    for c in WELLNESS:
        frames.append(read_wide(os.path.join(BASE, "wellness", f"{c}.csv"), c))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["date", "player"], how="outer")
    merged = merged.sort_values(["player", "date"]).reset_index(drop=True)
    for c in CHANNELS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    return merged


def load_injury():
    df = pd.read_csv(os.path.join(BASE, "injury", "injury.csv"))
    df["date"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    return df[["player_name", "date"]].rename(columns={"player_name": "player"})


def build_windows(panel, injuries, win=WIN, horizon=HORIZON):
    inj_set = set(zip(injuries["player"], injuries["date"]))
    X_list, y_list, meta = [], [], []
    for player, g in panel.groupby("player"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[CHANNELS].to_numpy(dtype=np.float32)
        mask = ~np.isnan(vals).all(axis=1)
        g = g[mask].reset_index(drop=True)
        vals = g[CHANNELS].to_numpy(dtype=np.float32)
        if len(g) < win + horizon:
            continue
        col_mean = np.nanmean(vals, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        nan_idx = np.where(np.isnan(vals))
        vals[nan_idx] = np.take(col_mean, nan_idx[1])
        for i in range(len(g) - win - horizon + 1):
            window = vals[i : i + win]
            pred_start = g["date"].iloc[i + win]
            pred_end = g["date"].iloc[i + win + horizon - 1]
            label = 0
            for d in pd.date_range(pred_start, pred_end):
                if (player, d) in inj_set:
                    label = 1
                    break
            X_list.append(window)
            y_list.append(label)
            meta.append((player, g["date"].iloc[i + win - 1]))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y, meta


def main():
    panel = load_channels()
    injuries = load_injury()
    X, y, meta = build_windows(panel, injuries)
    print(f"samples={len(y)}  positives={int(y.sum())}  rate={y.mean():.4f}")
    print(f"X shape={X.shape}  (N, T, C)")
    out = os.path.dirname(__file__)
    players_arr = np.array([m[0] for m in meta])
    dates_arr = np.array([pd.Timestamp(m[1]) for m in meta])
    teams = np.array(["TeamA" if p.startswith("TeamA") else "TeamB" for p in players_arr])
    seasons = np.array([d.year for d in dates_arr])
    tasks = np.array([f"{t}-{s}" for t, s in zip(teams, seasons)])
    print("task counts:", dict(zip(*np.unique(tasks, return_counts=True))))
    np.savez_compressed(
        os.path.join(out, "windows.npz"),
        X=X,
        y=y,
        players=players_arr,
        dates=np.array([str(d) for d in dates_arr]),
        teams=teams,
        seasons=seasons,
        tasks=tasks,
    )
    print("saved windows.npz")
    print("saved windows.npz")


if __name__ == "__main__":
    main()
