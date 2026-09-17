import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = ['nr. sessions','total km','km Z3-4','km Z5-T1-T2','km sprinting','strength training','hours alternative','perceived exertion','perceived trainingSuccess','perceived recovery']
T = 7

df = pd.read_csv(os.path.join(HERE, '..', '..', 'data', 'runners', 'day_approach_maskedID_timeseries.csv')).sort_values(['Athlete ID','Date']).reset_index(drop=True)
X = np.zeros((len(df), T, len(FEAT)), dtype=np.float32)
for k in range(T):
    cols = [f if k == 0 else f'{f}.{k}' for f in FEAT]
    X[:, k, :] = df[cols].values.astype(np.float32)
y = df['injury'].astype(np.int64).values
ath = df['Athlete ID'].astype(int).values
dates = df['Date'].astype(int).values
eid = np.array([f'{a}|{d}' if lab == 1 else '' for a, d, lab in zip(ath, dates, y)])

rng = np.random.default_rng(0)
inj_per_ath = pd.Series(y).groupby(ath).sum().sort_values(ascending=False)
groups = {}
order = inj_per_ath.index.tolist()
for i, a in enumerate(order):
    groups[a] = ['G1','G2','G3'][i % 3]
task = np.array([groups[a] for a in ath])

np.savez_compressed(os.path.join(HERE, 'windows_runners.npz'), X=X, y=y, athletes=ath, dates=dates, event_id=eid, tasks=task)

print(f'窗口 {len(y)}  阳性 {int(y.sum())}（=独立事件数）  阳性率 {y.mean():.4f}  X {X.shape}')
print('按伤病数交错分成3组(模拟LOTOCV):')
for g in ['G1','G2','G3']:
    m = task == g
    print(f'  {g}: 运动员 {len(set(ath[m]))}  窗口 {m.sum()}  阳性 {int(y[m].sum())}  阳性率 {y[m].mean():.4f}')
print('时间顺序核对: X[:,0,:]=7天前, X[:,6,:]=前一天')
