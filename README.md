# Reproduction package — BZLR-090-1

Manuscript: *Injury-event-level evaluation of machine learning models for athlete injury risk prediction: a benchmark across two cohorts* (Scientific Reports, under revision).

This package contains everything needed to re-run the experiments: code, both datasets, and the result files reported in the manuscript. No manuscript text, review correspondence or internal notes are included.

## Layout

```
BZLR-090-1_Reproduction_Package_20260917/
├── README.md
├── DATA_SOURCES.md          provenance and licences for both datasets
├── code/
│   ├── pipeline_v2/         protocol v2: events.py, splits.py, run_v2.py, analyze_v2.py
│   ├── src/                 model implementations (LMVG, TCA-GNN and variants) + prepare_data.py
│   ├── scripts/             baselines and stage-1 validation scripts
│   ├── runners/             runner-cohort window construction and runs
│   └── data/                windows.npz, windows_runners.npz (derived arrays)
├── data/
│   ├── soccermon/           football cohort, raw CSVs (CC BY 4.0)
│   └── runners/             runner cohort, raw CSVs (CC0 1.0)
└── results/
    ├── v1/                  stage-1 protocol results (JSON, TXT, MD)
    └── v2/                  protocol v2 per-run metrics (CSV), 1128 runs
```

## Quick start

```bash
cd code/pipeline_v2

# football cohort, event-level protocol, one target, one method  (~1 min)
python3 run_v2.py --cohort soccermon --protocol event_v2_random \
                  --targets TeamB-2020 --methods XGBoost --seeds 42

# full runner-cohort run under the athlete-grouped protocol
python3 run_v2.py --cohort runners --protocol athlete_grouped5 --seeds 42,123,2026
python3 analyze_v2.py --cohort runners --protocol athlete_grouped5
```

Results are written to `code/output_v2/`. The reference outputs shipped in `results/v2/` were produced with the default ten seeds and all eight methods.

Key arguments of `run_v2.py`: `--cohort` (`soccermon` or `runners`), `--protocol` (`event_v2_random` or `athlete_grouped5`), `--targets`, `--methods` (default: ours, LSTM, XGBoost, GAT, Transformer, ProtoNet, DANN, AthleteRate), `--seeds`, `--episodes` (default 120), `--embargo` (default 20), `--threads` (default 2).

Rebuilding the derived arrays from raw CSVs is optional:

```bash
python3 code/src/prepare_data.py            # → windows.npz
python3 code/runners/build_windows_runners.py   # → windows_runners.npz
```

## Protocol v2 in brief

Event reconstruction uses three independent symbols: `g`, the merge gap for injury reports (default 7 days); `h`, the label horizon (7 days for football, 1 day for runners); and `e`, the embargo (default 20 days). Splitting fixes the test side first, then applies a one-directional purge of support-side positive windows overlapping the observation interval of any test event. Negative windows are drawn as whole blocks of consecutive dates within an athlete, with a ±13-day buffer on the test side. Every split emits purge counts, residual overlap rates and cross-side assertions.

The athlete-grouped protocol (`athlete_grouped5`) partitions athletes into five folds and removes held-out athletes from the source side as well. Fold predictions are concatenated before metrics are computed, which is equivalent to stratified leave-one-athlete-out cross-validation.

## Metrics

Besides overall AUC and average precision, the package computes within-athlete AUC (positive versus negative windows compared only inside the same athlete) and between-athlete AUC (each athlete aggregated to one point). The first asks whether the model predicts *when* an athlete gets injured; the second measures how much it only separates *who* is injury-prone. Threshold-dependent metrics use an alert-budget threshold calibrated on the support side, reading no test label; the test-set-optimal threshold is reported separately as an optimistic upper bound.

## Environment

Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, scikit-learn 1.6.1, pandas, scipy, xgboost. CPU only; no GPU required.

## Note on file paths

Data paths in six scripts were rewritten for this package so that everything resolves inside the package directory (`run_v2.py`, `event_split.py`, `prepare_data.py`, `run_runners_probe.py`, `run_runners_probe_tp.py`, `run_runners_10seed.py`). No analysis logic was changed. The quick-start command above was executed against this package to confirm it runs end to end.
