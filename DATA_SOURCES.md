# Data sources and licences

Both datasets bundled here are public and redistributable. Neither was collected by the authors.

## 1. SoccerMon (football cohort) — `data/soccermon/`

Midoglu C, et al. *A large-scale multivariate soccer athlete health, performance, and position monitoring dataset.* Scientific Data, 2024.

- Record: https://zenodo.org/records/10033832
- DOI: 10.5281/zenodo.10033832
- Licence: **CC BY 4.0** (Zenodo record metadata, `"license": {"id": "cc-by-4.0"}`)
- Bundled subset: `injury/`, `wellness/`, `training-load/`, `illness/`, `game-performance/`
- Two Norwegian elite women's teams (anonymised as TeamA and TeamB), two seasons, 2020-01-14 to 2021-11-03

Wide-format CSVs: rows are dates, columns are pseudonymous player identifiers of the form `TeamA-<uuid>`. The injury table `injury/injury.csv` has 162 rows and three columns (player, body-part/severity JSON, date). Merging reports from the same player within 7 days gives **40 independent injury events** (TeamA 36, TeamB 4).

## 2. Competitive runners (endurance cohort) — `data/runners/`

Lövdal S, den Hartigh RJR, Azzopardi G. *Injury prediction in competitive runners with machine learning.* International Journal of Sports Physiology and Performance, 2021.

- Record: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/UWU9PV
- DOI: 10.34894/UWU9PV
- Licence: **CC0 1.0 Universal** (DataverseNL metadata, `"rightsIdentifier": "CC0-1.0"`; identical in V1 and V2)
- Bundled files: `day_approach_maskedID_timeseries.csv`, `week_approach_maskedID_timeseries.csv`, `README.txt` (the authors' own field documentation)
- 74 competitive runners, 583 independent injury events

The DataverseNL V2 release (2024-06-05) corrects three lines of the README only; the three data files are byte-identical to V1.

## Derived arrays — `code/data/`

| File | Built by | Shape | Contents |
|---|---|---|---|
| `windows.npz` | `code/src/prepare_data.py` | X (35550, 14, 10) | Football cohort: 14-day windows over 10 channels (4 load, 6 wellness), labels, player/date/team/season/task keys |
| `windows_runners.npz` | `code/runners/build_windows_runners.py` | X (42766, 7, 10) | Runner cohort: 7-day windows over 10 channels, labels, athlete/date/event-id/task keys |

Both are regenerable from the raw CSVs in `data/`; they are included so that experiments run without a preprocessing step.

## Attribution requirement

SoccerMon is CC BY 4.0, so any reuse must cite Midoglu et al. (2024). The runner dataset is CC0 and carries no legal attribution requirement, but the source paper is cited throughout this work.
