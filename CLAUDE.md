# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A small, self-contained fulfilment-centre (warehouse) operations analytics project. All data is synthetic. The project simulates hourly, per-station throughput data, then runs EDA, a bottleneck/disruption-cost analysis, and a gradient-boosting classifier to predict hours at risk of missing throughput targets. Everything lives under `fc-analytics/`.

See [fc-analytics/README.md](fc-analytics/README.md) for the full write-up of business questions and key findings.

## Commands

There is no dependency manifest (no `requirements.txt`/`pyproject.toml`) — install directly:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

Run from inside `fc-analytics/` (paths below assume this as cwd):

```bash
python src/generate_data.py   # regenerate data/fc_operations_90days.csv (seeded, reproducible via np.random.default_rng(42))
python src/analysis.py        # EDA + bottleneck/starvation analysis + ML model; writes charts/tables to outputs/
streamlit run src/app.py      # interactive dashboard reading data/fc_operations_90days.csv
```

There are no tests, linter, or build step in this repo.

A `venv/` at the repo root (`fc-analytics-project/venv`) already has pandas, numpy, scikit-learn, matplotlib, seaborn, and streamlit installed — activate it (`venv\Scripts\activate`) or call `venv/Scripts/python` directly instead of reinstalling.

All three scripts (`generate_data.py`, `analysis.py`, `app.py`) resolve their data/outputs paths relative to the script's own location (via `Path(__file__).resolve().parent.parent`), so they can be run/launched from any cwd.

## Architecture

Three scripts form a linear pipeline, connected only through the CSV file on disk (no shared modules/imports between them):

1. **`src/generate_data.py`** — builds the synthetic dataset from scratch. Encodes the domain model that everything downstream depends on: 6 stations (`Stow`, `Pick`, `Rebin`, `Pack-Singles`, `Pack-Multis`, `SLAM`) each with a base target UPH (units-per-hour), two shifts (`Day` 08:00–18:00, `Night` 19:00–05:00) with different baseline productivity, 90 simulated days with weekend demand surges and a mid-period "peak event" (days 60–67), random per-shift "pod starvation" windows that suppress UPH for downstream stations (`Pick`, `Rebin`, `Pack-Singles`, `Pack-Multis`), and a within-shift fatigue curve (productivity dips after hour 6). Output: `data/fc_operations_90days.csv`, one row per (date, hour, station) — 10,800 rows.
2. **`src/analysis.py`** — reads that CSV and does three independent things in sequence: (a) EDA plots (UPH vs. target by station, fatigue curve, daily timeline, day-of-week pattern) saved as `outputs/01_eda_overview.png`; (b) starvation cost analysis — compares mean UPH in starved vs. normal hours per station to estimate units lost, saved as `outputs/02_starvation_cost.png` and `outputs/starvation_cost_by_station.csv`; (c) a `GradientBoostingClassifier` predicting `miss_target` (`uph < target_uph`) from one-hot-encoded station/shift/day-of-week plus hour, headcount, starvation flag, and peak-period flag, with feature importances saved as `outputs/03_feature_importance.png`. Prints an ROC AUC and a key-findings summary to stdout.
3. **`src/app.py`** — a Streamlit dashboard over the same CSV, independent of `analysis.py` (recomputes its own aggregations rather than reading `outputs/`). Sidebar filters (station, shift, date range) drive KPI tiles and four charts (daily throughput, UPH-vs-target by station, fatigue curve, starvation cost comparison).

When changing the data schema in `generate_data.py`, `analysis.py` and `app.py` both need matching updates since they each independently re-derive features (e.g. `miss_target`, UPH-vs-target ratios) from the raw columns rather than sharing helper functions.
