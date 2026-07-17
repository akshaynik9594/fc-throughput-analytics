# Fulfilment Centre Throughput & Bottleneck Analytics

End-to-end operations analytics project: simulating, analysing and predicting throughput performance in a warehouse fulfilment operation — combining my first-hand experience working in large-scale fulfilment centres with my MSc Data Science skillset.

> **Data note:** all data is **synthetic**, generated to reflect publicly known warehouse dynamics (units-per-hour metrics, pick/pack/stow stations, shift patterns, upstream disruption events). No proprietary or employer data is used.

> **Methodology note:** because the underlying effects (fatigue, starvation impact, shift penalty, weekend/peak demand) are known parameters baked into `generate_data.py`, this project is a **validation exercise**, not a claim of discovering unknown patterns: it demonstrates that the analysis and model correctly recover effects whose ground truth is known in advance. The same pipeline — EDA → cost analysis → classifier with a linear baseline — is designed to carry over unchanged to real operational data, where those drivers would *not* be known ahead of time.

## The business questions

1. **Where is the bottleneck?** Which station most often misses its hourly UPH target, and when?
2. **What do disruptions cost?** Pod starvation events (upstream supply interruptions) suppress downstream throughput — how many units are lost?
3. **Can we predict trouble before it happens?** Flag hours at high risk of missing target so leaders can reallocate labour proactively.

## Key findings (90-day simulation)

- **Pack-Multis is the least reliable station**, missing its hourly UPH target ~53% of the time — a labour-planning problem, not an effort problem.
- **Pod starvation events cost ≈ 222k units** over 90 days, concentrated in Pick and downstream pack stations. Preventing even a quarter of starvation hours recovers ~55k units.
- **Night shift runs ~7% below day shift UPH**, and productivity dips measurably after hour 6 of any shift (fatigue effect) — break scheduling and rotation are levers.
- **Target-misses are predictable** — a gradient boosting classifier reaches ROC AUC 0.89, against a logistic regression baseline at 0.88. The gap is small, meaning most of the signal is close to linear once station and shift are encoded; gradient boosting is kept for its built-in feature-importance ranking and for headroom on messier real-world data, not because it's dramatically more accurate here.

## Project structure

```
├── data/
│   └── fc_operations_90days.csv     # 10,800 rows: hourly × station × 90 days
├── src/
│   ├── generate_data.py             # synthetic data generator (seeded, reproducible)
│   ├── analysis.py                  # EDA, bottleneck & starvation analysis, ML model
│   └── app.py                       # interactive Streamlit dashboard
├── outputs/                         # charts + starvation cost table
└── README.md
```

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
python src/generate_data.py
python src/analysis.py
streamlit run src/app.py
```

## Skills demonstrated

Python (pandas, NumPy, scikit-learn) · simulation & data generation · EDA · statistical thinking (seasonality, fatigue effects, disruption costing) · gradient boosting classification with feature importance · Streamlit dashboarding · translating operational domain knowledge into analytical questions.

## Why this project

I worked inside Amazon fulfilment centres while completing my MSc in Data Science. On the floor, I watched pod starvation events stall entire pick stations and saw how much of shift management is reactive. This project asks: what would it look like to manage that operation *proactively*, with data? It is the analysis I wished my site had.

---
*Akshay Shelke — MSc Data Science (Merit), Coventry University · [LinkedIn] · [Email](mailto:akshayraj2895@gmail.com)*
