"""
Fulfilment Centre Throughput & Bottleneck Analysis
---------------------------------------------------
1. EDA: throughput patterns by station, shift, and time
2. Bottleneck analysis: which station constrains flow, and what pod
   starvation events cost in lost units
3. Predictive model: flag hours at risk of missing UPH targets, with
   feature importance to explain the drivers
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

sns.set_theme(style="whitegrid", palette="deep")
BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE_DIR / "data" / "fc_operations_90days.csv", parse_dates=["date"])

# ---------- 1. EDA ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# a) UPH vs target by station
perf = df.groupby("station").apply(
    lambda g: (g["uph"] / g["target_uph"]).mean() * 100, include_groups=False
).sort_values()
perf.plot(kind="barh", ax=axes[0, 0], color=["#c0392b" if v < 100 else "#27ae60" for v in perf])
axes[0, 0].axvline(100, color="black", ls="--", lw=1)
axes[0, 0].set_title("Average UPH as % of Target, by Station")
axes[0, 0].set_xlabel("% of target UPH")

# b) shift comparison over hours into shift (fatigue curve)
fat = df.groupby(["shift", "hours_into_shift"])["uph"].mean().reset_index()
sns.lineplot(data=fat, x="hours_into_shift", y="uph", hue="shift", marker="o", ax=axes[0, 1])
axes[0, 1].set_title("Productivity Across the Shift (Fatigue Effect)")
axes[0, 1].set_xlabel("Hours into shift")

# c) daily throughput timeline with peak period highlighted
daily = df.groupby("date")["units_processed"].sum()
axes[1, 0].plot(daily.index, daily.values, color="#2c3e50")
peak = df[df["peak_period"] == 1]["date"]
axes[1, 0].axvspan(peak.min(), peak.max(), color="orange", alpha=0.25, label="Peak event")
axes[1, 0].set_title("Total Daily Units Processed (90 days)")
axes[1, 0].legend()

# d) day-of-week pattern
dow = df.groupby("day_of_week")["units_processed"].sum().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
dow.plot(kind="bar", ax=axes[1, 1], color="#2980b9")
axes[1, 1].set_title("Units by Day of Week (Weekend Surge)")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUT / "01_eda_overview.png", dpi=130)
plt.close()

# ---------- 2. Bottleneck & starvation cost ----------
# expected units in starved hours = headcount * target-adjusted rate; compare to actual
starved = df[df["pod_starvation_event"] == 1]
normal = df[df["pod_starvation_event"] == 0]
lost_by_station = []
for st in df["station"].unique():
    s, n = starved[starved["station"] == st], normal[normal["station"] == st]
    if len(s) == 0:
        continue
    lost = (n["uph"].mean() - s["uph"].mean()) * s["headcount"].sum()
    lost_by_station.append({"station": st, "starved_hours": len(s),
                            "avg_uph_normal": round(n["uph"].mean(), 1),
                            "avg_uph_starved": round(s["uph"].mean(), 1),
                            "est_units_lost": int(max(0, lost))})
lost_df = pd.DataFrame(lost_by_station).sort_values("est_units_lost", ascending=False)
lost_df.to_csv(OUT / "starvation_cost_by_station.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=lost_df, x="est_units_lost", y="station", hue="station", legend=False, ax=ax)
ax.set_title(f"Estimated Units Lost to Pod Starvation (90 days) — total ≈ {lost_df['est_units_lost'].sum():,}")
ax.set_xlabel("Estimated units lost")
plt.tight_layout()
plt.savefig(OUT / "02_starvation_cost.png", dpi=130)
plt.close()

# ---------- 3. Predictive model: will this hour miss target? ----------
model_df = df.copy()
model_df["miss_target"] = (model_df["uph"] < model_df["target_uph"]).astype(int)
model_df = pd.get_dummies(model_df, columns=["station", "shift", "day_of_week"], drop_first=True)
features = [c for c in model_df.columns if c.startswith(("station_", "shift_", "day_of_week_"))] + \
           ["hour", "hours_into_shift", "headcount", "pod_starvation_event", "peak_period"]
X, y = model_df[features], model_df["miss_target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)
print(f"\nTarget-miss prediction model — ROC AUC: {auc:.3f}")
print(classification_report(y_te, clf.predict(X_te), target_names=["hit target", "missed target"]))

# Baseline comparison: plain logistic regression (linear, no learned interactions)
scaler = StandardScaler()
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(scaler.fit_transform(X_tr), y_tr)
logreg_auc = roc_auc_score(y_te, logreg.predict_proba(scaler.transform(X_te))[:, 1])
print(f"Baseline logistic regression — ROC AUC: {logreg_auc:.3f} (GBM: {auc:.3f})")

imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9, 5))
imp.plot(kind="barh", ax=ax, color="#8e44ad")
ax.invert_yaxis()
ax.set_title(f"Top Drivers of Missed UPH Targets (Gradient Boosting, AUC {auc:.2f})")
plt.tight_layout()
plt.savefig(OUT / "03_feature_importance.png", dpi=130)
plt.close()

# ---------- summary ----------
print("\n=== KEY FINDINGS ===")
miss_rate = df.assign(miss=(df["uph"] < df["target_uph"])).groupby("station")["miss"].mean() * 100
worst = miss_rate.idxmax()
print(f"1. Least reliable station: {worst} — misses hourly UPH target {miss_rate.max():.0f}% of the time")
print(f"2. Pod starvation cost ≈ {lost_df['est_units_lost'].sum():,} units over 90 days")
night_gap = (1 - df[df["shift"] == "Night"]["uph"].mean() / df[df["shift"] == "Day"]["uph"].mean()) * 100
print(f"3. Night shift UPH is {night_gap:.1f}% below day shift on average")
print(f"4. Target-miss risk is predictable (GBM AUC {auc:.2f} vs. logistic regression baseline {logreg_auc:.2f}) "
      f"— starvation events and late-shift hours dominate. The two models are close here, so most of the "
      f"signal is near-linear once station/shift are encoded; GBM is kept for the free feature-importance "
      f"ranking and headroom on messier real-world data")
print("\nCharts saved to outputs/")
