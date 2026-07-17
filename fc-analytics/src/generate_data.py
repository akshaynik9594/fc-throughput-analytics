"""
Synthetic Fulfilment Centre Operations Data Generator
------------------------------------------------------
Generates 90 days of realistic hourly station-level data for a simulated
fulfilment centre. All data is synthetic — modelled on publicly known
warehouse concepts (UPH, pick/pack/stow stations, downtime events) and
contains no real company data.

Realistic patterns baked in:
- Day/night shift productivity differences
- Weekly seasonality (weekend order surges)
- A mid-period "peak event" (like Prime Day / Black Friday ramp)
- Station-specific base rates (pack is faster than pick, etc.)
- Pod starvation events that suppress downstream throughput
- Fatigue effect in final hours of a shift
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

rng = np.random.default_rng(42)

DAYS = 90
STATIONS = {
    "Stow": 180, "Pick": 240, "Rebin": 300, "Pack-Singles": 210, "Pack-Multis": 140, "SLAM": 900,
}
SHIFTS = {"Day": range(8, 18), "Night": list(range(19, 24)) + list(range(0, 5))}

rows = []
start = pd.Timestamp("2026-03-01")

for d in range(DAYS):
    date = start + pd.Timedelta(days=d)
    dow = date.dayofweek
    # weekend order surge + peak event around day 60-67
    demand_mult = 1.0 + (0.15 if dow >= 5 else 0.0)
    if 60 <= d <= 67:
        demand_mult *= 1.35
    for shift, hours in SHIFTS.items():
        shift_mult = 1.0 if shift == "Day" else 0.93  # night shift slightly lower baseline
        # each shift has ~risk of an upstream disruption window
        starvation_hours = set()
        if rng.random() < (0.22 if shift == "Night" else 0.12):
            h0 = rng.choice(list(hours))
            for k in range(rng.integers(1, 4)):
                starvation_hours.add((h0 + k) % 24)
        for i, hour in enumerate(hours):
            hours_into_shift = i
            fatigue = 1.0 - max(0, hours_into_shift - 6) * 0.02  # dip late in shift
            for station, base in STATIONS.items():
                headcount = max(2, int(rng.normal(12, 2)))
                starved = int(hour in starvation_hours and station in ("Pick", "Rebin", "Pack-Singles", "Pack-Multis"))
                starve_mult = 0.55 if starved else 1.0
                uph = base * shift_mult * fatigue * demand_mult * starve_mult * rng.normal(1.0, 0.07)
                units = max(0, int(uph * headcount))
                downtime = round(max(0, rng.normal(3 if starved else 1.2, 1.5)), 1)
                rows.append({
                    "date": date.date(), "day_of_week": date.day_name(), "shift": shift,
                    "hour": hour, "hours_into_shift": hours_into_shift, "station": station,
                    "headcount": headcount, "units_processed": units,
                    "uph": round(units / headcount, 1),
                    "target_uph": base, "downtime_minutes": downtime,
                    "pod_starvation_event": starved, "peak_period": int(60 <= d <= 67),
                })

df = pd.DataFrame(rows)
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_DIR / "fc_operations_90days.csv", index=False)
print(f"Generated {len(df):,} rows across {DAYS} days, {len(STATIONS)} stations")
print(df.head(3).to_string())
