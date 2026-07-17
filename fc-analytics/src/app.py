"""
FC Operations Dashboard — Streamlit app
Run locally:  streamlit run src/app.py
Deploy free:  Streamlit Community Cloud (share.streamlit.io) from your GitHub repo
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FC Operations Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "fc_operations_90days.csv"

@st.cache_data
def load():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

df = load()

st.title("Fulfilment Centre Operations Dashboard")
st.caption("Synthetic 90-day dataset • hourly station-level throughput, downtime and disruption events")

# ---- Sidebar filters ----
stations = st.sidebar.multiselect("Stations", sorted(df["station"].unique()), default=sorted(df["station"].unique()))
shifts = st.sidebar.multiselect("Shift", df["shift"].unique().tolist(), default=df["shift"].unique().tolist())
date_range = st.sidebar.date_input("Date range", [df["date"].min(), df["date"].max()])

f = df[df["station"].isin(stations) & df["shift"].isin(shifts)]
if len(date_range) == 2:
    f = f[(f["date"] >= pd.Timestamp(date_range[0])) & (f["date"] <= pd.Timestamp(date_range[1]))]

# ---- KPI row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total units", f"{f['units_processed'].sum():,}")
c2.metric("Avg UPH", f"{f['uph'].mean():.0f}")
hit = (f["uph"] >= f["target_uph"]).mean() * 100
c3.metric("Hours hitting target", f"{hit:.0f}%")
c4.metric("Starvation events", int(f["pod_starvation_event"].sum()))

# ---- Charts ----
left, right = st.columns(2)

with left:
    st.subheader("Daily throughput")
    daily = f.groupby("date")["units_processed"].sum()
    st.line_chart(daily)

    st.subheader("UPH vs target by station")
    perf = (f.groupby("station")
             .apply(lambda g: (g["uph"] / g["target_uph"]).mean() * 100, include_groups=False)
             .sort_values())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    perf.plot(kind="barh", ax=ax, color=["#c0392b" if v < 100 else "#27ae60" for v in perf])
    ax.axvline(100, color="black", ls="--", lw=1)
    ax.set_xlabel("% of target UPH")
    st.pyplot(fig)

with right:
    st.subheader("Fatigue curve — productivity across the shift")
    fat = f.groupby(["shift", "hours_into_shift"])["uph"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    sns.lineplot(data=fat, x="hours_into_shift", y="uph", hue="shift", marker="o", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Cost of pod starvation")
    s = f[f["pod_starvation_event"] == 1]
    n = f[f["pod_starvation_event"] == 0]
    if len(s):
        comp = pd.DataFrame({
            "Normal hours": n.groupby("station")["uph"].mean(),
            "Starved hours": s.groupby("station")["uph"].mean(),
        }).dropna()
        st.bar_chart(comp)
    else:
        st.info("No starvation events in the current filter.")

st.divider()
st.subheader("What-if: headcount reallocation")

worst_station = perf.index[0]
station_uph = f.groupby("station")["uph"].mean()
station_hours = f.groupby("station").size()
station_units = f.groupby("station")["units_processed"].sum()

st.caption(
    f"Worst-performing station in the current filter: **{worst_station}** "
    f"({perf.iloc[0]:.0f}% of target UPH). Projection assumes each station's average "
    f"units-per-person-per-hour rate stays constant regardless of headcount "
    f"(no diminishing returns or congestion modeled) — a simplification, not a validated staffing model."
)

other_stations = sorted(s for s in station_uph.index if s != worst_station)
if other_stations:
    col_a, col_b = st.columns(2)
    with col_a:
        source_station = st.selectbox("Move headcount from", other_stations)
    with col_b:
        n_people = st.slider("Headcount to move (N)", min_value=0, max_value=10, value=2)

    target_gain = n_people * station_uph[worst_station] * station_hours[worst_station]
    source_loss = n_people * station_uph[source_station] * station_hours[source_station]
    net_change = target_gain - source_loss

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{worst_station} (target)", f"{station_units[worst_station] + target_gain:,.0f}",
              f"+{target_gain:,.0f}")
    m2.metric(f"{source_station} (source)", f"{station_units[source_station] - source_loss:,.0f}",
              f"-{source_loss:,.0f}")
    m3.metric("Net total throughput", f"{f['units_processed'].sum() + net_change:,.0f}",
              f"{net_change:+,.0f}")

    if net_change < 0:
        st.warning(
            f"Net units would fall — {source_station} produces more per person than {worst_station}, "
            f"so this move raises {worst_station}'s reliability at the cost of total volume."
        )

    comp = pd.DataFrame({
        "Current": [station_units[worst_station], station_units[source_station]],
        "Projected": [station_units[worst_station] + target_gain, station_units[source_station] - source_loss],
    }, index=[worst_station, source_station])
    st.bar_chart(comp)
else:
    st.info("Select at least two stations in the sidebar to model reallocation.")

st.divider()
st.markdown(
    "**About:** built by Akshay Shelke — MSc Data Science, with first-hand fulfilment-centre "
    "operations experience. Data is fully synthetic; no proprietary information is used."
)
