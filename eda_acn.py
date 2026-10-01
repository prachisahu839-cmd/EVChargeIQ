# ============================================================
# EVChargeIQ
# Exploratory Data Analysis - ACN Caltech Dataset
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/caltech_sessions.csv"
)

OUTPUT_DIR = Path(
    "data/processed/eda"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("EVChargeIQ - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nLoading processed dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# 3. DATETIME CONVERSION
# ============================================================

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    errors="coerce",
    utc=True
)

df["disconnectTime"] = pd.to_datetime(
    df["disconnectTime"],
    errors="coerce",
    utc=True
)

df["doneChargingTime"] = pd.to_datetime(
    df["doneChargingTime"],
    errors="coerce",
    utc=True
)

df["connectionDate"] = pd.to_datetime(
    df["connectionDate"],
    errors="coerce"
)


# ============================================================
# 4. NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "kWhDelivered",
    "sessionDuration_min",
    "chargingDuration_min",
    "postDisconnectDuration_min",
    "energy_per_charging_min",
    "hour",
    "day_of_week",
    "month",
    "year",
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# 5. BASIC DATASET PROFILE
# ============================================================

print("\n")
print("=" * 60)
print("BASIC DATASET PROFILE")
print("=" * 60)

print(
    "\nShape:",
    df.shape
)

print(
    "\nStations:",
    df["stationID"].nunique()
)

print(
    "Spaces:",
    df["spaceID"].nunique()
)

print(
    "Sites:",
    df["siteID"].nunique()
)

print(
    "Users with available ID:",
    df["userID"].notna().sum()
)


# ============================================================
# 6. SUMMARY STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

summary_columns = [
    "kWhDelivered",
    "sessionDuration_min",
    "chargingDuration_min",
]


summary = df[
    summary_columns
].describe()

print(summary)


summary.to_csv(
    OUTPUT_DIR / "numerical_summary.csv"
)


# ============================================================
# 7. MISSING VALUE REPORT
# ============================================================

print("\n")
print("=" * 60)
print("MISSING VALUE REPORT")
print("=" * 60)

missing = (
    df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing = missing[
    missing > 0
]

print(missing)

missing.to_csv(
    OUTPUT_DIR / "missing_values.csv"
)


# ============================================================
# 8. SESSIONS PER DAY
# ============================================================

daily_sessions = (
    df.groupby(
        "connectionDate"
    )
    .size()
    .reset_index(
        name="sessions"
    )
)


daily_sessions.to_csv(
    OUTPUT_DIR / "daily_sessions.csv",
    index=False
)


plt.figure(
    figsize=(12, 5)
)

plt.plot(
    daily_sessions["connectionDate"],
    daily_sessions["sessions"]
)

plt.title(
    "EV Charging Sessions per Day"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Number of Sessions"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_daily_sessions.png",
    dpi=200
)

plt.close()


# ============================================================
# 9. DAILY ENERGY
# ============================================================

daily_energy = (
    df.groupby(
        "connectionDate"
    )["kWhDelivered"]
    .sum()
    .reset_index()
)


daily_energy.to_csv(
    OUTPUT_DIR / "daily_energy.csv",
    index=False
)


plt.figure(
    figsize=(12, 5)
)

plt.plot(
    daily_energy["connectionDate"],
    daily_energy["kWhDelivered"]
)

plt.title(
    "Daily EV Charging Energy Demand"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Energy Delivered (kWh)"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_daily_energy.png",
    dpi=200
)

plt.close()


# ============================================================
# 10. SESSIONS BY HOUR
# ============================================================

hourly_sessions = (
    df.groupby(
        "hour"
    )
    .size()
    .reset_index(
        name="sessions"
    )
)


hourly_sessions.to_csv(
    OUTPUT_DIR / "hourly_sessions.csv",
    index=False
)


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    hourly_sessions["hour"],
    hourly_sessions["sessions"]
)

plt.title(
    "EV Charging Sessions by Hour"
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Number of Sessions"
)

plt.xticks(
    range(0, 24)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_sessions_by_hour.png",
    dpi=200
)

plt.close()


# ============================================================
# 11. ENERGY BY HOUR
# ============================================================

hourly_energy = (
    df.groupby(
        "hour"
    )["kWhDelivered"]
    .sum()
    .reset_index()
)


hourly_energy.to_csv(
    OUTPUT_DIR / "hourly_energy.csv",
    index=False
)


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    hourly_energy["hour"],
    hourly_energy["kWhDelivered"]
)

plt.title(
    "EV Charging Energy Demand by Hour"
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Energy Delivered (kWh)"
)

plt.xticks(
    range(0, 24)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_energy_by_hour.png",
    dpi=200
)

plt.close()


# ============================================================
# 12. SESSIONS BY WEEKDAY
# ============================================================

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


weekday_sessions = (
    df["day_name"]
    .value_counts()
    .reindex(
        weekday_order
    )
)


weekday_sessions.to_csv(
    OUTPUT_DIR / "weekday_sessions.csv"
)


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    weekday_sessions.index,
    weekday_sessions.values
)

plt.title(
    "EV Charging Sessions by Day of Week"
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Number of Sessions"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_sessions_by_weekday.png",
    dpi=200
)

plt.close()


# ============================================================
# 13. WEEKDAY VS WEEKEND
# ============================================================

weekend_sessions = (
    df["is_weekend"]
    .value_counts()
)


weekend_sessions.to_csv(
    OUTPUT_DIR / "weekday_vs_weekend.csv"
)


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    weekend_sessions.index,
    weekend_sessions.values
)

plt.title(
    "Weekday vs Weekend Charging Sessions"
)

plt.xlabel(
    "Day Type"
)

plt.ylabel(
    "Number of Sessions"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_weekday_vs_weekend.png",
    dpi=200
)

plt.close()


# ============================================================
# 14. ENERGY DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.hist(
    df["kWhDelivered"].dropna(),
    bins=40
)

plt.title(
    "Distribution of Energy Delivered per Session"
)

plt.xlabel(
    "Energy Delivered (kWh)"
)

plt.ylabel(
    "Number of Sessions"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_energy_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 15. SESSION DURATION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.hist(
    df["sessionDuration_min"].dropna(),
    bins=40
)

plt.title(
    "Distribution of Session Duration"
)

plt.xlabel(
    "Session Duration (minutes)"
)

plt.ylabel(
    "Number of Sessions"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_session_duration_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 16. CHARGING DURATION DISTRIBUTION
# ============================================================

valid_charging = df[
    df["chargingDuration_min"] >= 0
][
    "chargingDuration_min"
]


plt.figure(
    figsize=(10, 5)
)

plt.hist(
    valid_charging.dropna(),
    bins=40
)

plt.title(
    "Distribution of Charging Duration"
)

plt.xlabel(
    "Charging Duration (minutes)"
)

plt.ylabel(
    "Number of Sessions"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_charging_duration_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 17. ENERGY VS CHARGING DURATION
# ============================================================

plot_df = df[
    (df["chargingDuration_min"] >= 0)
    &
    (df["kWhDelivered"] > 0)
].copy()


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    plot_df["chargingDuration_min"],
    plot_df["kWhDelivered"],
    alpha=0.4
)

plt.title(
    "Energy Delivered vs Charging Duration"
)

plt.xlabel(
    "Charging Duration (minutes)"
)

plt.ylabel(
    "Energy Delivered (kWh)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "10_energy_vs_charging_duration.png",
    dpi=200
)

plt.close()


# ============================================================
# 18. ENERGY VS SESSION DURATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    df["sessionDuration_min"],
    df["kWhDelivered"],
    alpha=0.4
)

plt.title(
    "Energy Delivered vs Session Duration"
)

plt.xlabel(
    "Session Duration (minutes)"
)

plt.ylabel(
    "Energy Delivered (kWh)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "11_energy_vs_session_duration.png",
    dpi=200
)

plt.close()


# ============================================================
# 19. STATION SESSION COUNT
# ============================================================

station_sessions = (
    df.groupby(
        "stationID"
    )
    .size()
    .sort_values(
        ascending=False
    )
    .reset_index(
        name="sessions"
    )
)


station_sessions.to_csv(
    OUTPUT_DIR / "station_sessions.csv",
    index=False
)


top_stations = (
    station_sessions
    .head(15)
    .sort_values(
        "sessions"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_stations["stationID"],
    top_stations["sessions"]
)

plt.title(
    "Top 15 Stations by Number of Sessions"
)

plt.xlabel(
    "Number of Sessions"
)

plt.ylabel(
    "Station ID"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "12_top_stations_by_sessions.png",
    dpi=200
)

plt.close()


# ============================================================
# 20. STATION ENERGY
# ============================================================

station_energy = (
    df.groupby(
        "stationID"
    )["kWhDelivered"]
    .sum()
    .sort_values(
        ascending=False
    )
    .reset_index()
)


station_energy.columns = [
    "stationID",
    "total_kWh"
]


station_energy.to_csv(
    OUTPUT_DIR / "station_energy.csv",
    index=False
)


top_energy_stations = (
    station_energy
    .head(15)
    .sort_values(
        "total_kWh"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_energy_stations["stationID"],
    top_energy_stations["total_kWh"]
)

plt.title(
    "Top 15 Stations by Energy Delivered"
)

plt.xlabel(
    "Total Energy (kWh)"
)

plt.ylabel(
    "Station ID"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "13_top_stations_by_energy.png",
    dpi=200
)

plt.close()


# ============================================================
# 21. STATION UTILIZATION PROXY
# ============================================================

# For now we calculate:
#
# sessions per station
#
# This is a demand/activity measure.
#
# It is NOT yet true utilization percentage because that
# requires a defined observation window and occupied time.

station_activity = (
    df.groupby(
        "stationID"
    )
    .agg(
        sessions=(
            "sessionID",
            "count"
        ),

        total_energy_kWh=(
            "kWhDelivered",
            "sum"
        ),

        avg_session_duration_min=(
            "sessionDuration_min",
            "mean"
        ),

        avg_charging_duration_min=(
            "chargingDuration_min",
            "mean"
        ),
    )
    .reset_index()
)


station_activity.to_csv(
    OUTPUT_DIR / "station_activity.csv",
    index=False
)


# ============================================================
# 22. DAILY DEMAND TABLE
# ============================================================

daily_demand = (
    df.groupby(
        "connectionDate"
    )
    .agg(
        sessions=(
            "sessionID",
            "count"
        ),

        total_energy_kWh=(
            "kWhDelivered",
            "sum"
        ),

        average_energy_kWh=(
            "kWhDelivered",
            "mean"
        ),

        average_session_duration_min=(
            "sessionDuration_min",
            "mean"
        ),
    )
    .reset_index()
)


daily_demand.to_csv(
    OUTPUT_DIR / "daily_demand.csv",
    index=False
)


# ============================================================
# 23. HOURLY DEMAND TABLE
# ============================================================

hourly_demand = (
    df.groupby(
        "hour"
    )
    .agg(
        sessions=(
            "sessionID",
            "count"
        ),

        total_energy_kWh=(
            "kWhDelivered",
            "sum"
        ),

        average_energy_kWh=(
            "kWhDelivered",
            "mean"
        ),

        average_session_duration_min=(
            "sessionDuration_min",
            "mean"
        ),
    )
    .reset_index()
)


hourly_demand.to_csv(
    OUTPUT_DIR / "hourly_demand.csv",
    index=False
)


# ============================================================
# 24. OUTLIER SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)


def iqr_outlier_report(
    data,
    column
):

    series = data[column].dropna()

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    outliers = series[
        (series < lower)
        |
        (series > upper)
    ]

    return {
        "column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": len(outliers),
        "outlier_percentage":
            len(outliers) / len(series) * 100
    }


outlier_results = []

for column in [
    "kWhDelivered",
    "sessionDuration_min",
    "chargingDuration_min",
]:

    outlier_results.append(
        iqr_outlier_report(
            df,
            column
        )
    )


outlier_df = pd.DataFrame(
    outlier_results
)


print(
    outlier_df
)


outlier_df.to_csv(
    OUTPUT_DIR / "outlier_report.csv",
    index=False
)


# ============================================================
# 25. RESEARCH-RELEVANT SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("RESEARCH-RELEVANT SUMMARY")
print("=" * 60)


peak_hour = (
    hourly_sessions
    .loc[
        hourly_sessions["sessions"].idxmax()
    ]
)


peak_energy_hour = (
    hourly_energy
    .loc[
        hourly_energy["kWhDelivered"].idxmax()
    ]
)


highest_session_station = (
    station_sessions.iloc[0]
)


highest_energy_station = (
    station_energy.iloc[0]
)


print(
    "\nPeak session hour:"
)

print(
    f"Hour {int(peak_hour['hour'])}: "
    f"{int(peak_hour['sessions'])} sessions"
)


print(
    "\nPeak energy hour:"
)

print(
    f"Hour {int(peak_energy_hour['hour'])}: "
    f"{peak_energy_hour['kWhDelivered']:.2f} kWh"
)


print(
    "\nMost active station:"
)

print(
    f"{highest_session_station['stationID']} "
    f"-> "
    f"{int(highest_session_station['sessions'])} sessions"
)


print(
    "\nHighest energy station:"
)

print(
    f"{highest_energy_station['stationID']} "
    f"-> "
    f"{highest_energy_station['total_kWh']:.2f} kWh"
)


# ============================================================
# 26. FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print(
    "\nEDA files saved in:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nGenerated plots:"
)

for file in sorted(
    OUTPUT_DIR.glob("*.png")
):

    print(
        " -",
        file.name
    )


print(
    "\nGenerated CSV reports:"
)

for file in sorted(
    OUTPUT_DIR.glob("*.csv")
):

    print(
        " -",
        file.name
    )


print("\n")
print("=" * 60)