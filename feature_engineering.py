# ============================================================
# EVChargeIQ
# Feature Engineering
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/caltech_sessions.csv"
)

OUTPUT_DIR = Path(
    "data/processed/features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "caltech_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("EVChargeIQ - FEATURE ENGINEERING")
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
# 3. DATETIME
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
# 4. BASIC NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "kWhDelivered",
    "sessionDuration_min",
    "chargingDuration_min",
    "postDisconnectDuration_min",
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
# 5. TIME FEATURES
# ============================================================

print("\nCreating temporal features...")


# Hour as cyclical feature
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)


# Day of week as cyclical feature
df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


# Month as cyclical feature
df["month_sin"] = np.sin(
    2 * np.pi * (df["month"] - 1) / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * (df["month"] - 1) / 12
)


# ============================================================
# 6. TIME-OF-DAY CATEGORY
# ============================================================

def time_period(hour):

    if pd.isna(hour):
        return "Unknown"

    if 0 <= hour < 6:
        return "Night"

    if 6 <= hour < 12:
        return "Morning"

    if 12 <= hour < 17:
        return "Afternoon"

    if 17 <= hour < 21:
        return "Evening"

    return "Late Evening"


df["time_period"] = (
    df["hour"]
    .apply(time_period)
)


# ============================================================
# 7. VALID CHARGING DURATION
# ============================================================

print(
    "Creating valid charging-duration features..."
)


# Preserve original chargingDuration_min.
#
# Invalid negative values are not deleted.
# They are converted to NaN ONLY for features that require
# a physically valid duration.

df["chargingDuration_min_valid"] = (
    df["chargingDuration_min"]
    .where(
        df["chargingDuration_min"] >= 0
    )
)


# ============================================================
# 8. ENERGY RATE
# ============================================================

df["energy_per_min_valid"] = (

    df["kWhDelivered"]
    /
    df["chargingDuration_min_valid"]

)


df["energy_per_min_valid"] = (
    df["energy_per_min_valid"]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


# Energy per hour
df["average_charging_power_kW"] = (

    df["kWhDelivered"]
    /
    (
        df["chargingDuration_min_valid"]
        / 60
    )

)


df["average_charging_power_kW"] = (
    df["average_charging_power_kW"]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


# ============================================================
# 9. SESSION OCCUPANCY PROXY
# ============================================================

# Session duration represents how long the EV remained
# connected according to the session timestamps.
#
# This is useful as an occupancy/usage proxy.
#
# It is NOT yet a true station utilization percentage.

df["occupancy_duration_min"] = (
    df["sessionDuration_min"]
)


# ============================================================
# 10. ENERGY PER SESSION HOUR
# ============================================================

df["energy_per_session_hour"] = (

    df["kWhDelivered"]
    /
    (
        df["sessionDuration_min"]
        / 60
    )

)


df["energy_per_session_hour"] = (
    df["energy_per_session_hour"]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


# ============================================================
# 11. STATION HISTORICAL FEATURES
# ============================================================

print(
    "Creating station-level features..."
)


# Number of sessions handled by each station
station_session_count = (
    df.groupby(
        "stationID"
    )["sessionID"]
    .transform("count")
)


df["station_total_sessions"] = (
    station_session_count
)


# Total energy delivered by station
station_total_energy = (
    df.groupby(
        "stationID"
    )["kWhDelivered"]
    .transform("sum")
)


df["station_total_energy_kWh"] = (
    station_total_energy
)


# Average station energy per session
station_avg_energy = (
    df.groupby(
        "stationID"
    )["kWhDelivered"]
    .transform("mean")
)


df["station_avg_energy_kWh"] = (
    station_avg_energy
)


# Average session duration at station
station_avg_duration = (
    df.groupby(
        "stationID"
    )["sessionDuration_min"]
    .transform("mean")
)


df["station_avg_session_duration_min"] = (
    station_avg_duration
)


# ============================================================
# 12. DAILY DEMAND FEATURES
# ============================================================

print(
    "Creating daily demand features..."
)


daily_sessions = (
    df.groupby(
        "connectionDate"
    )["sessionID"]
    .transform("count")
)


df["daily_session_count"] = (
    daily_sessions
)


daily_energy = (
    df.groupby(
        "connectionDate"
    )["kWhDelivered"]
    .transform("sum")
)


df["daily_energy_kWh"] = (
    daily_energy
)


daily_average_energy = (
    df.groupby(
        "connectionDate"
    )["kWhDelivered"]
    .transform("mean")
)


df["daily_avg_energy_kWh"] = (
    daily_average_energy
)


# ============================================================
# 13. HOURLY DEMAND FEATURES
# ============================================================

print(
    "Creating hourly demand features..."
)


# IMPORTANT:
# We use hour as a general demand pattern.
# These are descriptive features, not future information
# for a forecasting model.

hourly_sessions = (
    df.groupby(
        "hour"
    )["sessionID"]
    .transform("count")
)


df["historical_hour_session_count"] = (
    hourly_sessions
)


hourly_energy = (
    df.groupby(
        "hour"
    )["kWhDelivered"]
    .transform("sum")
)


df["historical_hour_energy_kWh"] = (
    hourly_energy
)


# ============================================================
# 14. SESSION ENERGY CATEGORY
# ============================================================

def energy_category(value):

    if pd.isna(value):
        return "Unknown"

    if value < 5:
        return "Low"

    if value < 15:
        return "Medium"

    if value < 25:
        return "High"

    return "Very High"


df["energy_category"] = (
    df["kWhDelivered"]
    .apply(energy_category)
)


# ============================================================
# 15. SESSION DURATION CATEGORY
# ============================================================

def duration_category(value):

    if pd.isna(value):
        return "Unknown"

    if value < 60:
        return "Short"

    if value < 240:
        return "Medium"

    if value < 480:
        return "Long"

    return "Very Long"


df["session_duration_category"] = (
    df["sessionDuration_min"]
    .apply(duration_category)
)


# ============================================================
# 16. ANOMALY-READY FEATURES
# ============================================================

df["is_duration_anomaly"] = (

    (
        df["chargingDuration_min"]
        < 0
    )

    |

    df["chargingDuration_min"].isna()

)


df["is_temporal_anomaly"] = (

    df["charging_before_connection"]

    |

    df["disconnect_before_connection"]

)


# ============================================================
# 17. SORT
# ============================================================

df = df.sort_values(
    "connectionTime"
).reset_index(
    drop=True
)


# ============================================================
# 18. SAVE
# ============================================================

print(
    "\nSaving feature dataset..."
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. REPORT
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE ENGINEERING REPORT")
print("=" * 60)

print(
    f"\nRows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    "\nNew feature examples:"
)

new_features = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "time_period",
    "chargingDuration_min_valid",
    "energy_per_min_valid",
    "average_charging_power_kW",
    "occupancy_duration_min",
    "station_total_sessions",
    "station_total_energy_kWh",
    "station_avg_energy_kWh",
    "daily_session_count",
    "daily_energy_kWh",
    "historical_hour_session_count",
    "historical_hour_energy_kWh",
    "energy_category",
    "session_duration_category",
    "is_duration_anomaly",
    "is_temporal_anomaly",
]


for feature in new_features:

    print(
        " -",
        feature
    )


print(
    "\nFeature dataset saved at:"
)

print(
    OUTPUT_FILE.resolve()
)


print("\n")
print("=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)