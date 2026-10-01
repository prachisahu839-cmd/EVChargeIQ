import os
import pandas as pd
import numpy as np


# ============================================================
# EVChargeIQ - Forecasting Dataset Creation
# ============================================================

INPUT_FILE = "data/processed/caltech_sessions.csv"
OUTPUT_DIR = "data/processed/forecasting"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. Load preprocessed session data
# ------------------------------------------------------------

print("=" * 60)
print("EVChargeIQ - FORECASTING DATASET CREATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
# Create valid charging duration feature if not already present
if "chargingDuration_min_valid" not in df.columns:

    if "chargingDuration_min" in df.columns:
        df["chargingDuration_min_valid"] = (
            pd.to_numeric(
                df["chargingDuration_min"],
                errors="coerce"
            )
        )

        # Invalid negative durations are excluded
        df.loc[
            df["chargingDuration_min_valid"] < 0,
            "chargingDuration_min_valid"
        ] = np.nan

    else:
        print("WARNING: charging duration column not found.")
        df["chargingDuration_min_valid"] = np.nan

print(f"\nInput rows: {len(df)}")
print(f"Input columns: {len(df.columns)}")


# ------------------------------------------------------------
# 2. Parse timestamps
# ------------------------------------------------------------

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    utc=True,
    errors="coerce"
)

df["connectionTime_local"] = (
    df["connectionTime"]
    .dt.tz_convert("America/Los_Angeles")
)


# ------------------------------------------------------------
# 3. Create hourly time index
# ------------------------------------------------------------

df["hour"] = df["connectionTime_local"].dt.floor("h")


# ------------------------------------------------------------
# 4. Aggregate charging sessions by hour
# ------------------------------------------------------------

hourly = (
    df.groupby("hour")
      .agg(
          session_count=("sessionID", "count"),
          energy_kWh=("kWhDelivered", "sum"),
          avg_energy_kWh=("kWhDelivered", "mean"),
          avg_session_duration_min=("sessionDuration_min", "mean"),
          avg_charging_duration_min=("chargingDuration_min_valid", "mean"),
          unique_stations=("stationID", "nunique")
      )
      .reset_index()
)


# ------------------------------------------------------------
# 5. Create continuous hourly timeline
# ------------------------------------------------------------

start_time = hourly["hour"].min()
end_time = hourly["hour"].max()

full_hours = pd.date_range(
    start=start_time,
    end=end_time,
    freq="h",
    tz=start_time.tz
)

hourly = (
    hourly
    .set_index("hour")
    .reindex(full_hours)
    .rename_axis("hour")
    .reset_index()
)


# ------------------------------------------------------------
# 6. Fill demand values for hours with no sessions
# ------------------------------------------------------------

hourly["session_count"] = (
    hourly["session_count"]
    .fillna(0)
)

hourly["energy_kWh"] = (
    hourly["energy_kWh"]
    .fillna(0)
)

# These are descriptive variables.
# Do not assume zero means the actual average was zero.
hourly["avg_energy_kWh"] = (
    hourly["avg_energy_kWh"]
    .fillna(0)
)

hourly["avg_session_duration_min"] = (
    hourly["avg_session_duration_min"]
    .fillna(0)
)

hourly["avg_charging_duration_min"] = (
    hourly["avg_charging_duration_min"]
    .fillna(0)
)

hourly["unique_stations"] = (
    hourly["unique_stations"]
    .fillna(0)
)


# ------------------------------------------------------------
# 7. Calendar features
# ------------------------------------------------------------

hourly["hour_of_day"] = hourly["hour"].dt.hour

hourly["day_of_week"] = hourly["hour"].dt.dayofweek

hourly["day_of_month"] = hourly["hour"].dt.day

hourly["month"] = hourly["hour"].dt.month

hourly["is_weekend"] = (
    hourly["day_of_week"] >= 5
).astype(int)


# ------------------------------------------------------------
# 8. Cyclical time features
# ------------------------------------------------------------

hourly["hour_sin"] = np.sin(
    2 * np.pi * hourly["hour_of_day"] / 24
)

hourly["hour_cos"] = np.cos(
    2 * np.pi * hourly["hour_of_day"] / 24
)

hourly["dow_sin"] = np.sin(
    2 * np.pi * hourly["day_of_week"] / 7
)

hourly["dow_cos"] = np.cos(
    2 * np.pi * hourly["day_of_week"] / 7
)


# ------------------------------------------------------------
# 9. Lag features
# ------------------------------------------------------------

# Previous hour
hourly["sessions_lag_1h"] = (
    hourly["session_count"].shift(1)
)

hourly["energy_lag_1h"] = (
    hourly["energy_kWh"].shift(1)
)

# Same hour previous day
hourly["sessions_lag_24h"] = (
    hourly["session_count"].shift(24)
)

hourly["energy_lag_24h"] = (
    hourly["energy_kWh"].shift(24)
)

# Same hour previous week
hourly["sessions_lag_168h"] = (
    hourly["session_count"].shift(168)
)

hourly["energy_lag_168h"] = (
    hourly["energy_kWh"].shift(168)
)


# ------------------------------------------------------------
# 10. Rolling historical features
# ------------------------------------------------------------

# IMPORTANT:
# shift(1) ensures that the current target is NOT included.

hourly["sessions_rolling_24h"] = (
    hourly["session_count"]
    .shift(1)
    .rolling(24)
    .mean()
)

hourly["energy_rolling_24h"] = (
    hourly["energy_kWh"]
    .shift(1)
    .rolling(24)
    .mean()
)

hourly["sessions_rolling_168h"] = (
    hourly["session_count"]
    .shift(1)
    .rolling(168)
    .mean()
)

hourly["energy_rolling_168h"] = (
    hourly["energy_kWh"]
    .shift(1)
    .rolling(168)
    .mean()
)


# ------------------------------------------------------------
# 11. Rolling standard deviation
# ------------------------------------------------------------

hourly["sessions_std_24h"] = (
    hourly["session_count"]
    .shift(1)
    .rolling(24)
    .std()
)

hourly["energy_std_24h"] = (
    hourly["energy_kWh"]
    .shift(1)
    .rolling(24)
    .std()
)


# ------------------------------------------------------------
# 12. Target variables
# ------------------------------------------------------------

# Main target
hourly["target_sessions_next_hour"] = (
    hourly["session_count"].shift(-1)
)

# Alternative target
hourly["target_energy_next_hour"] = (
    hourly["energy_kWh"].shift(-1)
)


# ------------------------------------------------------------
# 13. Remove rows where lag features are unavailable
# ------------------------------------------------------------

forecast_df = hourly.dropna(
    subset=[
        "sessions_lag_1h",
        "sessions_lag_24h",
        "sessions_lag_168h",
        "energy_lag_1h",
        "energy_lag_24h",
        "energy_lag_168h",
        "sessions_rolling_24h",
        "energy_rolling_24h",
        "sessions_rolling_168h",
        "energy_rolling_168h",
        "target_sessions_next_hour",
        "target_energy_next_hour"
    ]
).copy()


# ------------------------------------------------------------
# 14. Save dataset
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "hourly_forecasting_dataset.csv"
)

forecast_df.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 15. Create summary
# ------------------------------------------------------------

summary_file = os.path.join(
    OUTPUT_DIR,
    "forecasting_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("EVChargeIQ - Forecasting Dataset Summary\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Original sessions: {len(df)}\n")
    f.write(f"Hourly rows before feature removal: {len(hourly)}\n")
    f.write(f"Final forecasting rows: {len(forecast_df)}\n")

    f.write(
        f"Time range: {hourly['hour'].min()} "
        f"to {hourly['hour'].max()}\n"
    )

    f.write(
        f"Number of stations: {df['stationID'].nunique()}\n"
    )

    f.write(
        f"Average hourly sessions: "
        f"{hourly['session_count'].mean():.2f}\n"
    )

    f.write(
        f"Average hourly energy: "
        f"{hourly['energy_kWh'].mean():.2f} kWh\n"
    )

    f.write("\nTarget variables:\n")
    f.write("- target_sessions_next_hour\n")
    f.write("- target_energy_next_hour\n")

    f.write("\nImportant lag features:\n")
    f.write("- lag 1 hour\n")
    f.write("- lag 24 hours\n")
    f.write("- lag 168 hours (7 days)\n")

    f.write("\nRolling features:\n")
    f.write("- previous 24-hour mean\n")
    f.write("- previous 168-hour mean\n")
    f.write("- previous 24-hour standard deviation\n")

    f.write("\nNo current-target leakage is intentionally introduced.\n")


# ------------------------------------------------------------
# 16. Print results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FORECASTING DATASET CREATED")
print("=" * 60)

print(f"\nOriginal sessions       : {len(df)}")
print(f"Hourly rows             : {len(hourly)}")
print(f"Final forecasting rows  : {len(forecast_df)}")

print(
    f"\nTime range              : "
    f"{hourly['hour'].min()} → {hourly['hour'].max()}"
)

print(
    f"Average hourly sessions : "
    f"{hourly['session_count'].mean():.2f}"
)

print(
    f"Average hourly energy   : "
    f"{hourly['energy_kWh'].mean():.2f} kWh"
)

print("\nTarget columns:")
print("  target_sessions_next_hour")
print("  target_energy_next_hour")

print("\nImportant features:")
print("  sessions_lag_1h")
print("  sessions_lag_24h")
print("  sessions_lag_168h")
print("  energy_lag_1h")
print("  energy_lag_24h")
print("  energy_lag_168h")
print("  sessions_rolling_24h")
print("  sessions_rolling_168h")
print("  energy_rolling_24h")
print("  energy_rolling_168h")

print(f"\nSaved:")
print(f"  {output_file}")
print(f"  {summary_file}")

print("\nFirst 5 rows:")
print(
    forecast_df[
        [
            "hour",
            "session_count",
            "energy_kWh",
            "sessions_lag_1h",
            "sessions_lag_24h",
            "sessions_lag_168h",
            "target_sessions_next_hour"
        ]
    ].head()
)

print("\nDone.")