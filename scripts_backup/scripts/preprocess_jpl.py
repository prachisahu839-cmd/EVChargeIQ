
import json
from pathlib import Path

import pandas as pd

# Input and output paths
input_path = Path("data/raw/jpl_sessions.json")
output_dir = Path("data/processed/jpl")
output_dir.mkdir(parents=True, exist_ok=True)

# Load raw JPL sessions
with input_path.open("r", encoding="utf-8") as file:
    df = pd.DataFrame(json.load(file))

print("Raw sessions:", len(df))

# Parse timestamps consistently in UTC
for col in ["connectionTime", "disconnectTime"]:
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"], errors="coerce"
)

# Remove records missing essential forecasting fields
df = df.dropna(
    subset=["sessionID", "connectionTime",
            "disconnectTime", "kWhDelivered"]
).copy()

# Keep valid durations and non-negative energy
df["session_duration_min"] = (
    df["disconnectTime"] - df["connectionTime"]
).dt.total_seconds() / 60

df = df[
    (df["session_duration_min"] > 0)
    & (df["kWhDelivered"] >= 0)
].copy()

# Convert connection time to the site's local timezone
df["connectionTime_local"] = (
    df["connectionTime"].dt.tz_convert("America/Los_Angeles")
)

# Create hourly session and energy summaries
df["hour"] = df["connectionTime_local"].dt.floor("h")

hourly = (
    df.groupby("hour")
    .agg(
        sessions=("sessionID", "nunique"),
        energy_kwh=("kWhDelivered", "sum"),
    )
    .sort_index()
)

# Fill missing hours so the time series is continuous
full_hours = pd.date_range(
    start=hourly.index.min(),
    end=hourly.index.max(),
    freq="h",
)

hourly = hourly.reindex(full_hours, fill_value=0)
hourly.index.name = "hour"

# Save cleaned sessions and hourly dataset
df.to_csv(output_dir / "jpl_sessions_cleaned.csv", index=False)
hourly.reset_index().to_csv(
    output_dir / "jpl_hourly_dataset.csv", index=False
)

# Save a compact quality report
report = [
    f"Valid sessions: {len(df)}",
    f"Unique session IDs: {df['sessionID'].nunique()}",
    f"Hourly rows: {len(hourly)}",
    f"Date range: {hourly.index.min()} to {hourly.index.max()}",
    f"Missing hourly values: {int(hourly.isna().sum().sum())}",
]

(output_dir / "quality_report.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n".join(report))
print("\nSaved files in:", output_dir)
