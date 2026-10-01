"""
EVChargeIQ
Forecasting Leakage Audit

Purpose:
Check whether the forecasting dataset contains obvious
target leakage or temporal leakage before advanced
forecasting models are trained.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "hourly_forecasting_dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "forecasting_leakage_audit.txt"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("EVChargeIQ - FORECASTING LEAKAGE AUDIT")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"\nRows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 3. DATETIME CHECK
# ============================================================

print("\n" + "=" * 70)
print("1. DATETIME CHECK")
print("=" * 70)

if "hour" not in df.columns:
    print("ERROR: 'hour' column not found.")
    raise SystemExit

df["hour"] = pd.to_datetime(
    df["hour"],
    errors="coerce"
)

missing_dates = df["hour"].isna().sum()

print(f"Missing datetime values : {missing_dates}")

is_sorted = df["hour"].is_monotonic_increasing

print(f"Chronologically sorted   : {is_sorted}")

duplicate_hours = df["hour"].duplicated().sum()

print(f"Duplicate hours          : {duplicate_hours}")

print(f"First timestamp          : {df['hour'].min()}")
print(f"Last timestamp           : {df['hour'].max()}")


# ============================================================
# 4. TARGET CHECK
# ============================================================

print("\n" + "=" * 70)
print("2. TARGET CHECK")
print("=" * 70)

TARGETS = [
    "target_sessions_next_hour",
    "target_energy_next_hour"
]

for target in TARGETS:

    if target not in df.columns:
        print(f"ERROR: Missing target: {target}")
    else:
        print(
            f"{target}: "
            f"min={df[target].min()}, "
            f"max={df[target].max()}, "
            f"missing={df[target].isna().sum()}"
        )


# ============================================================
# 5. TARGET LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("3. TARGET LEAKAGE CHECK")
print("=" * 70)

numeric_features = [
    column
    for column in df.columns
    if column not in TARGETS
    and column != "hour"
    and pd.api.types.is_numeric_dtype(df[column])
]

print(f"Numeric feature count: {len(numeric_features)}")

print("\nChecking whether target columns are included as features...")

for target in TARGETS:

    if target in numeric_features:
        print(
            f"WARNING: {target} appears in feature set."
        )
    else:
        print(
            f"OK: {target} excluded from feature set."
        )


# ============================================================
# 6. FEATURE NAME AUDIT
# ============================================================

print("\n" + "=" * 70)
print("4. FEATURE NAME AUDIT")
print("=" * 70)

future_keywords = [
    "future",
    "next",
    "target"
]

suspicious_features = []

for feature in numeric_features:

    feature_lower = feature.lower()

    for keyword in future_keywords:

        if keyword in feature_lower:

            suspicious_features.append(
                (feature, keyword)
            )

            break


if suspicious_features:

    print("Potentially suspicious feature names:")

    for feature, keyword in suspicious_features:

        print(
            f"  {feature} "
            f"(contains '{keyword}')"
        )

else:

    print(
        "No suspicious future/target feature names found."
    )


# ============================================================
# 7. LAG FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("5. LAG FEATURE CHECK")
print("=" * 70)

lag_features = [
    column
    for column in df.columns
    if "_lag_" in column
]

if lag_features:

    print("Lag features found:")

    for feature in lag_features:

        print(
            f"  {feature}"
        )

else:

    print("No lag features found.")


# ============================================================
# 8. ROLLING FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("6. ROLLING FEATURE CHECK")
print("=" * 70)

rolling_features = [
    column
    for column in df.columns
    if "rolling" in column.lower()
]

if rolling_features:

    print("Rolling features found:")

    for feature in rolling_features:

        print(
            f"  {feature}"
        )

else:

    print("No rolling features found.")


# ============================================================
# 9. SUSPICIOUS RAW FEATURES
# ============================================================

print("\n" + "=" * 70)
print("7. RAW FEATURE CHECK")
print("=" * 70)

raw_time_features = [
    column
    for column in df.columns
    if any(
        word in column.lower()
        for word in [
            "disconnect",
            "chargingduration",
            "donecharging",
            "sessionduration"
        ]
    )
]

if raw_time_features:

    print(
        "Time/session-related features found:"
    )

    for feature in raw_time_features:

        print(
            f"  {feature}"
        )

else:

    print(
        "No raw session-time features detected."
    )


# ============================================================
# 10. TRAIN / TEST TEMPORAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("8. CHRONOLOGICAL SPLIT CHECK")
print("=" * 70)

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

train_last = train["hour"].max()
test_first = test["hour"].min()

print(f"Training rows : {len(train)}")
print(f"Testing rows  : {len(test)}")

print(f"Training end  : {train_last}")
print(f"Testing start : {test_first}")

if train_last < test_first:

    print(
        "OK: Training period ends before test period."
    )

else:

    print(
        "WARNING: Possible temporal overlap!"
    )


# ============================================================
# 11. GAP CHECK
# ============================================================

print("\n" + "=" * 70)
print("9. TIME CONTINUITY CHECK")
print("=" * 70)

time_diff = (
    df["hour"]
    .sort_values()
    .diff()
    .dropna()
)

one_hour = pd.Timedelta(hours=1)

continuous_points = (
    time_diff == one_hour
).sum()

gaps = (
    time_diff > one_hour
).sum()

print(
    f"Consecutive 1-hour intervals : "
    f"{continuous_points}"
)

print(
    f"Detected time gaps            : "
    f"{gaps}"
)

if gaps > 0:

    print(
        "NOTE: Time gaps exist. "
        "This is not automatically leakage, "
        "but should be considered during forecasting."
    )

else:

    print(
        "OK: No gaps detected."
    )


# ============================================================
# 12. CORRELATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. TARGET CORRELATION CHECK")
print("=" * 70)

for target in TARGETS:

    correlations = (
        df[numeric_features + [target]]
        .corr(numeric_only=True)[target]
        .drop(target)
        .abs()
        .sort_values(
            ascending=False
        )
    )

    print(
        f"\nTop correlations with {target}:"
    )

    print(
        correlations.head(10).to_string()
    )


# ============================================================
# 13. FINAL ASSESSMENT
# ============================================================

print("\n" + "=" * 70)
print("11. AUDIT SUMMARY")
print("=" * 70)

issues = []

if missing_dates > 0:
    issues.append(
        "Missing datetime values detected."
    )

if not is_sorted:
    issues.append(
        "Dataset is not chronologically sorted."
    )

if duplicate_hours > 0:
    issues.append(
        "Duplicate hourly timestamps detected."
    )

if suspicious_features:
    issues.append(
        "Potentially suspicious feature names detected."
    )

if train_last >= test_first:
    issues.append(
        "Potential temporal overlap between train and test."
    )


if issues:

    print(
        "\nPotential issues requiring review:"
    )

    for issue in issues:

        print(
            f"  - {issue}"
        )

else:

    print(
        "\nNo obvious structural leakage detected."
    )

print(
    "\nIMPORTANT:"
)

print(
    "This script checks for obvious leakage patterns."
)

print(
    "It does NOT mathematically prove that every "
    "feature is leakage-free."
)


# ============================================================
# 14. SAVE REPORT
# ============================================================

report_lines = []

report_lines.append(
    "EVChargeIQ - Forecasting Leakage Audit"
)

report_lines.append(
    "=" * 60
)

report_lines.append(
    f"Rows: {len(df)}"
)

report_lines.append(
    f"Columns: {len(df.columns)}"
)

report_lines.append(
    f"Chronologically sorted: {is_sorted}"
)

report_lines.append(
    f"Duplicate hours: {duplicate_hours}"
)

report_lines.append(
    f"Training rows: {len(train)}"
)

report_lines.append(
    f"Testing rows: {len(test)}"
)

report_lines.append(
    f"Training end: {train_last}"
)

report_lines.append(
    f"Testing start: {test_first}"
)

report_lines.append(
    f"Detected gaps: {gaps}"
)

if issues:

    report_lines.append(
        "\nPotential issues:"
    )

    report_lines.extend(
        [
            f"- {issue}"
            for issue in issues
        ]
    )

else:

    report_lines.append(
        "\nNo obvious structural leakage detected."
    )


REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


print(
    f"\nAudit report saved to:"
)

print(
    REPORT_FILE
)

print("\nAudit completed.")