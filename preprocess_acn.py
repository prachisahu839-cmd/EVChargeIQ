# ============================================================
# EVChargeIQ
# ACN-Data Final Preprocessing Pipeline
# ============================================================

import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

RAW_FILE = Path("data/raw/caltech_sessions.json")

PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = PROCESSED_DIR / "caltech_sessions.csv"


# ============================================================
# 2. START
# ============================================================

print("=" * 60)
print("EVChargeIQ - ACN DATA PREPROCESSING")
print("=" * 60)


# ============================================================
# 3. CHECK RAW FILE
# ============================================================

if not RAW_FILE.exists():

    raise FileNotFoundError(
        f"""
Raw dataset not found:

{RAW_FILE.resolve()}

Expected location:
data/raw/caltech_sessions.json
"""
    )


# ============================================================
# 4. LOAD RAW JSON
# ============================================================

print("\nLoading raw dataset...")

with open(
    RAW_FILE,
    "r",
    encoding="utf-8"
) as f:

    raw_data = json.load(f)


# ============================================================
# 5. HANDLE JSON STRUCTURE
# ============================================================

if isinstance(raw_data, list):

    sessions = raw_data

elif isinstance(raw_data, dict):

    if "sessions" in raw_data:

        sessions = raw_data["sessions"]

    elif "data" in raw_data:

        sessions = raw_data["data"]

    else:

        raise ValueError(
            """
Could not find session records in JSON.

Expected either:
- sessions
or
- data
"""
        )

else:

    raise ValueError(
        "Unexpected JSON structure."
    )


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(sessions)

print(
    f"Raw rows loaded: {len(df)}"
)

print(
    f"Raw columns: {len(df.columns)}"
)


# ============================================================
# 7. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "_id",
    "sessionID",
    "stationID",
    "spaceID",
    "siteID",
    "clusterID",
    "connectionTime",
    "disconnectTime",
    "doneChargingTime",
    "kWhDelivered",
    "timezone",
    "userID",
    "userInputs",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"""
Missing required columns:

{missing_columns}
"""
    )


# ============================================================
# 8. CREATE PROCESSED DIRECTORY
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. DUPLICATE SESSION CHECK
# ============================================================

print("\nChecking duplicate sessions...")

rows_before = len(df)


# userInputs contains nested dictionaries/lists.
# Therefore we do NOT use:
#
# df.drop_duplicates()
#
# Instead, sessionID is used as the session identifier.

duplicate_mask = df[
    "sessionID"
].duplicated(
    keep="first"
)


duplicate_sessions = int(
    duplicate_mask.sum()
)


print(
    f"Duplicate sessionIDs found: "
    f"{duplicate_sessions}"
)


# Remove duplicate sessions only if they exist

if duplicate_sessions > 0:

    df = df.loc[
        ~duplicate_mask
    ].copy()


rows_after = len(df)


print(
    f"Rows before duplicate removal: "
    f"{rows_before}"
)

print(
    f"Rows after duplicate removal: "
    f"{rows_after}"
)

print(
    f"Rows removed: "
    f"{rows_before - rows_after}"
)


# ============================================================
# 10. NUMERIC CONVERSION
# ============================================================

print("\nConverting numeric fields...")


df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"],
    errors="coerce"
)


# ============================================================
# 11. TIMESTAMP PROCESSING
# ============================================================

print("\nProcessing timestamps...")


# All three timestamps are converted to UTC.
#
# This puts them on the same time basis before calculating
# durations.

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


# ============================================================
# 12. TIMESTAMP MISSING VALUES
# ============================================================

missing_connection = int(
    df["connectionTime"].isna().sum()
)


missing_disconnect = int(
    df["disconnectTime"].isna().sum()
)


missing_done_charging = int(
    df["doneChargingTime"].isna().sum()
)


print("\nTimestamp missing values:")

print(
    f"connectionTime:   {missing_connection}"
)

print(
    f"disconnectTime:   {missing_disconnect}"
)

print(
    f"doneChargingTime: {missing_done_charging}"
)


# ============================================================
# 13. SESSION DURATION
# ============================================================

print(
    "\nCalculating session duration..."
)


df["sessionDuration_min"] = (

    df["disconnectTime"]
    -
    df["connectionTime"]

).dt.total_seconds() / 60


# ============================================================
# 14. CHARGING DURATION
# ============================================================

print(
    "Calculating charging duration..."
)


df["chargingDuration_min"] = (

    df["doneChargingTime"]
    -
    df["connectionTime"]

).dt.total_seconds() / 60


# ============================================================
# 15. POST-DISCONNECT DURATION
# ============================================================

print(
    "Calculating post-disconnect duration..."
)


df["postDisconnectDuration_min"] = (

    df["doneChargingTime"]
    -
    df["disconnectTime"]

).dt.total_seconds() / 60


# ============================================================
# 16. LOCAL CALTECH TIME
# ============================================================

print(
    "\nCreating Caltech local-time features..."
)


# UTC is retained for duration calculations.
#
# Local time is used only for calendar/time-of-day features.

local_connection = (
    df["connectionTime"]
    .dt
    .tz_convert(
        "America/Los_Angeles"
    )
)


# ============================================================
# 17. DATE FEATURES
# ============================================================

df["connectionDate"] = (
    local_connection.dt.date
)


df["hour"] = (
    local_connection.dt.hour
)


df["minute"] = (
    local_connection.dt.minute
)


df["day_of_week"] = (
    local_connection.dt.dayofweek
)


df["day_name"] = (
    local_connection.dt.day_name()
)


df["month"] = (
    local_connection.dt.month
)


df["year"] = (
    local_connection.dt.year
)


df["week_of_year"] = (
    local_connection
    .dt
    .isocalendar()
    .week
    .astype("Int64")
)


# ============================================================
# 18. WEEKDAY / WEEKEND
# ============================================================

df["is_weekend"] = df[
    "day_of_week"
].apply(
    lambda x:
        "Weekend"
        if pd.notna(x) and x >= 5
        else "Weekday"
)


# ============================================================
# 19. ENERGY FEATURES
# ============================================================

print(
    "\nCreating energy features..."
)


df["energy_per_charging_min"] = (

    df["kWhDelivered"]
    /
    df["chargingDuration_min"]

)


# Avoid infinity

df["energy_per_charging_min"] = (
    df["energy_per_charging_min"]
    .replace(
        [
            float("inf"),
            float("-inf")
        ],
        pd.NA
    )
)


# ============================================================
# 20. USER INPUT EXTRACTION
# ============================================================

print(
    "\nExtracting user input features..."
)


def get_user_input(value, key):

    """
    Safely extracts a value from userInputs.

    userInputs can contain:
    - dictionary
    - list
    - None
    - other structures
    """

    if isinstance(value, dict):

        return value.get(key)

    return None


df["WhPerMile"] = df[
    "userInputs"
].apply(
    lambda x:
        get_user_input(
            x,
            "WhPerMile"
        )
)


df["kWhRequested"] = df[
    "userInputs"
].apply(
    lambda x:
        get_user_input(
            x,
            "kWhRequested"
        )
)


df["milesRequested"] = df[
    "userInputs"
].apply(
    lambda x:
        get_user_input(
            x,
            "milesRequested"
        )
)


df["minutesAvailable"] = df[
    "userInputs"
].apply(
    lambda x:
        get_user_input(
            x,
            "minutesAvailable"
        )
)


df["requestedDeparture"] = df[
    "userInputs"
].apply(
    lambda x:
        get_user_input(
            x,
            "requestedDeparture"
        )
)


df["paymentRequired"] = df[
    "userInputs"
].apply(
    lambda x:
        get_user_input(
            x,
            "paymentRequired"
        )
)


# ============================================================
# 21. NUMERIC USER INPUTS
# ============================================================

numeric_user_columns = [

    "WhPerMile",

    "kWhRequested",

    "milesRequested",

    "minutesAvailable",

]


for column in numeric_user_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 22. REQUESTED DEPARTURE
# ============================================================

df["requestedDeparture"] = pd.to_datetime(
    df["requestedDeparture"],
    errors="coerce",
    utc=True
)


# ============================================================
# 23. DATA QUALITY FLAGS
# ============================================================

print(
    "\nCreating data quality flags..."
)


# ------------------------------------------------------------
# Negative session duration
# ------------------------------------------------------------

df["invalid_session_duration"] = (

    df["sessionDuration_min"] < 0

)


# ------------------------------------------------------------
# Negative charging duration
# ------------------------------------------------------------

df["invalid_charging_duration"] = (

    df["chargingDuration_min"] < 0

)


# ------------------------------------------------------------
# Missing charging duration
# ------------------------------------------------------------

df["missing_charging_duration"] = (

    df["chargingDuration_min"].isna()

)


# ------------------------------------------------------------
# Zero or negative energy
# ------------------------------------------------------------

df["zero_or_negative_energy"] = (

    df["kWhDelivered"] <= 0

)


# ------------------------------------------------------------
# Missing energy
# ------------------------------------------------------------

df["missing_energy"] = (

    df["kWhDelivered"].isna()

)


# ------------------------------------------------------------
# Charging before connection
# ------------------------------------------------------------

df["charging_before_connection"] = (

    df["doneChargingTime"]
    <
    df["connectionTime"]

)


# ------------------------------------------------------------
# Disconnect before connection
# ------------------------------------------------------------

df["disconnect_before_connection"] = (

    df["disconnectTime"]
    <
    df["connectionTime"]

)


# ------------------------------------------------------------
# Done charging after disconnect
#
# This is NOT automatically treated as an error.
#
# We preserve these records and flag them.
# ------------------------------------------------------------

df["charging_after_disconnect"] = (

    df["doneChargingTime"]
    >
    df["disconnectTime"]

)


# ============================================================
# 24. VALID CHARGING DURATION FLAG
# ============================================================

df["valid_charging_duration"] = (

    df["chargingDuration_min"].notna()
    &
    (df["chargingDuration_min"] >= 0)

)


# ============================================================
# 25. VALID SESSION DURATION FLAG
# ============================================================

df["valid_session_duration"] = (

    df["sessionDuration_min"].notna()
    &
    (df["sessionDuration_min"] >= 0)

)


# ============================================================
# 26. SORT DATA
# ============================================================

df = df.sort_values(
    by="connectionTime"
).reset_index(
    drop=True
)


# ============================================================
# 27. COLUMN ORGANIZATION
# ============================================================

preferred_columns = [

    # --------------------------------------------------------
    # Identification
    # --------------------------------------------------------

    "_id",

    "sessionID",

    "stationID",

    "spaceID",

    "siteID",

    "clusterID",

    "userID",


    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    "connectionTime",

    "disconnectTime",

    "doneChargingTime",

    "connectionDate",

    "hour",

    "minute",

    "day_of_week",

    "day_name",

    "week_of_year",

    "month",

    "year",

    "is_weekend",


    # --------------------------------------------------------
    # Charging
    # --------------------------------------------------------

    "kWhDelivered",

    "sessionDuration_min",

    "chargingDuration_min",

    "postDisconnectDuration_min",

    "energy_per_charging_min",


    # --------------------------------------------------------
    # User requirements
    # --------------------------------------------------------

    "WhPerMile",

    "kWhRequested",

    "milesRequested",

    "minutesAvailable",

    "requestedDeparture",

    "paymentRequired",


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    "timezone",


    # --------------------------------------------------------
    # Data quality flags
    # --------------------------------------------------------

    "valid_session_duration",

    "valid_charging_duration",

    "invalid_session_duration",

    "invalid_charging_duration",

    "missing_charging_duration",

    "zero_or_negative_energy",

    "missing_energy",

    "charging_before_connection",

    "disconnect_before_connection",

    "charging_after_disconnect",


    # --------------------------------------------------------
    # Original nested field
    # --------------------------------------------------------

    "userInputs",

]


# Keep only columns that exist

final_columns = [

    column

    for column in preferred_columns

    if column in df.columns

]


# Add remaining columns

remaining_columns = [

    column

    for column in df.columns

    if column not in final_columns

]


df = df[
    final_columns +
    remaining_columns
]


# ============================================================
# 28. VALIDATION REPORT
# ============================================================

print("\n")

print("=" * 60)

print("DATA VALIDATION REPORT")

print("=" * 60)


# ============================================================
# DATASET
# ============================================================

print(
    "\n========== DATASET =========="
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# ENERGY
# ============================================================

print(
    "\n========== ENERGY ANALYSIS =========="
)

print(
    df["kWhDelivered"].describe()
)


# ============================================================
# SESSION DURATION
# ============================================================

print(
    "\n========== SESSION DURATION =========="
)

print(
    df["sessionDuration_min"].describe()
)


# ============================================================
# CHARGING DURATION
# ============================================================

print(
    "\n========== CHARGING DURATION =========="
)

print(
    df["chargingDuration_min"].describe()
)


# ============================================================
# POST-DISCONNECT DURATION
# ============================================================

print(
    "\n========== POST-DISCONNECT DURATION =========="
)

print(
    df["postDisconnectDuration_min"].describe()
)


# ============================================================
# DURATION VALIDATION
# ============================================================

negative_session = int(
    (
        df["sessionDuration_min"] < 0
    ).sum()
)


negative_charging = int(
    (
        df["chargingDuration_min"] < 0
    ).sum()
)


missing_charging = int(
    df["chargingDuration_min"].isna().sum()
)


print(
    "\n========== DURATION VALIDATION =========="
)

print(
    f"Negative session duration: "
    f"{negative_session}"
)

print(
    f"Negative charging duration: "
    f"{negative_charging}"
)

print(
    f"Missing charging duration: "
    f"{missing_charging}"
)


# ============================================================
# TIMESTAMP LOGIC
# ============================================================

charging_before_connection = int(
    df["charging_before_connection"].sum()
)


disconnect_before_connection = int(
    df["disconnect_before_connection"].sum()
)


charging_after_disconnect = int(
    df["charging_after_disconnect"].sum()
)


print(
    "\n========== TIMESTAMP LOGIC =========="
)

print(
    f"Charging before connection: "
    f"{charging_before_connection}"
)

print(
    f"Disconnect before connection: "
    f"{disconnect_before_connection}"
)

print(
    f"Charging after disconnect: "
    f"{charging_after_disconnect}"
)


# ============================================================
# ENERGY VALIDATION
# ============================================================

zero_energy = int(
    df["zero_or_negative_energy"].sum()
)


missing_energy = int(
    df["missing_energy"].sum()
)


print(
    "\n========== ENERGY VALIDATION =========="
)

print(
    f"Zero/negative energy: "
    f"{zero_energy}"
)

print(
    f"Missing energy: "
    f"{missing_energy}"
)


# ============================================================
# DUPLICATES
# ============================================================

print(
    "\n========== DUPLICATES =========="
)

print(
    "Duplicate sessionIDs:",
    df["sessionID"].duplicated().sum()
)


# ============================================================
# MISSING VALUES
# ============================================================

print(
    "\n========== MISSING VALUES =========="
)

missing_values = (
    df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


missing_values = (
    missing_values[
        missing_values > 0
    ]
)


if len(missing_values) > 0:

    print(
        missing_values
    )

else:

    print(
        "No missing values."
    )


# ============================================================
# TIME RANGE
# ============================================================

print(
    "\n========== TIME RANGE =========="
)

print(
    "First connection:",
    df["connectionTime"].min()
)

print(
    "Last connection:",
    df["connectionTime"].max()
)


# ============================================================
# INFRASTRUCTURE
# ============================================================

print(
    "\n========== INFRASTRUCTURE =========="
)

print(
    "Unique stations:",
    df["stationID"].nunique()
)

print(
    "Unique spaces:",
    df["spaceID"].nunique()
)

print(
    "Unique sites:",
    df["siteID"].nunique()
)


# ============================================================
# WEEKDAY / WEEKEND
# ============================================================

print(
    "\n========== WEEKDAY / WEEKEND =========="
)

print(
    df["is_weekend"].value_counts()
)


# ============================================================
# SESSIONS BY HOUR
# ============================================================

print(
    "\n========== SESSIONS BY HOUR =========="
)

print(
    df["hour"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SESSIONS BY DAY
# ============================================================

print(
    "\n========== SESSIONS BY DAY =========="
)

print(
    df["day_name"]
    .value_counts()
)


# ============================================================
# TOP STATIONS
# ============================================================

print(
    "\n========== TOP STATIONS =========="
)

print(
    df["stationID"]
    .value_counts()
    .head(10)
)


# ============================================================
# FINAL ANOMALY COUNTS
# ============================================================

print(
    "\n========== DATA QUALITY FLAGS =========="
)

print(
    "Invalid session duration:",
    df["invalid_session_duration"].sum()
)

print(
    "Invalid charging duration:",
    df["invalid_charging_duration"].sum()
)

print(
    "Missing charging duration:",
    df["missing_charging_duration"].sum()
)

print(
    "Charging before connection:",
    df["charging_before_connection"].sum()
)

print(
    "Charging after disconnect:",
    df["charging_after_disconnect"].sum()
)


# ============================================================
# 29. SAVE PROCESSED DATA
# ============================================================

print("\n")

print("=" * 60)

print("SAVING PROCESSED DATA")

print("=" * 60)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nProcessed dataset saved at:"
)

print(
    OUTPUT_FILE.resolve()
)


# ============================================================
# 30. FINAL STATUS
# ============================================================

print("\n")

print("=" * 60)

print("PREPROCESSING COMPLETE")

print("=" * 60)

print(
    f"\nFinal rows: {len(df)}"
)

print(
    f"Final columns: {len(df.columns)}"
)

print(
    "\nImportant:"
)

print(
    "- No sessions were removed because of timestamp anomalies."
)

print(
    "- Invalid timestamps are flagged for later ML filtering."
)

print(
    "- doneChargingTime > disconnectTime is preserved."
)

print(
    "- Missing doneChargingTime remains missing."
)

print(
    "- Raw JSON remains untouched."
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE.resolve()
)

print(
    "\nNext stage: EDA and visualization."
)

print("=" * 60)