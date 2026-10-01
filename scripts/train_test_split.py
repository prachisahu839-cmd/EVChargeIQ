import pandas as pd
import os

# ============================================================
# EVChargeIQ - Chronological Train/Test Split
# ============================================================

INPUT = "data/processed/forecasting/hourly_forecasting_dataset.csv"
OUTPUT_DIR = "data/processed/forecasting"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("EVChargeIQ - CHRONOLOGICAL TRAIN/TEST SPLIT")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load forecasting dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

df["hour"] = pd.to_datetime(df["hour"])

print(f"\nTotal rows: {len(df)}")
print(f"Start: {df['hour'].min()}")
print(f"End:   {df['hour'].max()}")


# ------------------------------------------------------------
# 2. Sort chronologically
# ------------------------------------------------------------

df = df.sort_values("hour").reset_index(drop=True)


# ------------------------------------------------------------
# 3. 80/20 chronological split
# ------------------------------------------------------------

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


# ------------------------------------------------------------
# 4. Save datasets
# ------------------------------------------------------------

train_file = os.path.join(
    OUTPUT_DIR,
    "train_forecasting.csv"
)

test_file = os.path.join(
    OUTPUT_DIR,
    "test_forecasting.csv"
)

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)


# ------------------------------------------------------------
# 5. Print information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SPLIT COMPLETED")
print("=" * 60)

print(f"\nTotal rows : {len(df)}")
print(f"Train rows : {len(train_df)}")
print(f"Test rows  : {len(test_df)}")

print("\nTraining period:")
print(f"  {train_df['hour'].min()} → {train_df['hour'].max()}")

print("\nTesting period:")
print(f"  {test_df['hour'].min()} → {test_df['hour'].max()}")

print("\nTarget statistics:")

print("\nTraining - next hour sessions:")
print(
    train_df["target_sessions_next_hour"].describe()
)

print("\nTesting - next hour sessions:")
print(
    test_df["target_sessions_next_hour"].describe()
)

print("\nSaved files:")
print(f"  {train_file}")
print(f"  {test_file}")

print("\nDONE")