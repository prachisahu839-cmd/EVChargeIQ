import pandas as pd
import numpy as np
import os

# ============================================================
# EVChargeIQ - Baseline Forecasting Models
# ============================================================

INPUT = "data/processed/forecasting/hourly_forecasting_dataset.csv"
OUTPUT_DIR = "data/processed/forecasting"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("EVChargeIQ - BASELINE FORECASTING")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

df["hour"] = pd.to_datetime(df["hour"])

df = df.sort_values("hour").reset_index(drop=True)

print(f"\nTotal rows: {len(df)}")


# ------------------------------------------------------------
# 2. Chronological 80/20 split
# ------------------------------------------------------------

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(f"Training rows: {len(train_df)}")
print(f"Testing rows : {len(test_df)}")


# ------------------------------------------------------------
# 3. Define evaluation metrics
# ------------------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    # MAPE can become problematic when actual = 0.
    # Therefore, calculate it only for non-zero actual values.
    non_zero = actual != 0

    if np.sum(non_zero) > 0:
        mape = np.mean(
            np.abs(
                (actual[non_zero] - predicted[non_zero])
                / actual[non_zero]
            )
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


# ============================================================
# PART A - SESSION DEMAND FORECASTING
# ============================================================

print("\n" + "=" * 60)
print("SESSION DEMAND FORECASTING")
print("=" * 60)

actual = test_df["target_sessions_next_hour"].values


# ------------------------------------------------------------
# Model 1 - Naive
# Next hour = current hour
# ------------------------------------------------------------

naive_prediction = test_df["session_count"].values

mae_naive, rmse_naive, mape_naive = calculate_metrics(
    actual,
    naive_prediction
)


# ------------------------------------------------------------
# Model 2 - 24-hour Seasonal Naive
# Next hour = same hour yesterday
# ------------------------------------------------------------

seasonal_24_prediction = df["session_count"].shift(23).iloc[split_index:].values

mae_24, rmse_24, mape_24 = calculate_metrics(
    actual,
    seasonal_24_prediction
)


# ------------------------------------------------------------
# Model 3 - 7-day Seasonal Naive
# Next hour = same hour last week
# ------------------------------------------------------------

seasonal_168_prediction = df["session_count"].shift(167).iloc[split_index:].values

mae_168, rmse_168, mape_168 = calculate_metrics(
    actual,
    seasonal_168_prediction
)


# ------------------------------------------------------------
# Print session results
# ------------------------------------------------------------

session_results = pd.DataFrame({

    "Model": [
        "Naive",
        "Seasonal Naive - 24h",
        "Seasonal Naive - 168h"
    ],

    "MAE": [
        mae_naive,
        mae_24,
        mae_168
    ],

    "RMSE": [
        rmse_naive,
        rmse_24,
        rmse_168
    ],

    "MAPE_percent": [
        mape_naive,
        mape_24,
        mape_168
    ]
})

print("\nSession Forecasting Results:")
print(
    session_results.to_string(index=False)
)


# ============================================================
# PART B - ENERGY DEMAND FORECASTING
# ============================================================

print("\n" + "=" * 60)
print("ENERGY DEMAND FORECASTING")
print("=" * 60)

actual_energy = test_df[
    "target_energy_next_hour"
].values


# ------------------------------------------------------------
# Model 1 - Naive
# ------------------------------------------------------------

naive_energy_prediction = test_df[
    "energy_kWh"
].values

mae_energy_naive, rmse_energy_naive, mape_energy_naive = (
    calculate_metrics(
        actual_energy,
        naive_energy_prediction
    )
)


# ------------------------------------------------------------
# Model 2 - 24-hour Seasonal Naive
# ------------------------------------------------------------

seasonal_energy_24_prediction = df["energy_kWh"].shift(23).iloc[split_index:].values

mae_energy_24, rmse_energy_24, mape_energy_24 = (
    calculate_metrics(
        actual_energy,
        seasonal_energy_24_prediction
    )
)


# ------------------------------------------------------------
# Model 3 - 7-day Seasonal Naive
# ------------------------------------------------------------

seasonal_energy_168_prediction = df["energy_kWh"].shift(167).iloc[split_index:].values

mae_energy_168, rmse_energy_168, mape_energy_168 = (
    calculate_metrics(
        actual_energy,
        seasonal_energy_168_prediction
    )
)


# ------------------------------------------------------------
# Print energy results
# ------------------------------------------------------------

energy_results = pd.DataFrame({

    "Model": [
        "Naive",
        "Seasonal Naive - 24h",
        "Seasonal Naive - 168h"
    ],

    "MAE": [
        mae_energy_naive,
        mae_energy_24,
        mae_energy_168
    ],

    "RMSE": [
        rmse_energy_naive,
        rmse_energy_24,
        rmse_energy_168
    ],

    "MAPE_percent": [
        mape_energy_naive,
        mape_energy_24,
        mape_energy_168
    ]
})

print("\nEnergy Forecasting Results:")
print(
    energy_results.to_string(index=False)
)


# ============================================================
# 4. Save results
# ============================================================

session_output = os.path.join(
    OUTPUT_DIR,
    "baseline_session_results.csv"
)

energy_output = os.path.join(
    OUTPUT_DIR,
    "baseline_energy_results.csv"
)

session_results.to_csv(
    session_output,
    index=False
)

energy_results.to_csv(
    energy_output,
    index=False
)


# ============================================================
# 5. Save predictions for later analysis
# ============================================================

predictions = test_df[
    [
        "hour",
        "session_count",
        "target_sessions_next_hour",
        "energy_kWh",
        "target_energy_next_hour"
    ]
].copy()

predictions["naive_session_prediction"] = (
    naive_prediction
)

predictions["seasonal_24h_session_prediction"] = (
    seasonal_24_prediction
)

predictions["seasonal_168h_session_prediction"] = (
    seasonal_168_prediction
)

predictions["naive_energy_prediction"] = (
    naive_energy_prediction
)

predictions["seasonal_24h_energy_prediction"] = (
    seasonal_energy_24_prediction
)

predictions["seasonal_168h_energy_prediction"] = (
    seasonal_energy_168_prediction
)

prediction_output = os.path.join(
    OUTPUT_DIR,
    "baseline_predictions.csv"
)

predictions.to_csv(
    prediction_output,
    index=False
)


# ============================================================
# 6. Final output
# ============================================================

print("\n" + "=" * 60)
print("BASELINE FORECASTING COMPLETED")
print("=" * 60)

print("\nFiles created:")

print(f"  {session_output}")
print(f"  {energy_output}")
print(f"  {prediction_output}")

print("\nNext step:")
print("Train ML forecasting model and compare it")
print("against these baseline results.")

print("\nDONE")