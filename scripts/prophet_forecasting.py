from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "hourly_forecasting_dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
PLOT_DIR = REPORT_DIR / "prophet_plots"

REPORT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def calculate_wape(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(y_true - y_pred))
        / denominator
        * 100
    )


def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)

    mask = denominator != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            2
            * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        )
        * 100
    )


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    wape = calculate_wape(
        np.array(y_true),
        np.array(y_pred)
    )

    smape = calculate_smape(
        np.array(y_true),
        np.array(y_pred)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "sMAPE_percent": smape,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EVChargeIQ - PROPHET FORECASTING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

df["hour"] = pd.to_datetime(df["hour"])

# Prophet does not support timezone-aware timestamps.
# Remove timezone information while preserving the local clock time.
df["hour"] = df["hour"].dt.tz_localize(None)

df = df.sort_values("hour").reset_index(drop=True)

print(f"Rows loaded: {len(df)}")
print(f"Start: {df['hour'].min()}")
print(f"End:   {df['hour'].max()}")


# ============================================================
# TARGETS
# ============================================================

targets = {
    "sessions": "target_sessions_next_hour",
    "energy": "target_energy_next_hour",
}


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nChronological split:")
print(f"Train rows: {len(train_df)}")
print(f"Test rows:  {len(test_df)}")

print(f"\nTrain end:  {train_df['hour'].max()}")
print(f"Test start: {test_df['hour'].min()}")


# ============================================================
# PROPHET FORECASTING FUNCTION
# ============================================================

def run_prophet(
    train_data,
    test_data,
    target_column,
    target_name
):

    print("\n" + "-" * 70)
    print(f"PROPHET MODEL: {target_name.upper()}")
    print("-" * 70)

    # Prophet requires ds and y
    prophet_train = pd.DataFrame({
        "ds": train_data["hour"],
        "y": train_data[target_column]
    })

    prophet_test = pd.DataFrame({
        "ds": test_data["hour"],
        "y": test_data[target_column]
    })

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05
    )

    # Explicit hourly seasonality
    model.add_seasonality(
        name="hourly",
        period=1,
        fourier_order=8
    )

    print("Training Prophet...")

    model.fit(prophet_train)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    future = prophet_test[["ds"]].copy()

    forecast = model.predict(future)

    predictions = forecast["yhat"].values

    # EV charging demand cannot be negative
    predictions = np.maximum(predictions, 0)

    actual = prophet_test["y"].values

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        actual,
        predictions
    )

    print("\nResults:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "hour": prophet_test["ds"],
        "actual": actual,
        "predicted": predictions
    })

    prediction_file = (
        REPORT_DIR
        / f"prophet_{target_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    print(f"\nPredictions saved:")
    print(prediction_file)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        prediction_df["hour"],
        prediction_df["actual"],
        label="Actual"
    )

    plt.plot(
        prediction_df["hour"],
        prediction_df["predicted"],
        label="Prophet Predicted"
    )

    plt.xlabel("Time")
    plt.ylabel(target_name.title())

    plt.title(
        f"EVChargeIQ - Prophet {target_name.title()} Forecast"
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_file = (
        PLOT_DIR
        / f"prophet_{target_name}_forecast.png"
    )

    plt.savefig(
        plot_file,
        dpi=150
    )

    plt.close()

    print(f"Plot saved:")
    print(plot_file)

    return metrics


# ============================================================
# RUN BOTH FORECASTS
# ============================================================

results = []

for target_name, target_column in targets.items():

    metrics = run_prophet(
        train_df,
        test_df,
        target_column,
        target_name
    )

    results.append({
        "Target": target_name,
        "Model": "Prophet",
        **metrics
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_file = (
    REPORT_DIR
    / "prophet_forecasting_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\n" + "=" * 70)
print("FINAL PROPHET RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nResults saved:")
print(results_file)

print("\nProphet forecasting completed successfully.")