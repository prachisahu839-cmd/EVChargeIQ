
# ============================================================
# EVChargeIQ - Cross-Site Forecasting Validation
# Train on Caltech data and evaluate on JPL data
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CALTECH_FILE = "data/processed/forecasting/hourly_forecasting_dataset.csv"
JPL_FILE = "data/processed/jpl/jpl_hourly_dataset.csv"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "hour_of_day", "day_of_week", "day_of_month", "month",
    "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "sessions_lag_1h", "energy_lag_1h",
    "sessions_lag_24h", "energy_lag_24h",
    "sessions_lag_168h", "energy_lag_168h",
    "sessions_rolling_24h", "energy_rolling_24h",
    "sessions_rolling_168h", "energy_rolling_168h",
    "sessions_std_24h", "energy_std_24h"
]


def build_features(source, time_col, sessions_col, energy_col):
    df = source[[time_col, sessions_col, energy_col]].copy()

    df[time_col] = pd.to_datetime(
        df[time_col], errors="coerce", utc=True
    )
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="last")

    df = df.rename(columns={
        time_col: "hour",
        sessions_col: "sessions",
        energy_col: "energy"
    })

    df["sessions"] = pd.to_numeric(
        df["sessions"], errors="coerce"
    ).fillna(0)

    df["energy"] = pd.to_numeric(
        df["energy"], errors="coerce"
    ).fillna(0)

    df = df.set_index("hour")

    # Build a continuous hourly timeline
    index = pd.date_range(
        start=df.index.min().floor("h"),
        end=df.index.max().floor("h"),
        freq="h",
        tz="UTC"
    )

    df = df.reindex(index)
    df.index.name = "hour"
    df[["sessions", "energy"]] = (
        df[["sessions", "energy"]].fillna(0)
    )

    # Calendar features
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour_of_day"] / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour_of_day"] / 24
    )
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Historical lag features
    for lag in [1, 24, 168]:
        df[f"sessions_lag_{lag}h"] = df["sessions"].shift(lag)
        df[f"energy_lag_{lag}h"] = df["energy"].shift(lag)

    # Rolling means use past hours only
    for window in [24, 168]:
        df[f"sessions_rolling_{window}h"] = (
            df["sessions"].shift(1).rolling(window).mean()
        )
        df[f"energy_rolling_{window}h"] = (
            df["energy"].shift(1).rolling(window).mean()
        )

    df["sessions_std_24h"] = (
        df["sessions"].shift(1).rolling(24).std()
    )
    df["energy_std_24h"] = (
        df["energy"].shift(1).rolling(24).std()
    )

    # Next-hour targets
    df["target_sessions"] = df["sessions"].shift(-1)
    df["target_energy"] = df["energy"].shift(-1)

    return df.reset_index()


def evaluate_target(train, test, target, label):
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(train[FEATURES], train[target])
    predictions = model.predict(test[FEATURES])
    actual = test[target].to_numpy()

    metrics = {
        "Target": label,
        "Training_Site": "Caltech",
        "Testing_Site": "JPL",
        "MAE": mean_absolute_error(actual, predictions),
        "RMSE": np.sqrt(mean_squared_error(actual, predictions)),
        "R2": (
            r2_score(actual, predictions)
            if len(actual) >= 2 and np.var(actual) > 0
            else np.nan
        ),
        "Test_Rows": len(actual)
    }

    predictions_df = pd.DataFrame({
        "hour": test["hour"].to_numpy(),
        "target": label,
        "actual": actual,
        "predicted": predictions,
        "training_site": "Caltech",
        "testing_site": "JPL"
    })

    return metrics, predictions_df


def main():
    print("=" * 60)
    print("EVChargeIQ CROSS-SITE VALIDATION")
    print("=" * 60)

    if not os.path.exists(CALTECH_FILE):
        raise FileNotFoundError(
            f"Caltech dataset not found: {CALTECH_FILE}"
        )

    if not os.path.exists(JPL_FILE):
        raise FileNotFoundError(
            f"JPL dataset not found: {JPL_FILE}"
        )

    print("\nLoading datasets...")
    caltech_raw = pd.read_csv(CALTECH_FILE)
    jpl_raw = pd.read_csv(JPL_FILE)

    print("Building matching features...")
    caltech = build_features(
        caltech_raw, "hour", "session_count", "energy_kWh"
    )

    jpl = build_features(
        jpl_raw, "hour", "sessions", "energy_kwh"
    )

    required = FEATURES + ["target_sessions", "target_energy"]

    caltech = caltech.dropna(subset=required).reset_index(drop=True)
    jpl = jpl.dropna(subset=required).reset_index(drop=True)

    if len(caltech) < 10 or len(jpl) < 10:
        raise ValueError(
            "Not enough rows after feature engineering. "
            f"Caltech rows: {len(caltech)}, JPL rows: {len(jpl)}"
        )

    # Chronological 80/20 split for Caltech; train only on its
    # earlier 80%. JPL is held out as the external evaluation site.
    split = int(len(caltech) * 0.80)
    train = caltech.iloc[:split].copy()

    print(f"\nCaltech usable rows: {len(caltech)}")
    print(f"Caltech training rows: {len(train)}")
    print(f"JPL evaluation rows: {len(jpl)}")

    results = []
    all_predictions = []

    for target, label in [
        ("target_sessions", "Next-hour sessions"),
        ("target_energy", "Next-hour energy (kWh proxy)")
    ]:
        print(f"\nTraining and evaluating: {label}")

        metrics, predictions = evaluate_target(
            train, jpl, target, label
        )

        results.append(metrics)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(results)
    predictions_df = pd.concat(
        all_predictions, ignore_index=True
    )

    metrics_file = os.path.join(
        OUTPUT_DIR, "cross_site_validation_metrics.csv"
    )
    predictions_file = os.path.join(
        OUTPUT_DIR, "cross_site_validation_predictions.csv"
    )

    metrics_df.to_csv(metrics_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    print("\n" + "=" * 60)
    print("CROSS-SITE RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    print("\nFiles saved:")
    print(metrics_file)
    print(predictions_file)

    print("\nInterpretation notes:")
    print("- Models were trained on Caltech and tested on JPL.")
    print("- MAE and RMSE are better when lower.")
    print("- R2 may be negative if predictions perform poorly.")
    print("- JPL energy is aggregated by session connection hour;")
    print("  treat its energy results as a proxy, not metered hourly use.")
    print("\nValidation script finished successfully.")


if __name__ == "__main__":
    main()
