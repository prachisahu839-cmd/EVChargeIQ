
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
JPL_FILE = "data/processed/jpl/jpl_hourly_dataset.csv"
RF_FILE = "reports/cross_site_validation_predictions.csv"

OUTPUT_METRICS = "reports/jpl_baseline_comparison_metrics.csv"
OUTPUT_PREDICTIONS = "reports/jpl_baseline_comparison_predictions.csv"


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not find any of these columns: {possible_names}. "
        f"Available columns: {list(df.columns)}"
    )


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def main():
    print("EVChargeIQ - JPL BASELINE COMPARISON")
    print("------------------------------------")

    # Load datasets
    jpl = pd.read_csv(JPL_FILE)
    rf = pd.read_csv(RF_FILE)

    jpl_time = find_column(jpl, ["hour"])
    jpl_sessions = find_column(jpl, ["sessions", "session_count"])
    jpl_energy = find_column(jpl, ["energy_kwh", "energy_kWh"])

    time_col = find_column(rf, ["hour", "timestamp"])
    target_col = find_column(rf, ["target", "Target"])
    actual_col = find_column(rf, ["actual", "Actual"])
    predicted_col = find_column(rf, ["predicted", "Predicted"])

    print("JPL dataset loaded:", jpl.shape)
    print("Random Forest predictions loaded:", rf.shape)
    print("Target values:", rf[target_col].unique())

    # Convert timestamps to UTC
    jpl[jpl_time] = pd.to_datetime(jpl[jpl_time], utc=True)
    rf[time_col] = pd.to_datetime(rf[time_col], utc=True)

    # Sort and remove duplicate timestamps
    jpl = (
        jpl.sort_values(jpl_time)
        .drop_duplicates(jpl_time)
        .set_index(jpl_time)
    )

    # Create continuous hourly timeline
    full_index = pd.date_range(
        start=jpl.index.min(),
        end=jpl.index.max(),
        freq="h",
        tz="UTC"
    )

    jpl = jpl.reindex(full_index)
    jpl.index.name = "hour"

    # Fill missing hourly demand with zero
    jpl[jpl_sessions] = jpl[jpl_sessions].fillna(0)
    jpl[jpl_energy] = jpl[jpl_energy].fillna(0)

    # Target: demand in the next hour
    jpl["actual_sessions"] = jpl[jpl_sessions].shift(-1)
    jpl["actual_energy"] = jpl[jpl_energy].shift(-1)

    # Baseline 1: previous-hour demand
    jpl["previous_hour_sessions"] = jpl[jpl_sessions]
    jpl["previous_hour_energy"] = jpl[jpl_energy]

    # Baseline 2: same target hour in previous week
    # Target is t+1, so previous week's matching hour is t-167
    jpl["weekly_sessions"] = jpl[jpl_sessions].shift(167)
    jpl["weekly_energy"] = jpl[jpl_energy].shift(167)

    jpl = jpl.reset_index()
    jpl["hour"] = pd.to_datetime(jpl["hour"], utc=True)

    # Target names are read from the actual prediction CSV.
    # Match using keywords to support different naming formats.
    target_values = rf[target_col].astype(str).str.lower()

    target_configs = [
        {
            "target_name": "Next-hour sessions",
            "keywords": ["session"],
            "actual_key": "actual_sessions",
            "previous_key": "previous_hour_sessions",
            "weekly_key": "weekly_sessions"
        },
        {
            "target_name": "Next-hour energy",
            "keywords": ["energy"],
            "actual_key": "actual_energy",
            "previous_key": "previous_hour_energy",
            "weekly_key": "weekly_energy"
        }
    ]

    results = []
    prediction_frames = []

    for config in target_configs:
        print("\nComparing:", config["target_name"])

        mask = target_values.str.contains(
            "|".join(config["keywords"]),
            regex=True,
            na=False
        )

        rf_target = rf.loc[mask].copy()

        if rf_target.empty:
            print("Available target names:", rf[target_col].unique())
            raise ValueError(
                f"No prediction rows found for {config['target_name']}."
            )

        # Merge predictions with JPL hourly observations
        merged = rf_target.merge(
            jpl[
                [
                    "hour",
                    config["actual_key"],
                    config["previous_key"],
                    config["weekly_key"]
                ]
            ],
            left_on=time_col,
            right_on="hour",
            how="inner"
        )

        merged = merged.dropna(
            subset=[
                actual_col,
                predicted_col,
                config["actual_key"],
                config["previous_key"],
                config["weekly_key"]
            ]
        )

        if merged.empty:
            raise ValueError(
                f"No common valid rows found for {config['target_name']}. "
                "Check timestamp alignment."
            )

        # Verify actual values match the JPL next-hour target
        actual = merged[actual_col].astype(float)
        reconstructed_actual = merged[config["actual_key"]].astype(float)

        if not np.allclose(
            actual.to_numpy(),
            reconstructed_actual.to_numpy(),
            rtol=1e-5,
            atol=1e-5
        ):
            raise ValueError(
                f"Actual values do not match JPL next-hour values for "
                f"{config['target_name']}. Check timestamp alignment."
            )

        # Predictions from all three methods
        model_predictions = {
            "Previous-hour baseline":
                merged[config["previous_key"]].astype(float),

            "Previous-week baseline":
                merged[config["weekly_key"]].astype(float),

            "Random Forest":
                merged[predicted_col].astype(float)
        }

        # Calculate evaluation metrics
        for model_name, predictions in model_predictions.items():
            metrics = calculate_metrics(actual, predictions)

            results.append({
                "Target": config["target_name"],
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Test_Rows": len(merged)
            })

        # Save individual predictions
        output = pd.DataFrame({
            "hour": merged[time_col],
            "Target": config["target_name"],
            "Actual": actual,
            "Previous_Hour": model_predictions["Previous-hour baseline"],
            "Previous_Week": model_predictions["Previous-week baseline"],
            "Random_Forest": model_predictions["Random Forest"]
        })

        prediction_frames.append(output)

        print("Rows evaluated:", len(merged))

    # Save output reports
    os.makedirs("reports", exist_ok=True)

    metrics_df = pd.DataFrame(results)
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

    print("\n========== BASELINE COMPARISON RESULTS ==========")
    print(metrics_df.to_string(index=False))

    print("\nFiles saved successfully:")
    print(OUTPUT_METRICS)
    print(OUTPUT_PREDICTIONS)


if __name__ == "__main__":
    main()
