from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.80
EPOCHS = 50
BATCH_SIZE = 32

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


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
PLOT_DIR = REPORT_DIR / "lstm_plots"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


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

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    wape = calculate_wape(
        np.asarray(y_true),
        np.asarray(y_pred)
    )

    smape = calculate_smape(
        np.asarray(y_true),
        np.asarray(y_pred)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "sMAPE_percent": smape
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EVChargeIQ - LSTM FORECASTING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

df["hour"] = pd.to_datetime(df["hour"])

df = (
    df
    .sort_values("hour")
    .reset_index(drop=True)
)

print(f"Rows loaded: {len(df)}")
print(f"Start: {df['hour'].min()}")
print(f"End:   {df['hour'].max()}")


# ============================================================
# TARGETS
# ============================================================

targets = {
    "sessions": "target_sessions_next_hour",
    "energy": "target_energy_next_hour"
}


# ============================================================
# FEATURE SELECTION
# ============================================================

# Only information available at the current hour or earlier.
# Target columns are deliberately excluded.

feature_columns = [
    "session_count",
    "energy_kWh",
    "unique_stations",

    "avg_session_duration_min",
    "avg_charging_duration_min",

    "hour_sin",
    "hour_cos",

    "sessions_lag_1h",
    "sessions_lag_24h",
    "sessions_lag_168h",

    "energy_lag_1h",
    "energy_lag_24h",
    "energy_lag_168h",

    "sessions_rolling_24h",
    "sessions_rolling_168h",

    "energy_rolling_24h",
    "energy_rolling_168h"
]

# Check that all features exist

missing_features = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing_features:

    print("\nMissing feature columns:")

    for col in missing_features:
        print(" -", col)

    raise ValueError(
        "Some required feature columns are missing."
    )


# ============================================================
# REMOVE ROWS WITH MISSING VALUES
# ============================================================

required_columns = (
    feature_columns
    + list(targets.values())
)

df_model = (
    df[
        ["hour"] + required_columns
    ]
    .dropna()
    .reset_index(drop=True)
)

print(
    f"\nRows after removing missing values: "
    f"{len(df_model)}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(df_model) * TRAIN_RATIO
)

train_df = df_model.iloc[
    :split_index
].copy()

test_df = df_model.iloc[
    split_index:
].copy()

print("\nChronological split:")
print(f"Train rows: {len(train_df)}")
print(f"Test rows:  {len(test_df)}")

print(
    f"Train end:  {train_df['hour'].max()}"
)

print(
    f"Test start: {test_df['hour'].min()}"
)


# ============================================================
# LSTM FUNCTION
# ============================================================

def run_lstm(
    train_data,
    test_data,
    target_column,
    target_name
):

    print("\n" + "-" * 70)
    print(
        f"LSTM MODEL: {target_name.upper()}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X_train_raw = train_data[
        feature_columns
    ].values

    X_test_raw = test_data[
        feature_columns
    ].values

    y_train_raw = train_data[
        target_column
    ].values.reshape(-1, 1)

    y_test_raw = test_data[
        target_column
    ].values.reshape(-1, 1)

    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    # IMPORTANT:
    # Fit scalers ONLY on training data.
    # This prevents test information leakage.

    X_scaler = MinMaxScaler()

    y_scaler = MinMaxScaler()

    X_train_scaled = X_scaler.fit_transform(
        X_train_raw
    )

    X_test_scaled = X_scaler.transform(
        X_test_raw
    )

    y_train_scaled = y_scaler.fit_transform(
        y_train_raw
    )

    y_test_scaled = y_scaler.transform(
        y_test_raw
    )

    # --------------------------------------------------------
    # CREATE SEQUENCES
    # --------------------------------------------------------

    def create_sequences(
        X,
        y,
        sequence_length
    ):

        X_sequences = []
        y_sequences = []

        for i in range(
            sequence_length,
            len(X)
        ):

            X_sequences.append(
                X[
                    i - sequence_length:i
                ]
            )

            y_sequences.append(
                y[i]
            )

        return (
            np.array(X_sequences),
            np.array(y_sequences)
        )

    # Training sequences

    X_train_seq, y_train_seq = (
        create_sequences(
            X_train_scaled,
            y_train_scaled,
            SEQUENCE_LENGTH
        )
    )

    # --------------------------------------------------------
    # TEST SEQUENCES
    # --------------------------------------------------------
    #
    # Include the final 24 training observations so that the
    # first test prediction has a complete historical window.
    #

    X_test_combined = np.vstack([
        X_train_scaled[
            -SEQUENCE_LENGTH:
        ],
        X_test_scaled
    ])

    y_test_combined = np.vstack([
        y_train_scaled[
            -SEQUENCE_LENGTH:
        ],
        y_test_scaled
    ])

    X_test_seq, y_test_seq = (
        create_sequences(
            X_test_combined,
            y_test_combined,
            SEQUENCE_LENGTH
        )
    )

    print(
        f"\nTraining sequences: "
        f"{X_train_seq.shape}"
    )

    print(
        f"Testing sequences: "
        f"{X_test_seq.shape}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Sequential([
        LSTM(
            64,
            input_shape=(
                X_train_seq.shape[1],
                X_train_seq.shape[2]
            ),
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(
            32,
            return_sequences=False
        ),

        Dropout(0.2),

        Dense(16, activation="relu"),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    print("\nModel summary:")

    model.summary()

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining LSTM...")

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping
        ],
        verbose=1,
        shuffle=False
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions_scaled = model.predict(
        X_test_seq,
        verbose=0
    )

    predictions = (
        y_scaler
        .inverse_transform(
            predictions_scaled
        )
        .flatten()
    )

    actual = (
        y_scaler
        .inverse_transform(
            y_test_seq
        )
        .flatten()
    )

    # EV charging quantities cannot be negative

    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        actual,
        predictions
    )

    print("\nResults:")

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.6f}"
        )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "hour": test_data["hour"].values,
        "actual": actual,
        "predicted": predictions
    })

    prediction_file = (
        REPORT_DIR
        / f"lstm_{target_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    print(
        f"\nPredictions saved:"
    )

    print(prediction_file)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR
        / f"lstm_{target_name}.keras"
    )

    model.save(
        model_file
    )

    print(
        f"Model saved:"
    )

    print(model_file)

    # --------------------------------------------------------
    # FORECAST PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        prediction_df["hour"],
        prediction_df["actual"],
        label="Actual"
    )

    plt.plot(
        prediction_df["hour"],
        prediction_df["predicted"],
        label="LSTM Predicted"
    )

    plt.xlabel("Time")

    plt.ylabel(
        target_name.title()
    )

    plt.title(
        f"EVChargeIQ - LSTM "
        f"{target_name.title()} Forecast"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plot_file = (
        PLOT_DIR
        / f"lstm_{target_name}_forecast.png"
    )

    plt.savefig(
        plot_file,
        dpi=150
    )

    plt.close()

    print(
        f"Forecast plot saved:"
    )

    print(plot_file)

    # --------------------------------------------------------
    # LOSS CURVE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("MSE Loss")

    plt.title(
        f"LSTM Training - "
        f"{target_name.title()}"
    )

    plt.legend()

    plt.tight_layout()

    loss_plot_file = (
        PLOT_DIR
        / f"lstm_{target_name}_loss.png"
    )

    plt.savefig(
        loss_plot_file,
        dpi=150
    )

    plt.close()

    print(
        f"Loss plot saved:"
    )

    print(loss_plot_file)

    return metrics


# ============================================================
# RUN BOTH TARGETS
# ============================================================

results = []

for target_name, target_column in targets.items():

    metrics = run_lstm(
        train_df,
        test_df,
        target_column,
        target_name
    )

    results.append({
        "Target": target_name,
        "Model": "LSTM",
        **metrics
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_file = (
    REPORT_DIR
    / "lstm_forecasting_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL LSTM RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved:")

print(results_file)

print(
    "\nLSTM forecasting completed successfully."
)