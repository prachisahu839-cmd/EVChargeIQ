import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/forecasting/hourly_forecasting_dataset.csv"

REPORT_DIR = "reports"
PLOT_DIR = os.path.join(REPORT_DIR, "gru_plots")
MODEL_DIR = "models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.80

EPOCHS = 50
BATCH_SIZE = 32

# Same 17 features used in LSTM
FEATURE_COLUMNS = [
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
    "energy_rolling_168h",
]

TARGETS = {
    "sessions": "target_sessions_next_hour",
    "energy": "target_energy_next_hour",
}


# ============================================================
# METRICS
# ============================================================

def calculate_wape(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return 0.0

    return (np.sum(np.abs(y_true - y_pred)) / denominator) * 100


def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return (
        np.mean(
            2 * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        )
        * 100
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EVChargeIQ - GRU Forecasting")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

# Parse datetime
df["hour"] = pd.to_datetime(df["hour"])

# Remove timezone if present
if df["hour"].dt.tz is not None:
    df["hour"] = df["hour"].dt.tz_localize(None)

# Sort chronologically
df = df.sort_values("hour").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['hour'].min()} -> {df['hour'].max()}")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURE_COLUMNS + list(TARGETS.values())

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * TRAIN_RATIO)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nChronological split:")
print(f"Train rows: {len(train_df)}")
print(f"Test rows : {len(test_df)}")

print(f"Train end : {train_df['hour'].max()}")
print(f"Test start: {test_df['hour'].min()}")

# ============================================================
# SCALE FEATURES
# IMPORTANT:
# Fit scaler ONLY on training data
# ============================================================

scaler = StandardScaler()

X_train_raw = train_df[FEATURE_COLUMNS].values
X_test_raw = test_df[FEATURE_COLUMNS].values

X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(X, y, sequence_length):
    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(X)):
        X_sequences.append(
            X[i - sequence_length:i]
        )

        y_sequences.append(y[i])

    return np.array(X_sequences), np.array(y_sequences)


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================

def train_gru(target_name, target_column):

    print("\n" + "=" * 70)
    print(f"TRAINING GRU - {target_name.upper()}")
    print("=" * 70)

    y_train = train_df[target_column].values
    y_test = test_df[target_column].values

    # --------------------------------------------------------
    # Create training sequences
    # --------------------------------------------------------

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train,
        SEQUENCE_LENGTH
    )

    # --------------------------------------------------------
    # For test sequence, include last 24 training observations
    # --------------------------------------------------------

    X_test_combined = np.vstack([
        X_train_scaled[-SEQUENCE_LENGTH:],
        X_test_scaled
    ])

    y_test_combined = np.concatenate([
        y_train[-SEQUENCE_LENGTH:],
        y_test
    ])

    X_test_seq, y_test_seq = create_sequences(
        X_test_combined,
        y_test_combined,
        SEQUENCE_LENGTH
    )

    print("\nSequence shapes:")
    print(f"X_train: {X_train_seq.shape}")
    print(f"y_train: {y_train_seq.shape}")
    print(f"X_test : {X_test_seq.shape}")
    print(f"y_test : {y_test_seq.shape}")

    # ========================================================
    # BUILD GRU MODEL
    # ========================================================

    model = tf.keras.Sequential([
        tf.keras.Input(
            shape=(
                SEQUENCE_LENGTH,
                len(FEATURE_COLUMNS)
            )
        ),

        tf.keras.layers.GRU(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.GRU(
            32
        ),

        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    print("\nModel summary:")
    model.summary()

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\nTraining...")

    history = model.fit(
        X_train_seq,
        y_train_seq,

        validation_split=0.10,

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        callbacks=[early_stopping],

        shuffle=False,

        verbose=1
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test_seq,
        verbose=0
    ).flatten()

    actual = y_test_seq

    # ========================================================
    # METRICS
    # ========================================================

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    wape = calculate_wape(
        actual,
        predictions
    )

    smape = calculate_smape(
        actual,
        predictions
    )

    print("\nResults:")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"WAPE : {wape:.6f}%")
    print(f"sMAPE: {smape:.6f}%")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = pd.DataFrame([
        {
            "Model": "GRU",
            "Target": target_column,
            "MAE": mae,
            "RMSE": rmse,
            "WAPE_percent": wape,
            "sMAPE_percent": smape
        }
    ])

    result_path = os.path.join(
        REPORT_DIR,
        "gru_forecasting_results.csv"
    )

    # Append if file already exists
    if os.path.exists(result_path):

        old_results = pd.read_csv(result_path)

        old_results = old_results[
            ~(
                (old_results["Model"] == "GRU") &
                (old_results["Target"] == target_column)
            )
        ]

        results = pd.concat(
            [old_results, results],
            ignore_index=True
        )

    results.to_csv(
        result_path,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({
        "hour": test_df["hour"].values,
        "actual": actual,
        "predicted": predictions
    })

    prediction_path = os.path.join(
        REPORT_DIR,
        f"gru_{target_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"gru_{target_name}.keras"
    )

    model.save(model_path)

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history.history
    )

    history_path = os.path.join(
        REPORT_DIR,
        f"gru_{target_name}_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # ========================================================
    # FORECAST PLOT
    # ========================================================

    plt.figure(figsize=(14, 6))

    plt.plot(
        prediction_df["hour"],
        actual,
        label="Actual"
    )

    plt.plot(
        prediction_df["hour"],
        predictions,
        label="GRU Predicted"
    )

    plt.title(
        f"GRU Forecast - {target_name.title()}"
    )

    plt.xlabel("Time")
    plt.ylabel(target_name.title())

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()

    forecast_plot_path = os.path.join(
        PLOT_DIR,
        f"gru_{target_name}_forecast.png"
    )

    plt.savefig(
        forecast_plot_path,
        dpi=150
    )

    plt.close()

    # ========================================================
    # LOSS PLOT
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        f"GRU Training Loss - {target_name.title()}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.legend()

    plt.tight_layout()

    loss_plot_path = os.path.join(
        PLOT_DIR,
        f"gru_{target_name}_loss.png"
    )

    plt.savefig(
        loss_plot_path,
        dpi=150
    )

    plt.close()

    print("\nSaved:")
    print(f"- {result_path}")
    print(f"- {prediction_path}")
    print(f"- {model_path}")
    print(f"- {history_path}")
    print(f"- {forecast_plot_path}")
    print(f"- {loss_plot_path}")

    return {
        "Target": target_column,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "sMAPE_percent": smape
    }


# ============================================================
# TRAIN BOTH TARGETS
# ============================================================

all_results = []

for target_name, target_column in TARGETS.items():

    result = train_gru(
        target_name,
        target_column
    )

    all_results.append(result)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRU FORECASTING COMPLETE")
print("=" * 70)

final_results = pd.DataFrame(all_results)

print("\nFinal GRU Results:")
print(final_results.to_string(index=False))

print("\nAll files saved successfully.")