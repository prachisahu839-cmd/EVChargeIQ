"""
EVChargeIQ
ML-Based EV Charging Demand Forecasting

Models:
1. Random Forest
2. XGBoost
3. LightGBM

Targets:
1. Next-hour session demand
2. Next-hour energy demand

Evaluation:
MAE
RMSE
WAPE
sMAPE

Important:
Chronological train/test split is preserved.
No random shuffling is used.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECASTING_DIR = PROJECT_ROOT / "data" / "processed" / "forecasting"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

DATA_FILE = FORECASTING_DIR / "hourly_forecasting_dataset.csv"

print("=" * 70)
print("EVChargeIQ - ML FORECASTING")
print("=" * 70)

print("\nLoading forecasting dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df)}")


# ============================================================
# 3. DATETIME HANDLING
# ============================================================

if "hour" in df.columns:
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")

    df = df.sort_values("hour").reset_index(drop=True)

print("\nTime range:")

if "hour" in df.columns:
    print(f"Start: {df['hour'].min()}")
    print(f"End  : {df['hour'].max()}")


# ============================================================
# 4. TARGETS
# ============================================================

TARGETS = [
    "target_sessions_next_hour",
    "target_energy_next_hour"
]


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

# Features that are known at prediction time.
#
# We intentionally exclude:
# - target columns
# - raw datetime
# - future target information
#
# Lag and rolling features created earlier are retained.

EXCLUDE_COLUMNS = {
    "hour",
    "target_sessions_next_hour",
    "target_energy_next_hour",
}


feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDE_COLUMNS
    and pd.api.types.is_numeric_dtype(df[column])
]


print("\nFeatures selected:")
for feature in feature_columns:
    print(f"  - {feature}")

print(f"\nTotal features: {len(feature_columns)}")


# ============================================================
# 6. CLEAN DATA
# ============================================================

required_columns = feature_columns + TARGETS

model_df = df[required_columns].copy()

model_df = model_df.replace([np.inf, -np.inf], np.nan)

before = len(model_df)

model_df = model_df.dropna(
    subset=required_columns
).reset_index(drop=True)

after = len(model_df)

print("\nData cleaning:")
print(f"Rows before cleaning : {before}")
print(f"Rows after cleaning  : {after}")
print(f"Rows removed         : {before - after}")


# ============================================================
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

print("\n" + "=" * 70)
print("CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Total rows : {len(model_df)}")
print(f"Train rows : {len(train_df)}")
print(f"Test rows  : {len(test_df)}")

print("\nNo random shuffle used.")


# ============================================================
# 8. EVALUATION FUNCTIONS
# ============================================================

def calculate_wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error.
    More stable than ordinary MAPE when demand contains zeros.
    """
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(y_true - y_pred))
        / denominator
    ) * 100


def calculate_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    """
    denominator = (
        np.abs(y_true) +
        np.abs(y_pred)
    )

    mask = denominator != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            2 * np.abs(
                y_pred[mask] - y_true[mask]
            ) / denominator[mask]
        )
        * 100
    )


def evaluate_model(y_true, y_pred):

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
        y_true,
        y_pred
    )

    smape = calculate_smape(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "sMAPE_percent": smape
    }


# ============================================================
# 9. MODEL DEFINITIONS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
}


# ============================================================
# 10. TRAIN MODELS
# ============================================================

all_results = []
all_predictions = []


for target in TARGETS:

    print("\n")
    print("=" * 70)
    print(f"TARGET: {target}")
    print("=" * 70)

    X_train = train_df[feature_columns]
    y_train = train_df[target]

    X_test = test_df[feature_columns]
    y_test = test_df[target]

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        metrics = evaluate_model(
            y_test.to_numpy(),
            predictions
        )

        result = {
            "Target": target,
            "Model": model_name,
            **metrics
        }

        all_results.append(result)

        prediction_df = pd.DataFrame({
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "Target": target,
            "Model": model_name
        })

        all_predictions.append(
            prediction_df
        )

        print(
            f"MAE   : {metrics['MAE']:.4f}"
        )

        print(
            f"RMSE  : {metrics['RMSE']:.4f}"
        )

        print(
            f"WAPE  : {metrics['WAPE_percent']:.2f}%"
        )

        print(
            f"sMAPE : {metrics['sMAPE_percent']:.2f}%"
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        safe_target = target.replace(
            "target_",
            ""
        )

        safe_model = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        model_path = (
            MODEL_DIR
            / f"{safe_model}_{safe_target}.pkl"
        )

        import joblib

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Model saved: {model_path.name}"
        )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


results_file = (
    REPORT_DIR
    / "ml_forecasting_results.csv"
)

predictions_file = (
    REPORT_DIR
    / "ml_forecasting_predictions.csv"
)


results_df.to_csv(
    results_file,
    index=False
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# 12. PRINT FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ML FORECASTING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


importance_records = []


for target in TARGETS:

    X_train = train_df[feature_columns]
    y_train = train_df[target]

    for model_name, model_template in models.items():

        model = model_template

        model.fit(
            X_train,
            y_train
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importances = (
                model.feature_importances_
            )

            for feature, importance in zip(
                feature_columns,
                importances
            ):

                importance_records.append({
                    "Target": target,
                    "Model": model_name,
                    "Feature": feature,
                    "Importance": importance
                })


importance_df = pd.DataFrame(
    importance_records
)

importance_file = (
    REPORT_DIR
    / "ml_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


print(
    f"Feature importance saved: "
    f"{importance_file}"
)


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("ML FORECASTING COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    f"  {results_file}"
)

print(
    f"  {predictions_file}"
)

print(
    f"  {importance_file}"
)

print(
    f"  Models → {MODEL_DIR}"
)

print("\nNext step:")
print(
    "Compare ML models against Naive "
    "and Seasonal Naive baselines."
)