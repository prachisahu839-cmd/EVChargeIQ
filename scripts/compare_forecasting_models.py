import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

REPORT_DIR = "reports"

# ============================================================
# EXISTING RESULTS
# ============================================================

files = [
    "reports/ml_forecasting_results.csv",
    "reports/prophet_forecasting_results.csv",
    "reports/lstm_forecasting_results.csv",
    "reports/gru_forecasting_results.csv",
]

dataframes = []

for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"Loaded: {file}")
    else:
        print(f"WARNING - Not found: {file}")

# ============================================================
# COMBINE ML RESULTS
# ============================================================

comparison = pd.concat(
    dataframes,
    ignore_index=True
)

# ============================================================
# ADD BASELINE RESULTS
# ============================================================

baseline = pd.DataFrame([
    {
        "Model": "Naive",
        "Target": "target_sessions_next_hour",
        "MAE": 1.997354,
        "RMSE": 3.083065,
        "WAPE_percent": None,
        "sMAPE_percent": None
    },
    {
        "Model": "Seasonal 24h",
        "Target": "target_sessions_next_hour",
        "MAE": 2.071429,
        "RMSE": 3.252594,
        "WAPE_percent": None,
        "sMAPE_percent": None
    },
    {
        "Model": "Seasonal 168h",
        "Target": "target_sessions_next_hour",
        "MAE": 1.965608,
        "RMSE": 3.052018,
        "WAPE_percent": None,
        "sMAPE_percent": None
    },
    {
        "Model": "Naive",
        "Target": "target_energy_next_hour",
        "MAE": 22.116478,
        "RMSE": 34.933037,
        "WAPE_percent": None,
        "sMAPE_percent": None
    },
    {
        "Model": "Seasonal 24h",
        "Target": "target_energy_next_hour",
        "MAE": 22.432287,
        "RMSE": 35.425860,
        "WAPE_percent": None,
        "sMAPE_percent": None
    },
    {
        "Model": "Seasonal 168h",
        "Target": "target_energy_next_hour",
        "MAE": 21.583900,
        "RMSE": 33.567401,
        "WAPE_percent": None,
        "sMAPE_percent": None
    }
])

comparison = pd.concat(
    [baseline, comparison],
    ignore_index=True
)
# Normalize target names across all forecasting result files
comparison["Target"] = comparison["Target"].replace({
    "sessions": "target_sessions_next_hour",
    "energy": "target_energy_next_hour"
})
# ============================================================
# MODEL TYPE
# ============================================================

model_type = {
    "Naive": "Baseline",
    "Seasonal 24h": "Baseline",
    "Seasonal 168h": "Baseline",
    "Random Forest": "Machine Learning",
    "XGBoost": "Machine Learning",
    "LightGBM": "Machine Learning",
    "Prophet": "Statistical",
    "LSTM": "Deep Learning",
    "GRU": "Deep Learning"
}

comparison["Model_Type"] = comparison["Model"].map(model_type)

# ============================================================
# IMPROVEMENT OVER SEASONAL 168h
# ============================================================

baseline_mae = comparison[
    comparison["Model"] == "Seasonal 168h"
][["Target", "MAE"]].rename(
    columns={"MAE": "Baseline_MAE"}
)

comparison = comparison.merge(
    baseline_mae,
    on="Target",
    how="left"
)

comparison["Improvement_over_168h_percent"] = (
    (
        comparison["Baseline_MAE"] -
        comparison["MAE"]
    )
    /
    comparison["Baseline_MAE"]
) * 100

# ============================================================
# ROUND VALUES
# ============================================================

comparison["MAE"] = comparison["MAE"].round(4)
comparison["RMSE"] = comparison["RMSE"].round(4)

comparison["WAPE_percent"] = pd.to_numeric(
    comparison["WAPE_percent"],
    errors="coerce"
).round(4)

comparison["sMAPE_percent"] = pd.to_numeric(
    comparison["sMAPE_percent"],
    errors="coerce"
).round(4)

comparison["Improvement_over_168h_percent"] = comparison[
    "Improvement_over_168h_percent"
].round(2)

# ============================================================
# SORT
# ============================================================

comparison = comparison.sort_values(
    ["Target", "MAE"]
).reset_index(drop=True)

# ============================================================
# SAVE COMPLETE COMPARISON
# ============================================================

output_file = os.path.join(
    REPORT_DIR,
    "forecasting_model_comparison.csv"
)

comparison.to_csv(
    output_file,
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("EVChargeIQ - FORECASTING MODEL COMPARISON")
print("=" * 80)

print(
    comparison[
        [
            "Model",
            "Model_Type",
            "Target",
            "MAE",
            "RMSE",
            "WAPE_percent",
            "sMAPE_percent",
            "Improvement_over_168h_percent"
        ]
    ].to_string(index=False)
)

# ============================================================
# BEST MODEL BY MAE
# ============================================================

print("\n" + "=" * 80)
print("BEST MODEL BY MAE")
print("=" * 80)

for target in comparison["Target"].unique():

    target_df = comparison[
        comparison["Target"] == target
    ]

    best = target_df.loc[
        target_df["MAE"].idxmin()
    ]

    print(f"\nTarget: {target}")
    print(f"Best Model: {best['Model']}")
    print(f"MAE: {best['MAE']:.4f}")
    print(
        f"Improvement over Seasonal 168h: "
        f"{best['Improvement_over_168h_percent']:.2f}%"
    )

# ============================================================
# PLOT - SESSION FORECASTING
# ============================================================

sessions = comparison[
    comparison["Target"] == "target_sessions_next_hour"
].copy()

plt.figure(figsize=(12, 6))

plt.bar(
    sessions["Model"],
    sessions["MAE"]
)

plt.title(
    "EVChargeIQ - Session Forecasting MAE Comparison"
)

plt.xlabel("Model")
plt.ylabel("MAE")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

sessions_plot = os.path.join(
    REPORT_DIR,
    "forecasting_sessions_mae_comparison.png"
)

plt.savefig(
    sessions_plot,
    dpi=150
)

plt.close()

# ============================================================
# PLOT - ENERGY FORECASTING
# ============================================================

energy = comparison[
    comparison["Target"] == "target_energy_next_hour"
].copy()

plt.figure(figsize=(12, 6))

plt.bar(
    energy["Model"],
    energy["MAE"]
)

plt.title(
    "EVChargeIQ - Energy Forecasting MAE Comparison"
)

plt.xlabel("Model")
plt.ylabel("MAE")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

energy_plot = os.path.join(
    REPORT_DIR,
    "forecasting_energy_mae_comparison.png"
)

plt.savefig(
    energy_plot,
    dpi=150
)

plt.close()

# ============================================================
# FINAL
# ============================================================

print("\nSaved:")
print(f"- {output_file}")
print(f"- {sessions_plot}")
print(f"- {energy_plot}")

print("\nForecasting comparison completed successfully.")