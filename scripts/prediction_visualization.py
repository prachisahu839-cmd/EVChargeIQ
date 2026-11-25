
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Input files
PREDICTIONS_FILE = "reports/jpl_baseline_comparison_predictions.csv"
METRICS_FILE = "reports/jpl_baseline_comparison_metrics.csv"

# Output folder
OUTPUT_DIR = "reports/figures"


def main():
    print("EVChargeIQ - PREDICTION VISUALIZATION")
    print("------------------------------------")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    predictions = pd.read_csv(PREDICTIONS_FILE)
    metrics = pd.read_csv(METRICS_FILE)

    predictions["hour"] = pd.to_datetime(
        predictions["hour"], utc=True
    )

    targets = predictions["Target"].unique()

    for target in targets:
        print("\nCreating graphs for:", target)

        data = (
            predictions[predictions["Target"] == target]
            .sort_values("hour")
            .copy()
        )

        # 1. Actual vs Random Forest prediction over time
        plt.figure(figsize=(14, 6))
        plt.plot(
            data["hour"], data["Actual"],
            label="Actual", linewidth=2
        )
        plt.plot(
            data["hour"], data["Random_Forest"],
            label="Random Forest Prediction", linewidth=1.5
        )
        plt.title(f"{target}: Actual vs Predicted")
        plt.xlabel("Hour")
        plt.ylabel("Demand")
        plt.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()

        filename = target.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{filename}_actual_vs_predicted.png"),
            dpi=300
        )
        plt.close()

        # 2. Actual vs predicted scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(
            data["Actual"],
            data["Random_Forest"],
            alpha=0.65
        )

        max_value = max(
            data["Actual"].max(),
            data["Random_Forest"].max()
        )

        plt.plot(
            [0, max_value],
            [0, max_value],
            linestyle="--",
            label="Perfect Prediction"
        )

        plt.title(f"{target}: Actual vs Predicted Scatter")
        plt.xlabel("Actual Demand")
        plt.ylabel("Predicted Demand")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{filename}_scatter.png"),
            dpi=300
        )
        plt.close()

        # 3. Model MAE comparison
        target_metrics = metrics[
            metrics["Target"] == target
        ].copy()

        target_metrics = target_metrics.sort_values("MAE")

        plt.figure(figsize=(9, 5))
        bars = plt.bar(
            target_metrics["Model"],
            target_metrics["MAE"]
        )

        plt.title(f"{target}: MAE Model Comparison")
        plt.ylabel("Mean Absolute Error (lower is better)")
        plt.xticks(rotation=15, ha="right")

        for bar, value in zip(bars, target_metrics["MAE"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{filename}_mae_comparison.png"),
            dpi=300
        )
        plt.close()

        print("Created 3 graphs for", target)

    print("\nAll graphs created successfully!")
    print("Saved inside:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
