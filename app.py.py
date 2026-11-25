
from pathlib import Path
import json

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

st.set_page_config(
    page_title="EVChargeIQ",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EVChargeIQ")
st.caption("AI-Driven EV Charging Analytics and Decision Support System")


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def find_file(candidates):
    """Return the first existing file from a list of paths."""
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def read_csv_safe(path):
    """Load a CSV safely."""
    if path is None or not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def read_json_safe(path):
    """Load JSON data and return session records when possible."""
    if path is None or not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            content = json.load(file)

        if isinstance(content, list):
            return content

        if isinstance(content, dict):
            # Support common JSON structures.
            for key in ("sessions", "data", "results", "items"):
                if isinstance(content.get(key), list):
                    return content[key]

            return [content]

        return []

    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return []


def format_number(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def load_model_comparison():
    candidates = [
        REPORTS_DIR / "forecasting_model_comparison.csv",
        REPORTS_DIR / "model_comparison.csv",
        REPORTS_DIR / "forecasting_results.csv",
    ]

    return read_csv_safe(find_file(candidates))


def load_ml_predictions(target_key, model_name):
    """Load tree-model predictions from the combined prediction file."""
    path = REPORTS_DIR / "ml_forecasting_predictions.csv"
    data = read_csv_safe(path)

    if data.empty:
        return pd.DataFrame()

    required = {"Actual", "Predicted", "Target", "Model"}
    if not required.issubset(data.columns):
        return pd.DataFrame()

    data = data[
        (data["Target"].astype(str) == target_key)
        & (data["Model"].astype(str).str.lower() == model_name.lower())
    ].copy()

    if data.empty:
        return pd.DataFrame()

    data = data.rename(
        columns={"Actual": "actual", "Predicted": "predicted"}
    )

    # These predictions do not have timestamps in the saved CSV.
    data["hour"] = range(len(data))
    data["hour"] = data["hour"].astype(str)

    return data


def load_sequence_predictions(model_name, target):
    """Load LSTM, GRU or Prophet prediction CSV."""
    suffix = "sessions" if target == "Sessions" else "energy"
    filename = f"{model_name.lower()}_{suffix}_predictions.csv"

    data = read_csv_safe(REPORTS_DIR / filename)

    if data.empty:
        return pd.DataFrame()

    required = {"hour", "actual", "predicted"}
    if not required.issubset(data.columns):
        return pd.DataFrame()

    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"], errors="coerce")
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce")
    data["predicted"] = pd.to_numeric(data["predicted"], errors="coerce")

    data = data.dropna(subset=["hour", "actual", "predicted"])
    data = data.sort_values("hour")

    return data


def get_processed_csvs():
    """Find processed CSVs without assuming a single folder layout."""
    if not PROCESSED_DIR.exists():
        return []

    return sorted(PROCESSED_DIR.rglob("*.csv"))


# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Dataset Analytics",
        "Model Comparison",
        "Forecasting Demo",
        "Cross-Site Validation",
        "Prediction Analysis",
        "Project Status",
    ],
)

st.sidebar.divider()
st.sidebar.caption("EVChargeIQ | Research Prototype")


# --------------------------------------------------
# 4. PROJECT OVERVIEW
# --------------------------------------------------

if page == "Project Overview":

    st.header("Project Overview")

    st.write(
        """
        EVChargeIQ is an AI-driven EV charging analytics and
        decision-support prototype. It uses historical EV charging
        sessions to analyze demand and evaluate forecasting models.
        """
    )

    caltech_path = RAW_DIR / "caltech_sessions.json"
    jpl_path = RAW_DIR / "jpl_sessions.json"

    caltech_records = read_json_safe(caltech_path)
    jpl_records = read_json_safe(jpl_path)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Caltech Sessions", f"{len(caltech_records):,}")

    with col2:
        st.metric("JPL Sessions", f"{len(jpl_records):,}")

    with col3:
        st.metric(
            "Total Raw Sessions",
            f"{len(caltech_records) + len(jpl_records):,}",
        )

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Analyze historical EV charging-session data.
        - Clean and preprocess charging records.
        - Explore hourly charging demand and energy usage.
        - Train and compare machine-learning and time-series models.
        - Demonstrate actual-versus-predicted demand.
        - Provide insights that can support charging-planning decisions.
        """
    )

    st.subheader("System Workflow")

    st.markdown(
        """
        **Data Collection** → **Data Cleaning** → **Exploratory Analysis**
        → **Feature Engineering** → **Forecasting Models**
        → **Model Evaluation** → **Dashboard**
        """
    )

    st.info(
        "This is a research prototype based on historical datasets. "
        "It does not currently monitor or control live charging stations."
    )


# --------------------------------------------------
# 5. DATASET ANALYTICS
# --------------------------------------------------

elif page == "Dataset Analytics":

    st.header("Dataset Analytics")

    st.write(
        "Explore raw charging sessions and the processed CSV datasets "
        "available in your project."
    )

    dataset_choice = st.radio(
        "Select dataset",
        ["Caltech Sessions", "JPL Sessions", "Processed CSV Files"],
        horizontal=True,
    )

    if dataset_choice == "Caltech Sessions":
        path = RAW_DIR / "caltech_sessions.json"
        records = read_json_safe(path)

        if records:
            data = pd.json_normalize(records)

            st.subheader("Caltech Dataset")
            st.metric("Number of Records", f"{len(data):,}")

            st.write("Dataset preview")
            st.dataframe(data.head(100), use_container_width=True)

            st.write("Columns")
            st.write(list(data.columns))

            numeric_data = data.select_dtypes(include="number")

            if not numeric_data.empty:
                selected_column = st.selectbox(
                    "Choose a numeric column to analyze",
                    list(numeric_data.columns),
                )

                st.subheader(f"Distribution: {selected_column}")
                st.bar_chart(
                    numeric_data[selected_column]
                    .dropna()
                    .value_counts()
                    .sort_index()
                    .head(50)
                )

        else:
            st.warning(
                f"No records found. Check that this file exists: {path}"
            )

    elif dataset_choice == "JPL Sessions":
        path = RAW_DIR / "jpl_sessions.json"
        records = read_json_safe(path)

        if records:
            data = pd.json_normalize(records)

            st.subheader("JPL Dataset")
            st.metric("Number of Records", f"{len(data):,}")

            st.write("Dataset preview")
            st.dataframe(data.head(100), use_container_width=True)

            st.write("Columns")
            st.write(list(data.columns))

            numeric_data = data.select_dtypes(include="number")

            if not numeric_data.empty:
                selected_column = st.selectbox(
                    "Choose a numeric column to analyze",
                    list(numeric_data.columns),
                )

                st.subheader(f"Distribution: {selected_column}")
                st.bar_chart(
                    numeric_data[selected_column]
                    .dropna()
                    .value_counts()
                    .sort_index()
                    .head(50)
                )

        else:
            st.warning(
                f"No records found. Check that this file exists: {path}"
            )

    else:
        csv_files = get_processed_csvs()

        if not csv_files:
            st.warning(
                "No processed CSV files were found under data/processed."
            )
        else:
            file_options = {
                str(path.relative_to(ROOT)): path for path in csv_files
            }

            selected_file = st.selectbox(
                "Choose a processed dataset",
                list(file_options.keys()),
            )

            data = read_csv_safe(file_options[selected_file])

            if not data.empty:
                st.subheader(selected_file)

                col1, col2 = st.columns(2)
                col1.metric("Rows", f"{len(data):,}")
                col2.metric("Columns", f"{len(data.columns):,}")

                st.dataframe(data.head(200), use_container_width=True)

                numeric_columns = list(
                    data.select_dtypes(include="number").columns
                )

                if numeric_columns:
                    selected_columns = st.multiselect(
                        "Choose numeric columns for a chart",
                        numeric_columns,
                        default=numeric_columns[:2],
                    )

                    if selected_columns:
                        st.line_chart(data[selected_columns])

                st.download_button(
                    "Download displayed dataset",
                    data=data.to_csv(index=False).encode("utf-8"),
                    file_name=Path(selected_file).name,
                    mime="text/csv",
                )


# --------------------------------------------------
# 6. MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("Model Comparison")

    comparison = load_model_comparison()

    if comparison.empty:
        st.warning(
            "Model comparison file was not found or contains no data."
        )
        st.write(
            "Expected file: reports/forecasting_model_comparison.csv"
        )

    else:
        st.subheader("Saved Model Evaluation Results")
        st.dataframe(comparison, use_container_width=True)

        st.download_button(
            "Download model comparison",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="forecasting_model_comparison.csv",
            mime="text/csv",
        )

        # Automatically identify likely metric columns.
        metric_candidates = [
            column
            for column in comparison.columns
            if any(
                metric in column.lower()
                for metric in ["mae", "rmse", "wape", "smape"]
            )
        ]

        if metric_candidates:
            selected_metric = st.selectbox(
                "Select metric to visualize",
                metric_candidates,
            )

            chart_data = comparison.copy()
            chart_data[selected_metric] = pd.to_numeric(
                chart_data[selected_metric], errors="coerce"
            )

            chart_data = chart_data.dropna(subset=[selected_metric])

            if not chart_data.empty:
                possible_labels = [
                    column
                    for column in chart_data.columns
                    if column != selected_metric
                    and not pd.api.types.is_numeric_dtype(
                        chart_data[column]
                    )
                ]

                if possible_labels:
                    label_column = st.selectbox(
                        "Select model/target label",
                        possible_labels,
                    )

                    chart_data = chart_data.set_index(label_column)
                    st.bar_chart(chart_data[selected_metric])

        st.caption(
            "Lower MAE and RMSE generally indicate smaller prediction errors. "
            "Compare metrics for the same forecasting target."
        )

    st.subheader("Saved Evaluation Plots")

    plot_files = [
        REPORTS_DIR / "forecasting_sessions_mae_comparison.png",
        REPORTS_DIR / "forecasting_energy_mae_comparison.png",
    ]

    for plot_path in plot_files:
        if plot_path.exists():
            st.image(str(plot_path), caption=plot_path.name)


# --------------------------------------------------
# 7. FORECASTING DEMO
# --------------------------------------------------

elif page == "Forecasting Demo":

    st.header("Actual vs Predicted Demand")

    st.write(
        """
        Compare saved model predictions with actual values from the
        existing test period. The dashboard reads saved prediction files;
        it does not train a model when you change these selections.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox(
            "Forecast target",
            ["Sessions", "Energy (kWh)"],
        )

    with col2:
        model = st.selectbox(
            "Select forecasting model",
            [
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "LSTM",
                "GRU",
                "Prophet",
            ],
        )

    target_key = (
        "target_sessions_next_hour"
        if target == "Sessions"
        else "target_energy_next_hour"
    )

    if model in ["Random Forest", "XGBoost", "LightGBM"]:
        data = load_ml_predictions(target_key, model)
    else:
        data = load_sequence_predictions(model, target)

    if data.empty:
        st.warning(
            "No prediction data was found for this selection. "
            "Check the relevant CSV file under the reports/ folder."
        )

        if model in ["Random Forest", "XGBoost", "LightGBM"]:
            st.code("reports/ml_forecasting_predictions.csv")
        else:
            suffix = "sessions" if target == "Sessions" else "energy"
            st.code(
                f"reports/{model.lower()}_{suffix}_predictions.csv"
            )

    else:
        data = data.copy()

        data["actual"] = pd.to_numeric(
            data["actual"], errors="coerce"
        )
        data["predicted"] = pd.to_numeric(
            data["predicted"], errors="coerce"
        )

        data = data.dropna(subset=["actual", "predicted"])

        if data.empty:
            st.warning("The prediction file has no valid numeric rows.")

        else:
            errors = data["actual"] - data["predicted"]

            mae = errors.abs().mean()
            rmse = (errors.pow(2).mean()) ** 0.5

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric("Model", model)
            metric2.metric("MAE", format_number(mae))
            metric3.metric("RMSE", format_number(rmse))

            st.subheader("Actual vs Predicted")

            chart_data = data[["actual", "predicted"]].copy()
            chart_data.index = range(len(chart_data))
            chart_data.index.name = "Observation"

            st.line_chart(chart_data)

            if model in ["Random Forest", "XGBoost", "LightGBM"]:
                st.caption(
                    "The tree-model prediction CSV does not contain "
                    "timestamps, so the horizontal axis represents "
                    "observation order rather than clock time."
                )
            else:
                st.caption(
                    "The saved prediction file includes timestamps, "
                    "but this chart uses observation order for comparison."
                )

            with st.expander("View prediction records"):
                st.dataframe(data, use_container_width=True)

                st.download_button(
                    "Download these predictions",
                    data=data.to_csv(index=False).encode("utf-8"),
                    file_name=f"{model.lower()}_{target.lower().replace(' ', '_')}_predictions.csv",
                    mime="text/csv",
                )

            st.caption(
                "MAE is the mean absolute prediction error. "
                "RMSE penalizes larger errors more strongly. "
                "These values describe the saved evaluation data, "
                "not guaranteed performance at a live charging station."
            )


# --------------------------------------------------
# 8. CROSS-SITE VALIDATION
# --------------------------------------------------

elif page == "Cross-Site Validation":

    st.header("Cross-Site Model Validation")

    st.write(
        """
        This experiment trains Random Forest models using Caltech
        historical data and evaluates them on the separate JPL dataset.
        It checks whether patterns learned at one site transfer to another.
        """
    )

    metrics_path = REPORTS_DIR / "cross_site_validation_metrics.csv"
    predictions_path = REPORTS_DIR / "cross_site_validation_predictions.csv"

    metrics = read_csv_safe(metrics_path)
    predictions = read_csv_safe(predictions_path)

    if metrics.empty:
        st.warning(
            "Cross-site metrics were not found. Run "
            "scripts/cross_site_validation.py first."
        )
        st.code("python scripts/cross_site_validation.py")
    else:
        st.subheader("Caltech → JPL Evaluation Results")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        metric_columns = {
            str(column).strip().lower(): column
            for column in metrics.columns
        }

        mae_col = metric_columns.get("mae")
        rmse_col = metric_columns.get("rmse")
        r2_col = metric_columns.get("r2")

        if mae_col and rmse_col:
            display_metrics = metrics.copy()
            for column in [mae_col, rmse_col, r2_col]:
                if column:
                    display_metrics[column] = pd.to_numeric(
                        display_metrics[column], errors="coerce"
                    )
            st.caption(
                "MAE and RMSE are error measures (lower is better). "
                "R² describes how much variation is explained; it is not "
                "an accuracy percentage."
            )

        st.download_button(
            "Download cross-site metrics",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="cross_site_validation_metrics.csv",
            mime="text/csv",
        )

    if not predictions.empty:
        st.subheader("JPL Actual vs Predicted")

        lower_columns = {
            str(column).strip().lower(): column
            for column in predictions.columns
        }
        target_col = lower_columns.get("target")
        actual_col = lower_columns.get("actual")
        predicted_col = lower_columns.get("predicted")
        hour_col = lower_columns.get("hour")

        if target_col and actual_col and predicted_col:
            target_values = predictions[target_col].astype(str).unique().tolist()
            selected_target = st.selectbox(
                "Select cross-site target",
                target_values,
                key="cross_site_target",
            )
            view = predictions[
                predictions[target_col].astype(str) == selected_target
            ].copy()

            view[actual_col] = pd.to_numeric(
                view[actual_col], errors="coerce"
            )
            view[predicted_col] = pd.to_numeric(
                view[predicted_col], errors="coerce"
            )
            view = view.dropna(subset=[actual_col, predicted_col])

            if not view.empty:
                chart = view[[actual_col, predicted_col]].copy()
                if hour_col:
                    chart.index = view[hour_col].astype(str)
                    chart.index.name = "Hour"
                else:
                    chart.index = range(len(chart))
                    chart.index.name = "Observation"
                st.line_chart(chart)

                st.metric(
                    "Prediction rows",
                    f"{len(view):,}",
                )
                with st.expander("View cross-site prediction records"):
                    st.dataframe(view, use_container_width=True)

                st.download_button(
                    "Download cross-site predictions",
                    data=view.to_csv(index=False).encode("utf-8"),
                    file_name="cross_site_validation_predictions.csv",
                    mime="text/csv",
                )
        else:
            st.warning(
                "Prediction file columns were not recognized. Expected "
                "hour, target, actual and predicted."
            )
    else:
        st.info(
            "Cross-site prediction file not found. Expected: "
            "reports/cross_site_validation_predictions.csv"
        )

    st.info(
        "Interpretation: this is a cross-site test, not proof of "
        "deployment-ready performance. JPL energy is an hourly "
        "connection-hour energy proxy, not exact metered hourly usage."
    )


# --------------------------------------------------
# 9. PREDICTION ANALYSIS
# --------------------------------------------------

elif page == "Prediction Analysis":

    st.header("Prediction Analysis & Visualizations")

    st.write(
        """
        View the graphs generated from the JPL baseline comparison.
        These plots compare the cross-site Random Forest predictions
        with actual values and simple historical baselines.
        """
    )

    figure_dir = REPORTS_DIR / "figures"

    plot_groups = [
        (
            "Next-hour sessions",
            [
                "next_hour_sessions_actual_vs_predicted.png",
                "next_hour_sessions_scatter.png",
                "next_hour_sessions_mae_comparison.png",
            ],
        ),
        (
            "Next-hour energy",
            [
                "next_hour_energy_actual_vs_predicted.png",
                "next_hour_energy_scatter.png",
                "next_hour_energy_mae_comparison.png",
            ],
        ),
    ]

    found_any_plot = False

    for group_title, filenames in plot_groups:
        existing = [
            figure_dir / filename
            for filename in filenames
            if (figure_dir / filename).exists()
        ]

        if existing:
            found_any_plot = True
            st.subheader(group_title)
            for plot_path in existing:
                st.image(str(plot_path), caption=plot_path.name, use_container_width=True)

    if not found_any_plot:
        st.warning(
            "No prediction-analysis plots were found in reports/figures. "
            "Run the visualization script first."
        )
        st.code("python scripts/prediction_visualization.py")

    baseline_metrics_path = (
        REPORTS_DIR / "jpl_baseline_comparison_metrics.csv"
    )
    baseline_predictions_path = (
        REPORTS_DIR / "jpl_baseline_comparison_predictions.csv"
    )

    baseline_metrics = read_csv_safe(baseline_metrics_path)
    if not baseline_metrics.empty:
        st.subheader("JPL Baseline Comparison Metrics")
        st.dataframe(
            baseline_metrics,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download JPL baseline metrics",
            data=baseline_metrics.to_csv(index=False).encode("utf-8"),
            file_name="jpl_baseline_comparison_metrics.csv",
            mime="text/csv",
        )

    baseline_predictions = read_csv_safe(baseline_predictions_path)
    if not baseline_predictions.empty:
        with st.expander("View JPL baseline comparison records"):
            st.dataframe(
                baseline_predictions,
                use_container_width=True,
            )
            st.download_button(
                "Download JPL baseline predictions",
                data=baseline_predictions.to_csv(index=False).encode("utf-8"),
                file_name="jpl_baseline_comparison_predictions.csv",
                mime="text/csv",
            )


# --------------------------------------------------
# 10. PROJECT STATUS
# --------------------------------------------------

elif page == "Project Status":

    st.header("Project Status")

    st.write(
        "Current implementation status of the EVChargeIQ prototype."
    )

    status_items = [
        ("Historical data collection", "Completed"),
        ("Data cleaning and preprocessing", "Completed"),
        ("Exploratory data analysis", "Completed"),
        ("Hourly feature engineering", "Completed"),
        ("Machine-learning forecasting", "Completed"),
        ("LSTM and GRU forecasting", "Completed"),
        ("Prophet forecasting", "Completed"),
        ("Model comparison reports", "Available if report files exist"),
        ("Streamlit dashboard", "Prototype"),
        ("Live charging-station integration", "Not implemented"),
        ("Cross-site model validation (Caltech → JPL)", "Completed — metrics saved"),
        ("JPL baseline comparison", "Completed — metrics and predictions saved"),
        ("Prediction visualization", "Completed — plots saved in reports/figures"),
    ]

    status_df = pd.DataFrame(
        status_items,
        columns=["Component", "Status"],
    )

    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.subheader("Project Limitations")

    st.markdown(
        """
        - Forecasting results are based on historical data.
        - Current saved forecasting results use the Caltech test period.
        - JPL data has been collected and preprocessed, but this does not
          establish that the models generalize to JPL.
        - The dashboard does not receive live charging-station telemetry.
        - Forecasts support analysis and decision-making; they are not
          guaranteed to represent future demand exactly.
        """
    )

    st.info(
        "For a final-year project demonstration, present this as a "
        "historical-data-based AI analytics and forecasting prototype."
    )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "EVChargeIQ | AI-Driven EV Charging Analytics and Decision Support System"
)
