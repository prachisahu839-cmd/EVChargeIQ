
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
    page_title="EVChargeIQ | EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional dark interface. All visual styling is local to this app.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      :root { --ev-bg:#07111f; --ev-panel:#0e1b2d; --ev-panel2:#12243a; --ev-cyan:#27d7e8; --ev-blue:#4388ff; --ev-text:#eaf2ff; --ev-muted:#9eb0c8; }
      html, body, [data-testid="stAppViewContainer"] { background:var(--ev-bg); color:var(--ev-text); font-family:'DM Sans',sans-serif; }
      [data-testid="stHeader"] { background:rgba(7,17,31,.88); }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#0c1b2e 0%,#091523 100%); border-right:1px solid rgba(125,170,220,.15); }
      [data-testid="stSidebar"] * { font-family:'DM Sans',sans-serif; }
      h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f3f8ff !important; letter-spacing:-.02em; }
      h1 { font-size:2.25rem !important; }
      h2 { font-size:1.55rem !important; }
      h3 { font-size:1.12rem !important; }
      p, label, li, [data-testid="stMarkdownContainer"] { color:#d7e4f5; }
      [data-testid="stCaptionContainer"] { color:var(--ev-muted); }
      [data-testid="stMetric"] { background:linear-gradient(145deg,#11243a,#0d1b2d); border:1px solid rgba(87,160,220,.22); padding:17px 18px; border-radius:15px; box-shadow:0 8px 24px rgba(0,0,0,.14); }
      [data-testid="stMetricLabel"] { color:#a9bdd5 !important; }
      [data-testid="stMetricValue"] { color:#f5fbff !important; font-family:'Space Grotesk',sans-serif; }
      [data-testid="stMetricDelta"] { color:#65dfc0 !important; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid rgba(125,170,220,.18); border-radius:12px; overflow:hidden; }
      [data-testid="stExpander"] { background:#0d1b2d; border:1px solid rgba(125,170,220,.18); border-radius:12px; }
      [data-testid="stAlert"] { border-radius:12px; }
      .stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid rgba(39,215,232,.45); background:linear-gradient(100deg,#146a91,#2459a5); color:white; font-weight:600; }
      .stButton>button:hover, .stDownloadButton>button:hover { border-color:#27d7e8; color:white; background:linear-gradient(100deg,#1785a8,#326fd0); }
      [data-testid="stRadio"] label, [data-testid="stSelectbox"] label { color:#c9d8eb !important; }
      hr { border-color:rgba(125,170,220,.18) !important; }
      .ev-hero { padding:24px 26px; border:1px solid rgba(39,215,232,.24); border-radius:18px; background:radial-gradient(circle at 85% 10%,rgba(39,215,232,.16),transparent 32%),linear-gradient(120deg,#10263d,#0b192b 70%); margin:4px 0 22px 0; }
      .ev-kicker { color:#55ddec; font-size:.78rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:7px; }
      .ev-hero-title { color:#f5fbff; font-family:'Space Grotesk',sans-serif; font-size:2.25rem; font-weight:700; line-height:1.1; }
      .ev-hero-sub { color:#afc3dc; font-size:1rem; margin-top:9px; }
      .ev-chip { display:inline-block; padding:5px 10px; border-radius:99px; background:rgba(39,215,232,.11); border:1px solid rgba(39,215,232,.25); color:#8debf3; font-size:.78rem; margin:8px 6px 0 0; }
      .ev-note { padding:14px 16px; border-radius:12px; background:#0d2033; border-left:3px solid #27d7e8; color:#c8d9ec; }
    </style>
    <div class="ev-hero">
      <div class="ev-kicker">EV charging intelligence platform</div>
      <div class="ev-hero-title">⚡ EVChargeIQ</div>
      <div class="ev-hero-sub">AI-Driven EV Charging Analytics and Demand Forecasting</div>
      <span class="ev-chip">Historical ACN datasets</span>
      <span class="ev-chip">Caltech + JPL</span>
      <span class="ev-chip">ML · Deep Learning · Analytics</span>
    </div>
    """,
    unsafe_allow_html=True,
)


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
st.sidebar.markdown("**⚡ EVChargeIQ**")
st.sidebar.caption("AI charging analytics · Research prototype")
st.sidebar.caption("Data scope: ACN Caltech + JPL")


# --------------------------------------------------
# 4. PROJECT OVERVIEW
# --------------------------------------------------

if page == "Project Overview":

    st.header("Project Overview")
    st.markdown(
        "EVChargeIQ analyzes historical electric-vehicle charging sessions and "
        "uses forecasting models to study near-term charging demand. The current "
        "prototype is built on ACN-Data from Caltech and JPL."
    )

    caltech_records = read_json_safe(RAW_DIR / "caltech_sessions.json")
    jpl_records = read_json_safe(RAW_DIR / "jpl_sessions.json")
    hourly = read_csv_safe(PROCESSED_DIR / "forecasting" / "hourly_forecasting_dataset.csv")
    comparison = load_model_comparison()

    total_sessions = len(caltech_records) + len(jpl_records)
    total_models = int(comparison["Model"].nunique()) if not comparison.empty and "Model" in comparison.columns else 0
    forecast_rows = len(hourly)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Caltech sessions", f"{len(caltech_records):,}")
    k2.metric("JPL sessions", f"{len(jpl_records):,}")
    k3.metric("Hourly records", f"{forecast_rows:,}")
    k4.metric("Models evaluated", f"{total_models:,}")

    st.subheader("Project at a glance")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("""
        **What EVChargeIQ does**
        - Explores historical EV charging-session patterns.
        - Prepares hourly demand features and next-hour targets.
        - Compares machine-learning, deep-learning and statistical models.
        - Evaluates transfer from Caltech training data to JPL test data.
        - Presents saved predictions and evaluation results in an interactive dashboard.
        """)
    with right:
        st.markdown("""
        <div class="ev-note">
          <b>Current project scope</b><br/>
          ACN-Data (Caltech + JPL) is the dataset used for this submission. Live
          Indian charging-station feeds and real-time charger availability are
          not part of the current implementation; BEE integration is future work.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("End-to-end workflow")
    workflow = st.columns(6)
    steps = ["Data collection", "Cleaning", "EDA", "Features", "Forecasting", "Evaluation"]
    for col, label in zip(workflow, steps):
        with col:
            st.markdown(
                f"<div style='min-height:76px;padding:13px 9px;text-align:center;border:1px solid rgba(39,215,232,.25);border-radius:12px;background:#10243a;color:#dff8ff;font-weight:600'>{label}</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Project objectives")
    obj1, obj2, obj3 = st.columns(3)
    obj1.markdown("**01 · Understand demand**\n\nAnalyze session counts, energy proxies, time patterns and site activity.")
    obj2.markdown("**02 · Forecast demand**\n\nCompare next-hour session and session-associated energy forecasts.")
    obj3.markdown("**03 · Evaluate transfer**\n\nTest how Caltech-trained patterns perform on separate JPL data.")

    st.caption("Research prototype · Historical-data analysis only · No live station monitoring or control")

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
            st.dataframe(data.head(100), width="stretch")

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
            st.dataframe(data.head(100), width="stretch")

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

                st.dataframe(data.head(200), width="stretch")

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
    st.write("Compare saved evaluation results. Metrics are separated by target so that session and energy errors are not mixed.")

    comparison = load_model_comparison()
    if comparison.empty:
        st.warning("Model comparison file was not found or contains no data.")
        st.code("reports/forecasting_model_comparison.csv")
    else:
        required = {"Target", "Model", "MAE", "RMSE"}
        if not required.issubset(comparison.columns):
            st.warning("The comparison CSV does not contain the expected Target, Model, MAE and RMSE columns.")
            st.dataframe(comparison, width="stretch", hide_index=True)
        else:
            comparison = comparison.copy()
            comparison["MAE"] = pd.to_numeric(comparison["MAE"], errors="coerce")
            comparison["RMSE"] = pd.to_numeric(comparison["RMSE"], errors="coerce")
            target_labels = {
                "target_sessions_next_hour": "Next-hour sessions",
                "target_energy_next_hour": "Next-hour energy proxy (kWh)",
            }
            comparison["Target display"] = comparison["Target"].map(target_labels).fillna(comparison["Target"].astype(str))
            selected_target = st.selectbox("Forecast target", sorted(comparison["Target display"].dropna().unique()))
            view = comparison[comparison["Target display"] == selected_target].copy()
            view = view.sort_values("MAE", ascending=True, na_position="last")

            best = view.dropna(subset=["MAE"]).head(1)
            if not best.empty:
                best_row = best.iloc[0]
                a, b, c = st.columns(3)
                a.metric("Lowest MAE model", str(best_row["Model"]))
                b.metric("Best MAE", format_number(best_row["MAE"]))
                c.metric("Best RMSE", format_number(view["RMSE"].min()))

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("MAE by model")
                mae_chart = view.dropna(subset=["MAE"]).set_index("Model")[["MAE"]]
                if not mae_chart.empty:
                    st.bar_chart(mae_chart)
            with chart_col2:
                st.subheader("RMSE by model")
                rmse_chart = view.dropna(subset=["RMSE"]).set_index("Model")[["RMSE"]]
                if not rmse_chart.empty:
                    st.bar_chart(rmse_chart)

            display_cols = [c for c in ["Model", "Model_Type", "MAE", "RMSE", "WAPE_percent", "sMAPE_percent", "Baseline_MAE", "Improvement_over_168h_percent"] if c in view.columns]
            st.subheader("Saved evaluation results")
            st.dataframe(view[display_cols], width="stretch", hide_index=True)
            st.download_button("Download filtered comparison", data=view.to_csv(index=False).encode("utf-8"), file_name="evchargeiq_model_comparison.csv", mime="text/csv")

            st.markdown("""
            <div class="ev-note"><b>How to interpret:</b> Lower MAE and RMSE indicate smaller errors for the same target. WAPE and sMAPE can become large when many actual hourly values are zero or close to zero. Improvement over the 168-hour seasonal baseline is based on MAE; it is not an accuracy percentage.</div>
            """, unsafe_allow_html=True)

    st.subheader("Saved evaluation plots")
    plot_files = [
        REPORTS_DIR / "figures" / "next_hour_sessions_mae_comparison.png",
        REPORTS_DIR / "figures" / "next_hour_energy_mae_comparison.png",
    ]
    available_plots = [p for p in plot_files if p.exists()]
    if available_plots:
        cols = st.columns(len(available_plots))
        for col, plot_path in zip(cols, available_plots):
            with col:
                st.image(str(plot_path), caption=plot_path.name, width="stretch")
    else:
        st.caption("No standalone comparison plots were found; the interactive charts above use the saved metrics CSV.")

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
            ["Sessions", "Energy proxy (kWh)"],
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

    if target != "Sessions":
        st.caption("Energy values represent session-associated energy grouped by connection hour. They are not metered electricity consumed strictly within each clock hour.")

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
                st.dataframe(data, width="stretch")

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

    st.header("Cross-Site Validation")
    st.write("A Random Forest model trained on Caltech data was evaluated on JPL data. This experiment assesses cross-site transfer; it does not establish production readiness.")

    metrics = read_csv_safe(REPORTS_DIR / "cross_site_validation_metrics.csv")
    baselines = read_csv_safe(REPORTS_DIR / "jpl_baseline_comparison_metrics.csv")
    predictions = read_csv_safe(REPORTS_DIR / "cross_site_validation_predictions.csv")

    if not metrics.empty:
        st.subheader("Caltech-trained model evaluated on JPL")
        st.dataframe(metrics, width="stretch", hide_index=True)
        if "Test_Rows" in metrics.columns:
            st.caption(f"Evaluation observations: {int(pd.to_numeric(metrics['Test_Rows'], errors='coerce').max()):,} hourly rows.")
    else:
        st.warning("Cross-site metrics not found. Expected reports/cross_site_validation_metrics.csv.")

    if not baselines.empty:
        st.subheader("JPL: model versus simple baselines")
        st.dataframe(baselines, width="stretch", hide_index=True)
        st.markdown("""
        <div class="ev-note"><b>Important result:</b> On the JPL evaluation set, the previous-week seasonal baseline has lower MAE than the Caltech-trained Random Forest for both next-hour sessions and next-hour energy proxy. This means the current Random Forest does not outperform that simple baseline on JPL. Report this as a limitation and an opportunity for future site-specific calibration.</div>
        """, unsafe_allow_html=True)
    else:
        st.info("JPL baseline comparison metrics were not found. Expected reports/jpl_baseline_comparison_metrics.csv.")

    if not predictions.empty:
        st.subheader("JPL actual versus predicted")
        lower_columns = {str(c).strip().lower(): c for c in predictions.columns}
        target_col = lower_columns.get("target")
        actual_col = lower_columns.get("actual")
        predicted_col = lower_columns.get("predicted")
        hour_col = lower_columns.get("hour")
        if target_col and actual_col and predicted_col:
            target_values = predictions[target_col].astype(str).unique().tolist()
            selected_target = st.selectbox("Select validation target", target_values, key="cross_site_target")
            view = predictions[predictions[target_col].astype(str) == selected_target].copy()
            view[actual_col] = pd.to_numeric(view[actual_col], errors="coerce")
            view[predicted_col] = pd.to_numeric(view[predicted_col], errors="coerce")
            view = view.dropna(subset=[actual_col, predicted_col])
            if not view.empty:
                chart = view[[actual_col, predicted_col]].copy()
                chart.columns = ["Actual", "Random Forest prediction"]
                if hour_col:
                    chart.index = view[hour_col].astype(str)
                    chart.index.name = "Hour"
                else:
                    chart.index = range(len(chart))
                    chart.index.name = "Observation"
                st.line_chart(chart)
                st.metric("Prediction rows", f"{len(view):,}")
                with st.expander("View prediction records"):
                    st.dataframe(view, width="stretch")
                st.download_button("Download cross-site predictions", data=view.to_csv(index=False).encode("utf-8"), file_name="cross_site_validation_predictions.csv", mime="text/csv")
        else:
            st.warning("Prediction CSV columns were not recognized. Expected hour, target, actual and predicted.")
    else:
        st.info("Cross-site prediction records were not found.")

    st.caption("The reported JPL test uses JPL lag-history features. It is a cross-site evaluation, not a completely history-free zero-shot deployment test. Energy values are session-associated connection-hour energy proxies, not metered electricity consumption in each clock hour.")

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
                st.image(str(plot_path), caption=plot_path.name, width="stretch")

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
            width="stretch",
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
                width="stretch",
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
        ("Live BEE charging-station integration", "Future scope — not included in this submission"),
        ("Cross-site model validation (Caltech → JPL)", "Completed — metrics saved"),
        ("JPL baseline comparison", "Completed — metrics and predictions saved"),
        ("Prediction visualization", "Completed — plots saved in reports/figures"),
    ]

    status_df = pd.DataFrame(
        status_items,
        columns=["Component", "Status"],
    )

    st.dataframe(status_df, width="stretch", hide_index=True)

    st.subheader("Project Limitations")

    st.markdown(
        """
        - Forecasting results are based on historical data.
        - Current saved forecasting results use the Caltech test period.
        - JPL data has been collected and preprocessed, but this does not
          establish that the models generalize to JPL.
        - The dashboard does not receive live charging-station telemetry or charger-availability updates.
        - Caltech-trained Random Forest results are weaker than the previous-week baseline on JPL for both evaluated targets.
        - Energy is a session-associated connection-hour proxy, not metered hourly electricity consumption.
        - Forecasts support historical analysis and planning research; they are not guaranteed to represent future demand exactly.
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
    "EVChargeIQ | ACN historical-data research prototype | Caltech + JPL"
)
