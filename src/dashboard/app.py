import sys
from datetime import date
from pathlib import Path

# Run from the project root: put both the repo root (for ``src.*`` imports, e.g.
# the pipeline runner) and this directory (for ``data_loader`` / ``charts``) on
# sys.path regardless of how Streamlit was launched.
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parents[1]
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from src.pipeline_runner import DATE_END as PIPELINE_END_DEFAULT
from src.pipeline_runner import read_run_status, run_pipeline

from data_loader import (
    build_common_forecast_dataset,
    build_common_sample_metrics,
    get_clark_west_summary,
    get_eda_findings,
    get_pipeline_metadata,
    get_round3_features,
    load_fevd_results,
    load_gold_features,
    load_irf_results,
    load_regime_metrics,
    load_shap_summary,
    load_xgboost_shap_summary,
)

from charts import (
    MODEL_COLUMNS,
    create_clark_west_chart,
    create_fevd_chart,
    create_forecast_error_distribution,
    create_forecast_vs_actual_chart,
    create_gold_target_chart,
    create_irf_chart,
    create_metric_comparison_chart,
    create_regime_comparison_chart,
    create_shap_summary_chart,
)




# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Canadian Yield Spread Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "Canadian 10Y–2Y Yield Spread Forecasting"
)

st.caption(
    "End-to-end analytics dashboard from public data acquisition "
    "through forecasting evaluation and operational model selection."
)


# =========================================================
# LOAD DATA
# =========================================================

forecast_df = build_common_forecast_dataset()
metrics_df = build_common_sample_metrics()

cw_all = None
shap_summary_df = load_shap_summary()
xgb_shap_summary_df = load_xgboost_shap_summary()
regime_metrics_df = load_regime_metrics()
irf_df = load_irf_results()
fevd_df = load_fevd_results()
gold_df = load_gold_features()

pipeline_metadata = get_pipeline_metadata()
eda_findings = get_eda_findings()
round3_features = get_round3_features()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Controls")

horizon = st.sidebar.radio(
    "Forecast horizon",
    options=[1, 5, 20],
    horizontal=True,
)

selected_models = st.sidebar.multiselect(
    "Models to display",
    options=list(MODEL_COLUMNS.keys()),
    default=[
        "Naïve Random Walk",
        "ARIMA",
        "VAR",
        "VECM",
        "LSTM",
        "XGBoost",
    ],
)

uncertainty_interval = st.sidebar.radio(
    "XGBoost Uncertainty Ribbon",
    options=["None", "90%", "95%"],
    horizontal=True,
    help="Displays 90% or 95% calibrated empirical prediction intervals for XGBoost.",
)



# =========================================================
# SIDEBAR — DATA PIPELINE TRIGGER (Path A: bronze → silver → gold)
# =========================================================

st.sidebar.divider()
st.sidebar.subheader("Data Pipeline")

_status = read_run_status()

if _status is None:
    st.sidebar.caption("No pipeline run recorded yet.")
else:
    _badge = {
        "success": "🟢",
        "warning": "🟡",
        "error": "🔴",
    }.get(_status.get("status"), "⚪")

    st.sidebar.caption(
        f"{_badge} Last run {_status.get('end_time', '?')} "
        f"· mode: {_status.get('mode', '?')}"
    )

    if any(
        source.get("mode") == "cached_fallback"
        for source in _status.get("sources", {}).values()
    ):
        st.sidebar.caption(
            "⚠️ a source used cached fallback on the last run"
        )

_pipeline_mode = st.sidebar.radio(
    "Run mode",
    options=["cache", "live"],
    format_func=lambda m: (
        "Cached rebuild (offline)"
        if m == "cache"
        else "Live API refresh (needs FRED key)"
    ),
    key="pipeline_mode",
)

_pipeline_end = st.sidebar.date_input(
    "Ingestion end date",
    value=date.fromisoformat(PIPELINE_END_DEFAULT),
    # The Gold frame needs 12 months of CPI before the first YoY row
    # (canonical start 2010-02-19); an earlier end date yields an empty frame.
    min_value=date(2010, 3, 1),
    max_value=date.today(),
    key="pipeline_end_date",
)

_pipeline_running = st.session_state.get("pipeline_running", False)

if st.sidebar.button(
    "Rebuild data & features",
    disabled=_pipeline_running,
    use_container_width=True,
):
    st.session_state["pipeline_running"] = True
    try:
        with st.status(
            "Running Path A pipeline…",
            expanded=True,
        ) as _status_box:
            st.write(
                f"Mode: **{_pipeline_mode}** · "
                f"window ends {_pipeline_end.isoformat()}"
            )

            _report = run_pipeline(
                mode=_pipeline_mode,
                end_date=_pipeline_end.isoformat(),
            )

            for _src in _report["sources"].values():
                _icon = {
                    "success": "✅",
                    "warning": "⚠️",
                    "failed": "❌",
                }.get(_src["status"], "•")
                st.write(
                    f"{_icon} {_src['name']}: "
                    f"{_src['status']} ({_src['mode']})"
                )

            _gold = _report["gold"]
            if _gold["status"] == "success":
                st.write(
                    f"✅ Gold: {_gold['rows']} rows "
                    f"through {_gold['date_range'][1]}"
                )
            else:
                st.write(
                    f"❌ Gold: {_gold['status']} — "
                    f"{_gold.get('error', _gold.get('reason', ''))}"
                )

            for _warning in _report["warnings"]:
                st.warning(_warning)

            _status_box.update(
                label=f"Pipeline finished: {_report['status']}",
                state=(
                    "error"
                    if _report["status"] == "error"
                    else "complete"
                ),
                expanded=_report["status"] == "error",
            )
    finally:
        st.session_state["pipeline_running"] = False

    # No @st.cache_data is used in this app, so the next rerun re-reads every
    # CSV. clear() is a harmless no-op today and stays correct if caching is
    # added later.
    st.cache_data.clear()

    # On error, keep the expanded st.status box (with the per-source failure
    # detail) on screen -- an st.rerun() here would wipe it and leave only the
    # 🔴 sidebar badge. On success / warning the rerun refreshes the dashboard
    # against the freshly rebuilt data.
    if _report["status"] != "error":
        st.rerun()

st.sidebar.caption(
    "Runs the Path A pipeline: rebuilds data/processed/*.csv and the Gold "
    "feature store. Forecast, Clark-West and SHAP/IRF/FEVD outputs are **not** "
    "re-run and may lag a fresh rebuild."
)


# =========================================================
# HORIZON-SPECIFIC RESULTS
# =========================================================

horizon_metrics = metrics_df[
    metrics_df["horizon"] == horizon
].copy()

best_rmse_row = horizon_metrics.loc[
    horizon_metrics["rmse"].idxmin()
]

best_mae_row = horizon_metrics.loc[
    horizon_metrics["mae"].idxmin()
]

cw_horizon = get_clark_west_summary(
    horizon
)

strongest_cw = cw_horizon.loc[
    cw_horizon["cw_p_value"].idxmin()
]

fdr_significant = bool(
    cw_horizon[
        "model_significantly_better_fdr_horizon"
    ].any()
)


# =========================================================
# PROJECT LIFECYCLE TABS
# =========================================================

(
    tab_data,
    tab_eda,
    tab_performance,
    tab_statistics,
    tab_errors,
    tab_interpretability,
    tab_decision,
) = st.tabs(
    [
        "1. Data & Pipeline",
        "2. EDA & Feature Engineering",
        "3. Model Performance",
        "4. Statistical Evidence",
        "5. Forecast Errors",
        "6. Model Interpretability",
        "7. Executive Decision",
    ]
)


# =========================================================
# 1 — DATA & PIPELINE
# =========================================================

with tab_data:

    st.header("Data Acquisition & Analytics Pipeline")

    st.markdown(
        """
        The project integrates **three institutional public-data sources** into an end-to-end,
        reproducible **Medallion Architecture** (Bronze $\\rightarrow$ Silver $\\rightarrow$ Gold)
        designed to benchmark econometric and machine learning yield curve forecasting models.
        """
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Institutional Sources",
            pipeline_metadata["data_sources"],
            help="Bank of Canada Valet, Federal Reserve FRED, and Statistics Canada WDS",
        )

    with col2:
        st.metric(
            "Candidate Models",
            pipeline_metadata["candidate_models"],
            help="Naïve Random Walk, ARIMA, VAR, VECM, LSTM, and XGBoost",
        )

    with col3:
        st.metric(
            "Forecast Horizons",
            pipeline_metadata["forecast_horizons"],
            help="1-day, 5-day (weekly), and 20-day (monthly) cumulative forward trading horizons",
        )

    with col4:
        st.metric(
            "Common Origins",
            pipeline_metadata["common_origins_per_horizon"],
            help="Strictly synchronized out-of-sample forecast origin dates per horizon across all models",
        )

    with col5:
        st.metric(
            "Total Evaluations",
            pipeline_metadata.get("total_evaluations", 2235),
            help=f"{pipeline_metadata['common_origins_per_horizon']} common origin dates × 3 horizons evaluated simultaneously",
        )

    st.markdown("---")

    # --- Section: Data Sources & Medallion Architecture ---
    st.subheader("Institutional Data Sources & Medallion Architecture")

    src_col1, src_col2, src_col3 = st.columns(3)

    with src_col1:
        st.markdown(
            """
            #### 🇨🇦 Bank of Canada (Valet API)
            - **GoC 10Y Benchmark Yield** (`BD.CDN.10YR.DQ.YLD`)
            - **GoC 2Y Benchmark Yield** (`BD.CDN.2YR.DQ.YLD`)
            - **Target Yield Spread** ($y_t = \\text{10Y} - \\text{2Y}$)
            - **Target First Difference** ($\\Delta y_t$)
            - **BoC Policy Overnight Rate** (`d_overnight_rate`)
            - **USD/CAD Spot Exchange Rate** (`d_usdcad`)
            """
        )

    with src_col2:
        st.markdown(
            """
            #### 🇺🇸 Federal Reserve (FRED API)
            - **U.S. 10Y Treasury Constant Maturity Yield** (`d_us_treasury_10y`)
            - **Effective Federal Funds Rate** (`d_fed_funds_rate`)
            - Captures cross-border term premium transmission, capital mobility, and bilateral monetary policy divergence.
            """
        )

    with src_col3:
        st.markdown(
            """
            #### Statistics Canada (WDS API)
            - **Consumer Price Index (CPI)** All-items monthly index (`d_cpi_yoy`)
            - Aligned by **official publication release dates** (rather than reference month) to ensure strict point-in-time realism with **zero look-ahead bias**.
            """
        )

    with st.expander("📂 Medallion Pipeline Architecture (Bronze → Silver → Gold)", expanded=False):
        st.markdown(
            """
            | Layer | Storage Format | Operations & Processing Logic | Key Artifacts |
            | :--- | :--- | :--- | :--- |
            | **Bronze** | Raw JSON / CSV | Immutable ingestion directly from institutional APIs. Raw response payloads preserved with date/source timestamps. | `data/raw/` |
            | **Silver** | Cleaned CSV | Business day calendar harmonization, non-synchronous holiday alignment via forward-fill, CPI release lag alignment, unit root / stationarity tests (ADF & KPSS). | `data/processed/` |
            | **Gold** | Standardized CSV | Analytical feature store: 6 stationary core series, 7 causal lag orders ($t-1, \\dots, t-20$), rolling volatility & momentum metrics, multi-horizon cumulative change targets ($\\Delta_h y_t$), cointegration level targets. | `data/processed/` |
            """
        )

    st.markdown("---")

    # --- Section: End-to-End Pipeline Diagram ---
    st.subheader("End-to-End Analytics & Modelling Flow")

    st.code(
        """
Bank of Canada API ─────┐
FRED API ───────────────┼──> [BRONZE LAYER] (Raw API Feeds)
Statistics Canada API ──┘           ↓
                                 [SILVER LAYER] (Calendar Harmonization & Point-in-Time CPI Alignment)
                                    ↓
                          [EDA & DIAGNOSTICS] (ADF/KPSS Stationarity, Johansen Cointegration, Cross-Correlation)
                                    ↓
                       [FEATURE ENGINEERING] (Causal Lags t-1..t-20, Rolling Volatility, Momentum)
                                    ↓
                                 [GOLD LAYER] (Canonical Analytical Feature Matrix)
                                    ↓
          ┌─────────────────────────┴─────────────────────────┐
          │                                                   │
  [Econometric Baselines]                               [Machine Learning]
  • Naïve Random Walk (Driftless)                       • LSTM Recurrent Neural Network
  • ARIMA(p,d,q) (Univariate AIC)                       • XGBoost Gradient Boosted Decision Trees
  • VAR(p) (Multivariate AIC)
  • VECM(p, r=1) (Cointegrated System)
          │                                                   │
          └─────────────────────────┬─────────────────────────┘
                                    ↓
                      [EVALUATION & BENCHMARKING]
                       • 5-Fold Expanding-Window Cross-Validation
                       • Multi-Horizon Cumulative Targets (h = 1, 5, 20 Days)
                       • Common Origin Alignment Across Synchronized Dates
                       • Out-of-Sample RMSE, MAE, R²oos, and R²oos,adj
                      • Clark-West (2007) Tests + Benjamini-Hochberg FDR Control
                      • Calibrated Conformal / Quantile Prediction Intervals
                                    ↓
                      [EXECUTIVE STREAMLIT DASHBOARD]
        """,
        language="text",
    )

    st.info(
        "💡 **Data Leakage Prevention Guarantee**: All lag features ($t-1$ to $t-20$), rolling volatilities, "
        "and momentum indicators are strictly backward-looking. For multi-step forecasting ($h > 1$), models "
        "predict cumulative change $\\Delta_h y_t = \\sum_{k=1}^h \\Delta y_{t+k}$ without access to contemporaneous "
        "macroeconomic shocks."
    )

    st.caption(
        "The sidebar **Data Pipeline** control re-runs this bronze → silver → "
        "gold sequence (Path A) on demand. It does not re-run the forecasting "
        "models or the Clark-West / SHAP / IRF / FEVD outputs."
    )

    st.markdown("---")

    # --- Section: Candidate Models Matrix ---
    st.subheader("Candidate Forecasting Models Matrix")

    models_table = pd.DataFrame([
        {
            "Model": "Naïve Random Walk",
            "Family": "Baseline Benchmark",
            "Target Formulation": "Level: y_hat_{t+h} = y_t",
            "Information Set": "Target history y_t",
            "Key Mechanism / Specification": "Driftless random walk benchmark; baseline for all relative gain metrics",
        },
        {
            "Model": "ARIMA(p,d,q)",
            "Family": "Linear Econometric",
            "Target Formulation": "Differences: Delta y_hat_{t+h}",
            "Information Set": "Univariate target lags",
            "Key Mechanism / Specification": "Auto-selected (p, d, q) minimizing AIC; single-series autoregressive moving-average",
        },
        {
            "Model": "VAR(p)",
            "Family": "Multivariate Econometric",
            "Target Formulation": "Differences: Delta y_hat_{t+h}",
            "Information Set": "6 stationary macro series",
            "Key Mechanism / Specification": "Multivariate vector autoregression; AIC-selected lag order with cross-variable transmission",
        },
        {
            "Model": "VECM(p)",
            "Family": "Cointegrated Econometric",
            "Target Formulation": "Levels & Differences",
            "Information Set": "6 non-stationary series (I(1))",
            "Key Mechanism / Specification": "Johansen cointegrating rank r=1; models long-run equilibrium + short-run error correction",
        },
        {
            "Model": "LSTM",
            "Family": "Deep Learning / RNN",
            "Target Formulation": "Cumulative Diff Delta_h y",
            "Information Set": "20-day sequence × 6 features",
            "Key Mechanism / Specification": "32 hidden units, AdamW optimizer, early stopping, sequence temporal memory",
        },
        {
            "Model": "XGBoost",
            "Family": "Tree-Based Machine Learning",
            "Target Formulation": "Cumulative Diff Delta_h y",
            "Information Set": "50 causal lag / rolling features",
            "Key Mechanism / Specification": "Gradient Boosted Trees (150 estimators, depth 3, lr 0.03) + Tree SHAP + Conformal PIs",
        },
    ])

    st.dataframe(models_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Section: Evaluation Coverage ---
    st.subheader("Out-of-Sample Evaluation & Common Sample Coverage")

    cov_col1, cov_col2 = st.columns(2)

    with cov_col1:
        st.markdown(
            f"""
            - **Target Variable:** `{pipeline_metadata['target']}`
            - **Common Evaluation Period:**  
              `{pipeline_metadata['common_start_date'].date()}` to `{pipeline_metadata['common_end_date'].date()}`
            - **Common Synchronized Origins:** `{pipeline_metadata['common_origins_per_horizon']} trading dates` per horizon
            - **Total Synchronized Forecasts:** `{pipeline_metadata.get('total_evaluations', 2235)} evaluations`
            """
        )

    with cov_col2:
        st.markdown(
            """
            - **Evaluation Protocol:** 5-fold expanding-window cross-validation
            - **Evaluation Scale:** Evaluated in yield spread levels (%) against actual realized spreads ($y_{t+h}$)
            - **Statistical Inference:** Clark-West (2007) test with HAC Bartlett kernel
            - **Multiple Testing Adjustment:** Benjamini-Hochberg False Discovery Rate (FDR $\le 0.05$)
            """
        )


# =========================================================
# 2 — EDA & FEATURE ENGINEERING
# =========================================================

with tab_eda:

    st.header("Exploratory Data Analysis & Feature Engineering")

    st.markdown(
        """
        EDA was used to understand temporal behaviour, missingness,
        cross-market relationships, stationarity, and feature
        suitability before constructing the Gold modelling layer.
        """
    )

    if gold_df is not None and (
        "yield_spread_10y_2y"
        in gold_df.columns
    ):
        spread_fig = create_gold_target_chart(
            gold_df
        )

        st.plotly_chart(
            spread_fig,
            use_container_width=True,
        )

    else:
        st.info(
            "The generated Gold CSV is not stored in the repository. "
            "The dashboard therefore summarizes the documented EDA "
            "findings below. Running the data pipeline locally will "
            "enable Gold-layer time-series visualization."
        )

    st.subheader("Key EDA Findings")

    st.dataframe(
        eda_findings,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("From Silver to Gold")

    silver_col, gold_col = st.columns(2)

    with silver_col:
        st.markdown(
            """
            ### Silver Layer

            Cleaned and aligned source datasets used for EDA:

            - Canadian yields and policy rates
            - U.S. rates
            - stitched USD/CAD
            - publication-date aligned CPI
            - missingness and calendar alignment reviewed
            """
        )

    with gold_col:
        st.markdown(
            """
            ### Gold Layer

            Modelling-ready transformations:

            - 10Y–2Y yield spread
            - additional yield spreads
            - CPI YoY
            - stationary first differences
            - common model-ready macro-financial features
            """
        )

    st.subheader(
        "Round 3 Shared Feature Set"
    )

    feature_cols = st.columns(3)

    for i, feature in enumerate(
        round3_features
    ):
        with feature_cols[i % 3]:
            st.code(feature)

    st.caption(
        "The shared Round 3 feature definition contains the "
        "10Y–2Y spread, overnight rate, U.S. Treasury 10Y, "
        "Federal Funds Rate, CPI YoY, and USD/CAD."
    )


# =========================================================
# 3 — MODEL PERFORMANCE
# =========================================================

with tab_performance:

    st.header("Forecast Performance")

    metric_choice = st.radio(
        "Performance metric",
        options=["RMSE", "MAE"],
        horizontal=True,
        key="metric_choice",
    )

    metric_fig = create_metric_comparison_chart(
        metrics_df,
        horizon=horizon,
        metric=metric_choice.lower(),
    )

    st.plotly_chart(
        metric_fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Forecasts vs Actuals")

    forecast_fig = create_forecast_vs_actual_chart(
        forecast_df,
        horizon=horizon,
        selected_models=selected_models,
        uncertainty_interval=uncertainty_interval,
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True,
    )

    st.caption(
        f"Cross-model comparisons use the {pipeline_metadata['common_origins_per_horizon']} forecast origins "
        "shared simultaneously across the core models and available benchmarks."
    )

    if regime_metrics_df is not None:
        st.divider()
        st.subheader("Macroeconomic Monetary Policy Regime Performance")
        st.markdown(
            """
            Evaluates model resilience across three distinct monetary policy regimes:
            1. **Regime 1: Rapid Tightening & Inversion** (2023)
            2. **Regime 2: Policy Plateau / Higher-for-Longer** (Jan–May 2024)
            3. **Regime 3: Easing Cycle & Un-inversion** (Jun 2024–2026)
            """
        )

        regime_fig = create_regime_comparison_chart(
            regime_metrics_df,
            horizon=horizon,
            metric=f"{metric_choice.lower()}_model",
        )
        st.plotly_chart(
            regime_fig,
            use_container_width=True,
        )

        reg_sub = regime_metrics_df[regime_metrics_df["horizon"] == horizon][
            [
                "regime",
                "model",
                "rmse_model",
                "mae_model",
                "r2_oos",
                "cw_stat",
                "cw_p_value",
            ]
        ].copy()
        reg_sub.columns = [
            "Regime",
            "Model",
            "RMSE",
            "MAE",
            "R²_OOS",
            "CW Stat",
            "CW p-value",
        ]
        st.dataframe(
            reg_sub,
            hide_index=True,
            use_container_width=True,
        )



# =========================================================
# 4 — STATISTICAL EVIDENCE
# =========================================================

with tab_statistics:

    st.header("Clark-West Statistical Evidence")

    cw_fig = create_clark_west_chart(
        cw_horizon,
        horizon=horizon,
    )

    st.plotly_chart(
        cw_fig,
        use_container_width=True,
    )

    if fdr_significant:
        significant_models = cw_horizon.loc[
            cw_horizon[
                "model_significantly_better_fdr_horizon"
            ],
            "model",
        ].tolist()

        st.success(
            "Models significant after horizon-level FDR: "
            + ", ".join(significant_models)
        )

    else:
        st.info(
            f"{strongest_cw['model']} provides the strongest "
            f"raw Clark-West evidence at h={horizon} "
            f"(p={strongest_cw['cw_p_value']:.4f}), "
            "but no model remains statistically significant "
            "after horizon-level FDR correction."
        )

    display_cols = [
        "model",
        "cw_stat",
        "cw_p_value",
        "cw_p_adj_horizon",
        "model_significantly_better",
        "model_significantly_better_fdr_horizon",
    ]

    cw_table = cw_horizon[
        display_cols
    ].copy()

    cw_table.columns = [
        "Model",
        "CW Statistic",
        "Raw p-value",
        "FDR q-value",
        "Raw Significant",
        "FDR Significant",
    ]

    st.dataframe(
        cw_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Clark-West statistics are loaded directly from the "
        "canonical project evaluation output and are not "
        "recalculated in Streamlit."
    )


# =========================================================
# 5 — FORECAST ERRORS
# =========================================================

with tab_errors:

    st.header("Forecast Error Distribution")

    error_chart_type = st.radio(
        "Distribution view",
        ["Box", "Violin"],
        horizontal=True,
    )

    error_fig = (
        create_forecast_error_distribution(
            forecast_df,
            horizon=horizon,
            chart_type=error_chart_type,
        )
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True,
    )

    st.caption(
        "Forecast error = Actual − Forecast. "
        "Distributions help reveal bias, dispersion, "
        "and extreme forecast errors."
    )


# =========================================================
# 6 — MODEL INTERPRETABILITY
# =========================================================

with tab_interpretability:

    st.header("Model Interpretability")

    shap_tab, xgb_shap_tab, irf_tab, fevd_tab = st.tabs(
        [
            "LSTM SHAP",
            "XGBoost SHAP",
            "VAR/VECM IRF",
            "VAR/VECM FEVD",
        ]
    )

    with shap_tab:

        st.markdown(
            """
            SHAP measures the relative contribution of each
            feature to the tuned LSTM forecasts.
            """
        )

        shap_fig = create_shap_summary_chart(
            shap_summary_df,
            model_name="LSTM",
        )

        st.plotly_chart(
            shap_fig,
            use_container_width=True,
        )

        top_feature = (
            shap_summary_df
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .iloc[0]
        )

        st.info(
            "Highest LSTM mean absolute SHAP feature: "
            f"`{top_feature['feature']}` "
            f"({top_feature['mean_abs_shap']:.4f})."
        )

        st.caption(
            "SHAP represents predictive contribution, "
            "not economic causality."
        )

    with xgb_shap_tab:

        st.markdown(
            """
            Tree SHAP measures the exact attribution of each lag and rolling volatility
            feature to the XGBoost gradient boosted tree predictions.
            """
        )

        if xgb_shap_summary_df is not None:
            xgb_sub = xgb_shap_summary_df[
                xgb_shap_summary_df["horizon"] == horizon
            ]
            xgb_shap_fig = create_shap_summary_chart(
                xgb_sub,
                model_name=f"XGBoost (h={horizon}d)",
                top_n=15,
            )

            st.plotly_chart(
                xgb_shap_fig,
                use_container_width=True,
            )

            top_xgb_feat = (
                xgb_sub.sort_values(
                    "mean_abs_shap",
                    ascending=False,
                ).iloc[0]
            )

            st.info(
                f"Highest XGBoost mean absolute SHAP feature at h={horizon}d: "
                f"`{top_xgb_feat['feature']}` "
                f"({top_xgb_feat['mean_abs_shap']:.6f})."
            )

            st.caption(
                "Tree SHAP values are computed deterministically "
                "across the sample space."
            )
        else:
            st.info("XGBoost SHAP summary not available.")


    with irf_tab:

        response_variable = st.selectbox(
            "Response variable",
            sorted(
                irf_df[
                    "response_variable"
                ].unique()
            ),
        )

        response_type = st.radio(
            "Response type",
            ["Instantaneous", "Cumulative"],
            horizontal=True,
        )

        irf_fig = create_irf_chart(
            irf_df,
            response_variable,
            response_type,
        )

        st.plotly_chart(
            irf_fig,
            use_container_width=True,
        )

        st.caption(
            "Impulse responses are model diagnostics and "
            "should not be interpreted as causal proof."
        )

    with fevd_tab:

        fevd_response = st.selectbox(
            "Response variable",
            sorted(
                fevd_df[
                    "response_variable"
                ].unique()
            ),
            key="fevd_response",
        )

        fevd_fig = create_fevd_chart(
            fevd_df,
            fevd_response,
        )

        st.plotly_chart(
            fevd_fig,
            use_container_width=True,
        )

        st.caption(
            "FEVD summarizes the share of forecast-error "
            "variance associated with model-system shocks."
        )


# =========================================================
# 7 — EXECUTIVE DECISION
# =========================================================

with tab_decision:

    st.header("Executive Model Decision")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best RMSE",
            f"{best_rmse_row['rmse']:.4f}",
        )
        st.caption(
            f"Model: {best_rmse_row['model']}"
        )

    with col2:
        st.metric(
            "Best MAE",
            f"{best_mae_row['mae']:.4f}",
        )
        st.caption(
            f"Model: {best_mae_row['model']}"
        )

    with col3:
        st.metric(
            "Strongest Clark-West",
            f"p = {strongest_cw['cw_p_value']:.4f}",
        )
        st.caption(
            f"Model: {strongest_cw['model']}"
        )

    with col4:
        st.metric(
            "FDR Significant?",
            "Yes" if fdr_significant else "No",
        )

    st.subheader(
        f"Decision at the {horizon}-Day Horizon"
    )

    if horizon == 1:

        st.info(
            """
            The Naïve Random Walk provides the lowest common-sample
            forecast error at the 1-day horizon. None of the more
            complex models demonstrates statistically reliable
            improvement after FDR correction.

            **Operational conclusion:** retain the Naïve benchmark
            for very short-term forecasting.
            """
        )

    elif horizon == 5:

        st.info(
            """
            The Naïve Random Walk remains the strongest common-sample
            benchmark at the 5-day horizon. LSTM is comparatively close,
            but no candidate model demonstrates statistically reliable
            improvement after FDR correction.

            **Operational conclusion:** additional model complexity is
            not justified at this horizon based on the available evidence.
            """
        )

    else:

        st.info(
            """
            VECM provides the lowest RMSE and MAE on the common 20-day
            evaluation sample and produces the strongest raw Clark-West
            evidence against the Naïve benchmark.

            However, the Clark-West result does not remain significant
            after horizon-level FDR correction.

            **Operational conclusion:** VECM is the most promising
            longer-horizon candidate, but the Naïve Random Walk remains
            the more defensible operational benchmark until the VECM
            improvement is validated more robustly.
            """
        )

    st.subheader("Overall Project Insight")

    st.success(
        """
        Increasing modelling complexity did not consistently produce
        more reliable forecasts. The Naïve Random Walk remained highly
        competitive at 1-day and 5-day horizons, while VECM showed the
        strongest evidence of improvement at 20 days. After multiple-
        testing adjustment, however, no candidate model demonstrated a
        statistically reliable advantage over the Naïve benchmark.
        """
    )

    st.subheader("Limitations")

    st.warning(
        """
        - Mixed data frequencies and publication schedules
        - CPI publication lag and temporal-alignment requirements
        - Structural and monetary-policy regime changes
        - Moderate neural-network sample size
        - Public-data-only feature coverage
        - Forecast performance may vary across market regimes
        - Common visual sample uses dynamic common origins, while canonical
          Clark-West evaluation uses 750 forecasts per horizon
        """
    )
