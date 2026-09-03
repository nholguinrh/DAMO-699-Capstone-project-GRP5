import streamlit as st

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
    load_shap_summary,
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
    ],
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
        The project integrates three institutional public-data
        sources into a reproducible analytics workflow.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Institutional Sources",
            pipeline_metadata["data_sources"],
        )

    with col2:
        st.metric(
            "Candidate Models",
            pipeline_metadata["candidate_models"],
        )

    with col3:
        st.metric(
            "Forecast Horizons",
            pipeline_metadata["forecast_horizons"],
        )

    with col4:
        st.metric(
            "Common Origins / Horizon",
            pipeline_metadata[
                "common_origins_per_horizon"
            ],
        )

    st.subheader("Data Sources")

    st.markdown(
        """
        **Bank of Canada Valet API**
        - Government of Canada yields
        - Overnight policy rate
        - USD/CAD exchange rate

        **Federal Reserve FRED API**
        - U.S. Treasury 10-Year Yield
        - Federal Funds Effective Rate

        **Statistics Canada Web Data Service**
        - Consumer Price Index
        - Publication-date aligned CPI YoY
        """
    )

    st.subheader("Project Pipeline")

    st.code(
        """
Bank of Canada API ─────┐
FRED API ───────────────┼──> BRONZE
Statistics Canada API ──┘
                              ↓
                           SILVER
                              ↓
                    EDA & DIAGNOSTICS
                              ↓
               FEATURE ENGINEERING / SELECTION
                              ↓
                            GOLD
                              ↓
         Naïve | ARIMA | VAR/VECM | LSTM
                              ↓
                RMSE / MAE + Clark-West
                              ↓
                  EXECUTIVE DASHBOARD
        """,
        language="text",
    )

    st.info(
        "In this project, exploratory analysis and diagnostics "
        "were performed using the cleaned Silver-layer data. "
        "Those findings informed the transformations and features "
        "retained in the canonical Gold-layer modelling dataset."
    )

    st.subheader("Evaluation Coverage")

    st.write(
        f"""
        **Target:** {pipeline_metadata['target']}

        **Common forecast comparison period:**  
        {pipeline_metadata['common_start_date'].date()}
        to
        {pipeline_metadata['common_end_date'].date()}
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
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True,
    )

    st.caption(
        "Cross-model comparisons use the 745 forecast origins "
        "shared simultaneously by ARIMA, VAR, VECM, and LSTM."
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

    shap_tab, irf_tab, fevd_tab = st.tabs(
        [
            "LSTM SHAP",
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
            shap_summary_df
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
        - Common visual sample uses 745 origins, while canonical
          Clark-West evaluation uses 750 forecasts per horizon
        """
    )