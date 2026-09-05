import pandas as pd
import plotly.graph_objects as go


MODEL_COLUMNS = {
    "Naïve Random Walk": "naive",
    "ARIMA": "arima_aic",
    "VAR": "var_aic",
    "VECM": "vecm",
    "LSTM": "lstm",
    "XGBoost": "xgboost",
}


# =========================================================
# FORECASTS VS ACTUALS
# =========================================================

def create_forecast_vs_actual_chart(
    df: pd.DataFrame,
    horizon: int,
    selected_models: list[str],
    uncertainty_interval: str = "None",
    window_label: str | None = None,
) -> go.Figure:

    plot_df = (
        df[df["horizon"] == horizon]
        .copy()
        .sort_values("origin_date")
    )

    fig = go.Figure()

    # Optional XGBoost Prediction Interval Ribbon
    if (
        uncertainty_interval in ["90%", "95%"]
        and "XGBoost" in selected_models
    ):
        l_col = "lower_90" if uncertainty_interval == "90%" else "lower_95"
        u_col = "upper_90" if uncertainty_interval == "90%" else "upper_95"
        if l_col in plot_df.columns and u_col in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["origin_date"],
                    y=plot_df[u_col],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_df["origin_date"],
                    y=plot_df[l_col],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255, 127, 14, 0.20)",
                    name=f"XGBoost {uncertainty_interval} Band",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=plot_df["origin_date"],
            y=plot_df["actual"],
            mode="lines",
            name="Actual",
            line=dict(width=3, color="#1f77b4"),
        )
    )

    for model_name in selected_models:
        column = MODEL_COLUMNS.get(model_name)

        if column is None or column not in plot_df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=plot_df["origin_date"],
                y=plot_df[column],
                mode="lines",
                name=model_name,
            )
        )

    fig.update_layout(
        title=(
            f"Forecasts vs Actuals — {horizon}-Day Horizon"
            + (
                f"<br><sub>Forecast origins {window_label}</sub>"
                if window_label
                else ""
            )
        ),
        xaxis_title="Forecast Origin Date",
        yaxis_title="10Y–2Y Yield Spread",
        hovermode="x unified",
        legend_title="Series",
    )

    return fig



# =========================================================
# RMSE / MAE
# =========================================================

def create_metric_comparison_chart(
    metrics_df: pd.DataFrame,
    horizon: int,
    metric: str,
) -> go.Figure:

    plot_df = (
        metrics_df[
            metrics_df["horizon"] == horizon
        ]
        .copy()
        .sort_values(metric)
    )

    metric_label = metric.upper()

    fig = go.Figure(
        go.Bar(
            x=plot_df["model"],
            y=plot_df[metric],
            text=plot_df[metric].map(
                lambda x: f"{x:.4f}"
            ),
            textposition="outside",
            customdata=plot_df[
                [
                    "rmse_improvement_pct",
                    "mae_improvement_pct",
                ]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{metric_label}: %{{y:.6f}}<br>"
                "RMSE improvement vs Naïve: "
                "%{customdata[0]:.2f}%<br>"
                "MAE improvement vs Naïve: "
                "%{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{metric_label} Comparison — {horizon}-Day Horizon",
        xaxis_title="Model",
        yaxis_title=metric_label,
        showlegend=False,
    )

    return fig


# =========================================================
# CLARK-WEST
# =========================================================

def create_clark_west_chart(
    cw_df: pd.DataFrame,
    horizon: int,
) -> go.Figure:

    plot_df = (
        cw_df[
            cw_df["horizon"] == horizon
        ]
        .copy()
        .sort_values("cw_p_value")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["model"],
            y=plot_df["cw_p_value"],
            name="Raw p-value",
            text=plot_df["cw_p_value"].map(
                lambda x: f"{x:.4f}"
            ),
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            x=plot_df["model"],
            y=plot_df["cw_p_adj_horizon"],
            name="FDR-adjusted q-value",
            text=plot_df["cw_p_adj_horizon"].map(
                lambda x: f"{x:.4f}"
            ),
            textposition="outside",
        )
    )

    fig.add_hline(
        y=0.05,
        line_dash="dash",
        annotation_text="α = 0.05",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"Clark-West Significance — {horizon}-Day Horizon",
        xaxis_title="Model",
        yaxis_title="p / q value",
        barmode="group",
        yaxis=dict(range=[0, 1]),
        legend_title="Statistical Evidence",
    )

    return fig


# =========================================================
# ERROR DISTRIBUTION
# =========================================================

def create_forecast_error_distribution(
    forecast_df: pd.DataFrame,
    horizon: int,
    chart_type: str,
    selected_models: list[str] | None = None,
    window_label: str | None = None,
) -> go.Figure:

    plot_df = forecast_df[
        forecast_df["horizon"] == horizon
    ].copy()

    fig = go.Figure()

    models_to_plot = (
        MODEL_COLUMNS.items()
        if selected_models is None
        else [
            (name, col)
            for name, col in MODEL_COLUMNS.items()
            if name in selected_models
        ]
    )

    for model_name, column in models_to_plot:
        if column not in plot_df.columns:
            continue

        valid = plot_df.dropna(subset=["actual", column])
        errors = valid["actual"] - valid[column]
        if len(errors) == 0:
            continue

        trace_name = f"{model_name} (n={len(errors)})"

        if chart_type == "Violin":
            fig.add_trace(
                go.Violin(
                    y=errors,
                    name=trace_name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )

        else:
            fig.add_trace(
                go.Box(
                    y=errors,
                    name=trace_name,
                    boxmean=True,
                    boxpoints="outliers",
                )
            )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Zero error",
    )

    fig.update_layout(
        title=(
            f"Forecast Error Distribution — "
            f"{horizon}-Day Horizon"
            + (
                f"<br><sub>Forecast origins {window_label}</sub>"
                if window_label
                else ""
            )
        ),
        xaxis_title="Model",
        yaxis_title=(
            "Forecast Error "
            "(Actual − Forecast)"
        ),
        showlegend=False,
    )

    return fig


# =========================================================
# SHAP
# =========================================================

def create_shap_summary_chart(
    shap_summary_df: pd.DataFrame,
    model_name: str = "LSTM",
    top_n: int = 15,
) -> go.Figure:

    plot_df = shap_summary_df.copy()
    if len(plot_df) > top_n:
        plot_df = (
            plot_df.sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .sort_values("mean_abs_shap", ascending=True)
        )
    else:
        plot_df = plot_df.sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=plot_df["mean_abs_shap"],
            y=plot_df["feature"],
            orientation="h",
            text=plot_df["mean_abs_shap"].map(
                lambda x: f"{x:.4f}"
            ),
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"{model_name} Feature Importance — Mean Absolute SHAP",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        showlegend=False,
    )

    return fig


# =========================================================
# REGIME SEGMENTED PERFORMANCE
# =========================================================

def create_regime_comparison_chart(
    regime_df: pd.DataFrame,
    horizon: int = 1,
    metric: str = "rmse_model",
) -> go.Figure:
    sub = regime_df[regime_df["horizon"] == horizon].copy()
    metric_label = "RMSE" if "rmse" in metric else "MAE"

    fig = go.Figure()

    for model_name in sub["model"].unique():
        m_df = sub[sub["model"] == model_name]
        fig.add_trace(
            go.Bar(
                x=m_df["regime"],
                y=m_df[metric],
                name=model_name,
                text=m_df[metric].map(lambda x: f"{x:.4f}"),
                textposition="outside",
            )
        )

    fig.update_layout(
        title=f"Monetary Policy Regime Performance ({metric_label}) — Horizon {horizon}d",
        xaxis_title="Monetary Policy Regime",
        yaxis_title=metric_label,
        barmode="group",
        legend_title="Model",
    )

    return fig



# =========================================================
# IRF
# =========================================================

def create_irf_chart(
    irf_df: pd.DataFrame,
    response_variable: str,
    response_type: str,
) -> go.Figure:

    plot_df = (
        irf_df[
            irf_df["response_variable"]
            == response_variable
        ]
        .copy()
        .sort_values("horizon_days")
    )

    value_col = (
        "cumulative_response"
        if response_type == "Cumulative"
        else "response"
    )

    fig = go.Figure(
        go.Scatter(
            x=plot_df["horizon_days"],
            y=plot_df[value_col],
            mode="lines+markers",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    fig.update_layout(
        title=(
            f"{response_type} Response to "
            "an Overnight-Rate Shock"
        ),
        xaxis_title="Horizon (days)",
        yaxis_title="Response",
        showlegend=False,
    )

    return fig


# =========================================================
# FEVD
# =========================================================

def create_fevd_chart(
    fevd_df: pd.DataFrame,
    response_variable: str,
) -> go.Figure:

    plot_df = (
        fevd_df[
            fevd_df["response_variable"]
            == response_variable
        ]
        .copy()
        .sort_values("horizon_days")
    )

    fig = go.Figure()

    for shock in (
        plot_df["shock_variable"]
        .unique()
    ):
        shock_df = plot_df[
            plot_df["shock_variable"] == shock
        ]

        fig.add_trace(
            go.Scatter(
                x=shock_df["horizon_days"],
                y=shock_df["proportion"],
                mode="lines",
                stackgroup="one",
                name=shock,
            )
        )

    fig.update_layout(
        title=f"FEVD — {response_variable}",
        xaxis_title="Horizon (days)",
        yaxis_title="Forecast Error Variance Share",
        yaxis_tickformat=".0%",
        legend_title="Shock Variable",
    )

    return fig


# =========================================================
# OPTIONAL GOLD FEATURE CHART
# =========================================================

def create_gold_target_chart(
    gold_df: pd.DataFrame,
) -> go.Figure:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=gold_df["date"],
            y=gold_df["yield_spread_10y_2y"],
            mode="lines",
            name="10Y–2Y spread",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Inversion threshold",
    )

    fig.update_layout(
        title="Canadian 10Y–2Y Yield Spread",
        xaxis_title="Date",
        yaxis_title="Spread (%)",
    )

    return fig
