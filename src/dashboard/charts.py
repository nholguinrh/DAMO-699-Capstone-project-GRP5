import pandas as pd
import plotly.graph_objects as go


MODEL_COLUMNS = {
    "Naïve Random Walk": "naive",
    "ARIMA": "arima_aic",
    "VAR": "var_aic",
    "VECM": "vecm",
    "LSTM": "lstm",
}


# =========================================================
# FORECASTS VS ACTUALS
# =========================================================

def create_forecast_vs_actual_chart(
    df: pd.DataFrame,
    horizon: int,
    selected_models: list[str],
) -> go.Figure:

    plot_df = (
        df[df["horizon"] == horizon]
        .copy()
        .sort_values("origin_date")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["origin_date"],
            y=plot_df["actual"],
            mode="lines",
            name="Actual",
            line=dict(width=3),
        )
    )

    for model_name in selected_models:
        column = MODEL_COLUMNS.get(model_name)

        if column is None:
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
        title=f"Forecasts vs Actuals — {horizon}-Day Horizon",
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
) -> go.Figure:

    plot_df = forecast_df[
        forecast_df["horizon"] == horizon
    ].copy()

    fig = go.Figure()

    for model_name, column in MODEL_COLUMNS.items():
        errors = (
            plot_df["actual"]
            - plot_df[column]
        )

        if chart_type == "Violin":
            fig.add_trace(
                go.Violin(
                    y=errors,
                    name=model_name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )

        else:
            fig.add_trace(
                go.Box(
                    y=errors,
                    name=model_name,
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
) -> go.Figure:

    plot_df = (
        shap_summary_df
        .copy()
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

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
        title="LSTM Feature Importance — Mean Absolute SHAP",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        showlegend=False,
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