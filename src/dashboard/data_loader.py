from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _read_csv(filename: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required dashboard file was not found: {path}"
        )

    return pd.read_csv(path)


# =========================================================
# FORECAST OUTPUTS
# =========================================================

def load_arima_forecasts() -> pd.DataFrame:
    df = _read_csv("r3_arima_forecasts.csv")
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


def load_var_forecasts() -> pd.DataFrame:
    df = _read_csv("r3_macroaug_var_aic_forecasts.csv")
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


def load_vecm_forecasts() -> pd.DataFrame:
    df = _read_csv("r3_vecm_6var_forecasts.csv")
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


def load_lstm_forecasts() -> pd.DataFrame:
    df = _read_csv("r3_lstm_forecasts.csv")
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


def load_xgboost_forecasts() -> pd.DataFrame | None:
    path = OUTPUTS_DIR / "r3_xgboost_forecasts.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


# =========================================================
# EVALUATION OUTPUTS
# =========================================================

def load_clark_west_results() -> pd.DataFrame:
    return _read_csv("clark_west_test_results.csv")


def load_arima_metrics() -> pd.DataFrame:
    return _read_csv("r3_arima_vs_naive.csv")


def load_var_metrics() -> pd.DataFrame:
    return _read_csv("r3_macroaug_rmse_mae_vs_naive.csv")


def load_vecm_metrics() -> pd.DataFrame:
    return _read_csv("vecm_rmse_mae_vs_naive.csv")


def load_lstm_metrics() -> pd.DataFrame:
    return _read_csv("r3_lstm_vs_naive.csv")


def load_xgboost_metrics() -> pd.DataFrame | None:
    path = OUTPUTS_DIR / "r3_xgboost_vs_naive.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# =========================================================
# INTERPRETABILITY OUTPUTS
# =========================================================

def load_shap_summary() -> pd.DataFrame:
    return _read_csv("r3_lstm_shap_summary.csv")


def load_xgboost_shap_summary() -> pd.DataFrame | None:
    path = OUTPUTS_DIR / "r3_xgboost_shap_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_fevd_results() -> pd.DataFrame:
    return _read_csv("fevd_results.csv")


def load_irf_results() -> pd.DataFrame:
    return _read_csv("irf_overnight_rate_shock.csv")


# =========================================================
# OPTIONAL GOLD DATA
# =========================================================

def load_gold_features() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "gold_features.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


# =========================================================
# COMMON FORECAST DATASET
# =========================================================

def build_common_forecast_dataset() -> pd.DataFrame:
    arima = load_arima_forecasts().copy()
    var = load_var_forecasts().copy()
    vecm = load_vecm_forecasts().copy()
    lstm = load_lstm_forecasts().copy()

    keys = ["origin_date", "horizon"]

    arima = arima[
        keys + ["actual", "naive", "arima_aic"]
    ].rename(
        columns={
            "actual": "actual_arima",
            "naive": "naive_arima",
        }
    )

    var = var[
        keys + ["actual", "naive", "var_aic"]
    ].rename(
        columns={
            "actual": "actual_var",
            "naive": "naive_var",
        }
    )

    vecm = vecm[
        keys + ["actual", "naive", "vecm"]
    ].rename(
        columns={
            "actual": "actual_vecm",
            "naive": "naive_vecm",
        }
    )

    lstm = lstm[
        keys + ["actual", "naive", "lstm"]
    ].rename(
        columns={
            "actual": "actual_lstm",
            "naive": "naive_lstm",
        }
    )

    common = (
        arima
        .merge(var, on=keys, how="inner")
        .merge(vecm, on=keys, how="inner")
        .merge(lstm, on=keys, how="inner")
    )

    xgb = load_xgboost_forecasts()
    has_xgb = False
    if xgb is not None and not xgb.empty:
        xgb_sub = xgb[
            keys + ["actual", "naive", "xgboost", "lower_90", "upper_90", "lower_95", "upper_95"]
        ].rename(
            columns={
                "actual": "actual_xgb",
                "naive": "naive_xgb",
            }
        )
        common = common.merge(xgb_sub, on=keys, how="inner")
        has_xgb = True

    check_actual_cols = ["actual_var", "actual_vecm", "actual_lstm"]
    check_naive_cols = ["naive_var", "naive_vecm", "naive_lstm"]
    if has_xgb:
        check_actual_cols.append("actual_xgb")
        check_naive_cols.append("naive_xgb")

    for col in check_actual_cols:
        if not np.allclose(
            common["actual_arima"],
            common[col],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"Actual values are inconsistent between ARIMA and {col}."
            )

    for col in check_naive_cols:
        if not np.allclose(
            common["naive_arima"],
            common[col],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"Naïve forecasts are inconsistent between ARIMA and {col}."
            )

    common["actual"] = common["actual_arima"]
    common["naive"] = common["naive_arima"]

    out_cols = [
        "origin_date",
        "horizon",
        "actual",
        "naive",
        "arima_aic",
        "var_aic",
        "vecm",
        "lstm",
    ]
    if has_xgb:
        out_cols.extend(["xgboost", "lower_90", "upper_90", "lower_95", "upper_95"])

    return (
        common[out_cols]
        .sort_values(["horizon", "origin_date"])
        .reset_index(drop=True)
    )


# =========================================================
# COMMON-SAMPLE METRICS
# =========================================================

def build_common_sample_metrics() -> pd.DataFrame:
    df = build_common_forecast_dataset()

    model_columns = {
        "Naïve Random Walk": "naive",
        "ARIMA": "arima_aic",
        "VAR": "var_aic",
        "VECM": "vecm",
        "LSTM": "lstm",
    }
    if "xgboost" in df.columns:
        model_columns["XGBoost"] = "xgboost"


    rows = []

    for horizon in sorted(df["horizon"].unique()):
        horizon_df = df[df["horizon"] == horizon]

        for model_name, forecast_col in model_columns.items():
            errors = (
                horizon_df["actual"]
                - horizon_df[forecast_col]
            )

            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "n_forecasts": len(horizon_df),
                    "rmse": float(
                        np.sqrt(np.mean(errors ** 2))
                    ),
                    "mae": float(
                        np.mean(np.abs(errors))
                    ),
                }
            )

    metrics = pd.DataFrame(rows)

    naive = (
        metrics[
            metrics["model"] == "Naïve Random Walk"
        ][["horizon", "rmse", "mae"]]
        .rename(
            columns={
                "rmse": "naive_rmse",
                "mae": "naive_mae",
            }
        )
    )

    metrics = metrics.merge(
        naive,
        on="horizon",
        how="left",
    )

    metrics["rmse_improvement_pct"] = (
        (metrics["naive_rmse"] - metrics["rmse"])
        / metrics["naive_rmse"]
        * 100
    )

    metrics["mae_improvement_pct"] = (
        (metrics["naive_mae"] - metrics["mae"])
        / metrics["naive_mae"]
        * 100
    )

    return (
        metrics
        .sort_values(["horizon", "rmse"])
        .reset_index(drop=True)
    )


# =========================================================
# CLARK-WEST
# =========================================================

def get_clark_west_summary(
    horizon: int,
) -> pd.DataFrame:
    cw = load_clark_west_results().copy()
    return cw[cw["horizon"] == horizon].copy()


# =========================================================
# PIPELINE METADATA
# =========================================================

def get_pipeline_metadata() -> dict:
    common = build_common_forecast_dataset()

    return {
        "data_sources": 3,
        "candidate_models": "5 (+ Naïve)",
        "forecast_horizons": "1 / 5 / 20 days",
        "common_origins_per_horizon": int(
            common.groupby("horizon").size().min()
        ),
        "common_start_date": common["origin_date"].min(),
        "common_end_date": common["origin_date"].max(),
        "target": "Canadian 10Y–2Y yield spread",
        "total_evaluations": len(common),
    }


# =========================================================
# FEATURE DEFINITIONS
# =========================================================

def get_round3_features() -> list[str]:
    return [
        "yield_spread_10y_2y",
        "overnight_rate",
        "us_treasury_10y",
        "fed_funds_rate",
        "cpi_yoy",
        "usdcad",
    ]


def get_eda_findings() -> pd.DataFrame:
    """
    EDA findings documented in the project EDA materials.
    """

    rows = [
        {
            "Topic": "Yield spread range",
            "Finding": (
                "10Y–2Y spread ranged approximately from "
                "-1.32% to +2.32%."
            ),
        },
        {
            "Topic": "Latest curve condition",
            "Finding": (
                "The 10Y–2Y spread was approximately +0.64% "
                "on 2026-06-30, indicating a normal upward slope."
            ),
        },
        {
            "Topic": "Cross-maturity co-movement",
            "Finding": (
                "2Y and 10Y yield first differences showed "
                "strong co-movement of approximately 0.80."
            ),
        },
        {
            "Topic": "Canada–U.S. long-end relationship",
            "Finding": (
                "Canadian 10Y and U.S. Treasury 10Y first "
                "differences showed approximately 0.85 correlation."
            ),
        },
        {
            "Topic": "Stationarity",
            "Finding": (
                "Core rate and yield series were non-stationary "
                "in levels and stationary after first differencing."
            ),
        },
        {
            "Topic": "CPI timing",
            "Finding": (
                "CPI was aligned using publication date to prevent "
                "look-ahead bias."
            ),
        },
    ]

    return pd.DataFrame(rows)


def load_regime_metrics() -> pd.DataFrame | None:
    """
    Loads macroeconomic monetary policy regime segmented metrics (Issue #101).
    """
    path = OUTPUTS_DIR / "r3_regime_segmented_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_xgboost_prediction_intervals() -> pd.DataFrame | None:
    """
    Loads XGBoost empirical coverage and interval width metrics (Issue #101).
    """
    path = OUTPUTS_DIR / "r3_xgboost_prediction_intervals.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)