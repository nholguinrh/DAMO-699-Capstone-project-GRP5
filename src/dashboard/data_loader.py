import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make ``project_paths`` importable whether the dashboard is launched as
# ``streamlit run src/dashboard/app.py`` (src/dashboard on sys.path) or via the
# repo-root ``streamlit_app.py`` shim.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from project_paths import OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT  # noqa: E402, F401


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

@st.cache_data(show_spinner=False)
def load_clark_west_results() -> pd.DataFrame:
    return _read_csv("clark_west_test_results.csv")


@st.cache_data(show_spinner=False)
def load_clark_west_sensitivity_results() -> pd.DataFrame | None:
    path = OUTPUTS_DIR / "clark_west_sensitivity_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


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

@st.cache_data(show_spinner=False)
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

    # Core 4-model parity check (strictly 745 common origins)
    for col in ["actual_var", "actual_vecm", "actual_lstm"]:
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

    for col in ["naive_var", "naive_vecm", "naive_lstm"]:
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
        # Left-join overlay preserves the core 4-model 745 common origins
        common = common.merge(xgb_sub, on=keys, how="left")
        has_xgb = True

        xgb_valid = common["actual_xgb"].notna()
        if xgb_valid.any():
            if not np.allclose(
                common.loc[xgb_valid, "actual_arima"],
                common.loc[xgb_valid, "actual_xgb"],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    "Actual values are inconsistent between ARIMA and actual_xgb."
                )
            if not np.allclose(
                common.loc[xgb_valid, "naive_arima"],
                common.loc[xgb_valid, "naive_xgb"],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    "Naïve forecasts are inconsistent between ARIMA and naive_xgb."
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
# ORIGIN-WINDOW FILTERING (sidebar date-range slider)
# =========================================================

def filter_by_origin_window(
    df: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Restrict a forecast frame to origins in the closed day-range [start, end].

    Implemented as a half-open interval on the underlying timestamps so the
    final day is retained even if ``origin_date`` ever carries an intraday
    component (``pd.Timestamp(date)`` normalizes to midnight, which a ``<=``
    bound would silently exclude). Row order and index are preserved and
    cardinality is non-increasing.
    """
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df["origin_date"] >= lo) & (df["origin_date"] < hi)]


def window_model_coverage(
    df: pd.DataFrame,
    model_columns: dict[str, str],
) -> pd.DataFrame:
    """Per-model non-null forecast counts within an already-filtered frame.

    Used to disclose which models have zero coverage in a user-selected
    origin window (e.g. XGBoost's later common-sample start date) and
    whether the remaining models share a common origin count -- i.e.
    whether a cross-model comparison in that window is still paired.
    """
    return pd.DataFrame(
        [
            {"model": name, "n_origins": int(df[col].notna().sum())}
            for name, col in model_columns.items()
            if col in df.columns
        ]
    )


def min_covered_origins(coverage_df: pd.DataFrame) -> int:
    """
    Smallest non-zero per-model origin count in a ``window_model_coverage()``
    result, or 0 if no model has any coverage.

    Used to gate distribution charts (quartiles/whiskers/KDE) on sample
    size: the weakest-covered model actually being plotted determines
    whether those statistics are still meaningful, not the best-covered
    one -- taking the max here would let one well-covered model wave
    through a box built from another plotted model's much smaller sample.
    """
    present = coverage_df.loc[coverage_df["n_origins"] > 0, "n_origins"]
    return int(present.min()) if not present.empty else 0


# =========================================================
# COMMON-SAMPLE METRICS
# =========================================================

@st.cache_data(show_spinner=False)
def build_common_sample_metrics() -> pd.DataFrame:
    df = build_common_forecast_dataset()

    model_columns = {
        "Naïve Random Walk": "naive",
        "ARIMA-AIC": "arima_aic",
        "VAR-AIC": "var_aic",
        "VECM (6-var)": "vecm",
        "LSTM (Tuned)": "lstm",
    }
    if "xgboost" in df.columns:
        model_columns["XGBoost"] = "xgboost"


    rows = []

    for horizon in sorted(df["horizon"].unique()):
        horizon_df = df[df["horizon"] == horizon]

        for model_name, forecast_col in model_columns.items():
            # Enforce strict paired evaluation: compute both model and naive errors on identical non-null origins
            valid_m = horizon_df.dropna(subset=["actual", "naive", forecast_col])
            errors = valid_m["actual"] - valid_m[forecast_col]
            naive_errors = valid_m["actual"] - valid_m["naive"]

            model_rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else 0.0
            model_mae = float(np.mean(np.abs(errors))) if len(errors) > 0 else 0.0
            paired_naive_rmse = float(np.sqrt(np.mean(naive_errors ** 2))) if len(naive_errors) > 0 else 0.0
            paired_naive_mae = float(np.mean(np.abs(naive_errors))) if len(naive_errors) > 0 else 0.0

            rmse_imp = (
                ((paired_naive_rmse - model_rmse) / paired_naive_rmse * 100.0)
                if paired_naive_rmse > 0 else 0.0
            )
            mae_imp = (
                ((paired_naive_mae - model_mae) / paired_naive_mae * 100.0)
                if paired_naive_mae > 0 else 0.0
            )

            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "n_forecasts": len(valid_m),
                    "rmse": model_rmse,
                    "mae": model_mae,
                    "naive_rmse": paired_naive_rmse,
                    "naive_mae": paired_naive_mae,
                    "rmse_improvement_pct": rmse_imp,
                    "mae_improvement_pct": mae_imp,
                }
            )

    metrics = pd.DataFrame(rows)

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


@st.cache_data(show_spinner=False)
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
