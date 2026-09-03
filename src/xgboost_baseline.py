"""
XGBoost / GBDT Tabular Benchmark + Rolling-Window CV (Issue #101)
DAMO-699 Capstone Project, Group 5

Trains Gradient Boosted Decision Trees on the canonical Gold-layer feature set
with causal lag engineering (t-1, ..., t-20) and rolling volatility/momentum features.
Implements expanding-window forward cross-validation matching the project conventions,
generates 90% and 95% prediction intervals (via quantile loss and empirical residual
calibration), and calculates exact Tree SHAP feature attribution.

XGBoost is the primary engine; if not installed, falls back to sklearn's
GradientBoostingRegressor so tests can run without the xgboost wheel.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

# Ensure src directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gold_feature_pipeline import build_gold_features  # noqa: E402
from project_paths import PROCESSED_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEVEL_TARGET = "yield_spread_10y_2y"
TARGET = "d_yield_spread_10y_2y"
BASE_FEATURES = [
    "d_yield_spread_10y_2y",
    "d_overnight_rate",
    "d_us_treasury_10y",
    "d_fed_funds_rate",
    "d_cpi_yoy",
    "d_usdcad",
]
HORIZONS = [1, 5, 20]
LAGS = [1, 2, 3, 4, 5, 10, 20]
MIN_TRAIN = 500
N_FOLDS = 5
SEED = 42

# Check for XGBoost availability; provide GradientBoostingRegressor fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingRegressor

_CACHED_GOLD_DF: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_common_sample(
    boc_path: Path | None = None,
    fred_path: Path | None = None,
    cpi_path: Path | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load the canonical Gold-layer feature dataset and slice to the common sample.
    Caches the result in memory so repeated calls in model comparison / diagnostics
    do not re-execute the feature engineering pipeline from disk.
    """
    global _CACHED_GOLD_DF
    if (
        _CACHED_GOLD_DF is not None
        and not force_reload
        and boc_path is None
        and fred_path is None
        and cpi_path is None
    ):
        return _CACHED_GOLD_DF.copy()

    df = build_gold_features(
        boc_path=boc_path,
        fred_path=fred_path,
        cpi_path=cpi_path,
        feature_type="all",
        save=False,
    )
    df = df.reset_index().rename(columns={"index": "date"})
    df = (
        df[["date", LEVEL_TARGET] + BASE_FEATURES]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    _CACHED_GOLD_DF = df.copy()
    return df


# ---------------------------------------------------------------------------
# 2. Tabular feature engineering
# ---------------------------------------------------------------------------

def engineer_tabular_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    horizons: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[int, str]]:
    """
    Construct strictly causal lag features, rolling volatility, momentum features,
    and multi-horizon cumulative change targets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing 'date', LEVEL_TARGET, and BASE_FEATURES.
    lags : list[int], optional
        Lag offsets (must all be >= 1). Defaults to module LAGS.
    horizons : list[int], optional
        Forecast horizons. Defaults to module HORIZONS.

    Returns
    -------
    data : pd.DataFrame
        Complete engineered dataset with dates, features, levels, and targets.
    feature_cols : list[str]
        List of engineered predictor column names.
    target_cols : dict[int, str]
        Mapping horizon -> target column name.
    """
    if lags is None:
        lags = LAGS
    if horizons is None:
        horizons = HORIZONS

    for lag in lags:
        if lag < 1:
            raise ValueError(
                f"All lag values must be >= 1 to prevent look-ahead bias, got {lag}"
            )

    feat_df = pd.DataFrame(index=df.index)
    feat_df["date"] = df["date"].values
    feat_df[LEVEL_TARGET] = df[LEVEL_TARGET].values

    feature_cols: list[str] = []

    # 2a. Causal lag features for all 6 base stationary series
    for col in BASE_FEATURES:
        for lag in lags:
            col_name = f"{col}_lag{lag}"
            feat_df[col_name] = df[col].shift(lag)
            feature_cols.append(col_name)

    # 2b. Rolling volatility and momentum features (strictly shifted by >= 1)
    target_s1 = df[TARGET].shift(1)
    feat_df["vol_target_5d"] = target_s1.rolling(5).std()
    feat_df["vol_target_20d"] = target_s1.rolling(20).std()
    feat_df["mom_target_5d"] = target_s1.rolling(5).sum()
    feat_df["mom_target_20d"] = target_s1.rolling(20).sum()

    on_s1 = df["d_overnight_rate"].shift(1)
    feat_df["vol_overnight_5d"] = on_s1.rolling(5).std()
    feat_df["vol_overnight_20d"] = on_s1.rolling(20).std()

    ust_s1 = df["d_us_treasury_10y"].shift(1)
    feat_df["vol_ust_5d"] = ust_s1.rolling(5).std()
    feat_df["vol_ust_20d"] = ust_s1.rolling(20).std()

    rolling_cols = [
        "vol_target_5d", "vol_target_20d", "mom_target_5d", "mom_target_20d",
        "vol_overnight_5d", "vol_overnight_20d", "vol_ust_5d", "vol_ust_20d",
    ]
    feature_cols.extend(rolling_cols)

    # 2c. Multi-horizon cumulative forward targets: Delta_h y_t = sum(d_t+1 .. d_t+h)
    target_cols: dict[int, str] = {}
    for h in horizons:
        t_col = f"target_h{h}"
        target_cols[h] = t_col
        feat_df[t_col] = sum(df[TARGET].shift(-k) for k in range(1, h + 1))
        feat_df[f"actual_level_h{h}"] = df[LEVEL_TARGET].shift(-h)

    # Drop rows with NaN due to lags or forward targets
    max_lag = max(max(lags), 20)  # rolling features use 20-day window
    max_h = max(horizons)
    valid_data = feat_df.iloc[max_lag:-max_h].copy().reset_index(drop=True)

    return valid_data, feature_cols, target_cols


# ---------------------------------------------------------------------------
# 3. CV fold construction
# ---------------------------------------------------------------------------

def make_rolling_folds(
    n_samples: int,
    min_train: int = MIN_TRAIN,
    n_folds: int = N_FOLDS,
) -> list[tuple[int, int]]:
    """Expanding-window rolling CV fold split indices."""
    test_pool = n_samples - min_train
    fold_size = test_pool // n_folds
    if fold_size < 1:
        raise ValueError(
            f"Insufficient samples {n_samples} for min_train={min_train} "
            f"and n_folds={n_folds}"
        )
    folds = []
    for k in range(n_folds):
        train_end = min_train + k * fold_size
        test_end = (
            n_samples if k == n_folds - 1
            else min_train + (k + 1) * fold_size
        )
        folds.append((train_end, test_end))
    return folds


# ---------------------------------------------------------------------------
# 4. Model factory
# ---------------------------------------------------------------------------

def create_model(
    loss: str = "squared_error",
    alpha: float | None = None,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    n_estimators: int = 150,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    random_state: int = SEED,
) -> Any:
    """
    Instantiate an XGBoost or sklearn GradientBoosting regressor
    for point or quantile estimation.
    """
    if HAS_XGBOOST:
        if loss == "quantile":
            return xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )
        else:
            return xgb.XGBRegressor(
                objective="reg:squarederror",
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )
    else:
        # sklearn fallback
        if loss == "quantile":
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                random_state=random_state,
            )
        else:
            return GradientBoostingRegressor(
                loss="squared_error",
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                random_state=random_state,
            )


# ---------------------------------------------------------------------------
# 5. Rolling CV pipeline
# ---------------------------------------------------------------------------

def run_rolling_cv(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    horizons: list[int] | None = None,
    min_train: int = MIN_TRAIN,
    n_folds: int = N_FOLDS,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    n_estimators: int = 150,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute expanding-window rolling CV across all horizons, generating level
    forecasts, 90% and 95% prediction intervals, and empirical coverage.

    Returns
    -------
    forecasts_df : pd.DataFrame
        Columns: [origin_date, horizon, actual, naive, xgboost, fold,
                  lower_90, upper_90, lower_95, upper_95].
    metrics_df : pd.DataFrame
        Summary RMSE, MAE, improvement vs Naive per horizon.
    intervals_df : pd.DataFrame
        Empirical coverage (%) and Mean Interval Width per horizon.
    """
    if lags is None:
        lags = LAGS
    if horizons is None:
        horizons = HORIZONS

    data, feature_cols, target_cols = engineer_tabular_features(
        df, lags=lags, horizons=horizons,
    )
    folds = make_rolling_folds(len(data), min_train=min_train, n_folds=n_folds)

    forecast_records: list[dict] = []

    for fold_id, (train_end, test_end) in enumerate(folds):
        test_data = data.iloc[train_end:test_end]
        X_test = test_data[feature_cols].values

        for h in horizons:
            t_col = target_cols[h]

            # 1. Causal h-step embargo:
            # Slicing at train_end - h guarantees that target Delta_h y_t (which spans t+1..t+h)
            # contains no realized returns from test_data (which starts at train_end).
            embargo_end = train_end - h
            if embargo_end <= 0:
                raise ValueError(
                    f"Insufficient samples for train_end={train_end} and embargo h={h}"
                )
            train_data = data.iloc[:embargo_end]

            X_train = train_data[feature_cols].values
            y_train = train_data[t_col].values

            # --- Full point prediction model fit on all pre-embargo training data ---
            model_point = create_model(
                loss="squared_error",
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
            )
            model_point.fit(X_train, y_train)
            pred_diff = model_point.predict(X_test)

            # --- Prediction intervals: Split-conformal residual calibration ---
            # To avoid in-sample residual optimism, hold out the tail of the training block as a calibration set.
            # Enforce an h-step gap between the fit set and calibration set to prevent label leakage.
            cal_ratio = 0.15
            n_tr = len(train_data)
            cal_size = max(int(n_tr * cal_ratio), 20)
            fit_end = n_tr - cal_size - h

            if fit_end >= 30:
                fit_data = train_data.iloc[:fit_end]
                cal_data = train_data.iloc[fit_end + h:]

                model_cal = create_model(
                    loss="squared_error",
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                )
                model_cal.fit(fit_data[feature_cols].values, fit_data[t_col].values)
                cal_preds = model_cal.predict(cal_data[feature_cols].values)
                cal_residuals = np.abs(cal_data[t_col].values - cal_preds)

                res_90 = float(np.quantile(cal_residuals, 0.90))
                res_95 = float(np.quantile(cal_residuals, 0.95))
            else:
                # Fallback for small fixtures in unit tests
                residuals = np.abs(y_train - model_point.predict(X_train))
                res_90 = float(np.quantile(residuals, 0.90))
                res_95 = float(np.quantile(residuals, 0.95))

            # Strictly enforce non-crossing monotonicity (95% interval >= 90% interval)
            res_95 = max(res_95, res_90)

            lower_90_diff = pred_diff - res_90
            upper_90_diff = pred_diff + res_90
            lower_95_diff = pred_diff - res_95
            upper_95_diff = pred_diff + res_95

            # Reconstruct levels: y_hat_{t+h} = y_t + Delta_hat
            current_levels = test_data[LEVEL_TARGET].values
            actual_future = test_data[f"actual_level_h{h}"].values

            pred_level = current_levels + pred_diff
            lower_90_lev = current_levels + lower_90_diff
            upper_90_lev = current_levels + upper_90_diff
            lower_95_lev = current_levels + lower_95_diff
            upper_95_lev = current_levels + upper_95_diff

            for i in range(len(test_data)):
                forecast_records.append({
                    "origin_date": str(test_data["date"].iloc[i])[:10],
                    "horizon": h,
                    "actual": float(actual_future[i]),
                    "naive": float(current_levels[i]),
                    "xgboost": float(pred_level[i]),
                    "fold": fold_id,
                    "lower_90": float(lower_90_lev[i]),
                    "upper_90": float(upper_90_lev[i]),
                    "lower_95": float(lower_95_lev[i]),
                    "upper_95": float(upper_95_lev[i]),
                })

    forecasts_df = pd.DataFrame(forecast_records)
    forecasts_df["origin_date"] = pd.to_datetime(forecasts_df["origin_date"])
    forecasts_df = forecasts_df.sort_values(
        ["origin_date", "horizon"]
    ).reset_index(drop=True)

    # --- Summary metrics vs Naive per horizon ---
    metrics_records: list[dict] = []
    interval_records: list[dict] = []

    for h in horizons:
        sub = forecasts_df[forecasts_df["horizon"] == h]
        act = sub["actual"].values
        naive = sub["naive"].values
        pred = sub["xgboost"].values

        rmse_m = np.sqrt(np.mean((act - pred) ** 2))
        mae_m = np.mean(np.abs(act - pred))
        rmse_n = np.sqrt(np.mean((act - naive) ** 2))
        mae_n = np.mean(np.abs(act - naive))

        rmse_imp = (rmse_n - rmse_m) / rmse_n * 100.0 if rmse_n > 0 else 0.0
        mae_imp = (mae_n - mae_m) / mae_n * 100.0 if mae_n > 0 else 0.0

        metrics_records.append({
            "horizon": h,
            "n_forecasts": len(sub),
            "rmse_xgboost": round(rmse_m, 5),
            "rmse_naive": round(rmse_n, 5),
            "rmse_improvement_pct": round(rmse_imp, 3),
            "mae_xgboost": round(mae_m, 5),
            "mae_naive": round(mae_n, 5),
            "mae_improvement_pct": round(mae_imp, 3),
        })

        # Coverage & MIW
        cov_90 = np.mean(
            (act >= sub["lower_90"].values) & (act <= sub["upper_90"].values)
        ) * 100.0
        cov_95 = np.mean(
            (act >= sub["lower_95"].values) & (act <= sub["upper_95"].values)
        ) * 100.0
        miw_90 = np.mean(sub["upper_90"].values - sub["lower_90"].values)
        miw_95 = np.mean(sub["upper_95"].values - sub["lower_95"].values)

        interval_records.append({
            "horizon": h,
            "target_nominal_90": 90.0,
            "empirical_coverage_90": round(cov_90, 2),
            "mean_interval_width_90": round(miw_90, 4),
            "target_nominal_95": 95.0,
            "empirical_coverage_95": round(cov_95, 2),
            "mean_interval_width_95": round(miw_95, 4),
        })

    metrics_df = pd.DataFrame(metrics_records)
    intervals_df = pd.DataFrame(interval_records)

    return forecasts_df, metrics_df, intervals_df


# ---------------------------------------------------------------------------
# 6. SHAP Interpretability
# ---------------------------------------------------------------------------

def compute_tree_shap_interpretability(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    horizons: list[int] | None = None,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    n_estimators: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train full-sample models for each horizon and compute exact SHAP values.

    Returns
    -------
    shap_summary_df : pd.DataFrame
        Mean absolute SHAP value per feature and horizon.
    shap_values_df : pd.DataFrame
        Full matrix of SHAP values.
    """
    if lags is None:
        lags = LAGS
    if horizons is None:
        horizons = HORIZONS

    data, feature_cols, target_cols = engineer_tabular_features(
        df, lags=lags, horizons=horizons,
    )
    X = data[feature_cols].values

    summary_records: list[dict] = []
    shap_val_dfs: list[pd.DataFrame] = []

    for h in horizons:
        t_col = target_cols[h]
        y = data[t_col].values

        model = create_model(
            loss="squared_error",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        vals = shap_values.values
        mean_abs_shap = np.mean(np.abs(vals), axis=0)

        for feat_name, imp in zip(feature_cols, mean_abs_shap):
            summary_records.append({
                "horizon": h,
                "feature": feat_name,
                "mean_abs_shap": round(float(imp), 6),
            })

        val_df = pd.DataFrame(
            vals, columns=[f"{col}_h{h}" for col in feature_cols]
        )
        val_df["date"] = data["date"].values
        val_df["horizon"] = h
        shap_val_dfs.append(val_df)

    shap_summary_df = pd.DataFrame(summary_records)
    shap_values_df = pd.concat(shap_val_dfs, ignore_index=True)

    return shap_summary_df, shap_values_df


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    output_dir: Path | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute the complete XGBoost tabular benchmark pipeline."""
    out_dir = output_dir if output_dir is not None else PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_common_sample()
    logger.info("Loaded Gold dataset with %d rows.", len(df))

    logger.info("Executing 5-fold expanding-window CV for XGBoost...")
    forecasts_df, metrics_df, intervals_df = run_rolling_cv(df)

    logger.info("Computing exact Tree SHAP feature attribution...")
    shap_summary_df, shap_values_df = compute_tree_shap_interpretability(df)

    if save:
        forecasts_df.to_csv(out_dir / "r3_xgboost_forecasts.csv", index=False)
        metrics_df.to_csv(out_dir / "r3_xgboost_vs_naive.csv", index=False)
        intervals_df.to_csv(
            out_dir / "r3_xgboost_prediction_intervals.csv", index=False
        )
        shap_summary_df.to_csv(
            out_dir / "r3_xgboost_shap_summary.csv", index=False
        )
        shap_values_df.to_csv(
            out_dir / "r3_xgboost_shap_values.csv", index=False
        )
        logger.info("Saved all XGBoost artifacts to %s", out_dir)

    return {
        "forecasts": forecasts_df,
        "metrics": metrics_df,
        "intervals": intervals_df,
        "shap_summary": shap_summary_df,
        "shap_values": shap_values_df,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_pipeline()
