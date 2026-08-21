"""
Round 3 pairwise Diebold-Mariano comparison across all baselines (issue #50)
DAMO-699 Capstone, Group 5

Compares Naive, ARIMA-AIC, ARIMA-BIC, VAR-AIC, VAR-BIC, VECM (6-var), and LSTM
pairwise at 1-/5-/20-day horizons, per proposal Section 5.4/Section 7.

ARIMA/VAR-AIC/VAR-BIC share one data pipeline (src/EDA_VAR_AIC _lag _order.py's
load_levels(), inner-joined, 698 origins) and agree bit-for-bit on origin_date,
actual, and naive. VECM and LSTM are built on a different pipeline
(src/gold_feature_pipeline.py's build_gold_features(), outer-joined + ffilled),
which does not share the same business-day grid -- "5 trading days ahead" of
the same origin_date can land on a different real calendar date depending on
which grid produced it. Verified empirically: merging VECM/LSTM's forecasts
against ARIMA's on (origin_date, horizon) alone produces "actual" mismatches in
up to 74% of rows at h=20.

So any comparison crossing the two pipeline families is filtered to rows where
BOTH sides' recorded `actual` value agree (see `merge_cross_pipeline` below) --
that is proof the two origins really do refer to the same calendar span, not
just the same label. This shrinks the usable sample for those pairs (reported
per-pair as n_forecasts, not assumed constant across the comparison table) but
keeps every surviving pair genuinely paired. Reconciling the two pipelines onto
one shared calendar end-to-end is a bigger fix, tracked separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dieboldmariano import dm_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROJECT_ROOT  # noqa: E402

ALPHA = 0.05
HORIZONS = [1, 5, 20]
ACTUAL_TOL = 1e-9  # float-equality tolerance when confirming cross-pipeline pairs

OUT_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# 1. Load each baseline's per-origin forecasts into a common set of "arms"
# ---------------------------------------------------------------------------

def load_core() -> pd.DataFrame:
    """
    Naive, ARIMA-AIC, ARIMA-BIC, VAR-AIC, VAR-BIC all share one pipeline
    (load_levels(), 698 origins). Merges them into one frame -- exact-matching
    on (origin_date, horizon), verified bit-identical actual/naive across all
    three source files, so no row is dropped here.
    """
    arima = pd.read_csv(OUT_DIR / "r3_arima_forecasts.csv", parse_dates=["origin_date"])
    var_aic = pd.read_csv(OUT_DIR / "r3_patha_var_aic_forecasts.csv", parse_dates=["origin_date"])
    var_bic = pd.read_csv(OUT_DIR / "r3_pathb_var_bic_forecasts.csv", parse_dates=["origin_date"])

    core = arima[["origin_date", "horizon", "actual", "naive", "arima_aic", "arima_bic"]].merge(
        var_aic[["origin_date", "horizon", "actual", "naive", "var_aic"]],
        on=["origin_date", "horizon", "actual", "naive"], how="inner",
    ).merge(
        var_bic[["origin_date", "horizon", "actual", "naive", "var_bic"]],
        on=["origin_date", "horizon", "actual", "naive"], how="inner",
    )
    expected = len(arima)
    if len(core) != expected:
        raise AssertionError(
            f"core merge dropped rows ({len(core)} of {expected}) -- ARIMA/VAR-AIC/VAR-BIC "
            f"were expected to be bit-identical on (origin_date, horizon, actual, naive). "
            f"Re-verify load_levels() hasn't diverged across the three scripts."
        )
    return core


def load_vecm() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "r3_vecm_6var_forecasts.csv", parse_dates=["origin_date"])
    return df[["origin_date", "horizon", "actual", "naive", "vecm"]]


def load_lstm(date_min, date_max) -> pd.DataFrame:
    """
    LSTM's rolling-CV folds cover ~3,728 origins, far more than the other
    baselines' 698 -- restrict to the same calendar span before comparison
    (see issue #50 decision log) so LSTM isn't scored on years the other
    baselines were never evaluated on.
    """
    df = pd.read_csv(OUT_DIR / "r3_lstm_forecasts.csv", parse_dates=["origin_date"])
    df = df[(df["origin_date"] >= date_min) & (df["origin_date"] <= date_max)]
    return df[["origin_date", "horizon", "actual", "naive", "lstm"]]


def merge_cross_pipeline(
    left: pd.DataFrame, left_col: str, right: pd.DataFrame, right_col: str,
) -> pd.DataFrame:
    """
    Join two forecast frames from different pipelines on (origin_date, horizon),
    then keep only rows where both sides' `actual` agree within ACTUAL_TOL --
    proof the origin+horizon really points at the same calendar-date target on
    both sides, not just a matching label. See module docstring.
    """
    m = left[["origin_date", "horizon", "actual", "naive", left_col]].merge(
        right[["origin_date", "horizon", "actual", "naive", right_col]],
        on=["origin_date", "horizon"], suffixes=("_l", "_r"),
    )
    agree = (m["actual_l"] - m["actual_r"]).abs() < ACTUAL_TOL
    m = m[agree].copy()
    m["actual"] = m["actual_l"]
    # naive is anchored on the origin's own last_level, not the h-step-ahead
    # target, so it's far less prone to cross-pipeline drift -- but confirm it
    # rather than assume it, same as actual.
    naive_agree = (m["naive_l"] - m["naive_r"]).abs() < ACTUAL_TOL
    m = m[naive_agree]
    m["naive"] = m["naive_l"]
    return m[["origin_date", "horizon", "actual", "naive", left_col, right_col]]


# ---------------------------------------------------------------------------
# 2. Generic pairwise Diebold-Mariano test
# ---------------------------------------------------------------------------

def pairwise_dm(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, h: int) -> dict:
    """
    Runs the DM test for both squared-error and absolute-error loss, per the
    #43/#44 fix (both loss functions' verdicts reported independently, never
    one derived from the other). Convention: dm_stat < 0 means `pred_a` has
    the lower loss (is "better"); matches src/EDA_VAR_AIC _lag _order.py's
    dm_report() and src/johansen_vecm.py's dm_report_vecm().
    """
    dm_sq, p_sq = dm_test(
        actual, pred_a, pred_b,
        loss=lambda u, v: (u - v) ** 2,
        h=h, harvey_correction=True, variance_estimator="bartlett",
    )
    dm_abs, p_abs = dm_test(
        actual, pred_a, pred_b,
        loss=lambda u, v: abs(u - v),
        h=h, harvey_correction=True, variance_estimator="bartlett",
    )
    a_better_rmse = bool(p_sq < ALPHA and dm_sq < 0)
    b_better_rmse = bool(p_sq < ALPHA and dm_sq > 0)
    a_better_mae = bool(p_abs < ALPHA and dm_abs < 0)
    b_better_mae = bool(p_abs < ALPHA and dm_abs > 0)
    return {
        "n_forecasts": len(actual),
        "dm_stat_squared_loss": round(dm_sq, 3),
        "dm_p_value_squared_loss": round(p_sq, 4),
        "dm_stat_absolute_loss": round(dm_abs, 3),
        "dm_p_value_absolute_loss": round(p_abs, 4),
        "a_significantly_better_rmse": a_better_rmse,
        "b_significantly_better_rmse": b_better_rmse,
        "a_significantly_better_mae": a_better_mae,
        "b_significantly_better_mae": b_better_mae,
    }


def plain_language_verdict(row: dict, name_a: str, name_b: str) -> str:
    a_rmse, b_rmse = row["a_significantly_better_rmse"], row["b_significantly_better_rmse"]
    a_mae, b_mae = row["a_significantly_better_mae"], row["b_significantly_better_mae"]

    if a_rmse and a_mae:
        return f"{name_a} significantly better than {name_b} (RMSE and MAE agree)"
    if b_rmse and b_mae:
        return f"{name_b} significantly better than {name_a} (RMSE and MAE agree)"
    if (a_rmse and b_mae) or (b_rmse and a_mae):
        return (
            f"mixed result: RMSE and MAE are both significant but disagree on which of "
            f"{name_a}/{name_b} wins -- don't headline this as a robust result"
        )
    if a_rmse or b_rmse:
        winner = name_a if a_rmse else name_b
        p = row["dm_p_value_squared_loss"]
        return f"{winner} significantly better than the rival on RMSE only (p={p:.4f}); MAE not significant"
    if a_mae or b_mae:
        winner = name_a if a_mae else name_b
        p = row["dm_p_value_absolute_loss"]
        return f"{winner} significantly better than the rival on MAE only (p={p:.4f}); RMSE not significant"
    return "no significant difference on either RMSE or MAE"
