"""
Round 3 pairwise Diebold-Mariano comparison across all baselines (issue #50)
DAMO-699 Capstone, Group 5

Compares Naive, ARIMA-AIC, ARIMA-BIC, VAR-AIC, VAR-BIC, VECM (6-var), and LSTM
pairwise at 1-/5-/20-day horizons, per proposal Section 5.4/Section 7.

As of issue #63, ARIMA/VAR-AIC/VAR-BIC (src/EDA_VAR_AIC _lag _order.py's load_levels())
and VECM/LSTM (src/gold_feature_pipeline.py's build_gold_features()) all read the same
canonical data/processed/gold_features.csv, and share one identical business-day
calendar (verified: core_calendar()/vecm_calendar()/lstm_calendar() are index-identical
over their common range) -- 750 origins each for ARIMA/VAR-AIC/VAR-BIC/VECM, LSTM's own
larger rolling-CV calendar covers it as a strict superset. Before #63, ARIMA/VAR-AIC/
VAR-BIC instead independently rebuilt an inner-joined, no-fill frame (698 origins) that
silently disagreed with build_gold_features()'s outer-join + ffill(limit=2) -- "5
trading days ahead" of the same origin_date could land on a different real calendar
date depending on which pipeline produced it.

Any cross-pipeline comparison is still verified by walking each side's own calendar
forward `horizon` positions from `origin_date` and requiring the resulting real target
date to match on both sides (see `attach_target_date` / `merge_cross_pipeline` below)
-- not by comparing the `actual` values themselves. An earlier version of this module
did compare `actual` with a float tolerance instead; that check is not sufficient proof
of a same-calendar target, because build_gold_features() forward-fills yield levels
before the target spread is computed (~16% of rows in gold_features.csv repeat their
immediately-prior value), so two genuinely different real target dates can
coincidentally carry the same `actual` value and pass a value-based check. Verified on
the shipped PR #64 output: 12 of 30 ARIMA-AIC-vs-VECM rows at h=20 had matching `actual`
values but different real target dates.

#63 also fixed a separate, deeper disagreement in what "origin" means, found while
fixing the calendar-*source* one: ARIMA/VAR-AIC/VAR-BIC's loops used to define an origin
by how many *differenced* observations it had trained on (`train_diff = diffed.iloc[:origin]`),
while evaluate_vecm()'s loop defines it by how many *level* observations it has trained
on (`train_levels = levels.iloc[:origin]`) -- a one-row phase offset for the identical
`origin` index number, since one differenced observation is "used up" reconstructing the
first level. On the newly-unified calendar this meant origin_date values from the two
loop families stopped coinciding almost entirely (0 of 750x750 pairs shared an
origin_date, down from the ~19%/133 that happened to coincide against the old, less
regular inner-joined calendar -- ffill(limit=2) makes the calendar *more* strictly
periodic, which removes the holiday irregularities that used to occasionally break the
phase alignment back into sync). merge_cross_pipeline()'s inner join is on `origin_date`
first, so this phase offset -- not target-date mismatch -- was what would have zeroed out
every ARIMA/VAR-vs-VECM/LSTM pairwise comparison. Fixed by re-anchoring
ARIMA/VAR-AIC/VAR-BIC's loops onto the same level-counting convention evaluate_vecm()
uses (`src/EDA_VAR_AIC _lag _order.py`'s `evaluate()`, `arima_baseline.ipynb`,
`var_bic_baseline.ipynb`) -- verified: origin_date now matches 750/750 against VECM.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import scipy.stats as stats
from dieboldmariano import (
    InvalidParameterException,
    NegativeVarianceException,
    ZeroVarianceException,
    dm_test,
)
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import OUTPUTS_DIR as OUT_DIR, PROCESSED_DIR, PROJECT_ROOT, write_data_manifest  # noqa: E402

ALPHA = 0.05
HORIZONS = [1, 5, 20]
ACTUAL_TOL = 1e-9  # float tolerance for the actual-value sanity check, not the alignment check

_DM_EXCEPTIONS = (InvalidParameterException, ZeroVarianceException, NegativeVarianceException)
_RESERVED_COLS = {"origin_date", "horizon", "actual", "naive", "target_date"}


# ---------------------------------------------------------------------------
# 1. Load each baseline's per-origin forecasts into a common set of "arms"
# ---------------------------------------------------------------------------

def load_core() -> pd.DataFrame:
    """
    Naive, ARIMA-AIC, ARIMA-BIC, VAR-AIC, VAR-BIC all share one pipeline
    (load_levels(), 750 origins as of issue #63 -- was 698 before the Gold-layer
    calendar migration). Merges them into one frame -- exact-matching on
    (origin_date, horizon), verified bit-identical actual/naive across all
    three source files, so no row is dropped here.
    """
    arima = pd.read_csv(OUT_DIR / "r3_arima_forecasts.csv", parse_dates=["origin_date"])
    if arima.empty:
        raise AssertionError(
            "outputs/r3_arima_forecasts.csv is empty -- re-run "
            "`src/EDA_VAR_AIC _lag _order.py` before comparing models."
        )
    var_aic = pd.read_csv(OUT_DIR / "r3_patha_var_aic_forecasts.csv", parse_dates=["origin_date"])
    var_bic = pd.read_csv(OUT_DIR / "r3_pathb_var_bic_forecasts.csv", parse_dates=["origin_date"])

    core = arima[["origin_date", "horizon", "actual", "naive", "arima_aic", "arima_bic"]].merge(
        var_aic[["origin_date", "horizon", "actual", "naive", "var_aic"]],
        on=["origin_date", "horizon", "actual", "naive"], how="inner",
    ).merge(
        var_bic[["origin_date", "horizon", "actual", "naive", "var_bic"]],
        on=["origin_date", "horizon", "actual", "naive"], how="inner",
    )
    if len(core) != len(arima):
        raise AssertionError(
            f"core merge dropped rows ({len(core)} of {len(arima)}) -- ARIMA/VAR-AIC/VAR-BIC "
            f"were expected to be bit-identical on (origin_date, horizon, actual, naive). "
            f"Re-verify load_levels() hasn't diverged across the three scripts."
        )
    return core


def load_vecm(date_min=None, date_max=None) -> pd.DataFrame:
    """
    date_min/date_max restrict to a shared comparison window -- pass core's
    date range so every non-core arm (VECM, LSTM) is judged over the same span,
    rather than only restricting whichever arm happens to have the larger range.
    """
    df = pd.read_csv(OUT_DIR / "r3_vecm_6var_forecasts.csv", parse_dates=["origin_date"])
    if date_min is not None:
        df = df[(df["origin_date"] >= date_min) & (df["origin_date"] <= date_max)]
    return df[["origin_date", "horizon", "actual", "naive", "vecm"]]


def load_lstm(date_min=None, date_max=None) -> pd.DataFrame:
    """
    LSTM's rolling-CV folds cover ~3,728 origins, far more than the other
    baselines' 750 -- restrict to a shared calendar span before comparison
    (see issue #50 decision log) so LSTM isn't scored on years the other
    baselines were never evaluated on. Same date_min/date_max contract as
    load_vecm(), applied consistently rather than only to this one arm.
    """
    df = pd.read_csv(OUT_DIR / "r3_lstm_forecasts.csv", parse_dates=["origin_date"])
    if date_min is not None:
        df = df[(df["origin_date"] >= date_min) & (df["origin_date"] <= date_max)]
    return df[["origin_date", "horizon", "actual", "naive", "lstm"]]


def load_xgboost(date_min=None, date_max=None) -> pd.DataFrame:
    """
    Load XGBoost experimental benchmark forecasts (Issue #101).
    Returns None-like empty frame if the forecast file doesn't exist yet
    (the xgboost pipeline hasn't been run), so callers can skip gracefully.
    """
    path = OUT_DIR / "r3_xgboost_forecasts.csv"
    if not path.exists():
        return pd.DataFrame(columns=["origin_date", "horizon", "actual", "naive", "xgboost"])
    df = pd.read_csv(path, parse_dates=["origin_date"])
    if date_min is not None:
        df = df[(df["origin_date"] >= date_min) & (df["origin_date"] <= date_max)]
    return df[["origin_date", "horizon", "actual", "naive", "xgboost"]]



# ---------------------------------------------------------------------------
# 2. Each pipeline's real evaluation calendar (for cross-pipeline alignment)
# ---------------------------------------------------------------------------

def core_calendar() -> pd.DatetimeIndex:
    """The exact daily calendar ARIMA/VAR-AIC/VAR-BIC evaluate against."""
    path = Path(__file__).resolve().parent / "EDA_VAR_AIC _lag _order.py"
    spec = importlib.util.spec_from_file_location("eda_var_aic_lag_order", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_levels().index


def vecm_calendar() -> pd.DatetimeIndex:
    """The exact daily calendar the 6-variable VECM evaluates against."""
    import johansen_vecm as jv
    return jv.load_gold_levels(jv.FEATURE_SET_6VAR).index


def lstm_calendar() -> pd.DatetimeIndex:
    """The exact daily calendar LSTM's rolling-CV evaluates against."""
    from lstm_baseline import load_common_sample
    return pd.DatetimeIndex(load_common_sample()["date"])


def xgboost_calendar() -> pd.DatetimeIndex:
    """The exact daily calendar XGBoost's rolling-CV evaluates against.
    Uses the same Gold-layer common sample as LSTM (Issue #101)."""
    from xgboost_baseline import load_common_sample
    return pd.DatetimeIndex(load_common_sample()["date"])


def attach_target_date(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each (origin_date, horizon) row, resolve the real calendar date the
    forecast actually targets by walking `horizon` positions forward in
    `calendar` from origin_date's own position -- the same
    `future_pos = level_pos + h` arithmetic every evaluate()/evaluate_vecm()
    loop already uses internally to build `actual`. See module docstring.
    """
    pos = calendar.get_indexer(df["origin_date"])
    target_pos = pos + df["horizon"].to_numpy()
    valid = (pos >= 0) & (target_pos < len(calendar))
    target_date = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    target_date[valid] = calendar.values[target_pos[valid]]
    out = df.copy()
    out["target_date"] = target_date
    return out


def merge_cross_pipeline(
    left: pd.DataFrame, left_col: str, left_calendar: pd.DatetimeIndex,
    right: pd.DataFrame, right_col: str, right_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Join two forecast frames from different pipelines on (origin_date, horizon),
    keeping only rows where both sides' forecast genuinely targets the same real
    calendar date. See module docstring for why this can't be done by comparing
    `actual` values alone.
    """
    if left_col == right_col:
        raise ValueError(f"left_col and right_col must differ, got {left_col!r} for both")
    if left_col in _RESERVED_COLS or right_col in _RESERVED_COLS:
        raise ValueError(
            f"left_col/right_col must not collide with the reserved columns "
            f"{sorted(_RESERVED_COLS)}, got left_col={left_col!r}, right_col={right_col!r}"
        )

    l = attach_target_date(left, left_calendar)
    r = attach_target_date(right, right_calendar)
    m = l[["origin_date", "horizon", "actual", "naive", "target_date", left_col]].merge(
        r[["origin_date", "horizon", "actual", "naive", "target_date", right_col]],
        on=["origin_date", "horizon"], suffixes=("_l", "_r"),
    )
    same_target = m["target_date_l"].notna() & (m["target_date_l"] == m["target_date_r"])
    m = m[same_target].copy()

    # Sanity check, not a filter: once target_date genuinely matches, both sides'
    # `actual` should already agree (same real date, same underlying target series).
    # A mismatch here would mean the target series itself differs between pipelines
    # on the same date -- a data-quality bug worth failing loudly on, not silently
    # dropping the way the old actual-value filter did.
    mismatch = (m["actual_l"] - m["actual_r"]).abs() > ACTUAL_TOL
    if mismatch.any():
        raise AssertionError(
            f"{int(mismatch.sum())} row(s) have matching target_date but disagreeing "
            f"`actual` values between {left_col} and {right_col} -- investigate before "
            f"trusting this comparison; the target series may differ between pipelines "
            f"even on dates both claim to cover."
        )

    m["actual"] = m["actual_l"]
    m["naive"] = m["naive_l"]
    m["target_date"] = m["target_date_l"]
    return m[["origin_date", "horizon", "actual", "naive", "target_date", left_col, right_col]]


# ---------------------------------------------------------------------------
# 3. Generic pairwise Diebold-Mariano test
# ---------------------------------------------------------------------------

def cumulative_var_level_interval(
    var_fit,
    target_idx: int,
    max_h: int,
    alphas: tuple[float, ...] = (0.10, 0.05),
) -> dict[float, np.ndarray]:
    """
    Analytical (1 - alpha) prediction-interval half-widths for the *level*
    series reconstructed from a VAR(X) fitted on first differences (issue #103).

    `EDA_VAR_AIC _lag _order.py` and `johansen_vecm.py`'s VAR-AIC baseline both
    fit a VAR on differences and reconstruct the h-day-ahead level forecast as
    last_level + cumsum(d_hat_1, ..., d_hat_h). The reconstructed level's
    forecast-error variance is therefore the variance of that CUMULATIVE SUM of
    per-step differenced forecast errors, not the per-step differenced forecast
    variance the textbook VAR interval formula and statsmodels'
    `VARResults.forecast_interval()` / `.mse()` give you -- those answer "how
    uncertain is d_hat_h alone", not "how uncertain is the sum d_hat_1 + ... +
    d_hat_h". Naively summing per-step MSEs (or intervals) across h would also
    be wrong, since the per-step forecast errors are serially correlated (they
    share the same underlying shocks) -- the two errors are not proportional to
    each other for h > 1.

    Derivation (Lutkepohl 2005, ch. 2-3 Wold/MA representation): write the VAR's
    d-step-ahead forecast error as e_d = sum_{j=0}^{d-1} Phi_j u_{t+d-j}, where
    Phi_0 = I and Phi_1, Phi_2, ... are the Wold MA coefficient matrices and u
    is the (co)variance-Sigma_u innovation. Summing e_1..e_h and collecting
    terms by shock time t+k (k = 1..h) gives:

        sum_{d=1}^{h} e_d = sum_{k=1}^{h} Psi_{h-k} u_{t+k},   Psi_m := sum_{j=0}^{m} Phi_j

    so, since the u_{t+k} are uncorrelated across k:

        Sigma_level_h = Var(sum_{d=1}^h e_d) = sum_{m=0}^{h-1} Psi_m @ Sigma_u @ Psi_m.T

    Sigma_level_h is a running (cumulative) sum over m, computed once for
    m = 0..max_h-1 and sliced per horizon -- O(max_h) matrix products, no
    refitting. For a model with exogenous regressors (e.g. `d_cpi_yoy` in
    EDA_VAR_AIC's VARX, forecast at its own expanding-window mean since a real
    CPI release isn't known ahead of time), `Sigma_u` and `Phi_j` come from the
    fitted endogenous system conditional on the supplied exog path -- this
    formula treats that path as given, so the resulting band does not include
    uncertainty in the exog forecast itself and is, to that extent, a lower
    bound on total uncertainty for VARX models.

    Parameters
    ----------
    var_fit : statsmodels VARResultsWrapper
        A fitted `VAR(...).fit(lag)` result (differenced series).
    target_idx : int
        Column position of the target series in `var_fit`'s endogenous system.
    max_h : int
        Largest horizon needed; half-widths for every h in 1..max_h are
        returned at once.
    alphas : tuple[float, ...]
        Two-sided significance levels, e.g. (0.10, 0.05) for 90%/95% bands.

    Returns
    -------
    dict[float, np.ndarray]
        alpha -> array of shape (max_h,), the half-width at each horizon
        1..max_h (index h-1), i.e. the level forecast at horizon h has
        interval [level_hat_h - half_width[h-1], level_hat_h + half_width[h-1]].
    """
    phi = var_fit.ma_rep(max_h - 1)  # (max_h, k, k): Phi_0 (=I), Phi_1, ..., Phi_{max_h-1}
    psi_cum = np.cumsum(phi, axis=0)  # Psi_m = sum_{j=0}^{m} Phi_j, shape (max_h, k, k)
    sigma_u = np.asarray(var_fit.sigma_u)

    k = sigma_u.shape[0]
    sigma_level_h = np.zeros((max_h, k, k))
    running = np.zeros((k, k))
    for m in range(max_h):
        running = running + psi_cum[m] @ sigma_u @ psi_cum[m].T
        sigma_level_h[m] = running

    # Numerical floor: guards against a tiny negative diagonal from
    # floating-point cancellation in the matrix accumulation above.
    target_var = np.maximum(sigma_level_h[:, target_idx, target_idx], 0.0)
    se = np.sqrt(target_var)

    return {alpha: float(stats.norm.ppf(1.0 - alpha / 2.0)) * se for alpha in alphas}


def interval_coverage_summary(
    df: pd.DataFrame,
    actual_col: str,
    lower_cols: dict[float, str],
    upper_cols: dict[float, str],
    horizon_col: str = "horizon",
) -> pd.DataFrame:
    """
    Empirical coverage rate and mean interval width (MIW) per horizon, for one
    or more nominal levels (issue #103's calibration/coverage requirement).

    Parameters
    ----------
    df : pd.DataFrame
        Per-origin forecasts, one row per (origin, horizon), with an actual
        column and a lower/upper bound column pair per nominal level.
    actual_col : str
        Column holding the realized value.
    lower_cols, upper_cols : dict[float, str]
        nominal_level (e.g. 90.0, 95.0) -> column name, one entry per band.
        Both dicts must share the same keys.
    horizon_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        One row per horizon, with `target_nominal_{lvl}`,
        `empirical_coverage_{lvl}`, and `mean_interval_width_{lvl}` columns for
        every nominal level in `lower_cols`.
    """
    if set(lower_cols) != set(upper_cols):
        raise ValueError("lower_cols and upper_cols must share the same nominal-level keys")

    # An empty groupby produces `rows == []`, and pd.DataFrame([]) has no
    # columns at all (not even horizon_col) -- .sort_values(horizon_col)
    # below would then raise KeyError instead of returning a well-formed,
    # zero-row summary. Return the correctly-shaped empty frame up front so
    # callers (e.g. a source with 0 successfully-evaluated origins) can
    # check `.empty` without a per-call try/except.
    if df.empty:
        cols = [horizon_col]
        for level in sorted(lower_cols):
            tag = str(int(level)) if float(level).is_integer() else str(level)
            cols += [f"target_nominal_{tag}", f"empirical_coverage_{tag}", f"mean_interval_width_{tag}"]
        return pd.DataFrame(columns=cols)

    rows = []
    for h, sub in df.groupby(horizon_col):
        row: dict = {horizon_col: h}
        actual = sub[actual_col].to_numpy()
        for level in sorted(lower_cols):
            lo = sub[lower_cols[level]].to_numpy()
            hi = sub[upper_cols[level]].to_numpy()
            # (actual >= lo) & (actual <= hi) is False, not NaN, when any of
            # the three is NaN -- comparisons against NaN never raise, so an
            # incomplete origin would otherwise silently dilute the coverage
            # rate and interval width instead of being excluded from them.
            valid = np.isfinite(actual) & np.isfinite(lo) & np.isfinite(hi)
            covered = (actual[valid] >= lo[valid]) & (actual[valid] <= hi[valid])
            tag = str(int(level)) if float(level).is_integer() else str(level)
            row[f"target_nominal_{tag}"] = float(level)
            row[f"empirical_coverage_{tag}"] = round(float(np.mean(covered) * 100.0), 2)
            row[f"mean_interval_width_{tag}"] = round(float(np.mean(hi[valid] - lo[valid])), 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(horizon_col).reset_index(drop=True)


def pairwise_dm(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, h: int) -> dict:
    """
    Runs the DM test for both squared-error and absolute-error loss, per the
    #43/#44 fix (both loss functions' verdicts reported independently, never
    one derived from the other). Convention: dm_stat < 0 means `pred_a` has
    the lower loss (is "better").

    Guards against dieboldmariano's dm_test() raising when the sample is too
    small for the requested horizon (InvalidParameterException when
    len(actual) < h) or degenerate (Zero/NegativeVarianceException when the
    two prediction series' loss differentials have ~no variance) -- both are
    real risks for the thinner cross-pipeline pairs (as few as ~30 forecasts
    at h=20). Returns an "insufficient sample" row instead of letting either
    exception crash the entire comparison loop.
    """
    n = len(actual)
    if n <= h:
        return {
            "n_forecasts": n,
            "dm_stat_squared_loss": None, "dm_p_value_squared_loss": None,
            "dm_stat_absolute_loss": None, "dm_p_value_absolute_loss": None,
            "a_significantly_better_rmse": False, "b_significantly_better_rmse": False,
            "a_significantly_better_mae": False, "b_significantly_better_mae": False,
            "insufficient_sample": True,
        }
    try:
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
    except _DM_EXCEPTIONS:
        return {
            "n_forecasts": n,
            "dm_stat_squared_loss": None, "dm_p_value_squared_loss": None,
            "dm_stat_absolute_loss": None, "dm_p_value_absolute_loss": None,
            "a_significantly_better_rmse": False, "b_significantly_better_rmse": False,
            "a_significantly_better_mae": False, "b_significantly_better_mae": False,
            "insufficient_sample": True,
        }

    a_better_rmse = bool(p_sq < ALPHA and dm_sq < 0)
    b_better_rmse = bool(p_sq < ALPHA and dm_sq > 0)
    a_better_mae = bool(p_abs < ALPHA and dm_abs < 0)
    b_better_mae = bool(p_abs < ALPHA and dm_abs > 0)
    return {
        "n_forecasts": n,
        "dm_stat_squared_loss": round(dm_sq, 3),
        "dm_p_value_squared_loss": round(p_sq, 4),
        "dm_stat_absolute_loss": round(dm_abs, 3),
        "dm_p_value_absolute_loss": round(p_abs, 4),
        "a_significantly_better_rmse": a_better_rmse,
        "b_significantly_better_rmse": b_better_rmse,
        "a_significantly_better_mae": a_better_mae,
        "b_significantly_better_mae": b_better_mae,
        "insufficient_sample": False,
    }


def plain_language_verdict(row: dict, name_a: str, name_b: str) -> str:
    if row.get("insufficient_sample"):
        return (
            f"insufficient sample (n={row['n_forecasts']}) to run the DM test at this horizon"
        )

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


# ---------------------------------------------------------------------------
# 4. Clark-West (2007) Adjusted MSPE Test for Nested Models (Issue #88)
# ---------------------------------------------------------------------------

def clark_west_test(
    actual: np.ndarray | pd.Series,
    pred_benchmark: np.ndarray | pd.Series,
    pred_model: np.ndarray | pd.Series,
    h: int,
    alpha: float = ALPHA,
    small_sample_adj: bool = True,
) -> dict:
    """
    Computes the Clark-West (2007) adjusted MSPE test for comparing a nested
    benchmark (Model 1: Naïve Random Walk) against an unrestricted larger model
    (Model 2: ARIMA, VAR, VECM, LSTM) at forecast horizon `h`.

    Econometric rationale (Clark & McCracken 2001, Clark & West 2006, 2007):
    Under the null hypothesis H0 that the competing model's additional parameters
    have zero population predictive value, in-sample parameter estimation introduces
    noise of order O_p(1/T), inflating the larger model's sample MSPE. The standard
    Diebold-Mariano loss differential d_t = e1_t^2 - e2_t^2 thus has a negative expected
    value under H0, causing spurious rejections in favor of the benchmark.

    Clark-West corrects this bias by defining the adjusted loss differential:
        f_hat_t = e1_t^2 - [ e2_t^2 - (y_hat1_t - y_hat2_t)^2 ]
                = 2 * e1_t * (y_hat2_t - y_hat1_t)
    where:
        e1_t = actual_t - y_hat1_t (benchmark error)
        e2_t = actual_t - y_hat2_t (competing model error)
        adj_t = (y_hat1_t - y_hat2_t)^2 (parameter estimation noise correction)

    Sample MSPEs:
        MSPE_1 = (1/N) * sum(e1_t^2)
        MSPE_2 = (1/N) * sum(e2_t^2)
        adj    = (1/N) * sum(adj_t)
        MSPE_2_adj = MSPE_2 - adj
        f_bar  = MSPE_1 - MSPE_2_adj = (1/N) * sum(f_hat_t)

    Variance Estimation of f_bar:
        - For h=1 (1-step ahead, no forecast overlap): sample variance s_f^2 / N.
        - For h>1 (multi-step overlapping horizons): Heteroskedasticity and
          Autocorrelation Consistent (HAC / Newey-West) variance estimator with
          Bartlett kernel and truncation lag J = h - 1, preserving calendar lag distances.
        - Harvey, Leybourne & Newbold (1997) small-sample modification factor:
          HLN = ((N - h) * (N - h + 1)) / (N^2).

    Hypothesis Testing:
        - One-sided test: H0: MSPE_1 <= MSPE_2 vs. H1: MSPE_1 > MSPE_2
        - CW = f_bar / SE(f_bar)
        - p_value = 1 - Phi(CW) = stats.norm.sf(CW)
    """
    y = np.asarray(actual, dtype=float)
    y1 = np.asarray(pred_benchmark, dtype=float)
    y2 = np.asarray(pred_model, dtype=float)

    if len(y) != len(y1) or len(y) != len(y2):
        raise ValueError("actual, pred_benchmark, and pred_model must have identical lengths")

    # Trim leading and trailing all-NaN / invalid periods to isolate active evaluation span
    valid_mask = np.isfinite(y) & np.isfinite(y1) & np.isfinite(y2)
    if not np.any(valid_mask):
        return {
            "n_forecasts": 0,
            "mspe_naive": None,
            "mspe_model": None,
            "cw_adjustment": None,
            "mspe_model_adj": None,
            "r2_oos": None,
            "r2_oos_raw": None,
            "r2_oos_adj": None,
            "r2_oos_adj_raw": None,
            "mean_f_stat": None,
            "se_f_stat": None,
            "cw_stat": None,
            "cw_stat_raw": None,
            "cw_p_value": None,
            "model_significantly_better": False,
            "insufficient_sample": True,
        }

    first_idx = int(np.argmax(valid_mask))
    last_idx = int(len(valid_mask) - 1 - np.argmax(valid_mask[::-1]))

    y_span = y[first_idx : last_idx + 1]
    y1_span = y1[first_idx : last_idx + 1]
    y2_span = y2[first_idx : last_idx + 1]
    valid_span = valid_mask[first_idx : last_idx + 1]
    n_valid = int(np.sum(valid_span))

    if n_valid <= h or n_valid < 3:
        return {
            "n_forecasts": n_valid,
            "mspe_naive": None,
            "mspe_model": None,
            "cw_adjustment": None,
            "mspe_model_adj": None,
            "r2_oos": None,
            "r2_oos_raw": None,
            "r2_oos_adj": None,
            "r2_oos_adj_raw": None,
            "mean_f_stat": None,
            "se_f_stat": None,
            "cw_stat": None,
            "cw_stat_raw": None,
            "cw_p_value": None,
            "model_significantly_better": False,
            "insufficient_sample": True,
        }

    e1_v = y_span[valid_span] - y1_span[valid_span]
    e2_v = y_span[valid_span] - y2_span[valid_span]
    adj_v = (y1_span[valid_span] - y2_span[valid_span]) ** 2

    mspe_1 = float(np.mean(e1_v ** 2))
    mspe_2 = float(np.mean(e2_v ** 2))
    adj = float(np.mean(adj_v))
    mspe_2_adj = float(mspe_2 - adj)

    f_valid = e1_v ** 2 - (e2_v ** 2 - adj_v)
    f_bar = float(np.mean(f_valid))

    # Place f on calendar time span to preserve exact lag distances across any interior gaps
    f_span = np.full(len(y_span), np.nan)
    f_span[valid_span] = f_valid
    z = np.where(valid_span, f_span - f_bar, 0.0)

    gamma0 = float(np.sum(z[valid_span] ** 2) / n_valid)

    if h <= 1:
        omega = gamma0
    else:
        omega = gamma0
        for k in range(1, h):
            weight = 1.0 - (k / h)
            # Both t and t-k must be valid for the lag-k product to be real
            valid_pairs = valid_span[k:] & valid_span[:-k]
            n_pairs = int(np.sum(valid_pairs))
            if n_pairs > 0:
                gamma_k = float(np.sum(z[k:] * z[:-k] * valid_pairs) / n_pairs)
            else:
                gamma_k = 0.0
            omega += 2.0 * weight * gamma_k

    omega = max(omega, 1e-14)
    var_f = omega / n_valid

    if small_sample_adj:
        hln = ((n_valid - h) * (n_valid - h + 1)) / (n_valid * n_valid)
        var_f = var_f / hln

    se_f = float(np.sqrt(var_f))
    cw_stat = float(f_bar / se_f) if se_f > 0 else 0.0
    cw_p_value = float(stats.norm.sf(cw_stat))

    model_better = bool(cw_p_value < alpha and cw_stat > 0)
    EPS = 1e-12
    if mspe_1 > EPS:
        r2_oos = float(1.0 - (mspe_2 / mspe_1))
        r2_oos_adj = float(f_bar / mspe_1)
    else:
        r2_oos = None
        r2_oos_adj = None

    return {
        "n_forecasts": n_valid,
        "mspe_naive": round(mspe_1, 6),
        "mspe_model": round(mspe_2, 6),
        "cw_adjustment": round(adj, 6),
        "mspe_model_adj": round(mspe_2_adj, 6),
        "r2_oos": round(r2_oos, 6) if r2_oos is not None else None,
        "r2_oos_raw": r2_oos,
        "r2_oos_adj": round(r2_oos_adj, 6) if r2_oos_adj is not None else None,
        "r2_oos_adj_raw": r2_oos_adj,
        "mean_f_stat": round(f_bar, 6),
        "se_f_stat": round(se_f, 6),
        "cw_stat": round(cw_stat, 3),
        "cw_stat_raw": cw_stat,
        "cw_p_value": round(cw_p_value, 4),
        "model_significantly_better": model_better,
        "insufficient_sample": False,
    }



def plain_language_verdict_cw(row: dict | pd.Series, model_name: str) -> str:
    """
    Generates plain-language econometric interpretation for a Clark-West test result.
    """
    if row.get("insufficient_sample"):
        return f"insufficient sample (n={row.get('n_forecasts', 0)}) to run Clark-West test"

    sig = bool(row.get("model_significantly_better", False))
    cw_stat = row.get("cw_stat")
    p_val = row.get("cw_p_value")
    mspe_m = row.get("mspe_model")
    mspe_adj = row.get("mspe_model_adj")
    mspe_n = row.get("mspe_naive")

    if cw_stat is None or p_val is None:
        return "insufficient data to compute verdict"

    mspe_n_str = f"{mspe_n:.5f}" if mspe_n is not None else "N/A"
    mspe_m_str = f"{mspe_m:.5f}" if mspe_m is not None else "N/A"
    mspe_adj_str = f"{mspe_adj:.5f}" if mspe_adj is not None else "N/A"

    if sig:
        return (
            f"{model_name} significantly outperforms Naïve benchmark after Clark-West MSPE adjustment "
            f"(CW={cw_stat:.3f}, p={p_val:.4f}; MSPE_naive={mspe_n_str}, MSPE_adj={mspe_adj_str})"
        )
    else:
        if cw_stat <= 0:
            return (
                f"No significant improvement: {model_name} does not beat Naïve under Clark-West "
                f"(CW={cw_stat:.3f}, p={p_val:.4f}; MSPE_naive={mspe_n_str}, MSPE_model={mspe_m_str})"
            )
        else:
            return (
                f"Positive point gain not statistically significant: {model_name} fails to reject equal accuracy "
                f"(CW={cw_stat:.3f}, p={p_val:.4f}; MSPE_naive={mspe_n_str}, MSPE_adj={mspe_adj_str})"
            )


def run_clark_west_battery(
    alpha: float = ALPHA,
    date_min: pd.Timestamp | str | None = None,
    date_max: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the full Clark-West test battery against the Naïve benchmark (Issue #88).

    Primary 15-test battery (m=15):
        5 models (ARIMA-AIC, VAR-AIC, VECM-6var, LSTM, XGBoost) x 3 horizons (1, 5, 20)
        XGBoost is integrated into the primary research scope to mitigate methodological
        penalization for omitting a simpler tabular ML model alongside deep learning (LSTM).
    Sensitivity battery (m=6):
        2 BIC variants (ARIMA-BIC, VAR-BIC) x 3 horizons (1, 5, 20)

    Returns:
        (primary_df, sensitivity_df)
    """
    core = load_core()
    d_min = pd.Timestamp(date_min) if date_min is not None else core["origin_date"].min()
    d_max = pd.Timestamp(date_max) if date_max is not None else core["origin_date"].max()

    # Filter core to the same window applied to VECM/LSTM so all arms
    # are scored over an identical sample span (Nelson review point 2).
    core = core[(core["origin_date"] >= d_min) & (core["origin_date"] <= d_max)]

    vecm = load_vecm(d_min, d_max)
    lstm = load_lstm(d_min, d_max)

    m_lstm = merge_cross_pipeline(
        core, "arima_aic", core_calendar(),
        lstm, "lstm", lstm_calendar(),
    )

    # Incorporate XGBoost into the primary research scope
    xgb_raw = load_xgboost(d_min, d_max)
    m_xgb = None
    if not xgb_raw.empty:
        m_xgb = merge_cross_pipeline(
            core, "arima_aic", core_calendar(),
            xgb_raw, "xgboost", xgboost_calendar(),
        )

    # Align all candidate models to the strict common sample intersection
    # to guarantee an identical Naïve benchmark baseline across all arms (Review finding 4.3).
    common_origins = set(core["origin_date"]).intersection(vecm["origin_date"]).intersection(m_lstm["origin_date"])
    if m_xgb is not None:
        common_origins = common_origins.intersection(m_xgb["origin_date"])

    core = core[core["origin_date"].isin(common_origins)]
    vecm = vecm[vecm["origin_date"].isin(common_origins)]
    m_lstm = m_lstm[m_lstm["origin_date"].isin(common_origins)]
    if m_xgb is not None:
        m_xgb = m_xgb[m_xgb["origin_date"].isin(common_origins)]

    primary_models = [
        ("arima_aic", "ARIMA-AIC", "core"),
        ("var_aic", "VAR-AIC", "core"),
        ("vecm", "VECM (6-var)", "vecm"),
        ("lstm", "LSTM (Tuned)", "lstm"),
    ]
    if m_xgb is not None:
        primary_models.append(("xgboost", "XGBoost (Tuned)", "xgboost"))

    sensitivity_models = [
        ("arima_bic", "ARIMA-BIC", "core"),
        ("var_bic", "VAR-BIC", "core"),
    ]


    def _evaluate_models(model_list: list[tuple[str, str, str]]) -> pd.DataFrame:
        # Cache horizon slices to avoid repeated boolean indexing
        core_by_h = {h: core[core["horizon"] == h] for h in HORIZONS}
        vecm_by_h = {h: vecm[vecm["horizon"] == h] for h in HORIZONS}
        lstm_by_h = {h: m_lstm[m_lstm["horizon"] == h] for h in HORIZONS}
        xgb_by_h = {h: m_xgb[m_xgb["horizon"] == h] for h in HORIZONS} if m_xgb is not None else {}

        records = []
        for col_name, display_name, source in model_list:
            for h in HORIZONS:
                if source == "core":
                    df_sub = core_by_h[h]
                elif source == "vecm":
                    df_sub = vecm_by_h[h]
                elif source == "lstm":
                    df_sub = lstm_by_h[h]
                elif source == "xgboost":
                    df_sub = xgb_by_h[h]
                else:
                    raise ValueError(f"Unknown source {source}")

                act = df_sub["actual"].to_numpy()
                naive = df_sub["naive"].to_numpy()
                pred = df_sub[col_name].to_numpy()

                res = clark_west_test(act, naive, pred, h=h, alpha=alpha)
                rec = {
                    "model": display_name,
                    "model_key": col_name,
                    "horizon": h,
                    **res,
                }
                rec["verdict_raw"] = plain_language_verdict_cw(res, display_name)
                records.append(rec)
        return pd.DataFrame(records)

    primary_df = _evaluate_models(primary_models)
    sensitivity_df = _evaluate_models(sensitivity_models)

    # Apply FDR correction
    primary_df = apply_clark_west_fdr(primary_df, alpha=alpha)
    sensitivity_df = apply_clark_west_fdr(sensitivity_df, alpha=alpha)

    return primary_df, sensitivity_df


def apply_clark_west_fdr(cw_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Applies Benjamini-Hochberg (BH) FDR multiplicity correction to Clark-West test results.

    Implements two complementary correction tiers:
    1. Global FDR across all tests in the table (`cw_p_adj_global`, `model_significantly_better_fdr_global`)
    2. Horizon-Stratified FDR within each horizon subset (`cw_p_adj_horizon`, `model_significantly_better_fdr_horizon`),
       preventing long-horizon noise from masking short-horizon discoveries (Issue #81).
    """
    out = cw_df.copy()
    p_col = "cw_p_value"

    out["cw_p_adj_global"] = np.nan
    out["cw_p_adj_horizon"] = np.nan

    sign_col = "cw_stat_raw" if "cw_stat_raw" in out.columns else "cw_stat"
    out["model_significantly_better_fdr_global"] = False
    out["model_significantly_better_fdr_horizon"] = False

    # 1. Global FDR
    testable = out[p_col].notna() & (~out["insufficient_sample"])
    if testable.any():
        rej_glob, p_adj_glob, _, _ = multipletests(
            out.loc[testable, p_col].to_numpy(), alpha=alpha, method="fdr_bh",
        )
        out.loc[testable, "cw_p_adj_global"] = np.round(p_adj_glob, 4)
        out.loc[testable, "model_significantly_better_fdr_global"] = (
            rej_glob & (out.loc[testable, sign_col] > 0)
        )

    # 2. Horizon-Stratified FDR
    for h in out["horizon"].unique():
        h_mask = testable & (out["horizon"] == h)
        if h_mask.any():
            rej_h, p_adj_h, _, _ = multipletests(
                out.loc[h_mask, p_col].to_numpy(), alpha=alpha, method="fdr_bh",
            )
            out.loc[h_mask, "cw_p_adj_horizon"] = np.round(p_adj_h, 4)
            out.loc[h_mask, "model_significantly_better_fdr_horizon"] = (
                rej_h & (out.loc[h_mask, sign_col] > 0)
            )

    # Verdicts under FDR — single parameterized helper to avoid drift
    # between global and horizon tiers (Nelson review point 5).
    def _fdr_verdict(row: pd.Series, sig_col: str, p_adj_col: str) -> str:
        r_dict = {
            "insufficient_sample": row.get("insufficient_sample", False),
            "n_forecasts": row.get("n_forecasts", 0),
            "model_significantly_better": row.get(sig_col, False),
            "cw_stat": row.get("cw_stat"),
            "cw_p_value": row.get(p_adj_col),
            "mspe_naive": row.get("mspe_naive"),
            "mspe_model": row.get("mspe_model"),
            "mspe_model_adj": row.get("mspe_model_adj"),
        }
        return plain_language_verdict_cw(r_dict, row.get("model", "Model"))

    out["verdict_fdr_global"] = out.apply(
        lambda r: _fdr_verdict(r, "model_significantly_better_fdr_global", "cw_p_adj_global"), axis=1
    )
    out["verdict_fdr_horizon"] = out.apply(
        lambda r: _fdr_verdict(r, "model_significantly_better_fdr_horizon", "cw_p_adj_horizon"), axis=1
    )

    return out


# ---------------------------------------------------------------------------
# 6. Macroeconomic Monetary Policy Regime Segmentation (Issue #101 / #103)
# ---------------------------------------------------------------------------

# Macroeconomic monetary policy regimes based on Bank of Canada (BoC) policy rate announcements:
# - Regime 1: 2023-01-01 to 2023-12-31 (Aggressive Tightening & Yield Curve Inversion; BoC peak 5.00% reached July 2023)
# - Regime 2: 2024-01-01 to 2024-05-31 (Policy Plateau / Higher-for-Longer; BoC paused at 5.00% across Jan-Apr meetings)
# - Regime 3: 2024-06-01 to 2026-06-30 (Easing Cycle & Un-inversion; first BoC rate cut June 5 2024 to 4.75% through sample end)
REGIMES = [
    ("Regime 1: Rapid Tightening & Inversion", "2023-01-01", "2023-12-31"),
    ("Regime 2: Policy Plateau / Higher-for-Longer", "2024-01-01", "2024-05-31"),
    ("Regime 3: Easing Cycle & Un-inversion", "2024-06-01", "2026-06-30"),
]


def evaluate_regime_segmentation(
    output_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Computes out-of-sample forecasting performance (RMSE, MAE, R2_OOS, and Clark-West statistics)
    segmented across the three macroeconomic monetary policy regimes.
    """
    core = load_core()
    vecm = load_vecm()
    lstm = load_lstm()
    xgb_df = load_xgboost()

    m_lstm = merge_cross_pipeline(
        core, "arima_aic", core_calendar(),
        lstm, "lstm", lstm_calendar(),
    )

    has_xgb = not xgb_df.empty
    m_xgb = None
    if has_xgb:
        m_xgb = merge_cross_pipeline(
            core, "arima_aic", core_calendar(),
            xgb_df, "xgboost", xgboost_calendar(),
        )

    models_to_eval = [
        ("arima_aic", "ARIMA-AIC", "core"),
        ("var_aic", "VAR-AIC", "core"),
        ("vecm", "VECM (6-var)", "vecm"),
        ("lstm", "LSTM (Tuned)", "lstm"),
    ]
    if has_xgb and m_xgb is not None:
        models_to_eval.append(("xgboost", "XGBoost (Tuned)", "xgboost"))

    records = []
    for regime_name, start_date, end_date in REGIMES:
        s_date = pd.Timestamp(start_date)
        e_date = pd.Timestamp(end_date)

        for col_name, display_name, source in models_to_eval:
            for h in HORIZONS:
                if source == "core":
                    sub = core[(core["origin_date"] >= s_date) & (core["origin_date"] <= e_date) & (core["horizon"] == h)]
                elif source == "vecm":
                    sub = vecm[(vecm["origin_date"] >= s_date) & (vecm["origin_date"] <= e_date) & (vecm["horizon"] == h)]
                elif source == "lstm":
                    sub = m_lstm[(m_lstm["origin_date"] >= s_date) & (m_lstm["origin_date"] <= e_date) & (m_lstm["horizon"] == h)]
                elif source == "xgboost":
                    sub = m_xgb[(m_xgb["origin_date"] >= s_date) & (m_xgb["origin_date"] <= e_date) & (m_xgb["horizon"] == h)]
                else:
                    continue

                if len(sub) < 3:
                    continue

                act = sub["actual"].to_numpy()
                naive = sub["naive"].to_numpy()
                pred = sub[col_name].to_numpy()

                rmse_model = float(np.sqrt(np.mean((act - pred) ** 2)))
                mae_model = float(np.mean(np.abs(act - pred)))
                rmse_naive = float(np.sqrt(np.mean((act - naive) ** 2)))
                mae_naive = float(np.mean(np.abs(act - naive)))

                rmse_imp = (rmse_naive - rmse_model) / rmse_naive * 100.0 if rmse_naive > 0 else 0.0
                mae_imp = (mae_naive - mae_model) / mae_naive * 100.0 if mae_naive > 0 else 0.0

                # Statistical inference requires sufficient degrees of freedom under HAC Bartlett kernel (n >= 5*h)
                min_n_for_inference = 5 * h
                has_power = len(sub) >= min_n_for_inference

                if has_power:
                    cw_res = clark_west_test(act, naive, pred, h=h)
                    r2_oos = cw_res.get("r2_oos")
                    r2_oos_adj = cw_res.get("r2_oos_adj")
                    cw_stat = cw_res.get("cw_stat")
                    cw_p_value = cw_res.get("cw_p_value")
                    sig_better = cw_res.get("model_significantly_better", False)
                else:
                    cw_res = {}
                    # Calculate descriptive raw R2_OOS without asymptotic significance claim
                    mask = np.isfinite(act) & np.isfinite(naive) & np.isfinite(pred)
                    if np.any(mask):
                        mspe_n = float(np.mean((act[mask] - naive[mask]) ** 2))
                        mspe_m = float(np.mean((act[mask] - pred[mask]) ** 2))
                        EPS = 1e-12
                        r2_oos = round(1.0 - (mspe_m / mspe_n), 6) if mspe_n > EPS else None
                    else:
                        r2_oos = None
                    r2_oos_adj = None
                    cw_stat = None
                    cw_p_value = None
                    sig_better = False

                records.append({
                    "regime": regime_name,
                    "model": display_name,
                    "model_key": col_name,
                    "horizon": h,
                    "n_forecasts": len(sub),
                    "insufficient_sample": not has_power,
                    "rmse_model": round(rmse_model, 5),
                    "rmse_naive": round(rmse_naive, 5),
                    "rmse_improvement_pct": round(rmse_imp, 3),
                    "mae_model": round(mae_model, 5),
                    "mae_naive": round(mae_naive, 5),
                    "mae_improvement_pct": round(mae_imp, 3),
                    "r2_oos": r2_oos,
                    "r2_oos_adj": r2_oos_adj,
                    "cw_stat": cw_stat,
                    "cw_p_value": cw_p_value,
                    "model_significantly_better": sig_better,
                })

    regime_df = pd.DataFrame(records)
    if save:
        target = output_path if output_path is not None else OUT_DIR / "r3_regime_segmented_metrics.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        regime_df.to_csv(target, index=False)
        logger.info("Saved regime segmented metrics to %s", target)

    return regime_df


def export_all_model_comparisons(output_dir: Path | None = None) -> None:
    """
    Executes the full econometric comparison battery and exports all canonical result CSVs.
    """
    out = output_dir if output_dir is not None else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    primary_df, sensitivity_df = run_clark_west_battery()
    csv_cols_p = [c for c in primary_df.columns if not c.endswith("_raw")]
    csv_cols_s = [c for c in sensitivity_df.columns if not c.endswith("_raw")]
    primary_df[csv_cols_p].to_csv(out / "clark_west_test_results.csv", index=False)
    sensitivity_df[csv_cols_s].to_csv(out / "clark_west_sensitivity_results.csv", index=False)
    logger.info("Saved Clark-West battery results to %s", out)

    regime_df = evaluate_regime_segmentation(output_path=out / "r3_regime_segmented_metrics.csv")
    logger.info("Saved Regime Segmentation metrics (%d rows) to %s", len(regime_df), out)

    input_files = [
        PROCESSED_DIR / "gold_features.csv",
        PROCESSED_DIR / "bank_of_canada_data.csv",
        PROCESSED_DIR / "fred_rates.csv",
        PROCESSED_DIR / "statcan_cpi.csv",
    ]
    write_data_manifest(input_files, out / "data_manifest.json")
    logger.info("Saved input data manifest to %s", out / "data_manifest.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    export_all_model_comparisons()

