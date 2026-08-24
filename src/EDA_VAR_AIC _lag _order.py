"""
Round 3, Path A — VAR baseline, AIC lag order (issue #28)
DAMO-699 Capstone, Group 5

Scope of this script (per the issue #28 checklist, two boxes only):
  - Build Random Walk benchmark + AIC-selected-lag VAR on the Round 2-consolidated
    feature set
  - RMSE/MAE at 1-/5-/20-day horizons vs. the naive benchmark, per proposal §5.4

Feature set (per proposal §3.1/§5.3's committed predictor set):
    yield_spread_10y_2y (target), overnight_rate, us_treasury_10y, fed_funds_rate, cpi_yoy,
    usdcad

Aug 19 correction: `usdcad` had been left out of this script from the start (undocumented).
#29 briefly matched that omission to keep the #28-vs-#29 AIC/BIC comparison controlled, which
absorbed the gap as an accepted caveat in M2_CHECKLIST.md's Aug 16 decision instead of fixing
it. §3.1's research question names USD/CAD explicitly as a required transmission variable, so
it's restored here -- see M2_CHECKLIST.md's Aug 19 entry.

Important methodological note on horizons:
Proposal §5.4 defines the naive benchmark as "predicts tomorrow's [value] as today's
[value]" -- i.e. a random walk in LEVELS: level_hat(t+h) = level(t). For h=1 this is the
same as "predict zero change," but for h=5 and h=20 it is NOT the same as "predict zero
change on the differenced series" -- the actual level move over 5 or 20 days is the SUM
of five or twenty daily differences, not a single one. So both the naive and VAR forecasts
here are evaluated in levels: VAR is fit on first differences (required for stationarity,
confirmed in the Round 2 stationarity memo), then its h-step-ahead differenced forecasts
are cumulatively summed and added back onto the last observed level before scoring against
the actual level h days out. Scoring the naive/VAR forecasts directly on single-day
differences at multi-day horizons would silently misstate both models' errors.

Follow-up fix (issue #43): dm_report()'s significance verdict previously only checked
the squared-loss (RMSE) p-value, silently dropping absolute-loss (MAE) significant
results from the summary. Now reports both loss types' verdicts separately, plus an
overall verdict requiring agreement between them -- see dm_report() for detail.

Aug 22 correction (issue #63): load_levels() used to rebuild its own inner-joined,
no-fill daily frame from the raw BoC/FRED/CPI sources -- 4,010 rows, a calendar that
silently disagreed with VECM/LSTM's Gold-layer pipeline (4,268 rows, outer join +
ffill(limit=2) across US/CA market holidays), which #50's cross-model comparison had
to work around instead of every pipeline sharing one calendar. Verified the two were
otherwise identical: on the 4,009 overlapping dates every FEATURE_SET value matched
Gold's to floating-point precision, so this was 259 genuine trading days being dropped
for no modeling reason, not a real disagreement to reconcile. Now reads
data/processed/gold_features.csv directly, selecting the same FEATURE_SET columns --
same downstream differencing/evaluation logic, larger and more complete sample.

Aug 24 fix (issue #72): `d_cpi_yoy` (monthly CPI, forward-filled onto the daily grid
then first-differenced) is exactly 0.0 on ~95.5% of rows, with a jump only on the
~192 real release dates -- a spike train, not a continuous innovation like the other
five differenced series, and letting it sit inside the same endogenous VAR system
risked AIC/BIC lag-order search latching onto the release calendar's ~21-trading-day
periodicity as if it were real cross-series dynamics. `d_cpi_yoy` is now passed to
`VAR(..., exog=...)` as an exogenous regressor instead of a sixth endogenous series --
see `EXOG_COLS`/`_split_endog_exog()` below. It still enters each endogenous
equation's contemporaneous relationship, it just no longer participates in lag-order
search or has its own dynamics equation. Its true future value (a CPI print) isn't
knowable at forecast time, so `evaluate()` forecasts it with its own expanding-window
mean (issue #72's DM comparison against the prior all-endogenous treatment is in
`M2_CHECKLIST.md`'s Aug 24 entry).

Run: pip install dieboldmariano   (per reviewer comment 2, then)   python 03_var_baseline_path_a.py
Reads from data/processed/ via src/project_paths.py, same as the rest of the repo.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dieboldmariano import dm_test
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# The daily index has business-day gaps (weekends/holidays) with no fixed freq set,
# which statsmodels flags on every VAR fit/forecast call inside the evaluation loop.
# Harmless here -- forecasts are indexed by position (train.values[-lag:]), not by
# date -- so this is silenced rather than left to print hundreds of times.
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from model_comparison import pairwise_dm, plain_language_verdict  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

TARGET = "yield_spread_10y_2y"
FEATURE_SET = ["yield_spread_10y_2y", "overnight_rate", "us_treasury_10y",
               "fed_funds_rate", "cpi_yoy", "usdcad"]

# Issue #72: d_cpi_yoy is a release-day spike train (~95.5% exactly 0.0), not a
# continuous innovation -- treated as exogenous rather than a sixth endogenous
# series. See _split_endog_exog() and the module docstring's Aug 24 entry.
EXOG_COLS = ["d_cpi_yoy"]

HORIZONS = [1, 5, 20]     # trading days, per proposal §5.4
MAX_LAG_SEARCH = 15
MIN_TRAIN = 500           # minimum obs before the first out-of-sample origin
STEP = 5                  # re-fit every STEP origins (lower = slower, more thorough)
ALPHA = 0.05


# ----------------------------------------------------------------------------
# 1. Load the canonical Gold-layer feature set (issue #63)
# ----------------------------------------------------------------------------

def load_levels() -> pd.DataFrame:
    """
    FEATURE_SET, read from the canonical Gold-layer CSV (src/gold_feature_pipeline.py,
    issue #30) rather than rebuilt from raw sources -- the same calendar VECM and LSTM
    already evaluate against (issue #63), instead of a separately-maintained inner-join
    that silently dropped 259 genuine trading days neither of those two ever dropped.
    """
    gold = pd.read_csv(PROCESSED_DIR / "gold_features.csv", parse_dates=["date"])
    df = gold.set_index("date").sort_index()[FEATURE_SET].dropna()
    return df


# ----------------------------------------------------------------------------
# 2. Difference for stationarity (confirmed I(1) in the Round 2 stationarity memo)
# ----------------------------------------------------------------------------

def to_differenced(levels: pd.DataFrame) -> pd.DataFrame:
    diffed = levels.diff().dropna()
    diffed.columns = [f"d_{c}" for c in diffed.columns]
    return diffed


def confirm_stationary(diffed: pd.DataFrame) -> None:
    """
    Note (issue #72): ADF trivially "passes" for d_cpi_yoy -- a series that's
    constant 0.0 on ~95.5% of rows with a handful of release-day jumps will always
    look stationary to ADF, without being a well-behaved continuous innovation like
    the other five series. This check confirms no unit root, not that every column
    is equally well-behaved -- see _split_endog_exog()/EXOG_COLS for how d_cpi_yoy
    is actually handled downstream.
    """
    for col in diffed.columns:
        _, pval, *_ = adfuller(diffed[col], autolag="AIC")
        if pval >= ALPHA:
            raise RuntimeError(
                f"{col} is still non-stationary after one difference (p={pval:.3f}). "
                f"Do not proceed to VAR fitting/select_order() until this is resolved."
            )
    print(f"  All {len(diffed.columns)} series confirmed stationary post-differencing (p<0.05).")


def _split_endog_exog(diffed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Issue #72: pulls EXOG_COLS (d_cpi_yoy) out of the endogenous VAR system into a
    separate exogenous regressor frame. Returns (endog, exog), exog=None if no
    EXOG_COLS are present in `diffed` (e.g. the yield-only feature set in #70,
    which never includes cpi_yoy in the first place).
    """
    exog_cols = [c for c in EXOG_COLS if c in diffed.columns]
    endog = diffed.drop(columns=exog_cols)
    exog = diffed[exog_cols] if exog_cols else None
    return endog, exog


# ----------------------------------------------------------------------------
# 3. AIC lag order selection
# ----------------------------------------------------------------------------

def select_aic_lag(diffed: pd.DataFrame) -> int:
    endog, exog = _split_endog_exog(diffed)
    model = VAR(endog, exog=exog)
    order_results = model.select_order(maxlags=MAX_LAG_SEARCH)
    print(order_results.summary())
    aic_lag = order_results.aic
    print(f"\n  AIC-selected lag order: {aic_lag}")
    return aic_lag


# ----------------------------------------------------------------------------
# 4. Random Walk benchmark + AIC-VAR, evaluated in LEVELS at each horizon
# ----------------------------------------------------------------------------

def evaluate(levels: pd.DataFrame, diffed: pd.DataFrame, lag: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Expanding-window evaluation. At each origin t (indexed into `levels`, offset by 1
    to line up with `diffed`):
      - Naive forecast:  level_hat(t+h) = level(t)                      [proposal §5.4]
      - VAR forecast:    level_hat(t+h) = level(t) + sum(d_target forecasts 1..h)
      - Actual:          level(t+h)
    Both are then scored against the same actual level, so RMSE/MAE are directly
    comparable and expressed in the target's native units (spread, in percentage points).
    """
    endog_all, exog_all = _split_endog_exog(diffed)
    target_diff_col = f"d_{TARGET}"
    diff_target_idx = endog_all.columns.get_loc(target_diff_col)

    max_h = max(HORIZONS)
    records = {h: {"actual": [], "var_pred": [], "naive_pred": [], "origin_date": []} for h in HORIZONS}

    # Issue #63 follow-up: origin is now counted in LEVEL observations trained on
    # (train = levels.iloc[:origin], last_level/origin_date = levels.iloc[origin - 1]),
    # matching evaluate_vecm()'s convention exactly instead of counting DIFFERENCED
    # observations trained on (the previous train = diffed.iloc[:origin] anchored
    # last_level one row earlier for the same `origin` number -- a permanent one-row
    # phase offset from VECM's origin grid that no calendar fix could close, since it's
    # a training-size convention mismatch, not a data-source one). diffed.iloc[i] is the
    # change ending on levels.iloc[i+1], so training on `origin` levels uses `origin - 1`
    # diffs.
    n = len(levels)
    for origin in range(MIN_TRAIN, n - max_h, STEP):
        train_endog = endog_all.iloc[:origin - 1]
        train_exog = exog_all.iloc[:origin - 1] if exog_all is not None else None
        try:
            fitted = VAR(train_endog, exog=train_exog).fit(lag)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if train_exog is not None:
            # Issue #72: a future CPI release's surprise isn't knowable at forecast
            # time (that's what makes it a surprise) -- exog_future is set to the
            # expanding-window mean of train_exog (an unbiased, lookahead-free best
            # guess of a mean-zero shock), not its last observed value or an assumed
            # trend continuation.
            exog_future = np.tile(train_exog.mean().to_numpy(), (max_h, 1))
            fc = fitted.forecast(train_endog.values[-lag:], steps=max_h, exog_future=exog_future)
        else:
            fc = fitted.forecast(train_endog.values[-lag:], steps=max_h)
        fc_target_diffs = fc[:, diff_target_idx]
        cum_fc = np.cumsum(fc_target_diffs)  # cumulative reconstructed level change

        last_level = levels[TARGET].iloc[origin - 1]
        origin_date = levels.index[origin - 1]

        for h in HORIZONS:
            future_pos = origin - 1 + h
            if future_pos >= len(levels):
                continue
            actual_level = levels[TARGET].iloc[future_pos]
            var_level_pred = last_level + cum_fc[h - 1]
            naive_level_pred = last_level  # random walk: tomorrow = today, per §5.4

            records[h]["actual"].append(actual_level)
            records[h]["var_pred"].append(var_level_pred)
            records[h]["naive_pred"].append(naive_level_pred)
            records[h]["origin_date"].append(origin_date)

    rows = []
    raw = {}  # h -> (actual, var_pred, naive_pred) arrays, for the DM test in step 5
    forecast_parts = []  # per-origin forecasts, for cross-model comparison (issue #50)
    for h in HORIZONS:
        actual = np.array(records[h]["actual"])
        var_pred = np.array(records[h]["var_pred"])
        naive_pred = np.array(records[h]["naive_pred"])
        if len(actual) == 0:
            print(f"  WARNING: no forecasts generated for horizon {h} -- check MIN_TRAIN/STEP.")
            continue

        rmse_var = np.sqrt(np.mean((actual - var_pred) ** 2))
        mae_var = np.mean(np.abs(actual - var_pred))
        rmse_naive = np.sqrt(np.mean((actual - naive_pred) ** 2))
        mae_naive = np.mean(np.abs(actual - naive_pred))

        rows.append({
            "horizon_days": h,
            "n_forecasts": len(actual),
            "rmse_var": round(rmse_var, 4),
            "mae_var": round(mae_var, 4),
            "rmse_naive": round(rmse_naive, 4),
            "mae_naive": round(mae_naive, 4),
            "var_beats_naive_rmse": bool(rmse_var < rmse_naive),
            "var_beats_naive_mae": bool(mae_var < mae_naive),
        })
        raw[h] = (actual, var_pred, naive_pred)
        forecast_parts.append(pd.DataFrame({
            "origin_date": records[h]["origin_date"], "horizon": h,
            "actual": actual, "var_aic": var_pred, "naive": naive_pred,
        }))

    forecasts = pd.concat(forecast_parts, ignore_index=True) if forecast_parts else pd.DataFrame(
        columns=["origin_date", "horizon", "actual", "var_aic", "naive"]
    )
    return pd.DataFrame(rows), raw, forecasts


# ----------------------------------------------------------------------------
# 5. Diebold-Mariano test — is the RMSE/MAE gap statistically significant, or noise?
# ----------------------------------------------------------------------------
#  On the first draft's hand-rolled DM implementation:
#   "For the DM statistic's standard error at h=5 and h=20, are you using a
#    Newey-West/Bartlett-weighted long-run variance (lag = h-1), or a plain sample
#    variance divided by n?"
#   -> Answer at the time: plain sample variance / n. That understates the standard
#      error at h=5/h=20, because h-step-ahead forecast errors from adjacent,
#      overlapping origins are serially correlated by construction, and the plain
#      estimator doesn't account for that autocovariance structure.
#

#
# Adopted as-is below, with one explicit addition on top of the reviewer's call
# signature: variance_estimator="bartlett" (the library defaults to "acf" if left
# unset). "bartlett" is the Bartlett-kernel-weighted long-run variance estimator
# truncated at lag h-1 -- directly answering comment 1 -- and harvey_correction=True
# adds the Harvey, Leybourne & Newbold (1997) small-sample adjustment per comment 2.

def dm_report(raw: dict) -> pd.DataFrame:
    """
    Fixes issue #43: the original version derived var_significantly_better /
    naive_significantly_better from dm_p_value_squared_loss (RMSE test) only, even
    though dm_p_value_absolute_loss (MAE test) is computed and written to the CSV
    right alongside it -- reports both loss types' verdicts separately below, plus
    an overall verdict requiring agreement between them. (Exact p-values depend on
    the current evaluate() output -- see M2_CHECKLIST.md's decision log for the
    latest numbers and the Aug 20 off-by-one fix that changed them, rather than
    hardcoding an example here that would go stale the next time evaluate() changes.)

    Computes the actual test via the shared `pairwise_dm()` (src/model_comparison.py)
    so this and #47's dm_report_vecm() don't each hand-maintain their own copy of the
    same dm_test() call parameters (h, harvey_correction, variance_estimator, ALPHA).
    """
    rows = []
    for h, (actual, var_pred, naive_pred) in raw.items():
        r = pairwise_dm(actual, var_pred, naive_pred, h)
        rows.append({
            "horizon_days": h,
            "dm_stat_squared_loss": r["dm_stat_squared_loss"],
            "dm_p_value_squared_loss": r["dm_p_value_squared_loss"],
            "dm_stat_absolute_loss": r["dm_stat_absolute_loss"],
            "dm_p_value_absolute_loss": r["dm_p_value_absolute_loss"],
            # Per-loss-type verdicts -- neither is derived from the other.
            "var_significantly_better_rmse": r["a_significantly_better_rmse"],
            "naive_significantly_better_rmse": r["b_significantly_better_rmse"],
            "var_significantly_better_mae": r["a_significantly_better_mae"],
            "naive_significantly_better_mae": r["b_significantly_better_mae"],
            # Overall verdict: only "significant" when both loss functions agree.
            "var_significantly_better": bool(
                r["a_significantly_better_rmse"] and r["a_significantly_better_mae"]
            ),
            "naive_significantly_better": bool(
                r["b_significantly_better_rmse"] and r["b_significantly_better_mae"]
            ),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Round 3 Path A -- Random Walk benchmark + AIC-lag VAR (issue #28)")
    print("=" * 70)

    print("\n[1/5] Loading Round 2-consolidated feature set...")
    levels = load_levels()
    print(f"  {levels.shape}, {levels.index.min().date()} -> {levels.index.max().date()}")
    print(f"  Variables: {FEATURE_SET}")
    print(f"  Exogenous (issue #72): {EXOG_COLS}; endogenous: "
          f"{[c for c in FEATURE_SET if f'd_{c}' not in EXOG_COLS]}")

    print("\n[2/5] Differencing and confirming stationarity...")
    diffed = to_differenced(levels)
    confirm_stationary(diffed)

    print("\n[3/5] AIC lag order selection...")
    aic_lag = select_aic_lag(diffed)

    print(f"\n[4/5] Building Random Walk benchmark + VAR(AIC lag={aic_lag}), "
          f"scoring RMSE/MAE at {HORIZONS}-day horizons in levels (proposal §5.4)...")
    results, raw, forecasts = evaluate(levels, diffed, aic_lag)
    print("\n" + results.to_string(index=False))

    print("\n[5/5] Diebold-Mariano test: is the RMSE/MAE gap significant, or noise?")
    dm_results = dm_report(raw)
    print("\n" + dm_results.to_string(index=False))
    for _, row in dm_results.iterrows():
        h = int(row["horizon_days"])
        # Remap dm_report()'s var_*/naive_* field names onto plain_language_verdict()'s
        # generic a_*/b_* contract so the verdict text is built by the one shared
        # decision tree (src/model_comparison.py) instead of a second hand-maintained
        # copy of it -- see issue #60/PR #64 review, which found this exact
        # duplication let a "mixed result" branch go missing in a sibling copy.
        verdict_row = {
            "a_significantly_better_rmse": row["var_significantly_better_rmse"],
            "b_significantly_better_rmse": row["naive_significantly_better_rmse"],
            "a_significantly_better_mae": row["var_significantly_better_mae"],
            "b_significantly_better_mae": row["naive_significantly_better_mae"],
            "dm_p_value_squared_loss": row["dm_p_value_squared_loss"],
            "dm_p_value_absolute_loss": row["dm_p_value_absolute_loss"],
        }
        verdict = plain_language_verdict(verdict_row, "VAR", "naive")
        print(f"  h={h}: {verdict}")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_dir / "r3_patha_rmse_mae_vs_naive.csv", index=False)
    dm_results.to_csv(out_dir / "r3_patha_diebold_mariano.csv", index=False)
    forecasts.to_csv(out_dir / "r3_patha_var_aic_forecasts.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_patha_rmse_mae_vs_naive.csv, "
          f"{out_dir}/r3_patha_diebold_mariano.csv, "
          f"{out_dir}/r3_patha_var_aic_forecasts.csv")


if __name__ == "__main__":
    main()
