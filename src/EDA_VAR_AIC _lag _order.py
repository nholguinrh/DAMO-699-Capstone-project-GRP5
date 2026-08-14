"""
Round 3, Path A — VAR baseline, AIC lag order (issue #28)
DAMO-699 Capstone, Group 5

Scope of this script (per the issue #28 checklist, two boxes only):
  - Build Random Walk benchmark + AIC-selected-lag VAR on the Round 2-consolidated
    feature set
  - RMSE/MAE at 1-/5-/20-day horizons vs. the naive benchmark, per proposal §5.4

Round 2-consolidated feature set (per the stationarity/redundancy memo):
    yield_spread_10y_2y (target), overnight_rate, us_treasury_10y, fed_funds_rate, cpi_yoy

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

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from project_paths import PROCESSED_DIR  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

TARGET = "yield_spread_10y_2y"
FEATURE_SET = ["yield_spread_10y_2y", "overnight_rate", "us_treasury_10y",
               "fed_funds_rate", "cpi_yoy"]

HORIZONS = [1, 5, 20]     # trading days, per proposal §5.4
MAX_LAG_SEARCH = 15
MIN_TRAIN = 500           # minimum obs before the first out-of-sample origin
STEP = 5                  # re-fit every STEP origins (lower = slower, more thorough)
ALPHA = 0.05


# ----------------------------------------------------------------------------
# 1. Load + merge the three Round 1 processed sources
# ----------------------------------------------------------------------------

def load_levels() -> pd.DataFrame:
    boc = pd.read_csv(PROCESSED_DIR / "bank_of_canada_data.csv", parse_dates=["date"])
    fred = pd.read_csv(PROCESSED_DIR / "fred_rates.csv", parse_dates=["date"])
    cpi = pd.read_csv(PROCESSED_DIR / "statcan_cpi.csv",
                       parse_dates=["reference_month", "release_date"])

    df = (boc[["date", "overnight_rate", "yield_spread_10y_2y"]]
          .merge(fred[["date", "us_treasury_10y", "fed_funds_rate"]], on="date", how="inner")
          .sort_values("date")
          .set_index("date"))

    # cpi_yoy: pct_change(12) on reference-month order (not on the raw CPI index level),
    # then forward-filled onto the daily grid keyed on release_date -- a print only
    # becomes real information on the day StatCan actually publishes it. Mirrors
    # 01_eda.ipynb §2/§8; the raw statcan_cpi.csv only ships cpi_all_items, so cpi_yoy
    # is computed here rather than read directly.
    cpi_sorted = cpi.sort_values("reference_month").copy()
    cpi_sorted["cpi_yoy"] = cpi_sorted["cpi_all_items"].pct_change(12) * 100
    # Two reference months have no mapped release_date (the known unmappable-month
    # allowlist from Round 1's config/cpi_release_dates.csv) -- drop those rows before
    # building the daily grid; they can't be forward-filled without a release date.
    cpi_sorted = cpi_sorted.dropna(subset=["release_date"])
    cpi_by_release = (cpi_sorted.sort_values("release_date")
                       .drop_duplicates(subset="release_date", keep="last")
                       .set_index("release_date")["cpi_yoy"])
    daily_grid = pd.date_range(df.index.min(), df.index.max(), freq="D")
    cpi_yoy_daily = cpi_by_release.reindex(daily_grid).ffill()
    df["cpi_yoy"] = cpi_yoy_daily.reindex(df.index)

    df = df[FEATURE_SET].dropna()
    return df


# ----------------------------------------------------------------------------
# 2. Difference for stationarity (confirmed I(1) in the Round 2 stationarity memo)
# ----------------------------------------------------------------------------

def to_differenced(levels: pd.DataFrame) -> pd.DataFrame:
    diffed = levels.diff().dropna()
    diffed.columns = [f"d_{c}" for c in diffed.columns]
    return diffed


def confirm_stationary(diffed: pd.DataFrame) -> None:
    for col in diffed.columns:
        _, pval, *_ = adfuller(diffed[col], autolag="AIC")
        if pval >= ALPHA:
            raise RuntimeError(
                f"{col} is still non-stationary after one difference (p={pval:.3f}). "
                f"Do not proceed to VAR fitting/select_order() until this is resolved."
            )
    print("  All 5 series confirmed stationary post-differencing (p<0.05).")


# ----------------------------------------------------------------------------
# 3. AIC lag order selection
# ----------------------------------------------------------------------------

def select_aic_lag(diffed: pd.DataFrame) -> int:
    model = VAR(diffed)
    order_results = model.select_order(maxlags=MAX_LAG_SEARCH)
    print(order_results.summary())
    aic_lag = order_results.aic
    print(f"\n  AIC-selected lag order: {aic_lag}")
    return aic_lag


# ----------------------------------------------------------------------------
# 4. Random Walk benchmark + AIC-VAR, evaluated in LEVELS at each horizon
# ----------------------------------------------------------------------------

def evaluate(levels: pd.DataFrame, diffed: pd.DataFrame, lag: int) -> tuple[pd.DataFrame, dict]:
    """
    Expanding-window evaluation. At each origin t (indexed into `levels`, offset by 1
    to line up with `diffed`):
      - Naive forecast:  level_hat(t+h) = level(t)                      [proposal §5.4]
      - VAR forecast:    level_hat(t+h) = level(t) + sum(d_target forecasts 1..h)
      - Actual:          level(t+h)
    Both are then scored against the same actual level, so RMSE/MAE are directly
    comparable and expressed in the target's native units (spread, in percentage points).
    """
    target_diff_col = f"d_{TARGET}"
    diff_target_idx = diffed.columns.get_loc(target_diff_col)

    # diffed.index[i] corresponds to the change ending on levels.index[i+1]
    # (levels has one more row than diffed, since diff() drops the first obs).
    level_idx_for_diff_row = {i: levels.index.get_loc(diffed.index[i]) for i in range(len(diffed))}

    max_h = max(HORIZONS)
    records = {h: {"actual": [], "var_pred": [], "naive_pred": []} for h in HORIZONS}

    n = len(diffed)
    for origin in range(MIN_TRAIN, n - max_h, STEP):
        train = diffed.iloc[:origin]
        try:
            fitted = VAR(train).fit(lag)
        except (ValueError, np.linalg.LinAlgError):
            continue

        fc = fitted.forecast(train.values[-lag:], steps=max_h)
        fc_target_diffs = fc[:, diff_target_idx]
        cum_fc = np.cumsum(fc_target_diffs)  # cumulative reconstructed level change

        level_pos = level_idx_for_diff_row[origin]
        last_level = levels[TARGET].iloc[level_pos]

        for h in HORIZONS:
            future_pos = level_pos + h
            if future_pos >= len(levels):
                continue
            actual_level = levels[TARGET].iloc[future_pos]
            var_level_pred = last_level + cum_fc[h - 1]
            naive_level_pred = last_level  # random walk: tomorrow = today, per §5.4

            records[h]["actual"].append(actual_level)
            records[h]["var_pred"].append(var_level_pred)
            records[h]["naive_pred"].append(naive_level_pred)

    rows = []
    raw = {}  # h -> (actual, var_pred, naive_pred) arrays, for the DM test in step 5
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

    return pd.DataFrame(rows), raw


# ----------------------------------------------------------------------------
# 5. Diebold-Mariano test — is the RMSE/MAE gap statistically significant, or noise?
# ----------------------------------------------------------------------------
# Reviewer comment 1 (Nelson), on the first draft's hand-rolled DM implementation:
#   "For the DM statistic's standard error at h=5 and h=20, are you using a
#    Newey-West/Bartlett-weighted long-run variance (lag = h-1), or a plain sample
#    variance divided by n?"
#   -> Answer at the time: plain sample variance / n. That understates the standard
#      error at h=5/h=20, because h-step-ahead forecast errors from adjacent,
#      overlapping origins are serially correlated by construction, and the plain
#      estimator doesn't account for that autocovariance structure.
#
# Reviewer comment 2 (Nelson), follow-up:
#   "I think there is a library that can help me to compare results:
#    pip install dieboldmariano
#    from dieboldmariano import dm_test
#    stat, p_value = dm_test(actual, pred1, pred2, h=horizon, harvey_correction=True)"
#
# Adopted as-is below, with one explicit addition on top of the reviewer's call
# signature: variance_estimator="bartlett" (the library defaults to "acf" if left
# unset). "bartlett" is the Bartlett-kernel-weighted long-run variance estimator
# truncated at lag h-1 -- directly answering comment 1 -- and harvey_correction=True
# adds the Harvey, Leybourne & Newbold (1997) small-sample adjustment per comment 2.

def dm_report(raw: dict) -> pd.DataFrame:
    rows = []
    for h, (actual, var_pred, naive_pred) in raw.items():
        dm_rmse, p_rmse = dm_test(
            actual, var_pred, naive_pred,
            loss=lambda u, v: (u - v) ** 2,
            h=h, harvey_correction=True, variance_estimator="bartlett",
        )
        dm_mae, p_mae = dm_test(
            actual, var_pred, naive_pred,
            loss=lambda u, v: abs(u - v),
            h=h, harvey_correction=True, variance_estimator="bartlett",
        )
        rows.append({
            "horizon_days": h,
            "dm_stat_squared_loss": round(dm_rmse, 3),
            "dm_p_value_squared_loss": round(p_rmse, 4),
            "dm_stat_absolute_loss": round(dm_mae, 3),
            "dm_p_value_absolute_loss": round(p_mae, 4),
            "var_significantly_better": bool(p_rmse < ALPHA and dm_rmse < 0),
            "naive_significantly_better": bool(p_rmse < ALPHA and dm_rmse > 0),
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

    print("\n[2/5] Differencing and confirming stationarity...")
    diffed = to_differenced(levels)
    confirm_stationary(diffed)

    print("\n[3/5] AIC lag order selection...")
    aic_lag = select_aic_lag(diffed)

    print(f"\n[4/5] Building Random Walk benchmark + VAR(AIC lag={aic_lag}), "
          f"scoring RMSE/MAE at {HORIZONS}-day horizons in levels (proposal §5.4)...")
    results, raw = evaluate(levels, diffed, aic_lag)
    print("\n" + results.to_string(index=False))

    print("\n[5/5] Diebold-Mariano test: is the RMSE/MAE gap significant, or noise?")
    dm_results = dm_report(raw)
    print("\n" + dm_results.to_string(index=False))
    for _, row in dm_results.iterrows():
        if row["var_significantly_better"]:
            verdict = "VAR significantly better than naive"
        elif row["naive_significantly_better"]:
            verdict = "naive significantly better than VAR"
        else:
            verdict = "no significant difference -- don't headline either RMSE number"
        print(f"  h={int(row['horizon_days'])}: {verdict} "
              f"(p={row['dm_p_value_squared_loss']:.4f})")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_dir / "r3_patha_rmse_mae_vs_naive.csv", index=False)
    dm_results.to_csv(out_dir / "r3_patha_diebold_mariano.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_patha_rmse_mae_vs_naive.csv, "
          f"{out_dir}/r3_patha_diebold_mariano.csv")


if __name__ == "__main__":
    main()