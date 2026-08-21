"""
Johansen Cointegration Test + Conditional VECM (Issue #47)
DAMO-699 Capstone, Group 5

Purpose:
  Test for long-run cointegrating relationships among the Gold-layer level
  series using the Johansen (1991) procedure.  If cointegration is found
  (r >= 1), estimate a VECM alongside the existing stationary VAR baselines
  and evaluate forecast performance at 1-/5-/20-day horizons.

  Both VAR-AIC(10) and VAR-BIC(0) baselines fail to beat the naive Random
  Walk benchmark (issues #28, #29, #43).  The VECM's Error Correction Term
  (alpha * beta' * Y_{t-1}) may recover predictive signal that pure
  differencing discards.

Depends on:
  #30 (Gold-layer feature pipeline) — CLOSED.
  Data: data/processed/gold_features.csv

Methodology:
  - Johansen test across 3 deterministic specifications supported by
    statsmodels' coint_johansen (det_order in {-1, 0, 1}).
  - Primary specification: restricted constant (det_order=0, VECM det="ci").
  - Decision criterion: trace statistic at 95% significance.
  - Evaluation: expanding-window protocol matching VAR baselines
    (MIN_TRAIN=500, STEP=5, horizons {1,5,20} days, scored in levels).
  - 4-way Diebold-Mariano comparison: VECM vs naive, VAR-AIC, VAR-BIC.

Run:
  python src/johansen_vecm.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# The daily trading index has business-day gaps (weekends/holidays) with no fixed freq,
# which statsmodels flags on every VAR/VECM fit call as a ValueWarning. Suppress this
# specific warning while letting other warnings through.
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from model_comparison import pairwise_dm, plain_language_verdict  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET = "yield_spread_10y_2y"

FEATURE_SET_5VAR = [
    "yield_spread_10y_2y", "overnight_rate", "us_treasury_10y",
    "fed_funds_rate", "cpi_yoy",
]
FEATURE_SET_6VAR = FEATURE_SET_5VAR + ["usdcad"]

HORIZONS = [1, 5, 20]     # trading days, per proposal section 5.4
MAX_LAG_SEARCH = 15
MIN_TRAIN = 500            # minimum obs before first out-of-sample origin
STEP = 5                   # re-fit every STEP origins
ALPHA = 0.05

# Johansen deterministic specifications supported by statsmodels.
# Mapping: coint_johansen det_order -> (human label, VECM deterministic kwarg).
JOHANSEN_SPECS = {
    -1: {"label": "No deterministic terms",                       "vecm_det": "n"},
     0: {"label": "Restricted constant (intercept in CE only)",   "vecm_det": "ci"},
     1: {"label": "Restricted constant + restricted linear trend", "vecm_det": "li"},
}


# ---------------------------------------------------------------------------
# 1. Load Gold-layer level series
# ---------------------------------------------------------------------------

def load_gold_levels(feature_set: list[str]) -> pd.DataFrame:
    """
    Load the canonical Gold-layer CSV and return only the requested level
    columns, indexed by date, with NaN rows dropped.
    """
    gold = pd.read_csv(PROCESSED_DIR / "gold_features.csv", parse_dates=["date"])
    gold = gold.set_index("date").sort_index()
    levels = gold[feature_set].dropna()
    return levels


# ---------------------------------------------------------------------------
# 2. Confirm all series are I(1) & gate pipeline
# ---------------------------------------------------------------------------

def confirm_i1(
    levels: pd.DataFrame, alpha: float = ALPHA, enforce: bool = True
) -> pd.DataFrame:
    """
    ADF test on levels (expect non-stationary) and first differences (expect
    stationary).  A series is confirmed I(1) when p_level >= alpha AND
    p_diff < alpha.

    Parameters
    ----------
    levels : pd.DataFrame
        Level time series.
    alpha : float, default 0.05
        Significance level for ADF tests.
    enforce : bool, default True
        If True, raises RuntimeError if any series fails I(1) confirmation,
        preventing invalid rank estimation.
    """
    results = []
    for col in levels.columns:
        _, p_level, *_ = adfuller(levels[col], autolag="AIC")
        _, p_diff, *_ = adfuller(levels[col].diff().dropna(), autolag="AIC")
        confirmed = (p_level >= alpha) and (p_diff < alpha)
        results.append({
            "variable": col,
            "adf_p_level": round(p_level, 4),
            "adf_p_diff": round(p_diff, 4),
            "I1_confirmed": confirmed,
        })
    df = pd.DataFrame(results)
    n_ok = df["I1_confirmed"].sum()
    print(f"  {n_ok}/{len(df)} series confirmed I(1).")
    for _, row in df.iterrows():
        tag = "I(1)" if row["I1_confirmed"] else "NOT I(1)"
        print(f"    {row['variable']:25s}  p_level={row['adf_p_level']:.4f}  "
              f"p_diff={row['adf_p_diff']:.4f}  [{tag}]")

    failed = df[~df["I1_confirmed"]]
    if not failed.empty and enforce:
        failed_vars = failed["variable"].tolist()
        raise RuntimeError(
            f"Stationarity assumption violated: variable(s) failed I(1) confirmation: {failed_vars}. "
            "Johansen cointegration requires all input series to be I(1). "
            "Set enforce=False only for diagnostic overrides."
        )

    return df


# ---------------------------------------------------------------------------
# 3. Lag order selection on the levels VAR
# ---------------------------------------------------------------------------

def select_lag_levels(
    levels: pd.DataFrame, maxlags: int = MAX_LAG_SEARCH,
) -> tuple[int, object]:
    """
    Run VAR.select_order on the *level* series (not differenced).
    Returns the AIC-selected lag and the full order-selection object.
    """
    model = VAR(levels)
    order_results = model.select_order(maxlags=maxlags)
    print(order_results.summary())
    aic = order_results.aic
    print(f"\n  AIC={aic}, BIC={order_results.bic}, "
          f"HQIC={order_results.hqic}, FPE={order_results.fpe}")
    return aic, order_results


# ---------------------------------------------------------------------------
# 4. Johansen cointegration test — all supported deterministic specs
# ---------------------------------------------------------------------------

def run_johansen_all_specs(
    levels: pd.DataFrame, k_ar_diff: int,
) -> tuple[pd.DataFrame, int]:
    """
    Run coint_johansen for each deterministic specification.
    Returns a tidy DataFrame of all results and the cointegration rank
    under the primary specification (restricted constant, det_order=0).
    """
    all_rows: list[dict] = []
    primary_rank = 0
    n_vars = levels.shape[1]

    for det_order, spec in JOHANSEN_SPECS.items():
        result = coint_johansen(levels.values, det_order, k_ar_diff)

        # Sequential testing — trace statistic at 95%
        trace_rank = 0
        for i in range(n_vars):
            if result.lr2[i] > result.cvt[i, 1]:
                trace_rank = i + 1
            else:
                break

        # Sequential testing — max-eigenvalue at 95%
        maxeig_rank = 0
        for i in range(n_vars):
            if result.lr1[i] > result.cvm[i, 1]:
                maxeig_rank = i + 1
            else:
                break

        for i in range(n_vars):
            all_rows.append({
                "det_order": det_order,
                "det_label": spec["label"],
                "r_null": i,
                "trace_stat": round(float(result.lr2[i]), 4),
                "trace_cv_90": round(float(result.cvt[i, 0]), 4),
                "trace_cv_95": round(float(result.cvt[i, 1]), 4),
                "trace_cv_99": round(float(result.cvt[i, 2]), 4),
                "trace_reject_95": bool(result.lr2[i] > result.cvt[i, 1]),
                "max_eig_stat": round(float(result.lr1[i]), 4),
                "max_eig_cv_90": round(float(result.cvm[i, 0]), 4),
                "max_eig_cv_95": round(float(result.cvm[i, 1]), 4),
                "max_eig_cv_99": round(float(result.cvm[i, 2]), 4),
                "max_eig_reject_95": bool(result.lr1[i] > result.cvm[i, 1]),
                "eigenvalue": round(float(result.eig[i]), 6),
                "trace_rank": trace_rank,
                "max_eig_rank": maxeig_rank,
            })

        if det_order == 0:
            primary_rank = trace_rank

        agree = "AGREE" if trace_rank == maxeig_rank else "DISAGREE"
        print(f"\n  det_order={det_order:+d}  ({spec['label']})")
        print(f"    Trace rank: r={trace_rank},  Max-Eig rank: r={maxeig_rank}  [{agree}]")
        for i in range(n_vars):
            rej = "REJECT" if result.lr2[i] > result.cvt[i, 1] else "fail to reject"
            print(f"    H0: r<={i}:  trace={result.lr2[i]:9.2f}  "
                  f"cv_95={result.cvt[i, 1]:8.2f}  -> {rej}")

    print(f"\n  Primary (restricted constant, det_order=0): rank r = {primary_rank}")
    return pd.DataFrame(all_rows), primary_rank


# ---------------------------------------------------------------------------
# 5. VECM estimation + summary
# ---------------------------------------------------------------------------

def fit_and_summarize_vecm(
    levels: pd.DataFrame, k_ar_diff: int, coint_rank: int,
    det_spec: str = "ci",
) -> object:
    """
    Fit the VECM and print its summary, cointegrating vectors (beta),
    and adjustment coefficients (alpha).
    """
    model = VECM(
        levels, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
        deterministic=det_spec,
    )
    result = model.fit()
    print(result.summary())

    beta = pd.DataFrame(
        result.beta, index=levels.columns,
        columns=[f"CV_{j+1}" for j in range(coint_rank)],
    )
    alpha = pd.DataFrame(
        result.alpha, index=levels.columns,
        columns=[f"alpha_{j+1}" for j in range(coint_rank)],
    )
    print(f"\n  Cointegrating vectors (beta):\n{beta.to_string()}")
    print(f"\n  Adjustment coefficients (alpha):\n{alpha.to_string()}")
    return result


# ---------------------------------------------------------------------------
# 6. Expanding-window evaluation (VECM + VAR-AIC + VAR-BIC + naive)
# ---------------------------------------------------------------------------

def evaluate_vecm(
    levels: pd.DataFrame,
    diffed: pd.DataFrame,
    k_ar_diff: int,
    coint_rank: int,
    det_spec: str,
    aic_lag_diff: int,
    bic_lag_diff: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Expanding-window forecast evaluation matching the VAR baseline protocol.
    At each origin, all four models are evaluated on the same data so the
    Diebold-Mariano test receives truly paired forecast errors.

    Parameters
    ----------
    levels : pd.DataFrame
        Level time series.
    diffed : pd.DataFrame
        First-differenced time series (d_* columns).
    k_ar_diff : int
        Number of differenced lags for VECM.
    coint_rank : int
        Cointegrating rank for VECM.
    det_spec : str
        Deterministic specification for VECM (e.g. 'ci').
    aic_lag_diff : int
        Lag order for differenced VAR-AIC baseline.
    bic_lag_diff : int
        Lag order for differenced VAR-BIC baseline (if 0, uses drift model).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Metrics summary DataFrame and dict of raw arrays keyed by horizon.
    """
    target_idx = list(levels.columns).index(TARGET)
    target_diff_col = f"d_{TARGET}"
    diff_target_idx = list(diffed.columns).index(target_diff_col)

    max_h = max(HORIZONS)
    records = {
        h: {"actual": [], "vecm": [], "var_aic": [], "var_bic": [], "naive": [], "origin_date": []}
        for h in HORIZONS
    }

    n = len(levels)
    n_origins = 0
    n_failed = 0

    for origin in range(MIN_TRAIN, n - max_h, STEP):
        train_levels = levels.iloc[:origin]
        last_level = float(train_levels[TARGET].iloc[-1])

        # -- VECM forecast (returns levels directly) --
        try:
            vecm_fit = VECM(
                train_levels, k_ar_diff=k_ar_diff,
                coint_rank=coint_rank, deterministic=det_spec,
            ).fit()
            vecm_fc = vecm_fit.predict(steps=max_h)  # (max_h, n_vars) in levels
        except (ValueError, np.linalg.LinAlgError, IndexError):
            n_failed += 1
            continue

        # -- VAR-AIC forecast (on differences, reconstructed to levels) --
        train_diff = diffed.iloc[:origin - 1]
        if len(train_diff) < aic_lag_diff:
            n_failed += 1
            continue
        try:
            var_fit = VAR(train_diff).fit(aic_lag_diff)
            var_fc = var_fit.forecast(
                train_diff.values[-aic_lag_diff:], steps=max_h,
            )
            var_cum_target = np.cumsum(var_fc[:, diff_target_idx])
        except (ValueError, np.linalg.LinAlgError, IndexError):
            n_failed += 1
            continue

        # -- VAR-BIC forecast (dynamic lag: if lag=0 drift model; if lag>0 VAR) --
        if bic_lag_diff == 0:
            mean_diff = float(train_diff[target_diff_col].mean())
            var_bic_cum_target = np.array([h_i * mean_diff for h_i in range(1, max_h + 1)])
        else:
            if len(train_diff) < bic_lag_diff:
                n_failed += 1
                continue
            try:
                var_bic_fit = VAR(train_diff).fit(bic_lag_diff)
                var_bic_fc = var_bic_fit.forecast(
                    train_diff.values[-bic_lag_diff:], steps=max_h,
                )
                var_bic_cum_target = np.cumsum(var_bic_fc[:, diff_target_idx])
            except (ValueError, np.linalg.LinAlgError, IndexError):
                n_failed += 1
                continue

        origin_date = levels.index[origin - 1]
        for h in HORIZONS:
            future_pos = origin - 1 + h
            if future_pos >= n:
                continue

            actual_val = float(levels[TARGET].iloc[future_pos])
            records[h]["actual"].append(actual_val)
            records[h]["vecm"].append(float(vecm_fc[h - 1, target_idx]))
            records[h]["var_aic"].append(last_level + float(var_cum_target[h - 1]))
            records[h]["var_bic"].append(last_level + float(var_bic_cum_target[h - 1]))
            records[h]["naive"].append(last_level)
            records[h]["origin_date"].append(origin_date)

        n_origins += 1
        if n_origins % 100 == 0:
            pct = 100 * origin / (n - max_h)
            print(f"    ... {n_origins} origins evaluated  ({pct:.0f}%)")

    print(f"  Completed: {n_origins} origins, {n_failed} failures skipped.")

    # Compute per-horizon metrics
    rows: list[dict] = []
    raw: dict = {}
    forecast_parts: list[pd.DataFrame] = []  # per-origin forecasts, for comparison (issue #50)
    for h in HORIZONS:
        a = np.array(records[h]["actual"])
        v = np.array(records[h]["vecm"])
        va = np.array(records[h]["var_aic"])
        vb = np.array(records[h]["var_bic"])
        nv = np.array(records[h]["naive"])

        if len(a) == 0:
            print(f"  WARNING: no forecasts for h={h}")
            continue

        rmse_v = np.sqrt(np.mean((a - v) ** 2))
        mae_v = np.mean(np.abs(a - v))
        rmse_va = np.sqrt(np.mean((a - va) ** 2))
        mae_va = np.mean(np.abs(a - va))
        rmse_vb = np.sqrt(np.mean((a - vb) ** 2))
        mae_vb = np.mean(np.abs(a - vb))
        rmse_nv = np.sqrt(np.mean((a - nv) ** 2))
        mae_nv = np.mean(np.abs(a - nv))

        rows.append({
            "horizon_days": h,
            "n_forecasts": len(a),
            "rmse_vecm": round(rmse_v, 6),
            "mae_vecm": round(mae_v, 6),
            "rmse_var_aic": round(rmse_va, 6),
            "mae_var_aic": round(mae_va, 6),
            "rmse_var_bic": round(rmse_vb, 6),
            "mae_var_bic": round(mae_vb, 6),
            "rmse_naive": round(rmse_nv, 6),
            "mae_naive": round(mae_nv, 6),
            "vecm_beats_naive_rmse": bool(rmse_v < rmse_nv),
            "vecm_beats_naive_mae": bool(mae_v < mae_nv),
            "vecm_beats_var_aic_rmse": bool(rmse_v < rmse_va),
            "vecm_beats_var_aic_mae": bool(mae_v < mae_va),
            "vecm_beats_var_bic_rmse": bool(rmse_v < rmse_vb),
            "vecm_beats_var_bic_mae": bool(mae_v < mae_vb),
        })
        raw[h] = (a, v, va, vb, nv)
        forecast_parts.append(pd.DataFrame({
            "origin_date": records[h]["origin_date"], "horizon": h,
            "actual": a, "vecm": v, "var_aic": va, "var_bic": vb, "naive": nv,
        }))

    forecasts = pd.concat(forecast_parts, ignore_index=True) if forecast_parts else pd.DataFrame(
        columns=["origin_date", "horizon", "actual", "vecm", "var_aic", "var_bic", "naive"]
    )
    return pd.DataFrame(rows), raw, forecasts


# ---------------------------------------------------------------------------
# 7. Diebold-Mariano tests — 4-way comparison
# ---------------------------------------------------------------------------

def dm_report_vecm(raw: dict) -> pd.DataFrame:
    """
    Run DM tests comparing VECM against naive, VAR-AIC, and VAR-BIC.
    Uses Harvey-corrected, Bartlett-kernel long-run variance (h-1 truncation)
    matching the existing VAR baselines' dm_report() (issues #43, #44).

    Computes the actual test via the shared `pairwise_dm()` (src/model_comparison.py)
    so this and #28's dm_report() don't each hand-maintain their own copy of the same
    dm_test() call parameters.
    """
    rows: list[dict] = []

    for h, (actual, vecm, var_aic, var_bic, naive) in raw.items():
        comparisons = {"naive": naive, "var_aic": var_aic, "var_bic": var_bic}
        row: dict = {"horizon_days": h}

        for name, alt in comparisons.items():
            r = pairwise_dm(actual, vecm, alt, h)
            row[f"dm_vecm_{name}_stat_sq"] = r["dm_stat_squared_loss"]
            row[f"dm_vecm_{name}_p_sq"] = r["dm_p_value_squared_loss"]
            row[f"dm_vecm_{name}_stat_abs"] = r["dm_stat_absolute_loss"]
            row[f"dm_vecm_{name}_p_abs"] = r["dm_p_value_absolute_loss"]
            row[f"vecm_sig_better_{name}_rmse"] = r["a_significantly_better_rmse"]
            row[f"{name}_sig_better_vecm_rmse"] = r["b_significantly_better_rmse"]
            row[f"vecm_sig_better_{name}_mae"] = r["a_significantly_better_mae"]
            row[f"{name}_sig_better_vecm_mae"] = r["b_significantly_better_mae"]

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Full pipeline for one feature set
# ---------------------------------------------------------------------------

def run_pipeline(feature_set: list[str], label: str, enforce_i1: bool = True) -> dict:
    """
    Execute the complete Johansen/VECM analysis for a given feature set.
    Returns a dict containing all intermediate and final results.
    """
    print(f"\n{'=' * 70}")
    print(f"  Johansen / VECM Pipeline  —  {label}")
    print(f"{'=' * 70}")

    # -- Step 1: Load --
    print(f"\n[1/7] Loading Gold-layer level series ({label})...")
    levels = load_gold_levels(feature_set)
    print(f"  Shape: {levels.shape},  "
          f"{levels.index.min().date()} -> {levels.index.max().date()}")

    # -- Step 2: Confirm I(1) & Gate --
    print(f"\n[2/7] Confirming I(1) via ADF tests...")
    adf_results = confirm_i1(levels, enforce=enforce_i1)

    # Prepare differenced series once for downstream use
    diffed = levels.diff().dropna()
    diffed.columns = [f"d_{c}" for c in levels.columns]

    # -- Step 3: Lag selection (levels VAR + differenced VAR) --
    print(f"\n[3/7] Lag order selection on levels VAR (maxlags={MAX_LAG_SEARCH})...")
    aic_lag_levels, order_results = select_lag_levels(levels)
    k_ar_diff = max(1, aic_lag_levels - 1)

    # Dynamic AIC and BIC selection on differenced VAR for comparison baselines
    diff_order_results = VAR(diffed).select_order(maxlags=MAX_LAG_SEARCH)
    aic_lag_diff = max(1, diff_order_results.aic)
    bic_lag_diff = diff_order_results.bic  # Can be 0 (drift model) or >= 1
    print(f"  Levels VAR AIC lag: {aic_lag_levels}  ->  VECM k_ar_diff = {k_ar_diff}")
    print(f"  Differenced VAR AIC lag: {aic_lag_diff}, BIC lag: {bic_lag_diff} (baselines)")

    # -- Step 4: Johansen cointegration test --
    print(f"\n[4/7] Johansen cointegration test (3 deterministic specifications)...")
    johansen_df, primary_rank = run_johansen_all_specs(levels, k_ar_diff)

    result = {
        "label": label,
        "feature_set": feature_set,
        "levels": levels,
        "diffed": diffed,
        "adf_results": adf_results,
        "aic_lag_levels": aic_lag_levels,
        "k_ar_diff": k_ar_diff,
        "aic_lag_diff": aic_lag_diff,
        "bic_lag_diff": bic_lag_diff,
        "johansen_df": johansen_df,
        "primary_rank": primary_rank,
        "vecm_fit": None,
        "eval_metrics": None,
        "eval_raw": None,
        "eval_forecasts": None,
        "dm_results": None,
    }

    # -- Negative result: r = 0 --
    if primary_rank == 0:
        print(f"\n[5/7] No cointegrating vectors found (r=0).  VECM not estimated.")
        print(f"  {'─' * 64}")
        print(f"  SUBSTANTIVE FINDING: The {len(feature_set)}-variable system does")
        print(f"  not exhibit long-run equilibrium relationships at 95% significance")
        print(f"  under the primary specification (restricted constant) over the")
        print(f"  {levels.index.min().date()} to {levels.index.max().date()} sample.")
        print(f"  {'─' * 64}")
        print(f"  Possible explanations: structural breaks (QE 2010-14, COVID 2020,")
        print(f"  aggressive rate hiking cycle 2022-23), regime changes in monetary")
        print(f"  policy, or genuine absence of cointegration among these variables.")
        print(f"\n[6/7] Skipped (no VECM to evaluate).")
        print(f"[7/7] Skipped (no DM tests).")
        return result

    # -- Step 5: Estimate VECM --
    det_spec = JOHANSEN_SPECS[0]["vecm_det"]  # "ci" for restricted constant
    print(f"\n[5/7] Estimating VECM(k_ar_diff={k_ar_diff}, rank={primary_rank}, "
          f"det='{det_spec}')...")
    vecm_fit = fit_and_summarize_vecm(levels, k_ar_diff, primary_rank, det_spec)
    result["vecm_fit"] = vecm_fit

    # -- Step 6: Expanding-window evaluation --
    print(f"\n[6/7] Expanding-window evaluation "
          f"(MIN_TRAIN={MIN_TRAIN}, STEP={STEP}, h={HORIZONS})...")
    eval_metrics, eval_raw, eval_forecasts = evaluate_vecm(
        levels=levels,
        diffed=diffed,
        k_ar_diff=k_ar_diff,
        coint_rank=primary_rank,
        det_spec=det_spec,
        aic_lag_diff=aic_lag_diff,
        bic_lag_diff=bic_lag_diff,
    )
    print("\n" + eval_metrics.to_string(index=False))
    result["eval_metrics"] = eval_metrics
    result["eval_raw"] = eval_raw
    result["eval_forecasts"] = eval_forecasts

    # -- Step 7: Diebold-Mariano tests --
    print(f"\n[7/7] Diebold-Mariano significance tests (4-way comparison)...")
    dm_results = dm_report_vecm(eval_raw)
    result["dm_results"] = dm_results

    # Print verdicts. Built via the shared plain_language_verdict() decision tree
    # (src/model_comparison.py) instead of a second hand-maintained copy of it --
    # this file's own inline copy was missing the "mixed result" branch (RMSE and
    # MAE significant but disagreeing on the winner), which silently printed a
    # one-sided verdict when that happened. Found in PR #64 review.
    for _, row in dm_results.iterrows():
        h = int(row["horizon_days"])
        verdicts = []
        for rival in ["naive", "var_aic", "var_bic"]:
            verdict_row = {
                "a_significantly_better_rmse": row.get(f"vecm_sig_better_{rival}_rmse", False),
                "b_significantly_better_rmse": row.get(f"{rival}_sig_better_vecm_rmse", False),
                "a_significantly_better_mae": row.get(f"vecm_sig_better_{rival}_mae", False),
                "b_significantly_better_mae": row.get(f"{rival}_sig_better_vecm_mae", False),
                "dm_p_value_squared_loss": row.get(f"dm_vecm_{rival}_p_sq"),
                "dm_p_value_absolute_loss": row.get(f"dm_vecm_{rival}_p_abs"),
            }
            verdicts.append(f"vs {rival}: {plain_language_verdict(verdict_row, 'VECM', rival)}")
        print(f"  h={h}: {'; '.join(verdicts)}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # -- Run 5-variable system (primary) --
    r5 = run_pipeline(FEATURE_SET_5VAR, "5-variable system")

    # -- Run 6-variable robustness check --
    r6 = run_pipeline(FEATURE_SET_6VAR, "6-variable system (+ usdcad)")

    # -- Save outputs --
    print(f"\n{'=' * 70}")
    print("  Saving outputs")
    print(f"{'=' * 70}")

    # Johansen results (both systems)
    j5 = r5["johansen_df"].copy()
    j5.insert(0, "system", "5var")
    j6 = r6["johansen_df"].copy()
    j6.insert(0, "system", "6var")
    johansen_all = pd.concat([j5, j6], ignore_index=True)
    johansen_all.to_csv(out_dir / "johansen_cointegration_results.csv", index=False)
    print("  -> johansen_cointegration_results.csv")

    # Lag selection (including both AIC and dynamic BIC for differenced VAR)
    lag_df = pd.DataFrame([
        {
            "system": "5var",
            "aic_lag_levels": r5["aic_lag_levels"],
            "k_ar_diff": r5["k_ar_diff"],
            "aic_lag_diff": r5["aic_lag_diff"],
            "bic_lag_diff": r5["bic_lag_diff"],
        },
        {
            "system": "6var",
            "aic_lag_levels": r6["aic_lag_levels"],
            "k_ar_diff": r6["k_ar_diff"],
            "aic_lag_diff": r6["aic_lag_diff"],
            "bic_lag_diff": r6["bic_lag_diff"],
        },
    ])
    lag_df.to_csv(out_dir / "johansen_lag_selection.csv", index=False)
    print("  -> johansen_lag_selection.csv")

    # VECM metrics & DM results (if VECM was estimated)
    metrics_parts, dm_parts = [], []
    for r, sys_label in [(r5, "5var"), (r6, "6var")]:
        if r["eval_metrics"] is not None:
            m = r["eval_metrics"].copy()
            m.insert(0, "system", sys_label)
            metrics_parts.append(m)
        if r["dm_results"] is not None:
            d = r["dm_results"].copy()
            d.insert(0, "system", sys_label)
            dm_parts.append(d)

    if metrics_parts:
        pd.concat(metrics_parts).to_csv(
            out_dir / "vecm_rmse_mae_vs_naive.csv", index=False,
        )
        print("  -> vecm_rmse_mae_vs_naive.csv")
    if dm_parts:
        pd.concat(dm_parts).to_csv(
            out_dir / "vecm_diebold_mariano.csv", index=False,
        )
        print("  -> vecm_diebold_mariano.csv")

    # Per-origin forecasts for the 6-variable system, for cross-model comparison (issue #50) --
    # 5var isn't exported since #50 uses the 6var system to match ARIMA/VAR-AIC/VAR-BIC/LSTM's
    # proposal-correct feature set (see M2_CHECKLIST.md's Aug 19 usdcad entry).
    if r6["eval_forecasts"] is not None and not r6["eval_forecasts"].empty:
        r6["eval_forecasts"].to_csv(out_dir / "r3_vecm_6var_forecasts.csv", index=False)
        print("  -> r3_vecm_6var_forecasts.csv")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for r in [r5, r6]:
        rank = r["primary_rank"]
        if rank == 0:
            print(f"  {r['label']}: r=0 (no cointegration) -> VECM not estimated")
        else:
            print(f"  {r['label']}: r={rank} cointegrating vector(s) "
                  f"-> VECM estimated & evaluated")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
