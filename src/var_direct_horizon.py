"""
Round 3 -- Direct multi-horizon VAR estimation, vs. iterated-cumsum (issue #68)
DAMO-699 Capstone, Group 5

What this is: `EDA_VAR_AIC _lag _order.py`'s `evaluate()` (#28) forecasts h steps
ahead by iterating the one-step VAR forward and cumulatively summing the resulting
daily differenced forecasts (`np.cumsum`) back onto the last observed level. Issue
#68 flags that this compounds h one-step estimation errors into every multi-step
forecast -- variance the zero-parameter Naive benchmark (`level_hat(t+h) =
level(t)`) never carries -- and asks whether a *direct* h-step estimator (Marcellino/
Stock/Watson-style "direct" forecasting, closely related to Jordà's local
projections) changes the h=5/h=20 conclusions.

This script builds that direct-estimation comparison arm: for each horizon h, a
single OLS regression of the h-step-ahead level change `y_{t+h} - y_t` directly on
the same lag-block of endogenous differenced predictors the iterated VAR conditions
its one-step forecast on, refit at every expanding-window origin using the *same*
feature set and the *same* AIC-selected lag order as #28 -- so iterated-vs-direct is
the only thing that differs between this arm and #28's, not a different feature set
or a different lag search.

Limitation (deliberately scoped, not an oversight): the direct regression's
predictors are the endogenous lag-block only -- `d_cpi_yoy` (EXOG_COLS, issue #72)
is excluded. Folding a genuinely exogenous, mean-zero surprise regressor into a
direct h-step design would need its own h-step-ahead exog assumption (a separate
design question from the iterated-vs-direct comparison this script isolates), and
cpi_yoy's near-zero, sparse-jump distribution makes its omission a minor effect
either way (see #72).

Scope: VAR-AIC (#28) only. ARIMA-AIC/BIC (`arima_baseline.ipynb`) and VAR-BIC
(`var_bic_baseline.ipynb`) share the same cumsum pattern per issue #68 but are
notebooks -- fixing those means a fresh top-to-bottom re-run, not just a code
change, and is left as follow-up scope (same reasoning as #72's VAR-BIC/VECM
carve-out).

Reuses `EDA_VAR_AIC _lag _order.py`'s load_levels()/to_differenced()/
confirm_stationary()/select_aic_lag()/dm_report()/_split_endog_exog() as-is --
imported via importlib since the filename contains spaces, same pattern as
`model_comparison.py`'s core_calendar() and `yield_only_var.py`.

Run: python "src/var_direct_horizon.py"
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from model_comparison import pairwise_dm, plain_language_verdict, merge_cross_pipeline  # noqa: E402

_core_path = Path(__file__).resolve().parent / "EDA_VAR_AIC _lag _order.py"
_spec = importlib.util.spec_from_file_location("eda_var_aic_lag_order", _core_path)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)


# ----------------------------------------------------------------------------
# Direct multi-horizon estimation
# ----------------------------------------------------------------------------

def evaluate_direct(levels: pd.DataFrame, diffed: pd.DataFrame, lag: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Expanding-window evaluation, direct-h variant of #28's evaluate(). At each
    origin t and horizon h:
      - Naive forecast:  level_hat(t+h) = level(t)                     [unchanged]
      - Direct forecast: level_hat(t+h) = level(t) + OLS_h(lag-block ending at t)
        where OLS_h is refit at every origin on all fully-realized training pairs
        (i, y_{i+1+h} - y_{i+1}) available strictly before that origin (no lookahead)
      - Actual:          level(t+h)

    The lag-block predictor at the forecast origin is exactly the same `lag` most
    recent endogenous differenced rows the iterated VAR conditions its one-step
    forecast on (`train_endog.values[-lag:]` in evaluate()) -- same information set,
    different estimator.
    """
    endog_all, _ = _core._split_endog_exog(diffed)
    target_diff_col = f"d_{_core.TARGET}"
    n_endog = endog_all.shape[1]
    endog_vals = endog_all.values          # row i ends at levels.iloc[i+1]
    level_vals = levels[_core.TARGET].values
    n_diffed = len(endog_vals)

    # Precompute the full lag-block embedding once (row i valid for i >= lag - 1)
    # instead of rebuilding it inside the origin/horizon loop -- O(n_diffed) instead
    # of O(n_origins * n_horizons * training_size).
    X_all = np.full((n_diffed, lag * n_endog), np.nan)
    for i in range(lag - 1, n_diffed):
        X_all[i] = endog_vals[i - lag + 1: i + 1].reshape(-1)

    max_h = max(_core.HORIZONS)
    records = {h: {"actual": [], "direct_pred": [], "naive_pred": [], "origin_date": []} for h in _core.HORIZONS}

    n = len(levels)
    for origin in range(_core.MIN_TRAIN, n - max_h, _core.STEP):
        i_origin = origin - 2  # diffed row ending exactly at last_level (levels.iloc[origin-1])
        if i_origin < lag - 1:
            continue
        x_origin = np.concatenate([[1.0], X_all[i_origin]])

        last_level = level_vals[origin - 1]
        origin_date = levels.index[origin - 1]

        for h in _core.HORIZONS:
            future_pos = origin - 1 + h
            if future_pos >= n:
                continue

            # Training rows: full lag history (i >= lag - 1) AND a fully realized
            # target strictly before the current origin (i + 1 + h <= origin - 1) --
            # no row here ever sees a target beyond what's known at `origin`.
            i_max = origin - 2 - h
            i_min = lag - 1
            if i_max < i_min:
                continue

            X_train = X_all[i_min:i_max + 1]
            y_train = level_vals[i_min + 1 + h: i_max + 2 + h] - level_vals[i_min + 1: i_max + 2]
            X_train = np.column_stack([np.ones(len(X_train)), X_train])

            beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
            direct_h_change = float(x_origin @ beta)

            actual_level = level_vals[future_pos]
            direct_level_pred = last_level + direct_h_change
            naive_level_pred = last_level

            records[h]["actual"].append(actual_level)
            records[h]["direct_pred"].append(direct_level_pred)
            records[h]["naive_pred"].append(naive_level_pred)
            records[h]["origin_date"].append(origin_date)

    rows = []
    raw = {}
    forecast_parts = []
    for h in _core.HORIZONS:
        actual = np.array(records[h]["actual"])
        direct_pred = np.array(records[h]["direct_pred"])
        naive_pred = np.array(records[h]["naive_pred"])
        if len(actual) == 0:
            print(f"  WARNING: no forecasts generated for horizon {h} -- check MIN_TRAIN/STEP.")
            continue

        rmse_direct = np.sqrt(np.mean((actual - direct_pred) ** 2))
        mae_direct = np.mean(np.abs(actual - direct_pred))
        rmse_naive = np.sqrt(np.mean((actual - naive_pred) ** 2))
        mae_naive = np.mean(np.abs(actual - naive_pred))

        rows.append({
            "horizon_days": h,
            "n_forecasts": len(actual),
            "rmse_var": round(rmse_direct, 4),
            "mae_var": round(mae_direct, 4),
            "rmse_naive": round(rmse_naive, 4),
            "mae_naive": round(mae_naive, 4),
            "var_beats_naive_rmse": bool(rmse_direct < rmse_naive),
            "var_beats_naive_mae": bool(mae_direct < mae_naive),
        })
        raw[h] = (actual, direct_pred, naive_pred)
        forecast_parts.append(pd.DataFrame({
            "origin_date": records[h]["origin_date"], "horizon": h,
            "actual": actual, "var_direct": direct_pred, "naive": naive_pred,
        }))

    forecasts = pd.concat(forecast_parts, ignore_index=True) if forecast_parts else pd.DataFrame(
        columns=["origin_date", "horizon", "actual", "var_direct", "naive"]
    )
    return pd.DataFrame(rows), raw, forecasts


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Round 3 -- Direct multi-horizon VAR vs. iterated-cumsum VAR-AIC (issue #68)")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("\n[1/4] Loading #28's feature set and re-deriving its AIC lag "
          "(same feature set + lag as #28, so only the estimator differs)...")
    levels = _core.load_levels()
    diffed = _core.to_differenced(levels)
    _core.confirm_stationary(diffed)
    aic_lag = _core.select_aic_lag(diffed)
    print(f"  {levels.shape}, lag={aic_lag}")

    print(f"\n[2/4] Direct-h estimation, scoring RMSE/MAE at {_core.HORIZONS}-day horizons in levels...")
    results, raw, forecasts = evaluate_direct(levels, diffed, aic_lag)
    print("\n" + results.to_string(index=False))

    print("\n[3/4] Diebold-Mariano test: direct-h VAR vs. naive...")
    dm_results = _core.dm_report(raw)
    print("\n" + dm_results.to_string(index=False))
    for _, row in dm_results.iterrows():
        h = int(row["horizon_days"])
        verdict_row = {
            "a_significantly_better_rmse": row["var_significantly_better_rmse"],
            "b_significantly_better_rmse": row["naive_significantly_better_rmse"],
            "a_significantly_better_mae": row["var_significantly_better_mae"],
            "b_significantly_better_mae": row["naive_significantly_better_mae"],
            "dm_p_value_squared_loss": row["dm_p_value_squared_loss"],
            "dm_p_value_absolute_loss": row["dm_p_value_absolute_loss"],
        }
        verdict = plain_language_verdict(verdict_row, "direct-h VAR", "naive")
        print(f"  h={h}: {verdict}")

    results.to_csv(out_dir / "r3_direct_rmse_mae_vs_naive.csv", index=False)
    dm_results.to_csv(out_dir / "r3_direct_diebold_mariano.csv", index=False)
    forecasts.to_csv(out_dir / "r3_direct_var_forecasts.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_direct_rmse_mae_vs_naive.csv, "
          f"{out_dir}/r3_direct_diebold_mariano.csv, "
          f"{out_dir}/r3_direct_var_forecasts.csv")

    # ------------------------------------------------------------------
    # [4/4] The actual issue #68 question: does direct estimation change the
    # h=5/h=20 "Naive wins" conclusion vs. #28's iterated-cumsum VAR? Pairwise
    # DM between the two estimators, same feature set/lag, different forecast
    # construction. Uses merge_cross_pipeline() (issue #63) defensively even
    # though both arms are built from the same load_levels() calendar here --
    # cheap insurance against origin misalignment rather than assuming it.
    # ------------------------------------------------------------------
    print("\n[4/4] Direct-h VAR vs. iterated-cumsum VAR-AIC (#28) -- does the estimator matter?...")
    iterated_path = out_dir / "r3_patha_var_aic_forecasts.csv"
    if not iterated_path.exists():
        print(f"  Skipped -- {iterated_path} not found. Run `EDA_VAR_AIC _lag _order.py` first.")
        return

    iterated = pd.read_csv(iterated_path, parse_dates=["origin_date"])
    cal = levels.index  # same load_levels() calendar drives both arms in this script

    direct = forecasts.rename(columns={"var_direct": "var_direct_pred"})
    it = iterated.rename(columns={"var_aic": "var_iterated_pred"})

    merged = merge_cross_pipeline(
        direct[["origin_date", "horizon", "actual", "naive", "var_direct_pred"]],
        "var_direct_pred", cal,
        it[["origin_date", "horizon", "actual", "naive", "var_iterated_pred"]],
        "var_iterated_pred", cal,
    )

    comparison_rows = []
    for h in _core.HORIZONS:
        sub = merged[merged.horizon == h]
        row = pairwise_dm(
            sub["actual"].values, sub["var_direct_pred"].values, sub["var_iterated_pred"].values, h
        )
        verdict = plain_language_verdict(row, "direct-h VAR", "iterated-cumsum VAR (#28)")
        comparison_rows.append({"horizon_days": h, **row, "verdict": verdict})
        print(f"  h={h} (n={row['n_forecasts']}): {verdict}")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "r3_direct_vs_iterated_diebold_mariano.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_direct_vs_iterated_diebold_mariano.csv")


if __name__ == "__main__":
    main()
