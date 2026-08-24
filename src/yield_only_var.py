"""
Round 3 -- Yield-only VAR baseline (issue #70)
DAMO-699 Capstone, Group 5

Scope of this script (per issue #70's checklist):
  - Estimate an AIC-lag VAR on yield-curve-only variables (no macro transmission
    variables), evaluated with the same expanding-window/DM protocol as the existing
    6-variable VAR-AIC baseline (#28).
  - Compare yield-only vs. the existing 6-variable VAR-AIC via pairwise DM test at
    each horizon, to actually answer proposal Section 3.1's research question --
    "does macro add predictive value over yield-only dynamics" -- which no model had
    ever tested before this issue (every baseline shipped with the full 6-variable
    set from the start).

Feature set: yield_spread_10y_2y (target), yield_2y, yield_5y -- the Canadian yield
curve's own levels, with `overnight_rate`, `us_treasury_10y`, `fed_funds_rate`,
`cpi_yoy`, `usdcad` all excluded (per issue #70's proposed fix). `us_treasury_10y`/
`fed_funds_rate` are excluded here as US-market macro variables, consistent with
proposal Section 3.1 naming only the overnight rate/USD-CAD/CPI as the "macro
transmission variables" under test -- open to revisiting if the team wants a
stricter or looser line (see issue #70 discussion).

Deviates from issue #70's own suggested example set (`yield_spread_10y_2y, yield_2y,
yield_10y`): that trio is exactly collinear -- `yield_spread_10y_2y` is defined as
`yield_10y - yield_2y`, the same exact-linear-combination problem flagged in issue
#67 -- and produces a singular covariance matrix (`select_order()` raised
`LinAlgError: leading minor is not positive definite` when tried). `yield_5y` swapped
in for `yield_10y` instead: not algebraically determined by the other two, so the
system stays full rank, while still giving the VAR three genuinely distinct points on
the curve (short end, belly, slope) rather than silently collapsing to two.

Reuses `EDA_VAR_AIC _lag _order.py`'s to_differenced()/confirm_stationary()/
select_aic_lag()/evaluate()/dm_report() as-is (all parameterized on the levels/diffed
frame passed in, not hardcoded to the 6-variable FEATURE_SET) rather than
copy-pasting the same VAR-fitting/evaluation logic a third time -- imported via
importlib since the filename contains spaces, same pattern as
`model_comparison.py`'s core_calendar().

Run: python "src/yield_only_var.py"
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from model_comparison import pairwise_dm, plain_language_verdict, merge_cross_pipeline  # noqa: E402

# ----------------------------------------------------------------------------
# Reuse the AIC-path script's generic VAR machinery (issue #28) instead of a
# second hand-maintained copy of the same fitting/evaluation logic.
# ----------------------------------------------------------------------------
_core_path = Path(__file__).resolve().parent / "EDA_VAR_AIC _lag _order.py"
_spec = importlib.util.spec_from_file_location("eda_var_aic_lag_order", _core_path)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

YIELD_ONLY_FEATURE_SET = ["yield_spread_10y_2y", "yield_2y", "yield_5y"]


def load_levels() -> pd.DataFrame:
    gold = pd.read_csv(PROCESSED_DIR / "gold_features.csv", parse_dates=["date"])
    return gold.set_index("date").sort_index()[YIELD_ONLY_FEATURE_SET].dropna()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Round 3 -- Yield-only VAR baseline (issue #70)")
    print("=" * 70)

    print("\n[1/6] Loading yield-only feature set...")
    levels = load_levels()
    print(f"  {levels.shape}, {levels.index.min().date()} -> {levels.index.max().date()}")
    print(f"  Variables: {YIELD_ONLY_FEATURE_SET}")

    print("\n[2/6] Differencing and confirming stationarity...")
    diffed = _core.to_differenced(levels)
    _core.confirm_stationary(diffed)

    print("\n[3/6] AIC lag order selection...")
    aic_lag = _core.select_aic_lag(diffed)

    print(f"\n[4/6] Building Random Walk benchmark + VAR(AIC lag={aic_lag}), "
          f"scoring RMSE/MAE at {_core.HORIZONS}-day horizons in levels...")
    results, raw, forecasts = _core.evaluate(levels, diffed, aic_lag)
    print("\n" + results.to_string(index=False))

    print("\n[5/6] Diebold-Mariano test: yield-only VAR vs. naive...")
    dm_results = _core.dm_report(raw)
    print("\n" + dm_results.to_string(index=False))

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_dir / "r3_yieldonly_rmse_mae_vs_naive.csv", index=False)
    dm_results.to_csv(out_dir / "r3_yieldonly_diebold_mariano.csv", index=False)
    forecasts.to_csv(out_dir / "r3_yieldonly_var_aic_forecasts.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_yieldonly_rmse_mae_vs_naive.csv, "
          f"{out_dir}/r3_yieldonly_diebold_mariano.csv, "
          f"{out_dir}/r3_yieldonly_var_aic_forecasts.csv")

    # ------------------------------------------------------------------
    # [6/6] The actual research question (proposal Section 3.1, issue #70's
    # DoD bullet 2): does the 6-variable macro set add predictive value over
    # yield-only dynamics? Cross-pipeline DM comparison against the existing
    # 6-variable VAR-AIC baseline (#28), at each horizon. Uses
    # merge_cross_pipeline() (issue #63) rather than joining on origin_date
    # alone, since the two feature sets can drop different rows to NaN and
    # so don't share an identical calendar by construction.
    # ------------------------------------------------------------------
    print("\n[6/6] Yield-only vs. 6-variable macro set (proposal Section 3.1 research question)...")
    macro_path = out_dir / "r3_patha_var_aic_forecasts.csv"
    if not macro_path.exists():
        print(f"  Skipped -- {macro_path} not found. Run `EDA_VAR_AIC _lag _order.py` first.")
        return

    macro = pd.read_csv(macro_path, parse_dates=["origin_date"])
    yield_only_cal = levels.index
    macro_cal = _core.load_levels().index

    yo = forecasts.rename(columns={"var_aic": "var_aic_yieldonly"})
    mc = macro.rename(columns={"var_aic": "var_aic_6var"})

    merged = merge_cross_pipeline(
        yo[["origin_date", "horizon", "actual", "naive", "var_aic_yieldonly"]],
        "var_aic_yieldonly", yield_only_cal,
        mc[["origin_date", "horizon", "actual", "naive", "var_aic_6var"]],
        "var_aic_6var", macro_cal,
    )

    comparison_rows = []
    for h in _core.HORIZONS:
        sub = merged[merged.horizon == h]
        row = pairwise_dm(
            sub["actual"].values, sub["var_aic_yieldonly"].values, sub["var_aic_6var"].values, h
        )
        verdict = plain_language_verdict(row, "yield-only VAR", "6-variable VAR")
        comparison_rows.append({"horizon_days": h, **row, "verdict": verdict})
        print(f"  h={h} (n={row['n_forecasts']}): {verdict}")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "r3_yieldonly_vs_macro_diebold_mariano.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_yieldonly_vs_macro_diebold_mariano.csv")


if __name__ == "__main__":
    main()
