"""
Round 3 -- Yield-only vs. macro-augmented VAR (issue #70)
DAMO-699 Capstone, Group 5

Scope of this script (per issue #70's checklist):
  - Estimate an AIC-lag VAR on yield-curve-only variables (no macro transmission
    variables), evaluated with the same expanding-window/DM protocol as the existing
    6-variable VAR-AIC baseline (#28).
  - Compare yield-only vs. a macro-augmented VAR via pairwise DM test at each
    horizon, to actually answer proposal Section 3.1's research question -- "does
    macro add predictive value over yield-only dynamics" -- which no model had ever
    tested before this issue (every baseline shipped with the full 6-variable set
    from the start).

Yield-only feature set: yield_spread_10y_2y (target), yield_2y, yield_5y -- the
Canadian yield curve's own levels. Deviates from issue #70's own suggested example
set (`yield_spread_10y_2y, yield_2y, yield_10y`): that trio is exactly collinear --
`yield_spread_10y_2y` is defined as `yield_10y - yield_2y`, the same
exact-linear-combination problem flagged in issue #67 -- and produces a singular
covariance matrix (`select_order()` raised `LinAlgError: leading minor is not
positive definite` when tried). `yield_5y` swapped in for `yield_10y` instead: not
algebraically determined by the other two, so the system stays full rank, while
still giving the VAR three genuinely distinct points on the curve (short end, belly,
slope) rather than silently collapsing to two.

Aug 24 fix (PR #82 review, nholguinrh): the comparison arm originally reused the
existing 6-variable VAR-AIC baseline (#28) as-is -- `spread, overnight_rate,
us_treasury_10y, fed_funds_rate, cpi_yoy, usdcad` -- which has *no* curve-level
variables in it at all, not even yield_2y/yield_5y. That made the DM result
uninterpretable for Section 3.1's question: a loss there could be because the macro
variables genuinely hurt, or simply because that arm is missing yield_2y/yield_5y
and only has the spread's own lags to work with, independent of macro -- two
different variable systems changing in two ways at once, not an isolated macro
effect. Fixed by building a proper nested comparison instead: MACRO_AUGMENTED_
FEATURE_SET keeps the same three curve variables as YIELD_ONLY_FEATURE_SET and adds
only the three macro transmission variables proposal Section 3.1 actually names
(overnight rate, USD/CAD, CPI) -- `us_treasury_10y`/`fed_funds_rate` (US rates)
stay excluded from both arms, same rationale as before. This arm is a fresh AIC-lag
VAR built by run_pipeline() below, not #28's output -- #28's feature set doesn't
nest with either arm here, so it's no longer used as a comparison point in this
script. `cpi_yoy` is treated as fully endogenous here, matching this branch's
current EDA_VAR_AIC baseline -- issue #72's fix (treating it as exogenous) is on a
separate, not-yet-merged branch/PR (#83); worth reapplying here once that lands.

Reuses `EDA_VAR_AIC _lag _order.py`'s to_differenced()/confirm_stationary()/
select_aic_lag()/evaluate()/dm_report() as-is (all parameterized on the levels/diffed
frame passed in, not hardcoded to any one FEATURE_SET) rather than copy-pasting the
same VAR-fitting/evaluation logic a third time -- imported via importlib since the
filename contains spaces, same pattern as `model_comparison.py`'s core_calendar().

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
MACRO_AUGMENTED_FEATURE_SET = YIELD_ONLY_FEATURE_SET + ["overnight_rate", "usdcad", "cpi_yoy"]


def load_levels(feature_set: list) -> pd.DataFrame:
    gold = pd.read_csv(PROCESSED_DIR / "gold_features.csv", parse_dates=["date"])
    return gold.set_index("date").sort_index()[feature_set].dropna()


def run_pipeline(feature_set: list, label: str):
    """
    Shared AIC-lag VAR pipeline (load -> diff -> confirm stationary -> select lag
    -> evaluate -> DM vs naive), run once per feature set. Used identically for the
    yield-only and macro-augmented arms below so the two runs differ only in
    `feature_set` -- the nested comparison the Aug 24 PR #82 review asked for.
    """
    print(f"\n--- {label}: {feature_set} ---")
    levels = load_levels(feature_set)
    print(f"  {levels.shape}, {levels.index.min().date()} -> {levels.index.max().date()}")

    diffed = _core.to_differenced(levels)
    _core.confirm_stationary(diffed)

    aic_lag = _core.select_aic_lag(diffed)

    results, raw, forecasts = _core.evaluate(levels, diffed, aic_lag)
    print("\n" + results.to_string(index=False))

    dm_results = _core.dm_report(raw)
    print("\n" + dm_results.to_string(index=False))

    return levels, results, dm_results, forecasts


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Round 3 -- Yield-only vs. macro-augmented VAR (issue #70)")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("\n[1/3] Yield-only arm...")
    yo_levels, yo_results, yo_dm, yo_forecasts = run_pipeline(YIELD_ONLY_FEATURE_SET, "Yield-only")
    yo_results.to_csv(out_dir / "r3_yieldonly_rmse_mae_vs_naive.csv", index=False)
    yo_dm.to_csv(out_dir / "r3_yieldonly_diebold_mariano.csv", index=False)
    yo_forecasts.to_csv(out_dir / "r3_yieldonly_var_aic_forecasts.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_yieldonly_rmse_mae_vs_naive.csv, "
          f"{out_dir}/r3_yieldonly_diebold_mariano.csv, "
          f"{out_dir}/r3_yieldonly_var_aic_forecasts.csv")

    print("\n[2/3] Macro-augmented arm (same curve variables, + the 3 macro transmission "
          "variables proposal Section 3.1 names: overnight rate, USD/CAD, CPI)...")
    mc_levels, mc_results, mc_dm, mc_forecasts = run_pipeline(
        MACRO_AUGMENTED_FEATURE_SET, "Macro-augmented"
    )
    mc_results.to_csv(out_dir / "r3_macroaug_rmse_mae_vs_naive.csv", index=False)
    mc_dm.to_csv(out_dir / "r3_macroaug_diebold_mariano.csv", index=False)
    mc_forecasts.to_csv(out_dir / "r3_macroaug_var_aic_forecasts.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_macroaug_rmse_mae_vs_naive.csv, "
          f"{out_dir}/r3_macroaug_diebold_mariano.csv, "
          f"{out_dir}/r3_macroaug_var_aic_forecasts.csv")

    # ------------------------------------------------------------------
    # [3/3] The actual research question (proposal Section 3.1, issue #70's
    # DoD bullet 2): does adding the 3 named macro transmission variables add
    # predictive value over yield-only dynamics? Nested DM comparison (PR #82
    # review) -- both arms share the same 3 curve variables, differing only by
    # whether the 3 macro variables are added, so a loss here can actually be
    # attributed to the macro variables rather than to a confounded change in
    # which curve information each side has. Uses merge_cross_pipeline()
    # (issue #63) rather than joining on origin_date alone, since the two
    # feature sets can still drop different rows to NaN and so don't share an
    # identical calendar by construction.
    # ------------------------------------------------------------------
    print("\n[3/3] Yield-only vs. macro-augmented (proposal Section 3.1 research question)...")
    yo = yo_forecasts.rename(columns={"var_aic": "var_aic_yieldonly"})
    mc = mc_forecasts.rename(columns={"var_aic": "var_aic_macroaug"})

    merged = merge_cross_pipeline(
        yo[["origin_date", "horizon", "actual", "naive", "var_aic_yieldonly"]],
        "var_aic_yieldonly", yo_levels.index,
        mc[["origin_date", "horizon", "actual", "naive", "var_aic_macroaug"]],
        "var_aic_macroaug", mc_levels.index,
    )

    comparison_rows = []
    for h in _core.HORIZONS:
        sub = merged[merged.horizon == h]
        row = pairwise_dm(
            sub["actual"].values, sub["var_aic_yieldonly"].values, sub["var_aic_macroaug"].values, h
        )
        verdict = plain_language_verdict(row, "yield-only VAR", "macro-augmented VAR")
        comparison_rows.append({"horizon_days": h, **row, "verdict": verdict})
        print(f"  h={h} (n={row['n_forecasts']}): {verdict}")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "r3_yieldonly_vs_macroaug_diebold_mariano.csv", index=False)
    print(f"\nSaved: {out_dir}/r3_yieldonly_vs_macroaug_diebold_mariano.csv")


if __name__ == "__main__":
    main()
