"""
Impulse Response Functions, FEVD & Granger Causality (Issue #48)
DAMO-699 Capstone, Group 5

Purpose:
  proposal/sections/05_analytical_approach.md Section 5.3 commits to Impulse
  Response Functions (IRF), Forecast Error Variance Decomposition (FEVD), and
  Granger causality tests on the multivariate model, none of which had a
  TIMELINE.md task line until this issue. Runs on the converged primary VECM
  from issue #47: the 6-variable system (yield_spread_10y_2y, overnight_rate,
  us_treasury_10y, fed_funds_rate, cpi_yoy, usdcad), restricted-constant
  deterministic spec, cointegrating rank selected by the 95% trace test --
  the same system #50's cross-model DM scoring treats as primary.

Depends on:
  #47 (Johansen cointegration test + conditional VECM) -- CLOSED.

Methodology:
  - IRF: orthogonalized (Cholesky) one-SD shock, variables ordered per
    FEATURE_SET_6VAR (johansen_vecm.py) -- the same order #47's beta/alpha
    outputs already use, so a variable earlier in that list is treated as
    contemporaneously exogenous to variables later in the list. Concretely:
    yield_spread_10y_2y precedes overnight_rate, so a one-SD overnight_rate
    shock's contemporaneous (h=0) effect on the spread is mechanically zero
    by construction -- the transmission this test is actually measuring
    shows up from h=1 onward, through the VECM's lagged dynamics and error-
    correction term. This ordering choice (not re-derived here) is a known
    open question -- see issue #48's follow-on hardening work.
  - FEVD: statsmodels' VECMResults / IRAnalysis does not implement
    fevd_table() for VECM-derived IRFs (raises NotImplementedError), and
    VARResults.fevd() isn't exposed on VECMResults either. Replicated
    manually from the cumulative sum of squared orthogonalized IRF
    coefficients, normalized per horizon -- the same computation
    statsmodels' own VAR FEVD performs internally. Verified bit-for-bit
    (max abs diff ~1e-16) against statsmodels.tsa.vector_ar.var_model.FEVD
    on a VAR fit of the same data.
  - Granger causality: VECMResults.test_granger_causality(), which runs the
    F-test on the model's level-VAR representation (var_rep) -- valid for
    cointegrated I(1) systems, unlike a naive Granger F-test on raw
    non-stationary levels (Toda & Phillips, 1993).

Run:
  python src/irf_fevd_granger.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import PROJECT_ROOT  # noqa: E402
from johansen_vecm import (  # noqa: E402
    FEATURE_SET_6VAR,
    HORIZONS,
    confirm_i1,
    fit_and_summarize_vecm,
    load_gold_levels,
    run_johansen_all_specs,
    select_lag_levels,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "Bond yields" and "policy rate / exchange rate" per issue #48's checklist,
# mapped onto the 6-variable system's actual columns.
BOND_YIELDS = ["yield_spread_10y_2y", "us_treasury_10y"]
POLICY_AND_FX_DRIVERS = ["overnight_rate", "fed_funds_rate", "usdcad"]

IRF_SHOCK_VAR = "overnight_rate"
IRF_PERIODS = max(HORIZONS)   # 20 trading days, matching the project's shared horizon set
SIGNIF = 0.05


# ---------------------------------------------------------------------------
# 1. Refit the primary 6-variable VECM (issue #47's converged model)
# ---------------------------------------------------------------------------

def fit_primary_vecm(enforce_i1: bool = True) -> dict:
    """
    Refit the primary 6-variable VECM issue #47 converged on (restricted
    constant, cointegrating rank via the 95% trace test). Refits from the
    same steps run_pipeline() (johansen_vecm.py) uses for load/ADF/lag/rank
    selection, rather than hardcoding #47's k_ar_diff=10 / rank=1 as
    disconnected magic numbers -- if the Gold-layer data changes upstream,
    this will pick up the change instead of silently going stale.
    """
    levels = load_gold_levels(FEATURE_SET_6VAR)
    confirm_i1(levels, enforce=enforce_i1)

    aic_lag_levels, _ = select_lag_levels(levels)
    k_ar_diff = max(1, aic_lag_levels - 1)

    _, primary_rank = run_johansen_all_specs(levels, k_ar_diff)
    if primary_rank == 0:
        raise RuntimeError(
            "6-variable system has no cointegrating vectors (r=0) under the "
            "primary specification -- IRF/FEVD/Granger causality all require "
            "an estimated VECM. This contradicts issue #47's merged result; "
            "re-check the Gold-layer data before proceeding."
        )

    det_spec = "ci"  # restricted constant, matching #47's primary specification
    vecm_fit = fit_and_summarize_vecm(levels, k_ar_diff, primary_rank, det_spec)
    return {
        "levels": levels,
        "k_ar_diff": k_ar_diff,
        "coint_rank": primary_rank,
        "vecm_fit": vecm_fit,
    }


# ---------------------------------------------------------------------------
# 2. Impulse Response Functions
# ---------------------------------------------------------------------------

def compute_irf(vecm_fit, periods: int = IRF_PERIODS):
    """Orthogonalized (Cholesky) IRF object, ordered per FEATURE_SET_6VAR."""
    return vecm_fit.irf(periods=periods)


def one_sd_shock_response(
    irf, levels: pd.DataFrame, shock_var: str = IRF_SHOCK_VAR,
) -> pd.DataFrame:
    """
    Tidy long-format DataFrame of every variable's response to a one-SD
    orthogonalized shock in `shock_var`, at every horizon the IRF was
    computed for (0..periods).
    """
    var_names = list(levels.columns)
    shock_idx = var_names.index(shock_var)
    orth = irf.orth_irfs  # shape: (periods + 1, n_response_vars, n_shock_vars)

    rows = []
    for h in range(orth.shape[0]):
        for i, response_var in enumerate(var_names):
            rows.append({
                "horizon_days": h,
                "shock_variable": shock_var,
                "response_variable": response_var,
                "response": float(orth[h, i, shock_idx]),
                "cumulative_response": float(orth[:h + 1, i, shock_idx].sum()),
            })
    return pd.DataFrame(rows)


def summarize_irf_plain_language(irf_df: pd.DataFrame) -> list[str]:
    """
    One plain-language sentence per response variable: direction, peak
    magnitude/horizon, and whether the response has decayed back toward zero
    by the end of the window (mean-reverting) or stayed persistently
    displaced.
    """
    lines = []
    for response_var, g in irf_df.groupby("response_variable", sort=False):
        g = g.sort_values("horizon_days")
        peak_row = g.loc[g["response"].abs().idxmax()]
        peak, final = float(peak_row["response"]), float(g.iloc[-1]["response"])
        direction = "rises" if peak > 0 else "falls"
        decayed = abs(peak) < 1e-9 or abs(final) < 0.25 * abs(peak)
        persistence = "decays back toward zero" if decayed else "remains persistently displaced"
        lines.append(
            f"  {response_var}: {direction} to a peak of {peak:+.4f} at day "
            f"{int(peak_row['horizon_days'])}, then {persistence} "
            f"(day-{int(g['horizon_days'].max())} response = {final:+.4f})."
        )
    return lines


# ---------------------------------------------------------------------------
# 3. Forecast Error Variance Decomposition
# ---------------------------------------------------------------------------

def compute_fevd(irf, levels: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast error variance decomposition. See module docstring for why this
    is computed manually rather than via a fevd_table()/fevd() call.
    """
    var_names = list(levels.columns)
    orth = irf.orth_irfs                          # (periods+1, response, shock)
    cum_sq = (orth ** 2).cumsum(axis=0)
    totals = cum_sq.sum(axis=2, keepdims=True)
    proportions = cum_sq / totals

    rows = []
    for h in range(proportions.shape[0]):
        for i, response_var in enumerate(var_names):
            for j, shock_var in enumerate(var_names):
                rows.append({
                    "horizon_days": h,
                    "response_variable": response_var,
                    "shock_variable": shock_var,
                    "proportion": float(proportions[h, i, j]),
                })
    return pd.DataFrame(rows)


def summarize_fevd_plain_language(
    fevd_df: pd.DataFrame, response_vars: list[str], horizon: int,
) -> list[str]:
    """Top variance contributor for each response variable at one horizon."""
    lines = []
    for response_var in response_vars:
        g = fevd_df[
            (fevd_df["response_variable"] == response_var)
            & (fevd_df["horizon_days"] == horizon)
        ].sort_values("proportion", ascending=False)
        top = g.iloc[0]
        own_share = float(g.loc[g["shock_variable"] == response_var, "proportion"].iloc[0])
        note = "" if top["shock_variable"] == response_var else f" (own shocks explain {own_share:.1%})"
        lines.append(
            f"  {response_var} at day {horizon}: {top['shock_variable']} explains "
            f"{top['proportion']:.1%} of forecast error variance{note}."
        )
    return lines


# ---------------------------------------------------------------------------
# 4. Granger causality
# ---------------------------------------------------------------------------

def run_granger_causality(
    vecm_fit,
    causing_vars: list[str] = POLICY_AND_FX_DRIVERS,
    caused_vars: list[str] = BOND_YIELDS,
    signif: float = SIGNIF,
) -> pd.DataFrame:
    """
    F-test Granger causality for each (causing -> caused) pair via
    VECMResults.test_granger_causality() -- see module docstring for why
    this (not a raw-levels Granger test) is the valid test here.
    """
    rows = []
    for caused in caused_vars:
        for causing in causing_vars:
            r = vecm_fit.test_granger_causality(caused=caused, causing=causing, signif=signif)
            df_num, df_denom = r.df
            reject = bool(r.pvalue < signif)
            rows.append({
                "causing": causing,
                "caused": caused,
                "test_statistic": float(r.test_statistic),
                "crit_value": float(r.crit_value),
                "df_num": df_num,
                "df_denom": df_denom,
                "p_value": float(r.pvalue),
                "reject_h0": reject,
                "plain_language": (
                    f"{causing} Granger-causes {caused} (p={r.pvalue:.4f} < {signif})"
                    if reject else
                    f"No evidence {causing} Granger-causes {caused} (p={r.pvalue:.4f} >= {signif})"
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  IRF / FEVD / Granger Causality -- 6-variable primary VECM system")
    print("=" * 70)

    print("\n[1/4] Refitting primary VECM (issue #47's converged model)...")
    fit = fit_primary_vecm()
    levels, vecm_fit = fit["levels"], fit["vecm_fit"]
    print(f"  k_ar_diff={fit['k_ar_diff']}, coint_rank={fit['coint_rank']}")

    print(f"\n[2/4] Impulse response: one-SD shock to {IRF_SHOCK_VAR} "
          f"({IRF_PERIODS}-day horizon)...")
    irf = compute_irf(vecm_fit, periods=IRF_PERIODS)
    irf_df = one_sd_shock_response(irf, levels, shock_var=IRF_SHOCK_VAR)
    irf_df.to_csv(out_dir / "irf_overnight_rate_shock.csv", index=False)
    print("  -> irf_overnight_rate_shock.csv")
    for line in summarize_irf_plain_language(irf_df):
        print(line)

    print("\n[3/4] Forecast error variance decomposition for bond yields...")
    fevd_df = compute_fevd(irf, levels)
    fevd_df.to_csv(out_dir / "fevd_results.csv", index=False)
    print("  -> fevd_results.csv")
    for h in HORIZONS:
        print(f"  Horizon = {h} day(s):")
        for line in summarize_fevd_plain_language(fevd_df, BOND_YIELDS, h):
            print(" " + line)

    print("\n[4/4] Granger causality: policy rate / exchange rate -> bond yields...")
    granger_df = run_granger_causality(vecm_fit)
    granger_df.to_csv(out_dir / "granger_causality_results.csv", index=False)
    print("  -> granger_causality_results.csv")
    for _, row in granger_df.iterrows():
        print(f"  {row['plain_language']}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    n_sig = int(granger_df["reject_h0"].sum())
    print(f"  Granger causality: {n_sig}/{len(granger_df)} pairs significant at {SIGNIF:.0%}.")
    print("  Outputs: irf_overnight_rate_shock.csv, fevd_results.csv, "
          "granger_causality_results.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
