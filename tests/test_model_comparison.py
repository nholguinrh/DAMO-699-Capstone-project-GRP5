"""
Unit tests for the pairwise Diebold-Mariano comparison module (Issue #50)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_comparison import (
    apply_clark_west_fdr,
    attach_target_date,
    clark_west_test,
    cumulative_var_level_interval,
    interval_coverage_summary,
    merge_cross_pipeline,
    pairwise_dm,
    plain_language_verdict,
    plain_language_verdict_cw,
    run_clark_west_battery,
)


def test_pairwise_dm_equally_noisy_predictions_not_significant():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=200)
    pred_a = actual + rng.normal(scale=0.5, size=200)
    pred_b = actual + rng.normal(scale=0.5, size=200)
    row = pairwise_dm(actual, pred_a, pred_b, h=1)
    assert not row["a_significantly_better_rmse"]
    assert not row["b_significantly_better_rmse"]


def test_pairwise_dm_a_much_better_is_detected():
    rng = np.random.default_rng(1)
    n = 500
    actual = rng.normal(size=n)
    pred_a = actual + rng.normal(scale=0.05, size=n)  # tight
    pred_b = actual + rng.normal(scale=2.0, size=n)   # loose
    row = pairwise_dm(actual, pred_a, pred_b, h=1)
    assert row["a_significantly_better_rmse"]
    assert row["a_significantly_better_mae"]
    assert row["dm_p_value_squared_loss"] < 0.05


def test_plain_language_verdict_both_agree():
    row = {
        "a_significantly_better_rmse": True, "b_significantly_better_rmse": False,
        "a_significantly_better_mae": True, "b_significantly_better_mae": False,
        "dm_p_value_squared_loss": 0.01, "dm_p_value_absolute_loss": 0.02,
    }
    v = plain_language_verdict(row, "arima_aic", "naive")
    assert "arima_aic significantly better than naive" in v


def test_plain_language_verdict_mixed_result():
    row = {
        "a_significantly_better_rmse": True, "b_significantly_better_rmse": False,
        "a_significantly_better_mae": False, "b_significantly_better_mae": True,
        "dm_p_value_squared_loss": 0.01, "dm_p_value_absolute_loss": 0.02,
    }
    v = plain_language_verdict(row, "arima_aic", "naive")
    assert "mixed result" in v


def test_plain_language_verdict_no_significance():
    row = {
        "a_significantly_better_rmse": False, "b_significantly_better_rmse": False,
        "a_significantly_better_mae": False, "b_significantly_better_mae": False,
        "dm_p_value_squared_loss": 0.5, "dm_p_value_absolute_loss": 0.5,
    }
    v = plain_language_verdict(row, "arima_aic", "naive")
    assert v == "no significant difference on either RMSE or MAE"


def _levels_and_calendar(dates, seed):
    """A synthetic target series + its own calendar, so `actual` at any date is
    fully determined by that calendar's own position arithmetic -- mirrors how
    the real level series work."""
    rng = np.random.default_rng(seed)
    levels = pd.Series(rng.normal(size=len(dates)), index=dates)
    return levels, pd.DatetimeIndex(dates)


def _forecasts_from_calendar(levels, calendar, origin_dates, horizon, pred_offset=0.1):
    pos = calendar.get_indexer(origin_dates)
    target_pos = pos + horizon
    return pd.DataFrame({
        "origin_date": origin_dates,
        "horizon": horizon,
        "actual": levels.to_numpy()[target_pos],
        "naive": levels.to_numpy()[pos],
        "pred": levels.to_numpy()[target_pos] + pred_offset,
    })


def test_attach_target_date_walks_forward_by_horizon():
    calendar = pd.bdate_range("2020-01-01", periods=20)
    df = pd.DataFrame({"origin_date": [calendar[5], calendar[10]], "horizon": [3, 2]})
    out = attach_target_date(df, calendar)
    assert list(out["target_date"]) == [calendar[8], calendar[12]]


def test_attach_target_date_out_of_range_is_nat():
    calendar = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"origin_date": [calendar[4]], "horizon": [10]})
    out = attach_target_date(df, calendar)
    assert out["target_date"].isna().all()


def test_merge_cross_pipeline_drops_rows_whose_real_target_dates_diverge():
    """
    Two pipelines sharing most origin_dates but built on different calendars
    (the right side is missing one business day mid-series, like
    build_gold_features() vs load_levels() skipping different gap days) --
    only origins whose h-step-ahead target genuinely lands on the same real
    date in both calendars should survive, even though every row's
    origin_date label matches on both sides.
    """
    dates_left = pd.bdate_range("2020-01-01", periods=30)
    dates_right = dates_left.delete(15)  # right calendar is missing one business day

    levels_left, cal_left = _levels_and_calendar(dates_left, seed=0)
    levels_right = pd.Series(levels_left.reindex(dates_right).to_numpy(), index=dates_right)
    cal_right = pd.DatetimeIndex(dates_right)

    shared_origins = dates_left[:12]  # all before the gap, so both sides can resolve horizon=5
    left = _forecasts_from_calendar(levels_left, cal_left, shared_origins, horizon=5)
    right = _forecasts_from_calendar(levels_right, cal_right, shared_origins, horizon=5)
    right = right.rename(columns={"pred": "pred_right"})
    left = left.rename(columns={"pred": "pred_left"})

    merged = merge_cross_pipeline(left, "pred_left", cal_left, right, "pred_right", cal_right)

    # Origins whose 5-step target crosses the missing day land on different real
    # dates in each calendar and must be dropped; origins entirely before the gap
    # (target_pos + 5 still < 15) must survive.
    still_before_gap = shared_origins[cal_left.get_indexer(shared_origins) + 5 < 15]
    assert set(merged["origin_date"]) == set(still_before_gap)
    assert len(merged) == len(still_before_gap)
    assert len(merged) < len(shared_origins)


def test_merge_cross_pipeline_raises_on_actual_mismatch_at_matching_target_date():
    """
    If two sides claim to target the same real calendar date but disagree on
    the target series' own value there, that's a data-quality bug (the target
    series differs between pipelines), not a calendar-alignment issue -- it
    must fail loudly, not silently vanish from n_forecasts.
    """
    dates = pd.bdate_range("2020-01-01", periods=10)
    calendar = pd.DatetimeIndex(dates)
    left = pd.DataFrame({
        "origin_date": dates, "horizon": 1,
        "actual": np.arange(10, dtype=float),
        "naive": np.zeros(10),
        "pred_left": np.arange(10, dtype=float) + 0.1,
    })
    right = left.copy().rename(columns={"pred_left": "pred_right"})
    right["actual"] = right["actual"] + 100  # same target_date, but a different value

    with pytest.raises(AssertionError, match="disagreeing"):
        merge_cross_pipeline(left, "pred_left", calendar, right, "pred_right", calendar)


def test_merge_cross_pipeline_rejects_equal_columns():
    dates = pd.bdate_range("2020-01-01", periods=5)
    calendar = pd.DatetimeIndex(dates)
    df = pd.DataFrame({
        "origin_date": dates, "horizon": 1,
        "actual": np.ones(5), "naive": np.ones(5), "x": np.ones(5),
    })
    with pytest.raises(ValueError, match="must differ"):
        merge_cross_pipeline(df, "x", calendar, df, "x", calendar)


def test_merge_cross_pipeline_rejects_reserved_column_name():
    dates = pd.bdate_range("2020-01-01", periods=5)
    calendar = pd.DatetimeIndex(dates)
    df = pd.DataFrame({
        "origin_date": dates, "horizon": 1,
        "actual": np.ones(5), "naive": np.ones(5),
    })
    with pytest.raises(ValueError, match="reserved"):
        merge_cross_pipeline(df, "actual", calendar, df, "pred", calendar)


def test_pairwise_dm_insufficient_sample_does_not_raise():
    actual = np.array([1.0, 2.0, 3.0])
    pred_a = np.array([1.1, 2.1, 3.1])
    pred_b = np.array([0.9, 1.9, 2.9])
    row = pairwise_dm(actual, pred_a, pred_b, h=20)  # n=3 <= h=20
    assert row["insufficient_sample"]
    assert row["dm_p_value_squared_loss"] is None
    assert not row["a_significantly_better_rmse"]
    v = plain_language_verdict(row, "a", "b")
    assert "insufficient sample" in v


def test_clark_west_internal_gap_preserves_calendar_lags():
    """
    Tests that series with leading, trailing, and interior NaNs correctly isolate
    valid evaluation periods and preserve calendar-time lag distances for HAC
    autocovariance estimation without throwing errors or splicing non-adjacent
    time points.

    The key econometric invariant tested here: the variance estimate from the
    gappy series must NOT be smaller than the estimate from the same series with
    the gap filled — if it were, the HAC denominator would be artificially
    deflating the variance and inflating |cw_stat|, producing false significance.
    """
    n = 50
    rng = np.random.default_rng(42)
    actual = np.linspace(1.0, 5.0, n)
    naive = actual + rng.normal(scale=0.1, size=n)
    pred = actual + rng.normal(scale=0.05, size=n)

    # Insert leading, trailing, and interior NaNs
    actual_gap = actual.copy()
    actual_gap[0:2] = np.nan   # leading NaNs
    actual_gap[20:25] = np.nan # interior gap
    actual_gap[48:] = np.nan   # trailing NaNs

    res_gap = clark_west_test(actual_gap, naive, pred, h=5)
    assert not res_gap["insufficient_sample"]
    assert res_gap["n_forecasts"] == 50 - 2 - 5 - 2  # 41 valid observations
    assert res_gap["se_f_stat"] > 0
    assert np.isfinite(res_gap["cw_stat"])
    assert 0.0 <= res_gap["cw_p_value"] <= 1.0

    # Compare against the contiguous (gap-free) series trimmed to the same span
    # The gappy variance should be >= the contiguous variance (no false deflation)
    res_full = clark_west_test(actual[2:48], naive[2:48], pred[2:48], h=5)
    assert res_gap["se_f_stat"] >= res_full["se_f_stat"] * 0.95, (
        f"Gappy se_f ({res_gap['se_f_stat']:.6f}) is suspiciously smaller than "
        f"contiguous se_f ({res_full['se_f_stat']:.6f}), suggesting HAC denominator deflation"
    )


# ---------------------------------------------------------------------------
# Clark-West (2007) Adjusted MSPE Tests (Issue #88)
# ---------------------------------------------------------------------------

def test_clark_west_nested_null_hypothesis_no_spurious_rejection():
    """
    Under H0 where Model 2 adds zero-mean noise to Model 1 (e.g. parameter
    estimation noise), Model 2's sample MSPE will be higher than Model 1's MSPE.
    Standard DM on squared errors would produce DM > 0 (falsely favoring Naive),
    while Clark-West correctly adjusts for the noise: CW stat is close to 0,
    p-value >= 0.05, and model_significantly_better is False.
    """
    rng = np.random.default_rng(42)
    n = 1000
    actual = rng.normal(scale=1.0, size=n)
    naive = actual + rng.normal(scale=0.5, size=n)
    # Model 2 estimates noise on top of Naive (nested null DGP)
    pred_noisy = naive + rng.normal(scale=0.1, size=n)

    res = clark_west_test(actual, naive, pred_noisy, h=1)
    assert not res["insufficient_sample"]
    assert res["mspe_model"] > res["mspe_naive"]  # sample MSPE of noisy model is strictly higher
    assert res["cw_adjustment"] > 0
    assert abs(res["cw_stat"]) < 1.645  # Not significant at alpha=0.05
    assert res["cw_p_value"] > 0.05
    assert not res["model_significantly_better"]


def test_clark_west_superior_alternative_is_detected():
    """
    When Model 2 has genuine forecasting skill (smaller error variance than Naive),
    Clark-West should yield a strong positive test statistic (CW > 1.645) and
    p < 0.05.
    """
    rng = np.random.default_rng(101)
    n = 500
    actual = rng.normal(scale=1.0, size=n)
    naive = actual + rng.normal(scale=1.0, size=n)  # noisy benchmark
    pred_smart = actual + rng.normal(scale=0.2, size=n)  # superior model

    res = clark_west_test(actual, naive, pred_smart, h=1)
    assert not res["insufficient_sample"]
    assert res["cw_stat"] > 5.0
    assert res["cw_p_value"] < 0.001
    assert res["model_significantly_better"]


def test_clark_west_hac_bartlett_weighting_multi_horizon():
    """
    Tests that multi-step horizons (h=5, h=20) correctly invoke Bartlett HAC
    weighting without raising exceptions or producing negative variances.
    """
    rng = np.random.default_rng(202)
    n = 600
    actual = np.cumsum(rng.normal(scale=0.1, size=n))
    naive = np.roll(actual, 1)
    naive[0] = actual[0]
    pred = actual + rng.normal(scale=0.15, size=n)

    for h in [1, 5, 20]:
        res = clark_west_test(actual, naive, pred, h=h)
        assert not res["insufficient_sample"]
        assert res["se_f_stat"] > 0
        assert np.isfinite(res["cw_stat"])
        assert 0.0 <= res["cw_p_value"] <= 1.0


def test_clark_west_insufficient_sample_returns_safe_dict():
    actual = np.array([1.0, 2.0])
    naive = np.array([1.1, 2.1])
    pred = np.array([1.2, 2.2])

    res = clark_west_test(actual, naive, pred, h=5)
    assert res["insufficient_sample"]
    assert res["cw_stat"] is None
    assert res["cw_p_value"] is None
    assert not res["model_significantly_better"]

    v = plain_language_verdict_cw(res, "test_model")
    assert "insufficient sample" in v


def test_plain_language_verdict_cw_scenarios():
    # Significant improvement
    row_sig = {
        "insufficient_sample": False,
        "n_forecasts": 750,
        "model_significantly_better": True,
        "cw_stat": 2.15,
        "cw_p_value": 0.0158,
        "mspe_naive": 0.015,
        "mspe_model": 0.014,
        "mspe_model_adj": 0.013,
    }
    v_sig = plain_language_verdict_cw(row_sig, "VECM")
    assert "significantly outperforms Naïve" in v_sig

    # Not significant, positive CW
    row_pos = {
        "insufficient_sample": False,
        "n_forecasts": 750,
        "model_significantly_better": False,
        "cw_stat": 0.85,
        "cw_p_value": 0.1977,
        "mspe_naive": 0.015,
        "mspe_model": 0.016,
        "mspe_model_adj": 0.0145,
    }
    v_pos = plain_language_verdict_cw(row_pos, "VAR-AIC")
    assert "fails to reject equal accuracy" in v_pos

    # Negative CW
    row_neg = {
        "insufficient_sample": False,
        "n_forecasts": 750,
        "model_significantly_better": False,
        "cw_stat": -1.2,
        "cw_p_value": 0.8849,
        "mspe_naive": 0.015,
        "mspe_model": 0.018,
        "mspe_model_adj": 0.016,
    }
    v_neg = plain_language_verdict_cw(row_neg, "ARIMA-AIC")
    assert "does not beat Naïve" in v_neg


def test_apply_clark_west_fdr_global_and_horizon():
    """
    Tests both global and horizon-stratified Benjamini-Hochberg FDR adjustments.
    """
    rows = [
        {"model": "M1", "horizon": 1, "cw_stat": 2.5, "cw_p_value": 0.0062, "insufficient_sample": False, "n_forecasts": 500, "mspe_naive": 0.01, "mspe_model": 0.009, "mspe_model_adj": 0.008},
        {"model": "M2", "horizon": 1, "cw_stat": 0.5, "cw_p_value": 0.3085, "insufficient_sample": False, "n_forecasts": 500, "mspe_naive": 0.01, "mspe_model": 0.011, "mspe_model_adj": 0.010},
        {"model": "M1", "horizon": 5, "cw_stat": 1.8, "cw_p_value": 0.0359, "insufficient_sample": False, "n_forecasts": 500, "mspe_naive": 0.02, "mspe_model": 0.019, "mspe_model_adj": 0.018},
        {"model": "M2", "horizon": 5, "cw_stat": -0.2, "cw_p_value": 0.5793, "insufficient_sample": False, "n_forecasts": 500, "mspe_naive": 0.02, "mspe_model": 0.022, "mspe_model_adj": 0.021},
    ]
    df = pd.DataFrame(rows)
    df["model_significantly_better"] = df["cw_p_value"] < 0.05
    df["verdict_raw"] = df.apply(lambda r: plain_language_verdict_cw(r.to_dict(), r["model"]), axis=1)

    out = apply_clark_west_fdr(df, alpha=0.05)

    expected_cols = {
        "cw_p_adj_global", "cw_p_adj_horizon",
        "model_significantly_better_fdr_global", "model_significantly_better_fdr_horizon",
        "verdict_fdr_global", "verdict_fdr_horizon",
    }
    assert expected_cols.issubset(out.columns)
    # Adjusted p-values must always be >= raw p-values
    assert (out["cw_p_adj_global"] >= out["cw_p_value"] - 1e-6).all()
    assert (out["cw_p_adj_horizon"] >= out["cw_p_value"] - 1e-6).all()


def test_run_clark_west_battery_smoke():
    """
    Integration smoke test executing the full Clark-West 12-test battery
    against the actual outputs/ forecast files.
    """
    primary_df, sensitivity_df = run_clark_west_battery()

    # Primary battery is strictly 4 models x 3 horizons = 12 rows (m=12) with canonical names
    assert len(primary_df) == 12
    assert set(primary_df["horizon"]) == {1, 5, 20}
    assert set(primary_df["model"]) == {"ARIMA-AIC", "VAR-AIC", "VECM (6-var)", "LSTM (Tuned)"}

    # Sensitivity battery contains 2 BIC models (6 rows) or 3 models (9 rows with XGBoost Experimental)
    expected_sens = {"ARIMA-BIC", "VAR-BIC"}
    if "XGBoost (Experimental)" in set(sensitivity_df["model"]):
        expected_sens.add("XGBoost (Experimental)")
        assert len(sensitivity_df) == 9
    else:
        assert len(sensitivity_df) == 6
    assert set(sensitivity_df["model"]) == expected_sens

    # All forecasts should have ~745 to 750 sample size
    assert (primary_df["n_forecasts"] >= 745).all()
    assert (primary_df["n_forecasts"] <= 750).all()

    # Verify no NaN test statistics
    assert primary_df["cw_stat"].notna().all()
    assert primary_df["cw_p_value"].notna().all()
    assert primary_df["cw_p_adj_global"].notna().all()
    assert primary_df["cw_p_adj_horizon"].notna().all()


def test_clark_west_primary_battery_strictly_twelve_hypotheses():
    """Verify primary battery maintains exactly 4 canonical models x 3 horizons = 12 tests."""
    primary_df, sensitivity_df = run_clark_west_battery()

    assert len(primary_df) == 12, f"Primary battery must contain exactly 12 hypotheses, got {len(primary_df)}"
    assert set(primary_df["model"]) == {
        "ARIMA-AIC",
        "VAR-AIC",
        "VECM (6-var)",
        "LSTM (Tuned)",
    }
    # XGBoost must be strictly quarantined to sensitivity_df
    if "XGBoost (Experimental)" in set(sensitivity_df["model"]):
        assert len(sensitivity_df) == 9
    else:
        assert len(sensitivity_df) == 6


def test_regime_segmentation_hac_power_gating():
    """Assert Clark-West hypothesis claims are suppressed when sample size fails HAC regularity."""
    from model_comparison import evaluate_regime_segmentation, REGIMES

    regime_df = evaluate_regime_segmentation(save=False)
    assert len(regime_df) > 0

    expected_regimes = {r[0] for r in REGIMES}
    assert set(regime_df["regime"]) == expected_regimes

    # Assert small-sample gating rule: N < 5*h must suppress CW test claims
    underpowered = regime_df[regime_df["insufficient_sample"]]
    assert len(underpowered) > 0, "Expected underpowered regimes (e.g. Regime 2 at h=20)"

    for _, row in underpowered.iterrows():
        assert pd.isna(row["cw_stat"]) or row["cw_stat"] is None
        assert pd.isna(row["cw_p_value"]) or row["cw_p_value"] is None
        assert row["model_significantly_better"] is False


def test_common_sample_metrics_paired_baseline_parity():
    """Assert relative improvement metrics are computed over strictly identical observation pairs."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dashboard"))
    from data_loader import build_common_sample_metrics, build_common_forecast_dataset

    metrics_df = build_common_sample_metrics()
    forecast_df = build_common_forecast_dataset()

    for _, row in metrics_df.iterrows():
        model_name = row["model"]
        horizon = row["horizon"]
        if model_name == "Naïve Random Walk":
            continue

        # Map display name to forecast column
        col_map = {"ARIMA": "arima_aic", "VAR": "var_aic", "VECM": "vecm", "LSTM": "lstm", "XGBoost": "xgboost"}
        col = col_map.get(model_name)
        if col is None or col not in forecast_df.columns:
            continue

        sub = forecast_df[forecast_df["horizon"] == horizon].dropna(subset=["actual", "naive", col])
        expected_naive_rmse = np.sqrt(np.mean((sub["actual"] - sub["naive"]) ** 2))
        expected_model_rmse = np.sqrt(np.mean((sub["actual"] - sub[col]) ** 2))
        expected_imp = (expected_naive_rmse - expected_model_rmse) / expected_naive_rmse * 100.0

        assert np.isclose(row["naive_rmse"], expected_naive_rmse, atol=1e-5), \
            f"Naive RMSE for {model_name} at h={horizon} must match paired sample"
        assert np.isclose(row["rmse_improvement_pct"], expected_imp, atol=1e-3), \
            f"Improvement pct for {model_name} at h={horizon} must match paired calculation"


def test_forecast_error_distribution_missing_column_resilience():
    """Assert error distribution visualization renders without KeyError when optional models are missing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dashboard"))
    from charts import create_forecast_error_distribution

    # Synthetic forecast dataframe missing 'xgboost'
    df = pd.DataFrame({
        "horizon": [1, 1, 1],
        "origin_date": pd.date_range("2024-01-01", periods=3),
        "actual": [1.0, 1.1, 1.2],
        "naive": [1.0, 1.0, 1.1],
        "arima_aic": [1.01, 1.09, 1.21],
        "var_aic": [1.00, 1.11, 1.20],
        "vecm": [1.02, 1.10, 1.19],
        "lstm": [1.01, 1.08, 1.22],
    })

    fig_box = create_forecast_error_distribution(df, horizon=1, chart_type="Box")
    fig_violin = create_forecast_error_distribution(df, horizon=1, chart_type="Violin")
    assert fig_box is not None
    assert fig_violin is not None


# ---------------------------------------------------------------------------
# Prediction intervals (issue #103)
# ---------------------------------------------------------------------------

class _FakeVarFit:
    """Duck-types the subset of VARResultsWrapper cumulative_var_level_interval()
    needs, so its math can be checked against a hand-derived closed form without
    fitting a real (sampling-noisy) VAR."""

    def __init__(self, phi: np.ndarray, sigma_u: np.ndarray):
        self._phi = phi  # (maxn+1, k, k), phi[0] must be I_k
        self.sigma_u = sigma_u

    def ma_rep(self, maxn):
        return self._phi[: maxn + 1]


def test_cumulative_var_level_interval_white_noise_scales_linearly_with_h():
    """For an undifferenced-further, purely white-noise VAR (Phi_i = 0 for i > 0,
    i.e. the differenced series has no own dynamics), the h-step cumulative level
    variance must be exactly h * sigma_u -- summing h i.i.d. shocks. This is the
    textbook random-walk variance identity and the simplest exact check of the
    Psi-cumulative formula."""
    max_h = 5
    sigma_u = np.array([[2.0]])
    phi = np.zeros((max_h, 1, 1))
    phi[0] = np.eye(1)  # Phi_0 = I; Phi_1..Phi_{max_h-1} = 0

    fake = _FakeVarFit(phi, sigma_u)
    half_widths = cumulative_var_level_interval(fake, target_idx=0, max_h=max_h, alphas=(0.05,))

    z = 1.9599639845400545  # scipy.stats.norm.ppf(0.975)
    implied_var = (half_widths[0.05] / z) ** 2
    expected_var = sigma_u[0, 0] * np.arange(1, max_h + 1)  # h * sigma_u
    assert np.allclose(implied_var, expected_var, rtol=1e-6)


def test_cumulative_var_level_interval_matches_hand_derivation():
    """Non-trivial single lag (Phi_1 = 0.5, Phi_i>=2 = 0): Sigma_level_h must equal
    the closed-form sum_{m=0}^{h-1} Psi_m^2 * sigma_u derived in the function's
    docstring, computed independently here rather than by calling the code under
    test a second time."""
    max_h = 5
    sigma_u = np.array([[1.0]])
    phi = np.zeros((max_h, 1, 1))
    phi[0] = np.eye(1)
    phi[1] = np.array([[0.5]])
    # phi[2:] left at 0

    fake = _FakeVarFit(phi, sigma_u)
    half_widths = cumulative_var_level_interval(fake, target_idx=0, max_h=max_h, alphas=(0.05,))
    z = 1.9599639845400545
    implied_var = (half_widths[0.05] / z) ** 2

    psi = np.cumsum(phi[:, 0, 0])  # Psi_0..Psi_4 = 1, 1.5, 1.5, 1.5, 1.5
    expected_var = np.cumsum(psi ** 2) * sigma_u[0, 0]
    assert np.allclose(implied_var, expected_var, rtol=1e-6)
    assert np.allclose(expected_var, [1.0, 3.25, 5.5, 7.75, 10.0])


def test_cumulative_var_level_interval_widens_with_horizon_and_alpha():
    """Sanity properties that must hold regardless of the exact VAR dynamics:
    the band cannot shrink as h grows (variance accumulates), and the 95% band
    must never be narrower than the 90% band at the same horizon."""
    rng = np.random.default_rng(0)
    max_h = 6
    k = 2
    phi = np.zeros((max_h, k, k))
    phi[0] = np.eye(k)
    for i in range(1, max_h):
        phi[i] = 0.3 * rng.standard_normal((k, k)) * (0.6 ** i)  # decaying MA coefs
    a = rng.standard_normal((k, k)) * 0.1
    sigma_u = a @ a.T + np.eye(k) * 0.5  # guaranteed PSD

    fake = _FakeVarFit(phi, sigma_u)
    half_widths = cumulative_var_level_interval(fake, target_idx=0, max_h=max_h, alphas=(0.10, 0.05))

    hw90, hw95 = half_widths[0.10], half_widths[0.05]
    assert np.all(np.diff(hw90) >= -1e-10)
    assert np.all(np.diff(hw95) >= -1e-10)
    assert np.all(hw95 >= hw90 - 1e-10)


def test_interval_coverage_summary_basic():
    """A hand-constructed 2-horizon frame where coverage and width are known
    by inspection -- h=1: both origins covered by the 90% band, only the
    second by the (deliberately narrower, off-center) 95% band; h=5: neither
    origin covered by either band."""
    df = pd.DataFrame({
        "horizon": [1, 1, 5, 5],
        "actual": [1.0, 2.0, 10.0, 20.0],
        "lower_90": [0.5, 1.5, 100.0, 100.0],
        "upper_90": [1.5, 2.5, 101.0, 101.0],
        "lower_95": [1.05, 1.5, 100.0, 100.0],  # excludes actual=1.0
        "upper_95": [1.15, 2.5, 101.0, 101.0],
    })

    summary = interval_coverage_summary(
        df, actual_col="actual",
        lower_cols={90.0: "lower_90", 95.0: "lower_95"},
        upper_cols={90.0: "upper_90", 95.0: "upper_95"},
    )

    h1 = summary[summary["horizon"] == 1].iloc[0]
    assert h1["empirical_coverage_90"] == 100.0
    assert h1["empirical_coverage_95"] == 50.0  # only actual=2.0 falls inside [1.5, 2.5]
    assert np.isclose(h1["mean_interval_width_90"], 1.0)  # (1.5-0.5 + 2.5-1.5) / 2
    assert np.isclose(h1["mean_interval_width_95"], 0.55)  # (1.15-1.05 + 2.5-1.5) / 2

    h5 = summary[summary["horizon"] == 5].iloc[0]
    assert h5["empirical_coverage_90"] == 0.0
    assert h5["empirical_coverage_95"] == 0.0


def test_interval_coverage_summary_rejects_mismatched_level_keys():
    df = pd.DataFrame({"horizon": [1], "actual": [1.0], "lo": [0.5], "hi": [1.5]})
    with pytest.raises(ValueError, match="same nominal-level keys"):
        interval_coverage_summary(
            df, actual_col="actual",
            lower_cols={90.0: "lo"}, upper_cols={95.0: "hi"},
        )
