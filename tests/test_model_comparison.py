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
    apply_multiplicity_correction,
    attach_target_date,
    merge_cross_pipeline,
    pairwise_dm,
    plain_language_verdict,
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


# ---------------------------------------------------------------------------
# apply_multiplicity_correction() (issue #69)
# ---------------------------------------------------------------------------

def _synthetic_dm_all() -> pd.DataFrame:
    """
    A small synthetic table shaped like dm_pairwise_comparison.ipynb's `dm_all`:
    one clearly-significant pair, several borderline/non-significant pairs (the
    kind BH is expected to knock back out), and one insufficient-sample row.
    """
    rows = [
        # Clearly significant: tiny p-value, should usually survive BH too.
        {
            "horizon": 1, "model_a": "arima_aic", "model_b": "naive",
            "n_forecasts": 500, "insufficient_sample": False,
            "dm_stat_squared_loss": -6.0, "dm_p_value_squared_loss": 0.0001,
            "dm_stat_absolute_loss": -5.5, "dm_p_value_absolute_loss": 0.0002,
        },
        # Borderline: just under raw alpha=0.05, the kind BH across many tests
        # is expected to push back over the line.
        {
            "horizon": 1, "model_a": "var_aic", "model_b": "naive",
            "n_forecasts": 500, "insufficient_sample": False,
            "dm_stat_squared_loss": -2.0, "dm_p_value_squared_loss": 0.045,
            "dm_stat_absolute_loss": -1.9, "dm_p_value_absolute_loss": 0.048,
        },
    ]
    # Pad with clearly non-significant pairs so the borderline one above sits
    # inside a large-enough family for BH to actually matter.
    rng = np.random.default_rng(0)
    for i in range(10):
        rows.append({
            "horizon": 5, "model_a": f"model_{i}", "model_b": "naive",
            "n_forecasts": 500, "insufficient_sample": False,
            "dm_stat_squared_loss": float(rng.normal()),
            "dm_p_value_squared_loss": float(rng.uniform(0.3, 0.9)),
            "dm_stat_absolute_loss": float(rng.normal()),
            "dm_p_value_absolute_loss": float(rng.uniform(0.3, 0.9)),
        })
    # Insufficient-sample row: no p-value to correct.
    rows.append({
        "horizon": 20, "model_a": "vecm", "model_b": "lstm",
        "n_forecasts": 18, "insufficient_sample": True,
        "dm_stat_squared_loss": None, "dm_p_value_squared_loss": None,
        "dm_stat_absolute_loss": None, "dm_p_value_absolute_loss": None,
    })

    df = pd.DataFrame(rows)
    df["a_significantly_better_rmse"] = (df["dm_p_value_squared_loss"] < 0.05) & (
        df["dm_stat_squared_loss"] < 0
    )
    df["b_significantly_better_rmse"] = (df["dm_p_value_squared_loss"] < 0.05) & (
        df["dm_stat_squared_loss"] > 0
    )
    df["a_significantly_better_mae"] = (df["dm_p_value_absolute_loss"] < 0.05) & (
        df["dm_stat_absolute_loss"] < 0
    )
    df["b_significantly_better_mae"] = (df["dm_p_value_absolute_loss"] < 0.05) & (
        df["dm_stat_absolute_loss"] > 0
    )
    df["verdict"] = df.apply(
        lambda r: plain_language_verdict(r.to_dict(), r["model_a"], r["model_b"]), axis=1,
    )
    return df


def test_apply_multiplicity_correction_adds_expected_columns():
    dm_all = _synthetic_dm_all()
    out = apply_multiplicity_correction(dm_all)
    expected = {
        "dm_p_adj_squared_loss", "dm_p_adj_absolute_loss",
        "a_significantly_better_rmse_bh", "b_significantly_better_rmse_bh",
        "a_significantly_better_mae_bh", "b_significantly_better_mae_bh",
        "verdict_bh",
    }
    assert expected.issubset(out.columns)
    assert len(out) == len(dm_all)  # no rows dropped


def test_apply_multiplicity_correction_adjusted_pvalues_are_never_below_raw():
    """Defining property of BH: adjusted p-values are always >= raw p-values."""
    dm_all = _synthetic_dm_all()
    out = apply_multiplicity_correction(dm_all)
    testable = out["dm_p_value_squared_loss"].notna()
    assert (out.loc[testable, "dm_p_adj_squared_loss"] >= out.loc[testable, "dm_p_value_squared_loss"] - 1e-12).all()
    testable_abs = out["dm_p_value_absolute_loss"].notna()
    assert (out.loc[testable_abs, "dm_p_adj_absolute_loss"] >= out.loc[testable_abs, "dm_p_value_absolute_loss"] - 1e-12).all()


def test_apply_multiplicity_correction_borderline_result_loses_significance():
    """The borderline p=0.045 row should not survive BH correction once padded
    into a family of ~13 mostly-non-significant tests."""
    dm_all = _synthetic_dm_all()
    out = apply_multiplicity_correction(dm_all)
    borderline = out[(out["model_a"] == "var_aic") & (out["model_b"] == "naive")].iloc[0]
    assert borderline["a_significantly_better_rmse"]      # raw: significant
    assert not borderline["a_significantly_better_rmse_bh"]  # adjusted: not anymore


def test_apply_multiplicity_correction_strong_result_survives():
    dm_all = _synthetic_dm_all()
    out = apply_multiplicity_correction(dm_all)
    strong = out[(out["model_a"] == "arima_aic") & (out["model_b"] == "naive")].iloc[0]
    assert strong["a_significantly_better_rmse_bh"]
    assert "arima_aic significantly better than naive" in strong["verdict_bh"]


def test_apply_multiplicity_correction_insufficient_sample_row_stays_insufficient():
    dm_all = _synthetic_dm_all()
    out = apply_multiplicity_correction(dm_all)
    thin = out[out["insufficient_sample"]].iloc[0]
    assert pd.isna(thin["dm_p_adj_squared_loss"])
    assert not thin["a_significantly_better_rmse_bh"]
    assert not thin["b_significantly_better_rmse_bh"]
    assert "insufficient sample" in thin["verdict_bh"]


def test_apply_multiplicity_correction_does_not_mutate_input():
    dm_all = _synthetic_dm_all()
    original_cols = set(dm_all.columns)
    apply_multiplicity_correction(dm_all)
    assert set(dm_all.columns) == original_cols
