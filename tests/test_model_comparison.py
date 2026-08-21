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

from src.model_comparison import merge_cross_pipeline, pairwise_dm, plain_language_verdict


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


def test_merge_cross_pipeline_drops_mismatched_actuals():
    """
    Two 'pipelines' that agree on origin_date/horizon/naive but disagree on
    `actual` for half the rows (simulating the real ARIMA-vs-VECM/LSTM
    calendar-drift issue) -- only the agreeing rows should survive.
    """
    dates = pd.bdate_range("2020-01-01", periods=10)
    left = pd.DataFrame({
        "origin_date": dates, "horizon": 1,
        "actual": np.arange(10, dtype=float),
        "naive": np.zeros(10),
        "pred_left": np.arange(10, dtype=float) + 0.1,
    })
    right = left.copy().rename(columns={"pred_left": "pred_right"})
    # Desync half the "actual" values on the right side, as calendar drift would.
    right.loc[5:, "actual"] = right.loc[5:, "actual"] + 100

    merged = merge_cross_pipeline(left, "pred_left", right, "pred_right")
    assert len(merged) == 5
    assert (merged["origin_date"] < dates[5]).all()


def test_merge_cross_pipeline_drops_mismatched_naive():
    dates = pd.bdate_range("2020-01-01", periods=6)
    left = pd.DataFrame({
        "origin_date": dates, "horizon": 1,
        "actual": np.ones(6),
        "naive": np.ones(6),
        "pred_left": np.ones(6),
    })
    right = left.copy().rename(columns={"pred_left": "pred_right"})
    right.loc[3:, "naive"] = right.loc[3:, "naive"] + 5

    merged = merge_cross_pipeline(left, "pred_left", right, "pred_right")
    assert len(merged) == 3
