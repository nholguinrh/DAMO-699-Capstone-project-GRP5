"""
Unit tests for the sidebar date-range slider's frame-shaping logic
(Issue #100 follow-up, PR #115 review): ``filter_by_origin_window`` and
``window_model_coverage`` in ``src/dashboard/data_loader.py``.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dashboard"))

from data_loader import filter_by_origin_window, window_model_coverage  # noqa: E402


@pytest.fixture
def forecast_frame():
    dates = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame({
        "origin_date": list(dates) * 2,
        "horizon": [1] * 10 + [5] * 10,
        "actual": np.arange(20, dtype=float),
        "naive": np.arange(20, dtype=float),
    })


class TestFilterByOriginWindow:
    def test_inclusive_end_bound(self, forecast_frame):
        start, end = date(2024, 1, 1), date(2024, 1, 5)
        out = filter_by_origin_window(forecast_frame, start, end)
        assert out["origin_date"].max() == pd.Timestamp(end)

    def test_end_bound_inclusive_under_intraday_timestamp(self, forecast_frame):
        """A half-open upper bound must still include the end date even if
        origin_date ever carries a non-midnight component."""
        df = forecast_frame.copy()
        df.loc[df["origin_date"] == pd.Timestamp("2024-01-05"), "origin_date"] = (
            pd.Timestamp("2024-01-05 09:30:00")
        )
        out = filter_by_origin_window(df, date(2024, 1, 1), date(2024, 1, 5))
        assert (out["origin_date"].dt.date == date(2024, 1, 5)).any()

    def test_no_row_duplication_and_order_preserved(self, forecast_frame):
        out = filter_by_origin_window(forecast_frame, date(2024, 1, 1), date(2024, 1, 20))
        assert len(out) == len(forecast_frame)
        assert list(out.index) == sorted(out.index)

    def test_horizon_parity_preserved(self, forecast_frame):
        out = filter_by_origin_window(forecast_frame, date(2024, 1, 1), date(2024, 1, 5))
        assert set(out["horizon"].unique()) == {1, 5}
        assert (out["horizon"] == 1).sum() == (out["horizon"] == 5).sum()

    def test_empty_window_returns_empty_frame(self, forecast_frame):
        out = filter_by_origin_window(forecast_frame, date(2023, 1, 1), date(2023, 1, 2))
        assert out.empty
        assert list(out.columns) == list(forecast_frame.columns)

    def test_single_day_window_selects_only_that_origin(self, forecast_frame):
        out = filter_by_origin_window(forecast_frame, date(2024, 1, 3), date(2024, 1, 3))
        assert (out["origin_date"] == pd.Timestamp("2024-01-03")).all()
        assert out["origin_date"].nunique() == 1


class TestWindowModelCoverage:
    def test_counts_non_null_per_model(self):
        df = pd.DataFrame({
            "naive": [1.0, 2.0, np.nan],
            "xgboost": [np.nan, np.nan, np.nan],
        })
        coverage = window_model_coverage(
            df, {"Naïve Random Walk": "naive", "XGBoost": "xgboost"}
        )
        counts = coverage.set_index("model")["n_origins"].to_dict()
        assert counts["Naïve Random Walk"] == 2
        assert counts["XGBoost"] == 0

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"naive": [1.0, 2.0]})
        coverage = window_model_coverage(
            df, {"Naïve Random Walk": "naive", "XGBoost": "xgboost"}
        )
        assert set(coverage["model"]) == {"Naïve Random Walk"}
