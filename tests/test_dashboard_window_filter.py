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

from data_loader import (  # noqa: E402
    filter_by_origin_window,
    min_covered_origins,
    origin_window_label,
    window_model_coverage,
    window_origin_sets,
)


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

    def test_empty_selection_returns_typed_columns(self):
        """Regression guard (PR #115 review, C1): an empty model selection
        (e.g. the user clears every chip from 'Models to display') must
        yield a (0, 2) frame, not a bare (0, 0) frame from
        pd.DataFrame([]) on an empty list comprehension -- app.py indexes
        'n_origins' and 'model' unconditionally, so a column-less result
        is a KeyError that crashes the tab."""
        coverage = window_model_coverage(pd.DataFrame({"naive": [1.0, 2.0]}), {})
        assert list(coverage.columns) == ["model", "n_origins"]
        assert coverage.loc[coverage["n_origins"] == 0, "model"].tolist() == []
        assert min_covered_origins(coverage) == 0

    def test_no_matching_columns_returns_typed_columns(self):
        """Same guard for the deployment where r3_xgboost_forecasts.csv is
        absent and the user selects only XGBoost."""
        coverage = window_model_coverage(
            pd.DataFrame({"naive": [1.0]}), {"XGBoost": "xgboost"}
        )
        assert list(coverage.columns) == ["model", "n_origins"]
        assert min_covered_origins(coverage) == 0

    def test_counts_are_paired_on_actual(self):
        """n must match the chart's dropna(subset=["actual", column]) count,
        not a plain per-column notna() count (PR #115 review, M2) --
        otherwise the sample-size gate can overstate what's actually
        plotted whenever 'actual' itself has a gap a model column doesn't."""
        df = pd.DataFrame({
            "actual": [1.0, np.nan, 1.2],
            "naive": [1.0, 1.0, 1.1],
        })
        coverage = window_model_coverage(df, {"Naïve Random Walk": "naive"})
        assert coverage.set_index("model")["n_origins"]["Naïve Random Walk"] == 2


class TestWindowOriginSets:
    def test_equal_counts_over_different_subsets_are_detected(self):
        """PR #115 review, M1: cardinality equality does not imply the same
        origin set. Two models with the same count but a gap at a
        different date must compare as unpaired."""
        df = pd.DataFrame({
            "origin_date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15"]
            ),
            "actual": [1.0, 1.1, 1.2],
            "model_a": [1.0, np.nan, 1.2],
            "model_b": [1.0, 1.1, np.nan],
        })
        sets = window_origin_sets(df, {"A": "model_a", "B": "model_b"})
        assert len(sets["A"]) == len(sets["B"]) == 2
        assert sets["A"] != sets["B"]

    def test_identical_origin_sets_compare_equal(self):
        df = pd.DataFrame({
            "origin_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "actual": [1.0, 1.1],
            "model_a": [1.0, 1.1],
            "model_b": [1.0, 1.1],
        })
        sets = window_origin_sets(df, {"A": "model_a", "B": "model_b"})
        assert sets["A"] == sets["B"]


class TestOriginWindowLabel:
    def test_labels_the_realized_range_not_an_empty_one(self):
        df = pd.DataFrame({
            "origin_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
        })
        assert origin_window_label(df) == "2024-01-01 – 2024-01-08 (n=2)"

    def test_empty_frame_does_not_raise(self):
        assert origin_window_label(pd.DataFrame({"origin_date": []})) == \
            "no origins in range"


class TestMinCoveredOrigins:
    def test_returns_the_minimum_not_the_maximum(self):
        """Regression guard: taking the max here would let a well-covered
        model wave through a box/violin built from another plotted model's
        much smaller sample (e.g. 31 origins for most models but only 27
        for XGBoost, whose common-sample start is later)."""
        coverage = pd.DataFrame({
            "model": ["Naïve Random Walk", "ARIMA", "XGBoost"],
            "n_origins": [31, 31, 27],
        })
        assert min_covered_origins(coverage) == 27

    def test_ignores_zero_coverage_models(self):
        """A model with zero coverage is already disclosed/omitted
        separately -- it must not drag the sample-size floor to 0 for the
        models that ARE actually being plotted."""
        coverage = pd.DataFrame({
            "model": ["Naïve Random Walk", "XGBoost"],
            "n_origins": [31, 0],
        })
        assert min_covered_origins(coverage) == 31

    def test_all_zero_coverage_returns_zero(self):
        coverage = pd.DataFrame({
            "model": ["Naïve Random Walk"],
            "n_origins": [0],
        })
        assert min_covered_origins(coverage) == 0

    def test_empty_coverage_returns_zero(self):
        coverage = pd.DataFrame(columns=["model", "n_origins"])
        assert min_covered_origins(coverage) == 0
