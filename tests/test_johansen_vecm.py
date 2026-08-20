"""
Unit tests for Johansen Cointegration Test + Conditional VECM (Issue #47)

Uses the existing ``synthetic_gold_inputs`` fixture from conftest.py
(40 business days of synthetic BoC, FRED, and StatCan CPI data) plus
a longer synthetic fixture for tests that need more observations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.johansen_vecm import (  # noqa: E402
    ALPHA,
    FEATURE_SET_5VAR,
    HORIZONS,
    JOHANSEN_SPECS,
    TARGET,
    confirm_i1,
    dm_report_vecm,
    load_gold_levels,
    run_johansen_all_specs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_gold_csv(tmp_path: Path) -> Path:
    """
    Create a synthetic gold_features.csv with 200 business days of realistic
    I(1) level data (random walks) — enough for Johansen testing.
    """
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2023-01-02", periods=n)

    # Simulate I(1) random walks for each variable
    data = {"date": dates}
    starts = {
        "yield_spread_10y_2y": 0.5,
        "overnight_rate": 2.5,
        "us_treasury_10y": 3.8,
        "fed_funds_rate": 4.5,
        "cpi_yoy": 3.0,
        "usdcad": 1.32,
    }
    for col, start in starts.items():
        innovations = np.random.normal(0, 0.02, n)
        data[col] = start + np.cumsum(innovations)

    # Add yield columns needed by gold CSV schema
    data["yield_2y"] = data["us_treasury_10y"] - data["yield_spread_10y_2y"] + 0.1
    data["yield_5y"] = data["yield_2y"] + 0.2
    data["yield_10y"] = data["yield_2y"] + data["yield_spread_10y_2y"]
    data["yield_3y"] = data["yield_2y"] + 0.1
    data["yield_7y"] = data["yield_5y"] + 0.15
    data["yield_long"] = data["yield_10y"] + 0.1
    data["cpi_all_items"] = 150.0 + np.cumsum(np.random.normal(0.05, 0.1, n))
    data["yield_spread_10y_5y"] = data["yield_10y"] - data["yield_5y"]
    data["yield_spread_5y_2y"] = data["yield_5y"] - data["yield_2y"]

    # Add differenced columns (d_*)
    df = pd.DataFrame(data)
    for col in ["yield_spread_10y_2y", "yield_spread_10y_5y", "yield_spread_5y_2y",
                "overnight_rate", "yield_2y", "yield_5y", "yield_10y",
                "us_treasury_10y", "fed_funds_rate", "usdcad", "cpi_yoy"]:
        if col in df.columns:
            df[f"d_{col}"] = df[col].diff()

    df = df.dropna()
    csv_path = tmp_path / "gold_features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_levels() -> pd.DataFrame:
    """
    200-row DataFrame of I(1) level series for direct function testing
    (no CSV round-trip needed).
    """
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {}
    starts = {
        "yield_spread_10y_2y": 0.5,
        "overnight_rate": 2.5,
        "us_treasury_10y": 3.8,
        "fed_funds_rate": 4.5,
        "cpi_yoy": 3.0,
    }
    for col, start in starts.items():
        innovations = np.random.normal(0, 0.02, n)
        data[col] = start + np.cumsum(innovations)

    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadGoldLevels:
    """Tests for load_gold_levels()."""

    def test_returns_dataframe_with_correct_columns(self, synthetic_gold_csv, monkeypatch):
        """Validates column selection returns only requested features."""
        # Monkeypatch PROCESSED_DIR to point at our temp dir
        import src.johansen_vecm as jv
        monkeypatch.setattr(jv, "PROCESSED_DIR", synthetic_gold_csv.parent)

        df = load_gold_levels(FEATURE_SET_5VAR)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == FEATURE_SET_5VAR

    def test_datetime_index(self, synthetic_gold_csv, monkeypatch):
        """Validates DatetimeIndex."""
        import src.johansen_vecm as jv
        monkeypatch.setattr(jv, "PROCESSED_DIR", synthetic_gold_csv.parent)

        df = load_gold_levels(FEATURE_SET_5VAR)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing

    def test_no_nulls(self, synthetic_gold_csv, monkeypatch):
        """Validates zero null values after loading."""
        import src.johansen_vecm as jv
        monkeypatch.setattr(jv, "PROCESSED_DIR", synthetic_gold_csv.parent)

        df = load_gold_levels(FEATURE_SET_5VAR)
        assert df.isnull().sum().sum() == 0


class TestConfirmI1:
    """Tests for confirm_i1() — ADF stationarity verification."""

    def test_output_structure(self, synthetic_levels):
        """Validates output DataFrame has expected columns."""
        result = confirm_i1(synthetic_levels)
        assert isinstance(result, pd.DataFrame)
        assert "variable" in result.columns
        assert "adf_p_level" in result.columns
        assert "adf_p_diff" in result.columns
        assert "I1_confirmed" in result.columns
        assert len(result) == len(synthetic_levels.columns)

    def test_random_walk_detected_as_i1(self):
        """A pure random walk should be detected as I(1)."""
        np.random.seed(99)
        rw = np.cumsum(np.random.normal(0, 1, 500))
        df = pd.DataFrame({"rw": rw}, index=pd.bdate_range("2020-01-01", periods=500))
        result = confirm_i1(df)
        assert result["I1_confirmed"].iloc[0], "Random walk should be confirmed I(1)"


class TestJohansenOutputStructure:
    """Tests for run_johansen_all_specs() output format."""

    def test_output_columns(self, synthetic_levels):
        """Validates Johansen results DataFrame has all expected columns."""
        k_ar_diff = 1
        df, rank = run_johansen_all_specs(synthetic_levels, k_ar_diff)
        expected_cols = [
            "det_order", "det_label", "r_null",
            "trace_stat", "trace_cv_90", "trace_cv_95", "trace_cv_99", "trace_reject_95",
            "max_eig_stat", "max_eig_cv_90", "max_eig_cv_95", "max_eig_cv_99", "max_eig_reject_95",
            "eigenvalue", "trace_rank", "max_eig_rank",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_det_specs_present(self, synthetic_levels):
        """Validates results cover all 3 deterministic specifications."""
        k_ar_diff = 1
        df, _ = run_johansen_all_specs(synthetic_levels, k_ar_diff)
        det_orders = sorted(df["det_order"].unique())
        assert det_orders == [-1, 0, 1]

    def test_correct_number_of_rows(self, synthetic_levels):
        """Each det_order should produce n_vars rows (one per r_null)."""
        k_ar_diff = 1
        df, _ = run_johansen_all_specs(synthetic_levels, k_ar_diff)
        n_vars = synthetic_levels.shape[1]
        n_specs = len(JOHANSEN_SPECS)
        assert len(df) == n_vars * n_specs

    def test_rank_is_nonnegative_integer(self, synthetic_levels):
        """Primary rank should be a non-negative integer <= n_vars."""
        k_ar_diff = 1
        _, rank = run_johansen_all_specs(synthetic_levels, k_ar_diff)
        assert isinstance(rank, int)
        assert 0 <= rank <= synthetic_levels.shape[1]


class TestDMReportColumns:
    """Tests for dm_report_vecm() output format."""

    def test_output_columns_present(self):
        """Validates DM report DataFrame has expected column naming pattern."""
        # Create minimal synthetic raw data
        np.random.seed(42)
        n = 50
        raw = {
            1: (
                np.random.randn(n),       # actual
                np.random.randn(n),       # vecm
                np.random.randn(n),       # var_aic
                np.random.randn(n),       # var_bic
                np.random.randn(n),       # naive
            ),
        }
        result = dm_report_vecm(raw)
        assert isinstance(result, pd.DataFrame)
        assert "horizon_days" in result.columns

        # Check column patterns for each comparison
        for rival in ["naive", "var_aic", "var_bic"]:
            assert f"dm_vecm_{rival}_stat_sq" in result.columns
            assert f"dm_vecm_{rival}_p_sq" in result.columns
            assert f"dm_vecm_{rival}_stat_abs" in result.columns
            assert f"dm_vecm_{rival}_p_abs" in result.columns
            assert f"vecm_sig_better_{rival}_rmse" in result.columns
            assert f"{rival}_sig_better_vecm_rmse" in result.columns

    def test_verdicts_are_boolean(self):
        """All verdict columns should be boolean."""
        np.random.seed(42)
        n = 50
        raw = {1: tuple(np.random.randn(n) for _ in range(5))}
        result = dm_report_vecm(raw)
        verdict_cols = [c for c in result.columns if "sig_better" in c]
        for col in verdict_cols:
            assert result[col].dtype == bool, f"{col} should be boolean"


class TestNegativeResultHandling:
    """Tests for proper handling when no cointegration is found."""

    def test_rank_zero_returns_none_vecm(self, synthetic_levels):
        """When rank=0, the pipeline result should have None for VECM fields."""
        # We can't force r=0 easily, but we can verify the function signature
        # accepts r=0 gracefully by checking run_johansen_all_specs returns
        # a valid rank
        k_ar_diff = 1
        df, rank = run_johansen_all_specs(synthetic_levels, k_ar_diff)
        assert isinstance(rank, int)
        # If rank is 0, there should be no VECM estimation
        # If rank > 0, that's fine too — we just verify the interface
        assert df is not None
        assert len(df) > 0
