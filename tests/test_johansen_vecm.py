"""
Unit tests for Johansen Cointegration Test + Conditional VECM (Issue #47)

Comprehensive test suite covering:
1. Gold-layer levels loading
2. ADF I(1) confirmation and pipeline gating (enforce=True / False)
3. Johansen test output structure and all deterministic cases
4. VECM fitting and parameter extraction (beta, alpha)
5. Expanding-window evaluation mechanics with dynamic BIC lags (p=0 and p>0)
6. 4-way Diebold-Mariano test report formatting and boolean verdicts
7. Negative result handling (r=0)
8. End-to-end pipeline execution smoke test
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
    evaluate_vecm,
    fit_and_summarize_vecm,
    load_gold_levels,
    run_johansen_all_specs,
    run_pipeline,
    select_lag_levels,
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
    250-row DataFrame of I(1) level series for direct function testing.
    """
    np.random.seed(42)
    n = 250
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


@pytest.fixture
def synthetic_cointegrated_levels() -> pd.DataFrame:
    """
    Synthetic cointegrated system (n=120):
    Y1 and Y2 share a common random walk trend W_t.
    Y1_t = W_t + e1_t
    Y2_t = 2*W_t + e2_t
    -> Cointegrating vector: [2, -1]'
    """
    np.random.seed(123)
    n = 120
    dates = pd.bdate_range("2023-01-02", periods=n)
    w = np.cumsum(np.random.normal(0, 0.05, n))
    y1 = w + np.random.normal(0, 0.01, n)
    y2 = 2.0 * w + np.random.normal(0, 0.01, n)
    y3 = np.cumsum(np.random.normal(0, 0.02, n))

    return pd.DataFrame({
        "yield_spread_10y_2y": y1,
        "overnight_rate": y2,
        "us_treasury_10y": y3,
    }, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadGoldLevels:
    """Tests for load_gold_levels()."""

    def test_returns_dataframe_with_correct_columns(self, synthetic_gold_csv, monkeypatch):
        """Validates column selection returns only requested features."""
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
    """Tests for confirm_i1() — ADF stationarity verification and gating."""

    def test_output_structure(self, synthetic_levels):
        """Validates output DataFrame has expected columns."""
        result = confirm_i1(synthetic_levels, enforce=False)
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
        result = confirm_i1(df, enforce=True)
        assert result["I1_confirmed"].iloc[0], "Random walk should be confirmed I(1)"

    def test_gating_raises_runtime_error_on_stationary_data(self):
        """When a series is stationary I(0) in levels, enforce=True must raise RuntimeError."""
        np.random.seed(101)
        # White noise stationary series
        stationary = np.random.normal(0, 1, 300)
        df = pd.DataFrame({"stat_var": stationary}, index=pd.bdate_range("2020-01-01", periods=300))

        with pytest.raises(RuntimeError, match="Stationarity assumption violated"):
            confirm_i1(df, enforce=True)

    def test_gating_bypassed_when_enforce_false(self):
        """When enforce=False, non-I(1) series returns dataframe without error."""
        np.random.seed(101)
        stationary = np.random.normal(0, 1, 300)
        df = pd.DataFrame({"stat_var": stationary}, index=pd.bdate_range("2020-01-01", periods=300))

        res = confirm_i1(df, enforce=False)
        assert not res["I1_confirmed"].iloc[0]


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


class TestVECMEstimationAndEvaluation:
    """Tests for VECM fitting, summary, and expanding-window evaluation."""

    def test_fit_and_summarize_vecm(self, synthetic_cointegrated_levels):
        """Validates VECM estimation returns result with beta and alpha attributes."""
        res = fit_and_summarize_vecm(
            synthetic_cointegrated_levels, k_ar_diff=1, coint_rank=1, det_spec="ci"
        )
        assert hasattr(res, "beta")
        assert hasattr(res, "alpha")
        assert res.beta.shape[0] == synthetic_cointegrated_levels.shape[1]
        assert res.alpha.shape[0] == synthetic_cointegrated_levels.shape[1]

    def test_evaluate_vecm_with_drift_bic(self, synthetic_cointegrated_levels, monkeypatch):
        """Validates evaluate_vecm when bic_lag_diff=0 (drift model)."""
        import src.johansen_vecm as jv
        monkeypatch.setattr(jv, "MIN_TRAIN", 60)
        monkeypatch.setattr(jv, "STEP", 10)
        monkeypatch.setattr(jv, "HORIZONS", [1, 5])

        diffed = synthetic_cointegrated_levels.diff().dropna()
        diffed.columns = [f"d_{c}" for c in synthetic_cointegrated_levels.columns]

        metrics_df, raw, forecasts_df = evaluate_vecm(
            levels=synthetic_cointegrated_levels,
            diffed=diffed,
            k_ar_diff=1,
            coint_rank=1,
            det_spec="ci",
            aic_lag_diff=1,
            bic_lag_diff=0,  # Drift model test
        )

        assert isinstance(metrics_df, pd.DataFrame)
        assert not metrics_df.empty
        assert "rmse_vecm" in metrics_df.columns
        assert "rmse_var_bic" in metrics_df.columns
        assert 1 in raw and 5 in raw
        # Assert each raw tuple has 5 aligned arrays
        assert len(raw[1]) == 5
        actual, vecm, var_aic, var_bic, naive = raw[1]
        assert len(actual) == len(vecm) == len(var_aic) == len(var_bic) == len(naive)
        assert isinstance(forecasts_df, pd.DataFrame)
        assert "origin_date" in forecasts_df.columns

    def test_evaluate_vecm_with_lagged_bic(self, synthetic_cointegrated_levels, monkeypatch):
        """Validates evaluate_vecm when bic_lag_diff=1 (VAR model)."""
        import src.johansen_vecm as jv
        monkeypatch.setattr(jv, "MIN_TRAIN", 60)
        monkeypatch.setattr(jv, "STEP", 10)
        monkeypatch.setattr(jv, "HORIZONS", [1, 5])

        diffed = synthetic_cointegrated_levels.diff().dropna()
        diffed.columns = [f"d_{c}" for c in synthetic_cointegrated_levels.columns]

        metrics_df, raw, forecasts_df = evaluate_vecm(
            levels=synthetic_cointegrated_levels,
            diffed=diffed,
            k_ar_diff=1,
            coint_rank=1,
            det_spec="ci",
            aic_lag_diff=1,
            bic_lag_diff=1,  # Lagged VAR model test
        )

        assert isinstance(metrics_df, pd.DataFrame)
        assert not metrics_df.empty
        assert "rmse_var_bic" in metrics_df.columns


class TestDMReportColumns:
    """Tests for dm_report_vecm() output format."""

    def test_output_columns_present(self):
        """Validates DM report DataFrame has expected column naming pattern."""
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


class TestRunPipelineSmoke:
    """End-to-end smoke tests for run_pipeline."""

    def test_run_pipeline_execution(self, synthetic_gold_csv, monkeypatch):
        """Validates run_pipeline runs cleanly on synthetic gold CSV."""
        import src.johansen_vecm as jv
        monkeypatch.setattr(jv, "PROCESSED_DIR", synthetic_gold_csv.parent)
        monkeypatch.setattr(jv, "MIN_TRAIN", 60)
        monkeypatch.setattr(jv, "STEP", 20)
        monkeypatch.setattr(jv, "HORIZONS", [1, 5])
        monkeypatch.setattr(jv, "MAX_LAG_SEARCH", 3)

        features = ["yield_spread_10y_2y", "overnight_rate", "us_treasury_10y"]
        res = run_pipeline(features, "Smoke Test System", enforce_i1=False)

        assert isinstance(res, dict)
        assert "primary_rank" in res
        assert "johansen_df" in res
        assert "bic_lag_diff" in res
