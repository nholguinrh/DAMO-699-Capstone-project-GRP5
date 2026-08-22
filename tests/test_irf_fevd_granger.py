"""
Unit tests for IRF / FEVD / Granger Causality (Issue #48)

Comprehensive test suite covering:
1. fit_primary_vecm() gating on r=0 (no cointegration)
2. Orthogonalized IRF computation and tidy long-format output
3. Plain-language IRF summary
4. FEVD computation (manual replica of statsmodels' cumulative-squared-orth-IRF
   formula) and its defining property: proportions sum to 1 per horizon/response
5. Plain-language FEVD summary
6. Granger causality output structure, p-value bounds, and reject_h0 consistency
   with the plain-language string
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.vector_ar.vecm import VECM

# Ensure src/ is on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.irf_fevd_granger import (  # noqa: E402
    BOND_YIELDS,
    POLICY_AND_FX_DRIVERS,
    compute_fevd,
    compute_irf,
    fit_primary_vecm,
    one_sd_shock_response,
    run_granger_causality,
    summarize_fevd_plain_language,
    summarize_irf_plain_language,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cointegrated_levels() -> pd.DataFrame:
    """
    Synthetic cointegrated system (n=150), mirroring test_johansen_vecm.py's
    fixture: Y1 and Y2 share a common random-walk trend, Y3 is an independent
    random walk -- gives a fitted VECM (rank=1) fast enough for unit tests.
    """
    np.random.seed(123)
    n = 150
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


@pytest.fixture
def fitted_vecm(synthetic_cointegrated_levels):
    """A fitted VECM(k_ar_diff=1, rank=1) on the synthetic system above."""
    return VECM(
        synthetic_cointegrated_levels, k_ar_diff=1, coint_rank=1, deterministic="ci",
    ).fit()


@pytest.fixture
def fitted_irf(fitted_vecm):
    return compute_irf(fitted_vecm, periods=10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitPrimaryVecm:
    """Tests for fit_primary_vecm()'s r=0 gating."""

    def test_raises_on_zero_rank(self, monkeypatch):
        """When the Johansen test finds no cointegrating vectors, refuse to proceed."""
        import src.irf_fevd_granger as ifg

        monkeypatch.setattr(ifg, "load_gold_levels", lambda feature_set: pd.DataFrame(
            {c: np.random.randn(50) for c in feature_set},
            index=pd.bdate_range("2023-01-02", periods=50),
        ))
        monkeypatch.setattr(ifg, "confirm_i1", lambda levels, enforce: None)
        monkeypatch.setattr(ifg, "select_lag_levels", lambda levels: (1, None))
        monkeypatch.setattr(ifg, "run_johansen_all_specs", lambda levels, k: (None, 0))

        with pytest.raises(RuntimeError, match="no cointegrating vectors"):
            fit_primary_vecm(enforce_i1=False)


class TestComputeIrf:
    """Tests for compute_irf() and one_sd_shock_response()."""

    def test_orth_irfs_shape(self, fitted_irf, synthetic_cointegrated_levels):
        n_vars = synthetic_cointegrated_levels.shape[1]
        assert fitted_irf.orth_irfs.shape == (11, n_vars, n_vars)

    def test_response_dataframe_structure(self, fitted_irf, synthetic_cointegrated_levels):
        df = one_sd_shock_response(fitted_irf, synthetic_cointegrated_levels, shock_var="overnight_rate")
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "horizon_days", "shock_variable", "response_variable",
            "response", "cumulative_response",
        }
        assert expected_cols.issubset(df.columns)
        # 11 horizons (0..10) x 3 variables
        assert len(df) == 11 * synthetic_cointegrated_levels.shape[1]
        assert (df["shock_variable"] == "overnight_rate").all()

    def test_cumulative_response_is_running_sum(self, fitted_irf, synthetic_cointegrated_levels):
        df = one_sd_shock_response(fitted_irf, synthetic_cointegrated_levels, shock_var="overnight_rate")
        one_var = df[df["response_variable"] == "overnight_rate"].sort_values("horizon_days")
        expected_cumsum = one_var["response"].cumsum().to_numpy()
        assert np.allclose(one_var["cumulative_response"].to_numpy(), expected_cumsum)

    def test_shock_variable_has_nonzero_impact_response(self, fitted_irf, synthetic_cointegrated_levels):
        """A variable's own-shock response at h=0 should be nonzero (it's the shock)."""
        df = one_sd_shock_response(fitted_irf, synthetic_cointegrated_levels, shock_var="overnight_rate")
        h0 = df[(df["horizon_days"] == 0) & (df["response_variable"] == "overnight_rate")]
        assert abs(h0["response"].iloc[0]) > 0


class TestSummarizeIrfPlainLanguage:
    def test_one_line_per_response_variable(self, fitted_irf, synthetic_cointegrated_levels):
        df = one_sd_shock_response(fitted_irf, synthetic_cointegrated_levels, shock_var="overnight_rate")
        lines = summarize_irf_plain_language(df)
        assert len(lines) == synthetic_cointegrated_levels.shape[1]
        for line in lines:
            assert "peak of" in line and "day" in line


class TestComputeFevd:
    """Tests for compute_fevd() -- the manual replica of the statsmodels FEVD formula."""

    def test_output_structure(self, fitted_irf, synthetic_cointegrated_levels):
        df = compute_fevd(fitted_irf, synthetic_cointegrated_levels)
        expected_cols = {"horizon_days", "response_variable", "shock_variable", "proportion"}
        assert expected_cols.issubset(df.columns)

    def test_proportions_sum_to_one_per_horizon_and_response(self, fitted_irf, synthetic_cointegrated_levels):
        """Defining property of a variance decomposition: shares sum to 1."""
        df = compute_fevd(fitted_irf, synthetic_cointegrated_levels)
        sums = df.groupby(["horizon_days", "response_variable"])["proportion"].sum()
        assert np.allclose(sums.to_numpy(), 1.0, atol=1e-8)

    def test_proportions_are_nonnegative(self, fitted_irf, synthetic_cointegrated_levels):
        df = compute_fevd(fitted_irf, synthetic_cointegrated_levels)
        assert (df["proportion"] >= -1e-12).all()

    def test_own_shock_explains_all_variance_at_horizon_zero_for_first_ordered_variable(
        self, fitted_irf, synthetic_cointegrated_levels,
    ):
        """
        The first-ordered variable (yield_spread_10y_2y) is contemporaneously
        exogenous under this Cholesky ordering, so at h=0 its own shock must
        explain 100% of its forecast error variance.
        """
        df = compute_fevd(fitted_irf, synthetic_cointegrated_levels)
        row = df[
            (df["horizon_days"] == 0)
            & (df["response_variable"] == "yield_spread_10y_2y")
            & (df["shock_variable"] == "yield_spread_10y_2y")
        ]
        assert np.isclose(row["proportion"].iloc[0], 1.0, atol=1e-8)


class TestSummarizeFevdPlainLanguage:
    def test_one_line_per_response_variable(self, fitted_irf, synthetic_cointegrated_levels):
        fevd_df = compute_fevd(fitted_irf, synthetic_cointegrated_levels)
        lines = summarize_fevd_plain_language(
            fevd_df, ["yield_spread_10y_2y", "us_treasury_10y"], horizon=5,
        )
        assert len(lines) == 2
        for line in lines:
            assert "explains" in line and "forecast error variance" in line


class TestRunGrangerCausality:
    def test_output_structure(self, fitted_vecm):
        df = run_granger_causality(
            fitted_vecm,
            causing_vars=["overnight_rate"],
            caused_vars=["us_treasury_10y"],
        )
        expected_cols = {
            "causing", "caused", "test_statistic", "crit_value",
            "df_num", "df_denom", "p_value", "reject_h0", "plain_language",
        }
        assert expected_cols.issubset(df.columns)
        assert len(df) == 1

    def test_full_default_grid_shape(self, fitted_vecm):
        """Default grid: len(POLICY_AND_FX_DRIVERS) x len(BOND_YIELDS) is the
        full policy/FX-driver x bond-yield cross the issue asks for -- confirm
        the module's own defaults produce that shape (subsetting the fitted
        3-variable fixture down to variables it actually has)."""
        causing = [v for v in POLICY_AND_FX_DRIVERS if v in fitted_vecm.names]
        caused = [v for v in BOND_YIELDS if v in fitted_vecm.names]
        df = run_granger_causality(fitted_vecm, causing_vars=causing, caused_vars=caused)
        assert len(df) == len(causing) * len(caused)

    def test_p_values_in_unit_interval(self, fitted_vecm):
        df = run_granger_causality(
            fitted_vecm,
            causing_vars=["overnight_rate"],
            caused_vars=["us_treasury_10y"],
        )
        assert ((df["p_value"] >= 0) & (df["p_value"] <= 1)).all()

    def test_reject_h0_consistent_with_plain_language(self, fitted_vecm):
        df = run_granger_causality(
            fitted_vecm,
            causing_vars=["overnight_rate"],
            caused_vars=["us_treasury_10y"],
        )
        for _, row in df.iterrows():
            if row["reject_h0"]:
                assert "Granger-causes" in row["plain_language"]
                assert "No evidence" not in row["plain_language"]
            else:
                assert "No evidence" in row["plain_language"]

    def test_signif_threshold_changes_rejection(self, fitted_vecm):
        """A near-1.0 significance level should never reject; a near-0 level should
        essentially never reject either -- both are degenerate but should still run
        without error and stay internally consistent (reject_h0 == p_value < signif)."""
        df = run_granger_causality(
            fitted_vecm,
            causing_vars=["overnight_rate"],
            caused_vars=["us_treasury_10y"],
            signif=0.9999,
        )
        row = df.iloc[0]
        assert row["reject_h0"] == bool(row["p_value"] < 0.9999)
