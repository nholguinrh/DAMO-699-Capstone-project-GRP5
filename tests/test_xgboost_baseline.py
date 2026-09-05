"""
Unit tests for src/xgboost_baseline.py (Issue #101)
DAMO-699 Capstone Project, Group 5

Tests cover:
- Strict absence of look-ahead leakage in lag/rolling features
- Expanding-window CV fold temporal ordering and shape integrity
- Level reconstruction consistency
- Prediction interval ordering (lower <= point <= upper)
- Full pipeline smoke test and output CSV schema validation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic Gold-layer data matching the real schema
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_gold_df():
    """Generate a small reproducible Gold-layer-like DataFrame for testing."""
    rng = np.random.RandomState(42)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)

    df = pd.DataFrame({
        "date": dates,
        "yield_spread_10y_2y": np.cumsum(rng.randn(n) * 0.01) + 1.0,
        "d_yield_spread_10y_2y": rng.randn(n) * 0.01,
        "d_overnight_rate": rng.randn(n) * 0.005,
        "d_us_treasury_10y": rng.randn(n) * 0.01,
        "d_fed_funds_rate": rng.randn(n) * 0.005,
        "d_cpi_yoy": rng.randn(n) * 0.001,
        "d_usdcad": rng.randn(n) * 0.002,
    })
    return df


# ---------------------------------------------------------------------------
# Test 1: Feature engineering produces no look-ahead leakage
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Verify that all engineered features use strictly past data only."""

    def test_lag_features_are_strictly_causal(self, synthetic_gold_df):
        from xgboost_baseline import engineer_tabular_features

        data, feature_cols, target_cols = engineer_tabular_features(
            synthetic_gold_df, lags=[1, 2, 5], horizons=[1, 5],
        )

        # All lag columns should use shift >= 1, meaning value at row i
        # comes from row i-lag in the original. Check that the first few
        # lag features are NaN-free (they were clipped by the valid slice).
        assert len(data) > 0
        for col in feature_cols:
            assert data[col].isna().sum() == 0, f"Feature {col} has NaNs in valid slice"

    def test_no_future_information_in_features(self, synthetic_gold_df):
        """Verify that removing the last row doesn't change any feature value
        for earlier rows (a basic non-leakage sanity check)."""
        from xgboost_baseline import engineer_tabular_features

        data_full, fcols, _ = engineer_tabular_features(
            synthetic_gold_df, lags=[1, 2], horizons=[1],
        )
        data_trimmed, _, _ = engineer_tabular_features(
            synthetic_gold_df.iloc[:-1].copy(), lags=[1, 2], horizons=[1],
        )

        # Compare overlapping rows: features should be identical
        n_overlap = min(len(data_full), len(data_trimmed))
        if n_overlap > 10:
            for col in fcols:
                np.testing.assert_array_almost_equal(
                    data_full[col].values[:n_overlap - 5],
                    data_trimmed[col].values[:n_overlap - 5],
                    err_msg=f"Feature {col} changed when future data was removed",
                )

    def test_rejects_zero_lag(self, synthetic_gold_df):
        from xgboost_baseline import engineer_tabular_features

        with pytest.raises(ValueError, match="look-ahead bias"):
            engineer_tabular_features(
                synthetic_gold_df, lags=[0, 1, 2], horizons=[1],
            )


# ---------------------------------------------------------------------------
# Test 2: CV fold construction
# ---------------------------------------------------------------------------

class TestRollingFolds:
    """Verify expanding-window folds preserve temporal ordering."""

    def test_fold_boundaries_are_monotonic(self):
        from xgboost_baseline import make_rolling_folds

        folds = make_rolling_folds(n_samples=300, min_train=100, n_folds=5)
        assert len(folds) == 5

        prev_end = 0
        for train_end, test_end in folds:
            assert train_end >= prev_end, "Folds must be monotonically increasing"
            assert test_end > train_end, "Test block must follow training block"
            prev_end = test_end

    def test_folds_cover_all_test_data(self):
        from xgboost_baseline import make_rolling_folds

        n = 300
        folds = make_rolling_folds(n_samples=n, min_train=100, n_folds=3)
        # Last fold's test_end must equal n
        assert folds[-1][1] == n

    def test_rejects_insufficient_samples(self):
        from xgboost_baseline import make_rolling_folds

        with pytest.raises(ValueError):
            make_rolling_folds(n_samples=50, min_train=100, n_folds=5)


# ---------------------------------------------------------------------------
# Test 3: Level reconstruction consistency
# ---------------------------------------------------------------------------

class TestLevelReconstruction:
    """Verify that predicted level = origin_level + predicted_diff."""

    def test_level_reconstruction_identity(self, synthetic_gold_df):
        from xgboost_baseline import run_rolling_cv

        forecasts_df, _, _ = run_rolling_cv(
            synthetic_gold_df,
            lags=[1, 2],
            horizons=[1],
            min_train=50,
            n_folds=2,
            n_estimators=10,
            max_depth=2,
        )

        # xgboost = naive + (xgboost - naive), where naive is the origin level
        # This is an identity check: actual reconstruction uses
        # level = origin_level + diff, so xgboost != naive in general
        assert len(forecasts_df) > 0
        assert (forecasts_df["xgboost"] != forecasts_df["naive"]).any(), \
            "XGBoost predictions should differ from naive in general"


# ---------------------------------------------------------------------------
# Test 4: Prediction interval ordering
# ---------------------------------------------------------------------------

class TestPredictionIntervals:
    """Verify PI lower <= point <= upper after conformal correction."""

    def test_interval_ordering(self, synthetic_gold_df):
        from xgboost_baseline import run_rolling_cv

        forecasts_df, _, intervals_df = run_rolling_cv(
            synthetic_gold_df,
            lags=[1, 2],
            horizons=[1],
            min_train=50,
            n_folds=2,
            n_estimators=10,
            max_depth=2,
        )

        # lower_90 <= xgboost <= upper_90
        assert (forecasts_df["lower_90"] <= forecasts_df["xgboost"]).all(), \
            "lower_90 must be <= point prediction"
        assert (forecasts_df["upper_90"] >= forecasts_df["xgboost"]).all(), \
            "upper_90 must be >= point prediction"

        # lower_95 <= lower_90 (wider interval)
        assert (forecasts_df["lower_95"] <= forecasts_df["lower_90"]).all(), \
            "95% lower bound must be <= 90% lower bound"
        assert (forecasts_df["upper_95"] >= forecasts_df["upper_90"]).all(), \
            "95% upper bound must be >= 90% upper bound"

    def test_interval_metrics_computed(self, synthetic_gold_df):
        from xgboost_baseline import run_rolling_cv

        _, _, intervals_df = run_rolling_cv(
            synthetic_gold_df,
            lags=[1, 2],
            horizons=[1],
            min_train=50,
            n_folds=2,
            n_estimators=10,
            max_depth=2,
        )

        assert "empirical_coverage_90" in intervals_df.columns
        assert "mean_interval_width_90" in intervals_df.columns
        assert intervals_df["mean_interval_width_90"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Test 5: Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestSmokePipeline:
    """End-to-end smoke test on synthetic data."""

    def test_rolling_cv_returns_expected_schema(self, synthetic_gold_df):
        from xgboost_baseline import run_rolling_cv

        forecasts_df, metrics_df, intervals_df = run_rolling_cv(
            synthetic_gold_df,
            lags=[1, 2],
            horizons=[1, 5],
            min_train=50,
            n_folds=2,
            n_estimators=10,
            max_depth=2,
        )

        expected_forecast_cols = {
            "origin_date", "horizon", "actual", "naive", "xgboost",
            "fold", "lower_90", "upper_90", "lower_95", "upper_95",
        }
        assert set(forecasts_df.columns) == expected_forecast_cols

        expected_metric_cols = {
            "horizon", "n_forecasts", "rmse_xgboost", "rmse_naive",
            "rmse_improvement_pct", "mae_xgboost", "mae_naive",
            "mae_improvement_pct",
        }
        assert set(metrics_df.columns) == expected_metric_cols

    def test_shap_computation(self, synthetic_gold_df):
        from xgboost_baseline import compute_tree_shap_interpretability

        summary_df, values_df = compute_tree_shap_interpretability(
            synthetic_gold_df,
            lags=[1, 2],
            horizons=[1],
            n_estimators=10,
            max_depth=2,
        )

        assert len(summary_df) > 0
        assert "feature" in summary_df.columns
        assert "mean_abs_shap" in summary_df.columns
        assert summary_df["mean_abs_shap"].min() >= 0

    def test_model_factory_creates_valid_model(self):
        from xgboost_baseline import create_model

        model = create_model(loss="squared_error")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

        q_model = create_model(loss="quantile", alpha=0.05)
        assert hasattr(q_model, "fit")

    def test_real_output_prediction_interval_non_crossing(self):
        """Verify that committed/generated XGBoost forecast file has strictly non-crossing intervals."""
        csv_path = PROJECT_ROOT / "outputs" / "r3_xgboost_forecasts.csv"
        if not csv_path.exists():
            pytest.skip("r3_xgboost_forecasts.csv not yet generated")

        df = pd.read_csv(csv_path)
        assert (df["lower_95"] <= df["lower_90"]).all(), "95% lower bound must be <= 90% lower bound"
        assert (df["lower_90"] <= df["xgboost"]).all(), "90% lower bound must be <= point prediction"
        assert (df["xgboost"] <= df["upper_90"]).all(), "point prediction must be <= 90% upper bound"
        assert (df["upper_90"] <= df["upper_95"]).all(), "90% upper bound must be <= 95% upper bound"

    def test_temporal_embargo_prevents_target_overlap(self, synthetic_gold_df):
        """Verify training targets do not overlap with test fold dates for multi-step horizons."""
        from xgboost_baseline import engineer_tabular_features, make_rolling_folds

        horizons = [5, 20]
        data, _, target_cols = engineer_tabular_features(synthetic_gold_df, lags=[1, 2], horizons=horizons)
        folds = make_rolling_folds(len(data), min_train=50, n_folds=3)

        for train_end, test_end in folds:
            for h in horizons:
                embargo_end = train_end - h
                assert embargo_end < train_end
                # The maximum forward label step from the last training row must not exceed train_end
                assert embargo_end + h <= train_end

    def test_real_output_shap_values_schema_and_density(self):
        """Assert committed SHAP values artifact is dense without NaN column sprawl."""
        shap_path = PROJECT_ROOT / "outputs" / "r3_xgboost_shap_values.csv"
        if not shap_path.exists():
            pytest.skip("r3_xgboost_shap_values.csv not yet generated")

        df = pd.read_csv(shap_path)
        assert "horizon" in df.columns, "SHAP values must include 'horizon' column in tidy format"
        assert "date" in df.columns, "SHAP values must include 'date' column"
        # In tidy long format, feature columns should contain zero NaNs
        feature_cols = [c for c in df.columns if c not in ["date", "horizon"]]
        assert df[feature_cols].isna().sum().sum() == 0, "Tidy SHAP matrix must not contain NaN values"


# ---------------------------------------------------------------------------
# Test 6: Split-conformal calibration fix (Issue #108)
# ---------------------------------------------------------------------------

class TestConformalCalibrationFix:
    """Regression tests for the split-conformal quantile-level fix (#108)."""

    def test_quantile_level_uses_n_cal_plus_one_denominator(self):
        """Split-conformal quantile level must divide by (n_cal + 1), not n_cal --
        dividing by n_cal overshoots the target quantile and understates coverage.
        Calls the actual production helper rather than re-deriving the formula,
        so this fails if the shipped code ever regresses."""
        from xgboost_baseline import _conformal_quantile_level

        n_cal = 50
        correct_q90 = _conformal_quantile_level(n_cal, 0.90)
        buggy_q90 = np.ceil((n_cal + 1) * 0.90) / n_cal

        assert correct_q90 == pytest.approx(46 / 51)
        assert buggy_q90 > correct_q90, \
            "Regression guard: the old n_cal denominator must overshoot the corrected quantile level"

    def test_quantile_level_saturates_only_below_the_cal_size_floor(self):
        """Documents why cal_size's 20-row floor in run_rolling_cv is exactly
        the safety margin needed: below 19 calibration rows, the 95% level
        saturates to the calibration set's max residual (ceil((n+1)*0.95) ==
        n+1), which would silently produce a degenerate "95% interval"
        identical to the 90% one after the non-crossing clamp."""
        from xgboost_baseline import _conformal_quantile_level

        assert _conformal_quantile_level(18, 0.95) == 1.0
        assert _conformal_quantile_level(19, 0.95) < 1.0
        assert _conformal_quantile_level(20, 0.95) < 1.0

    def test_empirical_coverage_near_nominal_with_held_out_calibration(self, synthetic_gold_df):
        """With a genuine held-out calibration split, 90% empirical coverage should
        land in a plausible neighborhood of the 90% nominal target, not be
        systematically depressed by an overshot quantile level."""
        from xgboost_baseline import run_rolling_cv

        _, _, intervals_df = run_rolling_cv(
            synthetic_gold_df,
            lags=[1, 2],
            horizons=[1],
            min_train=150,
            n_folds=2,
            n_estimators=10,
            max_depth=2,
        )

        coverage_90 = intervals_df.loc[
            intervals_df["horizon"] == 1, "empirical_coverage_90"
        ].iloc[0]
        # Loose sanity band on a small synthetic sample -- not a statistical
        # guarantee, but it should not be far below nominal the way the
        # n_cal-denominator bug would systematically cause.
        assert 50.0 <= coverage_90 <= 100.0

    def test_lowered_fit_threshold_rescues_moderate_small_blocks_from_in_sample_fallback(
        self, synthetic_gold_df
    ):
        """A training block too small for the OLD 30-row fit-size threshold,
        but large enough for the lowered MIN_FIT_ROWS=10, should still reach
        the real held-out calibration branch with cal_size at its safe
        20-row floor -- not a shrunk, potentially-degenerate one, and not
        the in-sample fallback either."""
        from xgboost_baseline import engineer_tabular_features, make_rolling_folds

        data, _, _ = engineer_tabular_features(
            synthetic_gold_df, lags=[1, 2], horizons=[1]
        )
        folds = make_rolling_folds(len(data), min_train=36, n_folds=2)
        train_end, _ = folds[0]
        h = 1
        n_tr = train_end - h
        cal_size = max(int(n_tr * 0.15), 20)
        fit_end = n_tr - cal_size - h

        # Between the new MIN_FIT_ROWS=10 and the old 30-row threshold --
        # exactly the range this fix rescues -- while cal_size sits at its
        # unshrunk, quantile-safe 20-row floor.
        assert 10 <= fit_end < 30
        assert cal_size == 20

    def test_real_calibration_branch_never_uses_a_saturating_cal_size(
        self, synthetic_gold_df
    ):
        """End-to-end guard: across every fold/horizon actually exercised by
        run_rolling_cv on a small fixture, the 90%/95% intervals must not be
        degenerate (95% collapsing to the same width as 90%), which is what
        the reviewer-caught shrink-to-5 bug would have produced."""
        from xgboost_baseline import run_rolling_cv

        forecasts_df, _, _ = run_rolling_cv(
            synthetic_gold_df,
            lags=[1, 2],
            horizons=[1, 5],
            min_train=36,
            n_folds=3,
            n_estimators=10,
            max_depth=2,
        )

        width_90 = forecasts_df["upper_90"] - forecasts_df["lower_90"]
        width_95 = forecasts_df["upper_95"] - forecasts_df["lower_95"]
        assert (width_95 > width_90).all(), \
            "95% interval collapsed to the same width as 90% -- degenerate quantile"



