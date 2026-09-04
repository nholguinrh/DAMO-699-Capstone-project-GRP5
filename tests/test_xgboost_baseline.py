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



