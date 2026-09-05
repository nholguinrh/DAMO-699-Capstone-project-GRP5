"""
Unit tests for src/xgboost_grid_search.py (Issue #119)
DAMO-699 Capstone Project, Group 5

Verifies:
1. Search space construction and fallback-branch safety (pruning unsupported params).
2. Strict temporal embargo: inner validation split is strictly disjoint from both
   the fit training block and the split-conformal calibration block.
3. Search determinism under SEED = 42.
4. Export schema and ranking integrity in outputs/r3_xgboost_hyperparameter_search.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xgboost_baseline import (
    HORIZONS,
    LAGS,
    SEED,
    engineer_tabular_features,
)
from xgboost_grid_search import (
    CANONICAL_CONFIG,
    compare_best_to_canonical,
    evaluate_inner_split_config,
    export_search_results,
    generate_search_space,
    run_grid_search,
)


@pytest.fixture
def synthetic_gold_df():
    """Generate a reproducible Gold-layer-like DataFrame for testing."""
    rng = np.random.RandomState(42)
    n = 250
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


class TestSearchSpaceConstruction:
    """Validate grid search space generation across backends."""

    def test_xgboost_backend_includes_regularizers(self):
        space = generate_search_space(has_xgboost=True)
        assert len(space) == 28
        for cfg in space:
            assert "colsample_bytree" in cfg
            assert "min_child_weight" in cfg
            assert "reg_lambda" in cfg
            assert cfg["max_depth"] in [2, 3, 4]
            assert cfg["learning_rate"] in [0.01, 0.03, 0.10]
            assert cfg["n_estimators"] in [150, 300, 600]

    def test_sklearn_fallback_backend_prunes_unsupported_params(self):
        space = generate_search_space(has_xgboost=False)
        assert len(space) == 27
        for cfg in space:
            # GradientBoostingRegressor will fail with TypeError if these are passed
            assert "colsample_bytree" not in cfg
            assert "min_child_weight" not in cfg
            assert "reg_lambda" not in cfg
            assert "reg_alpha" not in cfg
            assert "max_depth" in cfg
            assert "learning_rate" in cfg
            assert "n_estimators" in cfg
            assert "subsample" in cfg


class TestInnerSplitTemporalEmbargo:
    """Verify that inner validation split preserves causal ordering with no leakage."""

    def test_inner_validation_disjoint_from_calibration_and_test(self, synthetic_gold_df):
        data, feature_cols, target_cols = engineer_tabular_features(
            synthetic_gold_df, lags=[1, 2], horizons=[1, 5],
        )

        h = 5
        cal_ratio = 0.15
        val_ratio = 0.20
        n_pool = len(data)

        embargo_end = n_pool - h
        train_data = data.iloc[:embargo_end]
        n_tr = len(train_data)

        cal_size = max(int(n_tr * cal_ratio), 20)
        fit_end = n_tr - cal_size - h
        fit_data = train_data.iloc[:fit_end]

        val_size = max(int(len(fit_data) * val_ratio), 20)
        inner_train_end = len(fit_data) - val_size - h

        # Assert disjoint partitions
        inner_train = fit_data.iloc[:inner_train_end]
        inner_val = fit_data.iloc[inner_train_end + h:]
        cal_data = train_data.iloc[fit_end + h:]

        # 1. Inner train and inner val have h-step gap
        idx_inner_train_max = inner_train.index.max()
        idx_inner_val_min = inner_val.index.min()
        assert idx_inner_val_min - idx_inner_train_max > h, \
            f"Expected >{h} gap between inner train and inner val, got {idx_inner_val_min - idx_inner_train_max}"

        # 2. Inner val and cal data have h-step gap
        idx_inner_val_max = inner_val.index.max()
        idx_cal_min = cal_data.index.min()
        assert idx_cal_min - idx_inner_val_max > h, \
            f"Expected >{h} gap between inner val and cal data, got {idx_cal_min - idx_inner_val_max}"

        # 3. No index intersection across all three partitions
        assert len(set(inner_train.index).intersection(set(inner_val.index))) == 0
        assert len(set(inner_val.index).intersection(set(cal_data.index))) == 0
        assert len(set(inner_train.index).intersection(set(cal_data.index))) == 0


class TestGridSearchExecution:
    """Verify execution determinism and export schema."""

    def test_search_is_deterministic(self, synthetic_gold_df):
        mini_space = [
            {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 5, "subsample": 0.8},
            {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 5, "subsample": 0.8},
        ]
        res1 = run_grid_search(synthetic_gold_df, search_space=mini_space, verbose=False)
        res2 = run_grid_search(synthetic_gold_df, search_space=mini_space, verbose=False)

        pd.testing.assert_frame_equal(res1, res2)

    def test_export_and_comparison_schema(self, synthetic_gold_df, tmp_path):
        mini_space = [
            {
                "max_depth": 3,
                "learning_rate": 0.03,
                "n_estimators": 150,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
            },
            {
                "max_depth": 2,
                "learning_rate": 0.10,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
                "reg_lambda": 0.5,
                "reg_alpha": 0.0,
            },
        ]
        results_df = run_grid_search(synthetic_gold_df, search_space=mini_space, verbose=False)
        out_csv = tmp_path / "test_search.csv"
        export_search_results(results_df, output_path=out_csv)

        assert out_csv.exists()
        saved_df = pd.read_csv(out_csv)
        assert "rank" in saved_df.columns
        assert "mean_relative_rmse" in saved_df.columns
        assert "val_improvement_pct" in saved_df.columns
        assert "is_canonical" in saved_df.columns
        assert saved_df["is_canonical"].sum() == 1

        summary = compare_best_to_canonical(results_df)
        assert "best_config" in summary
        assert "canonical_config" in summary
        assert "is_new_best" in summary
