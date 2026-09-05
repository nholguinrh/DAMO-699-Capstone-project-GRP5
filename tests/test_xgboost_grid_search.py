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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.xgboost_baseline import (
        HORIZONS,
        LAGS,
        SEED,
        engineer_tabular_features,
    )
    from src.xgboost_grid_search import (
        CANONICAL_CONFIG,
        OUTPUTS_DIR,
        compare_best_to_canonical,
        evaluate_inner_split_config,
        export_search_results,
        generate_search_space,
        run_grid_search,
    )
except ImportError:
    from xgboost_baseline import (  # type: ignore
        HORIZONS,
        LAGS,
        SEED,
        engineer_tabular_features,
    )
    from xgboost_grid_search import (  # type: ignore
        CANONICAL_CONFIG,
        OUTPUTS_DIR,
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

    def test_production_split_preserves_embargo(self, synthetic_gold_df):
        """
        Capture the partitions the SHIPPED function actually builds and assert the
        embargo on those -- not on a copy of the arithmetic maintained in the test (H-1).
        """
        data, feature_cols, target_cols = engineer_tabular_features(
            synthetic_gold_df, lags=[1, 2], horizons=[1, 5],
        )

        rec, partitions = evaluate_inner_split_config(
            data=data,
            feature_cols=feature_cols,
            target_cols=target_cols,
            config={"max_depth": 2, "learning_rate": 0.1, "n_estimators": 5},
            horizons=[5],
            selection_end_index=len(data),
        )
        assert len(partitions) == 1
        p = partitions[0]
        h = 5

        # 1. Inner train and inner val have causal h-step gap
        assert p["inner_val_start_idx"] - p["inner_train_end_idx"] >= h, "embargo collapsed"
        # 2. Inner val and calibration have causal h-step gap
        assert p["cal_start_idx"] - p["inner_val_end_idx"] >= h, "val/calibration overlap"
        # 3. Selection window ends before or at first scored origin
        assert p["first_scored_origin_idx"] >= p["cal_end_idx"], "selection window reaches into scored origins"

    def test_undersized_pool_raises_value_error_without_leaky_fallback(self, synthetic_gold_df):
        """Verify M-1: no silent fallbacks -- an undersized pool raises ValueError."""
        data, feature_cols, target_cols = engineer_tabular_features(
            synthetic_gold_df, lags=[1, 2], horizons=[1],
        )
        tiny_data = data.iloc[:40]
        with pytest.raises(ValueError, match="too small for a leakage-safe embargoed inner split"):
            evaluate_inner_split_config(
                data=tiny_data,
                feature_cols=feature_cols,
                target_cols=target_cols,
                config={"max_depth": 2, "learning_rate": 0.1, "n_estimators": 5},
                horizons=[1],
                selection_end_index=40,
            )


class TestGridSearchExecution:
    """Verify execution determinism and export schema."""

    def test_search_is_deterministic(self, synthetic_gold_df):
        mini_space = [
            {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 5, "subsample": 0.8},
            {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 5, "subsample": 0.8},
        ]
        res1, _ = run_grid_search(synthetic_gold_df, search_space=mini_space, seeds=(42,), verbose=False)
        res2, _ = run_grid_search(synthetic_gold_df, search_space=mini_space, seeds=(42,), verbose=False)

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
        results_df, partitions_df = run_grid_search(synthetic_gold_df, search_space=mini_space, seeds=(42,), verbose=False)
        out_csv = tmp_path / "test_search.csv"
        part_csv = tmp_path / "test_partitions.csv"
        export_search_results(results_df, partitions_df=partitions_df, output_path=out_csv, partitions_path=part_csv)

        assert out_csv.exists()
        assert part_csv.exists()
        saved_df = pd.read_csv(out_csv)
        assert "rank" in saved_df.columns
        assert "mean_relative_rmse" in saved_df.columns
        assert "std_relative_rmse" in saved_df.columns
        assert "n_selection_seeds" in saved_df.columns
        assert "val_improvement_pct" in saved_df.columns
        assert "is_canonical" in saved_df.columns
        assert saved_df["is_canonical"].sum() == 1

        saved_parts = pd.read_csv(part_csv)
        assert "inner_train_rows" in saved_parts.columns
        assert "inner_val_rows" in saved_parts.columns
        assert "cal_rows" in saved_parts.columns

        summary = compare_best_to_canonical(results_df)
        assert "best_config" in summary
        assert "canonical_config" in summary
        assert "is_new_best" in summary

    def test_search_csv_rows_reconcile_internally(self):
        """Every published row's per-horizon ratios must average to its published
        mean_relative_rmse. A mismatch means the row mixes single-seed detail with
        multi-seed aggregates, and the published ranking cannot be re-derived."""
        csv_path = OUTPUTS_DIR / "r3_xgboost_hyperparameter_search.csv"
        if not csv_path.exists():
            pytest.skip("r3_xgboost_hyperparameter_search.csv not present yet.")
        d = pd.read_csv(csv_path)
        per_h = d[[f"rel_rmse_h{h}" for h in (1, 5, 20)]].mean(axis=1)
        assert np.allclose(per_h, d["mean_relative_rmse"], atol=1e-4)

    def test_rank_is_monotonic_in_objective_outside_the_indifference_band(self):
        """Parsimony re-ordering applies only inside the 1-sd band. Every configuration
        outside it must be ranked by the selection objective, or `rank` stops meaning
        what §4.3 and the CSV header claim it means."""
        csv_path = OUTPUTS_DIR / "r3_xgboost_hyperparameter_search.csv"
        if not csv_path.exists():
            pytest.skip("r3_xgboost_hyperparameter_search.csv not present yet.")
        d = pd.read_csv(csv_path)
        outside = d[~d["within_1sd_of_leader"]].sort_values("rank")
        assert outside["mean_relative_rmse"].is_monotonic_increasing
        # The band leader must still be the global objective minimum.
        assert np.isclose(
            d["mean_relative_rmse"].min(),
            d.loc[d["within_1sd_of_leader"], "mean_relative_rmse"].min(),
        )
