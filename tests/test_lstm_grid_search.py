"""
Unit tests for the LSTM hyperparameter grid search module (Issue #90)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.lstm_baseline import (
    FEATURES,
    LEVEL_TARGET,
    TARGET,
)
from src.lstm_grid_search import (
    CANONICAL_CONFIG,
    compare_best_to_canonical,
    evaluate_cv_config,
    export_grid_results,
    run_grid_search,
    update_downstream_outputs,
)


def _make_synthetic_df(n: int = 220, seed: int = 42) -> pd.DataFrame:
    """Synthetic common-sample DataFrame for fast, isolated testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)

    level = 1.0 + np.cumsum(rng.normal(0, 0.02, size=n))
    level[0] = 1.0
    d_level = np.diff(level, prepend=level[0])
    d_level[0] = 0.0

    data = {"date": dates, LEVEL_TARGET: level, TARGET: d_level}
    for col in FEATURES:
        if col == TARGET:
            continue
        data[col] = rng.normal(0, 0.05, size=n)

    return pd.DataFrame(data)[["date", LEVEL_TARGET] + FEATURES]


def test_evaluate_cv_config_smoke():
    df = _make_synthetic_df(150, seed=1)

    result = evaluate_cv_config(
        df=df,
        hidden_size=4,
        dropout=0.1,
        lookback=5,
        learning_rate=1e-3,
        horizons=[1, 2],
        min_train=60,
        n_folds=2,
        max_epochs=3,
        patience=2,
        min_val_size=10,
    )

    assert result["hidden_size"] == 4
    assert result["dropout"] == 0.1
    assert result["lookback"] == 5
    assert result["learning_rate"] == 1e-3
    assert "fold_0_val_mse" in result
    assert "fold_1_val_mse" in result
    assert np.isfinite(result["mean_val_mse"])
    assert result["mean_val_mse"] > 0
    assert result["mean_epochs"] > 0


def test_run_grid_search_small_space():
    df = _make_synthetic_df(150, seed=2)

    small_space = {
        "hidden_size": [4],
        "dropout": [0.1, 0.2],
        "lookback": [5],
        "learning_rate": [1e-3],
    }

    results_df = run_grid_search(
        df=df,
        search_space=small_space,
        min_train=60,
        n_folds=2,
        max_epochs=2,
        patience=2,
        verbose=False,
    )

    assert len(results_df) == 2
    assert "mean_val_mse" in results_df.columns
    assert "rank" in results_df.columns
    assert list(results_df["rank"]) == [1, 2]
    # Verify sorted in ascending order of MSE
    assert results_df["mean_val_mse"].iloc[0] <= results_df["mean_val_mse"].iloc[1]


def test_export_grid_results_and_compare(tmp_path):
    # Construct synthetic results DataFrame containing the canonical config
    mock_results = pd.DataFrame([
        {
            "hidden_size": 16,
            "dropout": 0.2,
            "lookback": 20,
            "learning_rate": 1e-3,
            "fold_0_val_mse": 0.50,
            "fold_1_val_mse": 0.52,
            "mean_val_mse": 0.51,
            "std_val_mse": 0.01,
            "mean_epochs": 15.0,
            "rank": 2,
            "is_canonical": True,
        },
        {
            "hidden_size": 32,
            "dropout": 0.1,
            "lookback": 10,
            "learning_rate": 1e-3,
            "fold_0_val_mse": 0.45,
            "fold_1_val_mse": 0.47,
            "mean_val_mse": 0.46,
            "std_val_mse": 0.01,
            "mean_epochs": 20.0,
            "rank": 1,
            "is_canonical": False,
        },
    ]).sort_values("rank").reset_index(drop=True)

    out_file = tmp_path / "test_grid_results.csv"
    saved_path = export_grid_results(mock_results, out_file)
    assert saved_path.exists()

    loaded = pd.read_csv(saved_path)
    assert len(loaded) == 2

    summary = compare_best_to_canonical(mock_results)
    assert summary["is_new_best"] is True
    assert summary["best_config"]["hidden_size"] == 32
    assert summary["canonical_config"]["hidden_size"] == 16
    assert summary["improvement_pct"] > 0


def test_update_downstream_outputs_smoke(tmp_path):
    df = _make_synthetic_df(150, seed=3)

    best_config = {
        "hidden_size": 4,
        "dropout": 0.1,
        "lookback": 5,
        "learning_rate": 1e-3,
    }

    generated = update_downstream_outputs(
        best_config=best_config,
        df=df,
        output_dir=tmp_path,
        horizons=[1, 2],
        min_train=60,
        n_folds=2,
        max_epochs=2,
        patience=2,
        min_val_size=10,
    )

    expected_keys = {"final_model", "forecasts", "cv_folds", "metrics_vs_naive", "dm_results"}
    assert set(generated.keys()) == expected_keys
    for path in generated.values():
        assert path.exists()
