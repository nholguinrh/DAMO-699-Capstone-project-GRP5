"""
Unit tests for the LSTM baseline shared module (Issue #49)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.lstm_baseline import (
    LEVEL_TARGET,
    TARGET,
    FEATURES,
    load_common_sample,
    make_windows,
    make_rolling_folds,
    run_rolling_cv,
    train_final_model,
    save_final_model,
    load_final_model,
    ShallowLSTM,
)


def _make_synthetic_df(n: int, seed: int = 0) -> pd.DataFrame:
    """
    Small synthetic common-sample DataFrame with the exact schema
    `load_common_sample()` produces (date, LEVEL_TARGET, FEATURES), but built
    directly rather than through the Gold-layer pipeline so tests stay fast
    and independent of real processed data.
    """
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


def test_load_common_sample_uses_synthetic_gold_inputs(synthetic_gold_inputs):
    df = load_common_sample(
        boc_path=synthetic_gold_inputs["boc_path"],
        fred_path=synthetic_gold_inputs["fred_path"],
        cpi_path=synthetic_gold_inputs["cpi_path"],
    )

    assert list(df.columns) == ["date", LEVEL_TARGET] + FEATURES
    assert not df[["date", LEVEL_TARGET] + FEATURES].isna().any().any()
    assert df["date"].is_monotonic_increasing


def test_make_windows_shapes_and_cumulative_change():
    df = _make_synthetic_df(60)
    horizons = [1, 3, 5]
    lookback = 10

    X, Y, origin_idx = make_windows(df, lookback=lookback, horizons=horizons)

    n_expected = len(df) - lookback - max(horizons)
    assert X.shape == (n_expected, lookback, len(FEATURES))
    assert Y.shape == (n_expected, len(horizons))
    assert len(origin_idx) == n_expected

    # Y[i, j] must equal the cumulative sum of TARGET over the next horizons[j]
    # days, i.e. origin_level + Y[i, j] == actual future level.
    levels = df[LEVEL_TARGET].to_numpy()
    for i, origin in enumerate(origin_idx[:5]):
        for h_idx, h in enumerate(horizons):
            expected_level = levels[origin + h]
            reconstructed = levels[origin] + Y[i, h_idx]
            assert reconstructed == pytest.approx(expected_level, abs=1e-4)

    # each window is exactly the `lookback` rows immediately preceding its origin
    feat = df[FEATURES].to_numpy()
    assert np.allclose(X[0], feat[origin_idx[0] - lookback:origin_idx[0]])


def test_make_rolling_folds_expanding_and_covers_full_range():
    n_windows = 200
    min_train = 100
    n_folds = 4

    folds = make_rolling_folds(n_windows, min_train=min_train, n_folds=n_folds)

    assert len(folds) == n_folds
    assert folds[0][0] == min_train
    assert folds[-1][1] == n_windows

    prev_test_end = None
    for train_end, test_end in folds:
        assert train_end < test_end
        if prev_test_end is not None:
            # folds are sequential and non-overlapping: this fold's train
            # boundary is exactly where the previous fold's test block ended
            assert train_end == prev_test_end
        prev_test_end = test_end


def test_shallow_lstm_forward_output_shape():
    model = ShallowLSTM(n_features=len(FEATURES), hidden_size=4, n_outputs=3)
    x = torch.randn(8, 10, len(FEATURES))
    out = model(x)
    assert out.shape == (8, 3)


def test_run_rolling_cv_smoke():
    df = _make_synthetic_df(220, seed=1)

    results_df, fold_diag_df = run_rolling_cv(
        df,
        lookback=5, horizons=[1, 2, 3],
        min_train=100, n_folds=2,
        hidden_size=4, max_epochs=3, patience=2, min_val_size=10,
    )

    assert len(fold_diag_df) == 2
    assert set(results_df.columns) == {"origin_date", "horizon", "actual", "naive", "lstm", "fold"}
    assert not results_df[["actual", "naive", "lstm"]].isna().any().any()
    assert np.isfinite(results_df[["actual", "naive", "lstm"]].to_numpy()).all()
    # naive forecast must equal the last observed level at each origin, exactly
    assert (results_df["naive"] == results_df.groupby("origin_date")["naive"].transform("first")).all()


def test_make_rolling_folds_rejects_undersized_test_pool():
    # test_pool=10, n_folds=20 -> fold_size truncates to 0 via integer division;
    # every fold but the last would silently get zero test rows (bug caught in
    # review on PR #58) instead of raising.
    with pytest.raises(ValueError, match="n_folds"):
        make_rolling_folds(n_windows=110, min_train=100, n_folds=20)


def test_run_rolling_cv_rejects_min_val_size_that_empties_training_set():
    df = _make_synthetic_df(220, seed=1)

    # min_val_size=100 on a fold with train_end=100 leaves tr_end<=0, which used
    # to propagate as NaN into training instead of failing immediately.
    with pytest.raises(ValueError, match="min_val_size"):
        run_rolling_cv(
            df,
            lookback=5, horizons=[1, 2, 3],
            min_train=100, n_folds=2,
            hidden_size=4, max_epochs=3, patience=2, min_val_size=100,
        )


def test_train_with_early_stopping_raises_on_persistent_nan_val_loss():
    # A model that always outputs NaN (forced via a NaN-filled validation set)
    # never clears the `val_loss < best_val` bar, so best_state would stay None;
    # this must raise a diagnosable RuntimeError instead of crashing inside
    # load_state_dict(None) (bug caught in review on PR #58).
    n_features, n_outputs = len(FEATURES), 3
    Xtr = np.random.default_rng(0).normal(size=(20, 5, n_features)).astype(np.float32)
    Ytr = np.random.default_rng(1).normal(size=(20, n_outputs)).astype(np.float32)
    Xval = Xtr[:5].copy()
    Yval = np.full((5, n_outputs), np.nan, dtype=np.float32)

    from src.lstm_baseline import _train_with_early_stopping

    with pytest.raises(RuntimeError, match="no valid checkpoint"):
        _train_with_early_stopping(
            Xtr, Ytr, Xval, Yval, n_features=n_features, seed=0,
            hidden_size=4, max_epochs=3, patience=2,
        )


def test_load_final_model_rejects_mismatched_feature_set(tmp_path, monkeypatch):
    df = _make_synthetic_df(150, seed=2)

    model, scaling, X, origin_idx, diag = train_final_model(
        df, val_frac=0.2,
        lookback=5, horizons=[1, 2, 3],
        hidden_size=4, max_epochs=3, patience=2,
    )

    model_path = tmp_path / "lstm_test_model.pt"
    save_final_model(model, scaling, model_path)

    import src.lstm_baseline as lstm_baseline_module
    monkeypatch.setattr(lstm_baseline_module, "FEATURES", FEATURES[:-1] + ["some_other_feature"])

    with pytest.raises(RuntimeError, match="was trained on features"):
        load_final_model(model_path, n_features=len(FEATURES), hidden_size=4, n_outputs=3)


def test_train_final_model_save_and_load_roundtrip(tmp_path):
    df = _make_synthetic_df(150, seed=2)

    model, scaling, X, origin_idx, diag = train_final_model(
        df, val_frac=0.2,
        lookback=5, horizons=[1, 2, 3],
        hidden_size=4, max_epochs=3, patience=2,
    )

    model_path = tmp_path / "lstm_test_model.pt"
    save_final_model(model, scaling, model_path)
    assert model_path.exists()

    loaded_model, loaded_scaling = load_final_model(
        model_path, n_features=len(FEATURES), hidden_size=4, n_outputs=3
    )

    sample = torch.from_numpy(((X[:4] - scaling["x_mean"]) / scaling["x_std"]).astype(np.float32))
    with torch.no_grad():
        original_pred = model(sample).numpy()
        loaded_pred = loaded_model(sample).numpy()

    assert np.allclose(original_pred, loaded_pred)
    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        assert np.allclose(scaling[key], loaded_scaling[key])
