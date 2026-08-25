"""
LSTM Hyperparameter Tuning via Grid Search (Issue #90)
DAMO-699 Capstone Project, Group 5

Runs a 36-configuration grid search for ShallowLSTM over:
- hidden_size: [8, 16, 32]
- dropout: [0.1, 0.2, 0.3]
- lookback: [10, 20]
- learning_rate: [1e-3, 5e-4]

Total: 36 configurations x 5 expanding-window CV folds = 180 training runs.

Preserves time-series validation order without data leakage.
Exports results to `outputs/lstm_hyperparameter_grid_results.csv`.
If a new optimal configuration is identified, updates `outputs/lstm_final_model.pt`
and regenerates downstream forecast and evaluation outputs.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dieboldmariano import (
    InvalidParameterException,
    NegativeVarianceException,
    dm_test,
)

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.lstm_baseline import (
    DROPOUT,
    FEATURES,
    HIDDEN_SIZE,
    HORIZONS,
    LEVEL_TARGET,
    LOOKBACK,
    LR,
    MAX_EPOCHS,
    MIN_TRAIN,
    N_FOLDS,
    PATIENCE,
    SEED,
    _normalize_and_train,
    load_common_sample,
    make_rolling_folds,
    make_windows,
    run_rolling_cv,
    save_final_model,
    train_final_model,
)
from src.project_paths import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"

GRID_SEARCH_SPACE: Dict[str, List[Any]] = {
    "hidden_size": [8, 16, 32],
    "dropout": [0.1, 0.2, 0.3],
    "lookback": [10, 20],
    "learning_rate": [1e-3, 5e-4],
}

# The canonical baseline configuration prior to tuning (Issue #49)
CANONICAL_CONFIG: Dict[str, Any] = {
    "hidden_size": 16,
    "dropout": 0.2,
    "lookback": 20,
    "learning_rate": 1e-3,
}


def _safe_dm_test(
    actual: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    h: int,
    loss: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Guarded Diebold-Mariano test wrapper preventing unhandled exceptions
    on degenerate variance or undersized slices (matching model_comparison.py).
    """
    n = len(actual)
    if n <= h:
        return np.nan, np.nan
    try:
        kwargs: Dict[str, Any] = {
            "h": h,
            "one_sided": False,
            "harvey_correction": True,
            "variance_estimator": "bartlett",
        }
        if loss is not None:
            kwargs["loss"] = loss
        stat, p_val = dm_test(actual, pred_a, pred_b, **kwargs)
        return float(stat), float(p_val)
    except (InvalidParameterException, NegativeVarianceException, Exception):
        return np.nan, np.nan


def evaluate_cv_config(
    df: pd.DataFrame,
    hidden_size: int,
    dropout: float,
    lookback: int,
    learning_rate: float,
    horizons: List[int] = HORIZONS,
    min_train: int = MIN_TRAIN,
    n_folds: int = N_FOLDS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_val_size: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate a single hyperparameter configuration across K expanding-window CV folds.
    Returns validation MSE per fold, mean validation MSE, and average epochs run.
    """
    X, Y, origin_idx = make_windows(df, lookback=lookback, horizons=horizons)
    folds = make_rolling_folds(len(X), min_train=min_train, n_folds=n_folds)

    fold_val_losses: List[float] = []
    epochs_list: List[int] = []

    for fold_id, (train_end, test_end) in enumerate(folds):
        val_size = max(int(train_end * 0.15), min_val_size)
        tr_end = train_end - val_size
        if tr_end <= 0:
            raise ValueError(
                f"Fold {fold_id}: min_val_size={min_val_size} leaves no training data "
                f"(train_end={train_end}, val_size={val_size})."
            )

        Xtr, Ytr = X[:tr_end], Y[:tr_end]
        Xval, Yval = X[tr_end:train_end], Y[tr_end:train_end]

        model, scaling, best_val, epochs_run = _normalize_and_train(
            Xtr, Ytr, Xval, Yval,
            n_features=len(FEATURES),
            seed=SEED + fold_id,
            hidden_size=hidden_size,
            dropout=dropout,
            lr=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
        )

        fold_val_losses.append(float(best_val))
        epochs_list.append(epochs_run)

    result: Dict[str, Any] = {
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "lookback": int(lookback),
        "learning_rate": float(learning_rate),
    }

    for i, loss in enumerate(fold_val_losses):
        result[f"fold_{i}_val_mse"] = loss

    result["mean_val_mse"] = float(np.mean(fold_val_losses))
    result["std_val_mse"] = float(np.std(fold_val_losses))
    result["mean_epochs"] = float(np.mean(epochs_list))

    return result


def run_grid_search(
    df: pd.DataFrame,
    search_space: Optional[Dict[str, List[Any]]] = None,
    min_train: int = MIN_TRAIN,
    n_folds: int = N_FOLDS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full Cartesian product grid search over the specified search space.
    Returns a DataFrame of all configurations ranked by mean_val_mse.
    """
    if search_space is None:
        search_space = GRID_SEARCH_SPACE

    param_keys = ["hidden_size", "dropout", "lookback", "learning_rate"]
    param_values = [search_space[k] for k in param_keys]
    combinations = list(itertools.product(*param_values))
    total_combos = len(combinations)

    if verbose:
        print(f"Starting LSTM Grid Search: {total_combos} combinations x {n_folds} folds = {total_combos * n_folds} runs")
        print("=" * 80)

    start_time = time.time()
    results: List[Dict[str, Any]] = []

    for idx, (hs, dp, lb, lr) in enumerate(combinations, start=1):
        t0 = time.time()
        res = evaluate_cv_config(
            df=df,
            hidden_size=hs,
            dropout=dp,
            lookback=lb,
            learning_rate=lr,
            min_train=min_train,
            n_folds=n_folds,
            max_epochs=max_epochs,
            patience=patience,
        )
        t_elapsed = time.time() - t0
        results.append(res)

        if verbose:
            total_elapsed = time.time() - start_time
            avg_per_combo = total_elapsed / idx
            remaining_combos = total_combos - idx
            est_remaining_sec = avg_per_combo * remaining_combos
            print(
                f"[{idx:02d}/{total_combos:02d}] hs={hs:2d} | drop={dp:.2f} | lb={lb:2d} | lr={lr:.4f} "
                f"-> Mean Val MSE: {res['mean_val_mse']:.5f} (±{res['std_val_mse']:.5f}, ~{res['mean_epochs']:.1f} eps) "
                f"[{t_elapsed:.1f}s | ETA: {est_remaining_sec/60:.1f} min]"
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_val_mse", ascending=True).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)

    results_df["is_canonical"] = (
        (results_df["hidden_size"] == CANONICAL_CONFIG["hidden_size"])
        & (np.isclose(results_df["dropout"], CANONICAL_CONFIG["dropout"]))
        & (results_df["lookback"] == CANONICAL_CONFIG["lookback"])
        & (np.isclose(results_df["learning_rate"], CANONICAL_CONFIG["learning_rate"]))
    )

    return results_df


def export_grid_results(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Save grid search results table to CSV."""
    if output_path is None:
        output_path = OUTPUT_DIR / "lstm_hyperparameter_grid_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return output_path


def compare_best_to_canonical(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare the top-ranked configuration against the canonical baseline.
    """
    best_row = results_df.iloc[0]
    canonical_match = results_df[results_df["is_canonical"]]

    if canonical_match.empty:
        raise ValueError("Canonical baseline configuration not found in grid results.")

    canonical_row = canonical_match.iloc[0]

    best_mean_mse = float(best_row["mean_val_mse"])
    canonical_mean_mse = float(canonical_row["mean_val_mse"])
    improvement_pct = (canonical_mean_mse - best_mean_mse) / canonical_mean_mse * 100.0

    is_new_best = bool(best_row["rank"] == 1 and not best_row["is_canonical"])

    summary = {
        "best_config": {
            "hidden_size": int(best_row["hidden_size"]),
            "dropout": float(best_row["dropout"]),
            "lookback": int(best_row["lookback"]),
            "learning_rate": float(best_row["learning_rate"]),
            "mean_val_mse": best_mean_mse,
            "std_val_mse": float(best_row["std_val_mse"]),
            "rank": int(best_row["rank"]),
        },
        "canonical_config": {
            "hidden_size": int(canonical_row["hidden_size"]),
            "dropout": float(canonical_row["dropout"]),
            "lookback": int(canonical_row["lookback"]),
            "learning_rate": float(canonical_row["learning_rate"]),
            "mean_val_mse": canonical_mean_mse,
            "std_val_mse": float(canonical_row["std_val_mse"]),
            "rank": int(canonical_row["rank"]),
        },
        "improvement_pct": improvement_pct,
        "is_new_best": is_new_best,
    }

    return summary


def update_downstream_outputs(
    best_config: Dict[str, Any],
    df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    horizons: List[int] = HORIZONS,
    min_train: int = MIN_TRAIN,
    n_folds: int = N_FOLDS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_val_size: int = 50,
) -> Dict[str, Path]:
    """
    Regenerate all downstream LSTM model artifacts and forecasts using the best configuration:
    1. lstm_final_model.pt
    2. r3_lstm_forecasts.csv
    3. r3_lstm_cv_folds.csv
    4. r3_lstm_vs_naive.csv
    5. r3_lstm_diebold_mariano.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, Path] = {}

    hs = int(best_config["hidden_size"])
    dp = float(best_config["dropout"])
    lb = int(best_config["lookback"])
    lr = float(best_config["learning_rate"])

    print(f"\nRegenerating downstream outputs with best config: hs={hs}, dp={dp}, lb={lb}, lr={lr}")

    # 1. Final Model on full sample
    final_model, scaling, X_full, origin_idx_full, final_diag = train_final_model(
        df=df,
        lookback=lb,
        horizons=horizons,
        hidden_size=hs,
        dropout=dp,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
    )
    final_model_path = output_dir / "lstm_final_model.pt"
    save_final_model(final_model, scaling, final_model_path)
    generated_files["final_model"] = final_model_path
    print(f"  - Saved {final_model_path.name} (val_loss: {final_diag['best_val_loss']:.4f}, epochs: {final_diag['epochs_run']})")

    # 2. Rolling CV forecasts
    results_df, fold_diag_df = run_rolling_cv(
        df=df,
        lookback=lb,
        horizons=horizons,
        min_train=min_train,
        n_folds=n_folds,
        hidden_size=hs,
        dropout=dp,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        min_val_size=min_val_size,
    )
    forecasts_path = output_dir / "r3_lstm_forecasts.csv"
    results_df.to_csv(forecasts_path, index=False)
    generated_files["forecasts"] = forecasts_path
    print(f"  - Saved {forecasts_path.name} ({len(results_df)} forecast rows)")

    cv_folds_path = output_dir / "r3_lstm_cv_folds.csv"
    fold_diag_df.to_csv(cv_folds_path, index=False)
    generated_files["cv_folds"] = cv_folds_path
    print(f"  - Saved {cv_folds_path.name} ({len(fold_diag_df)} folds)")

    # 3. Metrics vs Naive (RMSE / MAE)
    metrics = []
    for h in horizons:
        subset = results_df[results_df["horizon"] == h]
        row = {"horizon": h}
        for name in ["lstm", "naive"]:
            row[f"{name}_rmse"] = np.sqrt(mean_squared_error(subset["actual"], subset[name]))
            row[f"{name}_mae"] = mean_absolute_error(subset["actual"], subset[name])
        row["lstm_rmse_improvement_pct"] = (row["naive_rmse"] - row["lstm_rmse"]) / row["naive_rmse"] * 100
        row["lstm_mae_improvement_pct"] = (row["naive_mae"] - row["lstm_mae"]) / row["naive_mae"] * 100
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics).round(6)
    metrics_path = output_dir / "r3_lstm_vs_naive.csv"
    metrics_df.to_csv(metrics_path, index=False)
    generated_files["metrics_vs_naive"] = metrics_path
    print(f"  - Saved {metrics_path.name}")

    # 4. Diebold-Mariano Tests
    ALPHA = 0.05
    dm_results = []
    for h in horizons:
        subset = results_df[results_df["horizon"] == h]
        actual = subset["actual"].to_numpy()
        naive = subset["naive"].to_numpy()
        lstm = subset["lstm"].to_numpy()

        dm_stat_sq, p_sq = _safe_dm_test(
            actual, lstm, naive, h=h,
        )
        dm_stat_abs, p_abs = _safe_dm_test(
            actual, lstm, naive, h=h, loss=lambda u, v: abs(u - v),
        )

        lstm_better_rmse = bool(not np.isnan(p_sq) and p_sq < ALPHA and dm_stat_sq < 0)
        naive_better_rmse = bool(not np.isnan(p_sq) and p_sq < ALPHA and dm_stat_sq > 0)
        lstm_better_mae = bool(not np.isnan(p_abs) and p_abs < ALPHA and dm_stat_abs < 0)
        naive_better_mae = bool(not np.isnan(p_abs) and p_abs < ALPHA and dm_stat_abs > 0)

        dm_results.append({
            "model": "lstm",
            "horizon_days": h,
            "dm_stat_squared_loss": dm_stat_sq,
            "dm_p_value_squared_loss": p_sq,
            "dm_stat_absolute_loss": dm_stat_abs,
            "dm_p_value_absolute_loss": p_abs,
            "lstm_significantly_better_rmse": lstm_better_rmse,
            "naive_significantly_better_rmse": naive_better_rmse,
            "lstm_significantly_better_mae": lstm_better_mae,
            "naive_significantly_better_mae": naive_better_mae,
            "lstm_significantly_better": bool(lstm_better_rmse and lstm_better_mae),
            "naive_significantly_better": bool(naive_better_rmse and naive_better_mae),
        })

    dm_results_df = pd.DataFrame(dm_results).round(4)
    dm_path = output_dir / "r3_lstm_diebold_mariano.csv"
    dm_results_df.to_csv(dm_path, index=False)
    generated_files["dm_results"] = dm_path
    print(f"  - Saved {dm_path.name}")

    return generated_files


def main() -> None:
    print("Loading common sample for LSTM hyperparameter tuning...")
    df = load_common_sample()
    print(f"Loaded common sample: {len(df)} rows, features: {FEATURES}")

    results_df = run_grid_search(df=df)
    csv_path = export_grid_results(results_df)
    print(f"\nSaved grid search results to {csv_path.relative_to(PROJECT_ROOT)}")

    summary = compare_best_to_canonical(results_df)
    best = summary["best_config"]
    canon = summary["canonical_config"]

    print("\n" + "=" * 80)
    print("GRID SEARCH SUMMARY & COMPARISON")
    print("=" * 80)
    print(f"Top-Ranked Config (Rank #{best['rank']}):")
    print(f"  hidden_size   : {best['hidden_size']}")
    print(f"  dropout       : {best['dropout']}")
    print(f"  lookback      : {best['lookback']}")
    print(f"  learning_rate : {best['learning_rate']}")
    print(f"  Mean Val MSE  : {best['mean_val_mse']:.6f} (±{best['std_val_mse']:.6f}) [CV early-stopping holdout]")

    print(f"\nCanonical Baseline Config (Rank #{canon['rank']}):")
    print(f"  hidden_size   : {canon['hidden_size']}")
    print(f"  dropout       : {canon['dropout']}")
    print(f"  lookback      : {canon['lookback']}")
    print(f"  learning_rate : {canon['learning_rate']}")
    print(f"  Mean Val MSE  : {canon['mean_val_mse']:.6f} (±{canon['std_val_mse']:.6f}) [CV early-stopping holdout]")

    print(f"\nRelative Validation MSE Improvement over Canonical: {summary['improvement_pct']:+.2f}%")
    print("NOTE: Metric is based on the 15% inner validation slice of training folds for early stopping.")

    if summary["is_new_best"]:
        print("\n>>> NEW OPTIMAL CONFIGURATION FOUND! Updating downstream model and forecast outputs...")
        update_downstream_outputs(best_config=best, df=df)
    else:
        print("\n>>> Canonical baseline remains optimal (Rank #1) or ties. Downstream models already aligned.")


if __name__ == "__main__":
    main()
