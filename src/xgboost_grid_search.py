"""
XGBoost Hyperparameter Tuning via Leakage-Safe Search (Issue #119)
DAMO-699 Capstone Project, Group 5

Executes a compact, reproducible, leakage-safe hyperparameter search for the
XGBoost tabular benchmark.

Search Design:
1. Selection window: Hyperparameters are selected strictly ONCE on historical
   burn-in data ending at index MIN_TRAIN = 500 (pre-2012-02-17), strictly
   before the very first scored rolling-CV test origin (index 500).
   This guarantees that 0% of evaluated test origins are touched during tuning.
2. Leakage-safe partitioning: The inner validation split is carved strictly
   out of the fit partition, preserving an h-step causal embargo gap
   and remaining completely disjoint from the split-conformal calibration block.
3. Fallback guard: The candidate search space conditionally adapts when
   XGBoost is unavailable, ensuring full compatibility with sklearn's
   GradientBoostingRegressor on the fallback branch.
4. Scale-standardized objective: Computes both raw RMSE and Naive-relative
   standardized RMSE across horizons h in {1, 5, 20} to prevent h=20 error
   scale dominance.
5. Multi-seed robustness: Evaluates configurations across SELECTION_SEEDS
   (42, 0, 1, 7, 2024) to report objective dispersion and avoid seed artifacts.

Exports results to `outputs/r3_xgboost_hyperparameter_search.csv`
and partition metadata to `outputs/r3_xgboost_search_partitions.csv`.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure src directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.project_paths import OUTPUTS_DIR
from src.xgboost_baseline import (
    COLSAMPLE_BYTREE,
    HAS_XGBOOST,
    HORIZONS,
    LAGS,
    LEARNING_RATE,
    MAX_DEPTH,
    MIN_CHILD_WEIGHT,
    MIN_TRAIN,
    N_ESTIMATORS,
    REG_ALPHA,
    REG_LAMBDA,
    SEED,
    SUBSAMPLE,
    create_model,
    engineer_tabular_features,
    load_common_sample,
)

logger = logging.getLogger(__name__)

# The rolling-CV evaluation scores every origin from index MIN_TRAIN onward
# (2012-02-17 .. 2026-06-02). Any row at or beyond that index is an evaluation
# origin, so hyperparameter selection must be confined strictly below it --
# guaranteeing zero overlap with scored test origins.
SELECTION_END_INDEX: int = MIN_TRAIN

# Seeds for multi-seed objective evaluation (H-2)
SELECTION_SEEDS: tuple[int, ...] = (42, 0, 1, 7, 2024)

# Canonical baseline configuration prior to tuning (Issue #101)
CANONICAL_CONFIG: dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 150,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}


def generate_search_space(has_xgboost: bool = HAS_XGBOOST) -> list[dict[str, Any]]:
    """
    Generate the curated candidate search space (~20-40 configurations).
    Couples learning_rate inversely with n_estimators to balance training speed
    and convergence stability.

    When running under the GradientBoostingRegressor fallback, parameters
    unsupported by sklearn (colsample_bytree, min_child_weight, reg_lambda)
    are gracefully omitted.
    """
    lr_n_est_pairs = [
        (0.01, 600),
        (0.03, 300),
        (0.10, 150),
    ]
    depths = [2, 3, 4]

    configs: list[dict[str, Any]] = []

    if has_xgboost:
        # 3 regularization / subsampling tiers:
        # Tier 0 (Conservative): strong L2 penalty and deeper bagging
        # Tier 1 (Balanced): close to baseline defaults
        # Tier 2 (Aggressive): unconstrained features and lighter L2 penalty
        reg_tiers = [
            {
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 5.0,
                "reg_lambda": 5.0,
            },
            {
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
            },
            {
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
                "reg_lambda": 0.5,
            },
        ]

        for lr, n_est in lr_n_est_pairs:
            for depth in depths:
                for reg in reg_tiers:
                    cfg = {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        "subsample": reg["subsample"],
                        "colsample_bytree": reg["colsample_bytree"],
                        "min_child_weight": reg["min_child_weight"],
                        "reg_lambda": reg["reg_lambda"],
                        "reg_alpha": 0.0,
                    }
                    configs.append(cfg)

        # Explicitly ensure canonical baseline is present for direct auditable comparison
        can_present = any(
            c["max_depth"] == CANONICAL_CONFIG["max_depth"]
            and np.isclose(c["learning_rate"], CANONICAL_CONFIG["learning_rate"])
            and c["n_estimators"] == CANONICAL_CONFIG["n_estimators"]
            and np.isclose(c["subsample"], CANONICAL_CONFIG["subsample"])
            and np.isclose(c["colsample_bytree"], CANONICAL_CONFIG["colsample_bytree"])
            and np.isclose(c["min_child_weight"], CANONICAL_CONFIG["min_child_weight"])
            and np.isclose(c["reg_lambda"], CANONICAL_CONFIG["reg_lambda"])
            for c in configs
        )
        if not can_present:
            configs.append(CANONICAL_CONFIG.copy())
    else:
        # Fallback space for sklearn GradientBoostingRegressor
        subsamples = [0.7, 0.8, 1.0]
        for lr, n_est in lr_n_est_pairs:
            for depth in depths:
                for sub in subsamples:
                    cfg = {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        "subsample": sub,
                    }
                    configs.append(cfg)

    return configs


def evaluate_inner_split_config(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: dict[int, str],
    config: dict[str, Any],
    horizons: list[int] | None = None,
    selection_end_index: int = SELECTION_END_INDEX,
    cal_ratio: float = 0.15,
    val_ratio: float = 0.20,
    seed: int = SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Evaluate a single hyperparameter configuration on the pre-evaluation hold-out.

    Enforces:
    - Hold-out strictly prior to selection_end_index (MIN_TRAIN = 500 rows, pre-2012).
      Every scored rolling-CV test origin starts at index MIN_TRAIN or later.
    - Inner validation split carved out of the fit block only.
    - Strict h-step causal embargo gap between inner_train and inner_val.
    - Zero overlap with the split-conformal calibration block.
    - Zero silent fallbacks: fails loudly if the partition cannot satisfy the embargo.
    """
    if horizons is None:
        horizons = HORIZONS

    # Isolate the pre-evaluation selection window: strictly the rows that are
    # never scored as forecast origins by make_rolling_folds().
    pool_size = min(len(data), selection_end_index)
    if pool_size < 60:
        raise ValueError(
            f"Selection pool of {pool_size} rows is too small for a leakage-safe "
            f"embargoed inner split; refusing to fall back to a leaky partition."
        )
    train_pool = data.iloc[:pool_size].copy()

    # Resolve all hyperparameters once against baseline constants (M-2)
    resolved = {
        "max_depth": int(config.get("max_depth", MAX_DEPTH)),
        "learning_rate": float(config.get("learning_rate", LEARNING_RATE)),
        "n_estimators": int(config.get("n_estimators", N_ESTIMATORS)),
        "subsample": float(config.get("subsample", SUBSAMPLE)),
        "colsample_bytree": float(config.get("colsample_bytree", COLSAMPLE_BYTREE)),
        "min_child_weight": float(config.get("min_child_weight", MIN_CHILD_WEIGHT)),
        "reg_lambda": float(config.get("reg_lambda", REG_LAMBDA)),
        "reg_alpha": float(config.get("reg_alpha", REG_ALPHA)),
    }

    rmse_per_h: dict[int, float] = {}
    naive_rmse_per_h: dict[int, float] = {}
    rel_rmse_per_h: dict[int, float] = {}
    partitions: list[dict[str, Any]] = []

    for h in horizons:
        t_col = target_cols[h]
        embargo_end = pool_size - h
        if embargo_end <= 0:
            raise ValueError(f"Insufficient samples {pool_size} for horizon {h}")

        train_data = train_pool.iloc[:embargo_end]
        n_tr = len(train_data)

        # Split-conformal calibration partition (untouched by model fitting)
        cal_size = max(int(n_tr * cal_ratio), 20)
        fit_end = n_tr - cal_size - h
        if fit_end < 20:
            raise ValueError(
                f"h={h}: fit block of {fit_end} rows cannot host an embargoed inner split "
                f"(pool={pool_size}, cal_size={cal_size}). Increase the selection window."
            )

        fit_data = train_data.iloc[:fit_end]
        n_fit = len(fit_data)
        val_size = max(int(n_fit * val_ratio), 20)
        inner_train_end = n_fit - val_size - h
        if inner_train_end < 10:
            raise ValueError(
                f"h={h}: only {inner_train_end} inner-train rows remain after the "
                f"{val_size}-row validation block and {h}-step embargo."
            )

        # Causal h-step embargo between inner_train and inner_val
        inner_train = fit_data.iloc[:inner_train_end]
        inner_val = fit_data.iloc[inner_train_end + h:]
        if len(inner_val) == 0:
            raise ValueError(f"h={h}: empty inner validation block after embargo.")

        cal_start = fit_end + h
        cal_data = train_data.iloc[cal_start:]

        # Log concrete partition boundaries (C-3 provenance artifact)
        p_info: dict[str, Any] = {
            "horizon": h,
            "pool_size": pool_size,
            "inner_train_start_idx": 0,
            "inner_train_end_idx": inner_train_end,
            "inner_train_rows": len(inner_train),
            "inner_val_start_idx": inner_train_end + h,
            "inner_val_end_idx": fit_end,
            "inner_val_rows": len(inner_val),
            "cal_start_idx": cal_start,
            "cal_end_idx": n_tr,
            "cal_rows": len(cal_data),
            "first_scored_origin_idx": selection_end_index,
        }
        if "date" in train_pool.columns:
            p_info["inner_train_start_date"] = str(inner_train["date"].iloc[0])
            p_info["inner_train_end_date"] = str(inner_train["date"].iloc[-1])
            p_info["inner_val_start_date"] = str(inner_val["date"].iloc[0])
            p_info["inner_val_end_date"] = str(inner_val["date"].iloc[-1])
            p_info["cal_start_date"] = str(cal_data["date"].iloc[0])
            p_info["cal_end_date"] = str(cal_data["date"].iloc[-1])
            if selection_end_index < len(data) and "date" in data.columns:
                p_info["first_scored_origin_date"] = str(data["date"].iloc[selection_end_index])
        partitions.append(p_info)

        X_tr = inner_train[feature_cols].values
        y_tr = inner_train[t_col].values
        X_val = inner_val[feature_cols].values
        y_val = inner_val[t_col].values

        # Differenced target naive benchmark is 0 (since naive random walk predicts Delta y = 0)
        naive_val_diff = np.zeros_like(y_val)

        # Build and fit model using fully resolved specification
        model = create_model(loss="squared_error", random_state=seed, **resolved)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        rmse_m = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        rmse_n = float(np.sqrt(np.mean((y_val - naive_val_diff) ** 2)))
        rel_rmse = rmse_m / rmse_n if rmse_n > 0 else 1.0

        rmse_per_h[h] = rmse_m
        naive_rmse_per_h[h] = rmse_n
        rel_rmse_per_h[h] = rel_rmse

    mean_val_rmse = float(np.mean(list(rmse_per_h.values())))
    mean_relative_rmse = float(np.mean(list(rel_rmse_per_h.values())))
    val_improvement_pct = float((1.0 - mean_relative_rmse) * 100.0)

    # Exactly the fitted specification is logged
    record: dict[str, Any] = dict(resolved)
    record["seed"] = seed
    for h in horizons:
        record[f"val_rmse_h{h}"] = round(rmse_per_h[h], 6)
        record[f"naive_rmse_h{h}"] = round(naive_rmse_per_h[h], 6)
        record[f"rel_rmse_h{h}"] = round(rel_rmse_per_h[h], 4)

    record["mean_val_rmse"] = round(mean_val_rmse, 6)
    record["mean_relative_rmse"] = round(mean_relative_rmse, 6)
    record["val_improvement_pct"] = round(val_improvement_pct, 3)

    return record, partitions


def evaluate_config_across_seeds(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: dict[int, str],
    config: dict[str, Any],
    horizons: list[int] | None = None,
    seeds: tuple[int, ...] = SELECTION_SEEDS,
    selection_end_index: int = SELECTION_END_INDEX,
    cal_ratio: float = 0.15,
    val_ratio: float = 0.20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Average the selection objective across multiple seeds and report dispersion.
    """
    seed_records: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []

    for s in seeds:
        rec, parts = evaluate_inner_split_config(
            data=data,
            feature_cols=feature_cols,
            target_cols=target_cols,
            config=config,
            horizons=horizons,
            selection_end_index=selection_end_index,
            cal_ratio=cal_ratio,
            val_ratio=val_ratio,
            seed=s,
        )
        seed_records.append(rec)
        if not partitions:
            partitions = parts

    out = dict(seed_records[0])
    mean_rel_vals = [r["mean_relative_rmse"] for r in seed_records]
    mean_val_vals = [r["mean_val_rmse"] for r in seed_records]

    out["mean_relative_rmse"] = round(float(np.mean(mean_rel_vals)), 6)
    out["std_relative_rmse"] = round(float(np.std(mean_rel_vals, ddof=1)), 6) if len(mean_rel_vals) > 1 else 0.0
    out["mean_val_rmse"] = round(float(np.mean(mean_val_vals)), 6)
    out["std_val_rmse"] = round(float(np.std(mean_val_vals, ddof=1)), 6) if len(mean_val_vals) > 1 else 0.0
    out["val_improvement_pct"] = round(float((1.0 - out["mean_relative_rmse"]) * 100.0), 3)
    out["n_selection_seeds"] = len(seeds)

    return out, partitions


def run_grid_search(
    df: pd.DataFrame,
    search_space: list[dict[str, Any]] | None = None,
    selection_end_index: int = SELECTION_END_INDEX,
    seeds: tuple[int, ...] = SELECTION_SEEDS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute grid search across all candidate configurations averaged over seeds.
    Returns:
    - results_df: Ranked DataFrame sorted by seed-mean relative RMSE
    - partitions_df: DataFrame documenting exact partition boundaries
    """
    if search_space is None:
        search_space = generate_search_space(has_xgboost=HAS_XGBOOST)

    data, feature_cols, target_cols = engineer_tabular_features(
        df, lags=LAGS, horizons=HORIZONS,
    )

    total_combos = len(search_space)
    if verbose:
        print(f"Starting XGBoost Hyperparameter Search ({total_combos} configurations x {len(seeds)} seeds)")
        print(f"Selection Window: strictly rows 0..{selection_end_index} (leakage-safe burn-in holdout)")
        print("=" * 80)

    start_time = time.time()
    records: list[dict[str, Any]] = []
    all_partitions: list[dict[str, Any]] = []

    for idx, cfg in enumerate(search_space, start=1):
        t0 = time.time()
        rec, parts = evaluate_config_across_seeds(
            data=data,
            feature_cols=feature_cols,
            target_cols=target_cols,
            config=cfg,
            seeds=seeds,
            selection_end_index=selection_end_index,
        )
        t_elapsed = time.time() - t0
        records.append(rec)
        if not all_partitions:
            all_partitions = parts

        if verbose:
            total_elapsed = time.time() - start_time
            avg_sec = total_elapsed / idx
            eta_min = (avg_sec * (total_combos - idx)) / 60.0
            print(
                f"[{idx:02d}/{total_combos:02d}] depth={rec['max_depth']} "
                f"| lr={rec['learning_rate']} | n_est={rec['n_estimators']} "
                f"| sub={rec['subsample']} | col={rec['colsample_bytree']} "
                f"| mcw={rec['min_child_weight']} | lam={rec['reg_lambda']} "
                f"-> Rel RMSE: {rec['mean_relative_rmse']:.4f} +/- {rec['std_relative_rmse']:.4f} "
                f"({rec['val_improvement_pct']:+.2f}% vs Naive) [{t_elapsed:.1f}s | ETA: {eta_min:.1f}m]"
            )

    results_df = pd.DataFrame(records)

    # Rank configurations by mean_relative_rmse, breaking ties toward parsimony
    results_df = results_df.sort_values(
        ["mean_relative_rmse", "max_depth", "n_estimators"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)

    # Flag canonical baseline configuration
    is_can = (
        (results_df["max_depth"] == CANONICAL_CONFIG["max_depth"])
        & np.isclose(results_df["learning_rate"].astype(float), CANONICAL_CONFIG["learning_rate"])
        & (results_df["n_estimators"] == CANONICAL_CONFIG["n_estimators"])
        & np.isclose(results_df["subsample"].astype(float), CANONICAL_CONFIG["subsample"])
    )
    if HAS_XGBOOST and "colsample_bytree" in results_df.columns:
        col_s = pd.to_numeric(results_df["colsample_bytree"], errors="coerce")
        mcw_s = pd.to_numeric(results_df["min_child_weight"], errors="coerce")
        lam_s = pd.to_numeric(results_df["reg_lambda"], errors="coerce")
        valid_mask = col_s.notna() & mcw_s.notna() & lam_s.notna()
        can_mask = np.zeros(len(results_df), dtype=bool)
        if valid_mask.any():
            can_mask[valid_mask] = (
                np.isclose(col_s[valid_mask], CANONICAL_CONFIG["colsample_bytree"])
                & np.isclose(mcw_s[valid_mask], CANONICAL_CONFIG["min_child_weight"])
                & np.isclose(lam_s[valid_mask], CANONICAL_CONFIG["reg_lambda"])
            )
        is_can = is_can & can_mask
    results_df["is_canonical"] = is_can

    partitions_df = pd.DataFrame(all_partitions)

    return results_df, partitions_df


def export_search_results(
    results_df: pd.DataFrame,
    partitions_df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    partitions_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Export ranked search results and partition metadata to CSV."""
    if output_path is None:
        output_path = OUTPUTS_DIR / "r3_xgboost_hyperparameter_search.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Saved hyperparameter search results to %s", output_path)

    part_out = None
    if partitions_df is not None:
        if partitions_path is None:
            partitions_path = OUTPUTS_DIR / "r3_xgboost_search_partitions.csv"
        partitions_path.parent.mkdir(parents=True, exist_ok=True)
        partitions_df.to_csv(partitions_path, index=False)
        logger.info("Saved search partitions to %s", partitions_path)
        part_out = partitions_path

    return output_path, part_out


def compare_best_to_canonical(results_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compare top-ranked winning configuration against canonical baseline defaults.
    """
    best_row = results_df.iloc[0]
    canonical_match = results_df[results_df["is_canonical"]]

    canonical_row = canonical_match.iloc[0] if not canonical_match.empty else None

    best_rel_rmse = float(best_row["mean_relative_rmse"])
    can_rel_rmse = float(canonical_row["mean_relative_rmse"]) if canonical_row is not None else None
    rel_gain = ((can_rel_rmse - best_rel_rmse) / can_rel_rmse * 100.0) if can_rel_rmse is not None else None

    return {
        "best_config": {
            "max_depth": int(best_row["max_depth"]),
            "learning_rate": float(best_row["learning_rate"]),
            "n_estimators": int(best_row["n_estimators"]),
            "subsample": float(best_row["subsample"]),
            "colsample_bytree": float(best_row["colsample_bytree"]) if pd.notna(best_row.get("colsample_bytree")) else None,
            "min_child_weight": float(best_row["min_child_weight"]) if pd.notna(best_row.get("min_child_weight")) else None,
            "reg_lambda": float(best_row["reg_lambda"]) if pd.notna(best_row.get("reg_lambda")) else None,
            "mean_val_rmse": float(best_row["mean_val_rmse"]),
            "mean_relative_rmse": float(best_row["mean_relative_rmse"]),
            "std_relative_rmse": float(best_row.get("std_relative_rmse", 0.0)),
            "val_improvement_pct": float(best_row["val_improvement_pct"]),
        },
        "canonical_config": {
            "mean_val_rmse": float(canonical_row["mean_val_rmse"]) if canonical_row is not None else None,
            "mean_relative_rmse": can_rel_rmse,
            "std_relative_rmse": float(canonical_row.get("std_relative_rmse", 0.0)) if canonical_row is not None else None,
            "val_improvement_pct": float(canonical_row["val_improvement_pct"]) if canonical_row is not None else None,
            "canonical_rank": int(canonical_row["rank"]) if canonical_row is not None else None,
        },
        "relative_gain_pct": round(rel_gain, 3) if rel_gain is not None else None,
        "is_new_best": bool(best_row["rank"] == 1 and not best_row["is_canonical"]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = load_common_sample()
    results_df, partitions_df = run_grid_search(df)
    out_path, part_path = export_search_results(results_df, partitions_df)
    summary = compare_best_to_canonical(results_df)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 80)
    print(f"Exported search results: {out_path}")
    print(f"Exported partition manifest: {part_path}")
    print(f"Top configuration (Rank 1): {summary['best_config']}")
    if summary["canonical_config"]["canonical_rank"] is not None:
        print(f"Canonical Baseline: Rank {summary['canonical_config']['canonical_rank']} (Rel RMSE {summary['canonical_config']['mean_relative_rmse']:.4f})")
        if summary["relative_gain_pct"] is not None:
            print(f"Relative Gain over Default: {summary['relative_gain_pct']:+.2f}%")


if __name__ == "__main__":
    main()

