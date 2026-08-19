"""
LSTM Univariate-Comparable Baseline + Rolling-Window CV (Issue #49)
DAMO-699 Capstone Project, Group 5

Shared model/data code for `notebooks/03_models/lstm_baseline.ipynb` (rolling-window
CV evaluation vs. the naive benchmark) and `notebooks/04_diagnostics/lstm_shap.ipynb`
(SHAP interpretability on a final full-sample model). Kept in one module so both
notebooks build the exact same feature set, windows, and architecture rather than
drifting apart.

Trains on the 6-variable Gold-layer stationary set (`yield_spread_10y_2y`,
`overnight_rate`, `us_treasury_10y`, `fed_funds_rate`, `cpi_yoy`, `usdcad`,
first-differenced), matching proposal §3.1/§5.3's committed predictor set --
`usdcad` is a required "macroeconomic transmission variable" per the research
question, not an optional one. #28/#29's VAR baselines had converged on a
5-variable set that dropped it (undocumented in #28, and #29 only matched #28 for
AIC-vs-BIC comparability, not because usdcad was found unnecessary -- see the
Aug 19 team decision to re-add it). Per proposal §5.3's requirement that the LSTM
receive "the same Gold-layer feature set as the VAR/VECM model" for a fair
comparison in #50, this module now leads with the proposal-correct set, so #28/#29
should re-converge to 6 variables rather than #49 converging down to 5.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from gold_feature_pipeline import build_gold_features  # noqa: E402
from project_paths import PROCESSED_DIR  # noqa: E402,F401

LEVEL_TARGET = "yield_spread_10y_2y"
TARGET = "d_yield_spread_10y_2y"
FEATURES = [
    "d_yield_spread_10y_2y",
    "d_overnight_rate",
    "d_us_treasury_10y",
    "d_fed_funds_rate",
    "d_cpi_yoy",
    "d_usdcad",
]
HORIZONS = [1, 5, 20]

LOOKBACK = 20          # trading days of history fed to the LSTM at each origin
MIN_TRAIN = 500        # first fold's training size, matching the VAR/ARIMA MIN_TRAIN convention
N_FOLDS = 5            # rolling-window CV folds (expanding window, sequential test blocks)
HIDDEN_SIZE = 16
DROPOUT = 0.2
MAX_EPOCHS = 100
PATIENCE = 8           # early-stopping patience on validation loss
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


def load_common_sample(boc_path=None, fred_path=None, cpi_path=None) -> pd.DataFrame:
    """
    Build the Gold-layer feature set (#30) and slice it down to the same
    5-variable common sample used by the VAR/ARIMA baselines, so RMSE/MAE/DM
    results are directly comparable per #50.

    boc_path/fred_path/cpi_path forward to `build_gold_features()` so tests can
    point this at the same synthetic fixtures used by `tests/test_gold_pipeline.py`
    instead of requiring real processed data on disk.
    """
    df = build_gold_features(
        boc_path=boc_path, fred_path=fred_path, cpi_path=cpi_path,
        feature_type="all", save=False,
    )
    df = df.reset_index().rename(columns={"index": "date"})
    df = df[["date", LEVEL_TARGET] + FEATURES].dropna().sort_values("date").reset_index(drop=True)
    return df


def make_windows(df: pd.DataFrame, lookback: int = LOOKBACK, horizons=HORIZONS):
    """
    Build supervised sequence-learning arrays.

    X[i]        = past `lookback` days of the stationary FEATURES ending at origin i
    Y[i, j]     = cumulative sum of d_yield_spread_10y_2y over the next horizons[j]
                  days, so origin_level + Y[i, j] == actual future level -- the same
                  "last_level + cumulative_change" convention the ARIMA/VAR baselines
                  use, which keeps evaluation directly comparable in levels.
    origin_idx  = row index into `df` that each window ends at (for date/level lookup)
    """
    feat = df[FEATURES].to_numpy(dtype=np.float32)
    target = df[TARGET].to_numpy(dtype=np.float32)
    n = len(df)
    max_h = max(horizons)

    X, Y, origin_idx = [], [], []
    for origin in range(lookback, n - max_h):
        X.append(feat[origin - lookback:origin])
        Y.append([target[origin + 1: origin + 1 + h].sum() for h in horizons])
        origin_idx.append(origin)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(origin_idx),
    )


class ShallowLSTM(nn.Module):
    """Single-layer LSTM with dropout, per proposal §5.3's "shallow architecture"."""

    def __init__(self, n_features, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, n_outputs=len(HORIZONS)):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(self.dropout(h_n[-1]))


def make_rolling_folds(n_windows: int, min_train: int = MIN_TRAIN, n_folds: int = N_FOLDS):
    """
    Expanding-window rolling folds over windowed-array indices. Fold k trains on
    everything before its test block and tests on the next sequential block --
    temporal order is preserved throughout, so no fold ever trains on future data.

    Retraining a fresh network at every 5-observation origin (the ARIMA/VAR
    convention) is intractable for an LSTM, so CV here means K sequential
    expanding-window folds rather than per-origin refits.
    """
    test_pool = n_windows - min_train
    fold_size = test_pool // n_folds
    folds = []
    for k in range(n_folds):
        train_end = min_train + k * fold_size
        test_end = n_windows if k == n_folds - 1 else min_train + (k + 1) * fold_size
        folds.append((train_end, test_end))
    return folds


def _train_with_early_stopping(
    Xtr, Ytr, Xval, Yval, n_features, seed,
    hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR,
):
    n_outputs = Ytr.shape[1]
    torch.manual_seed(seed)
    model = ShallowLSTM(n_features=n_features, hidden_size=hidden_size, dropout=dropout, n_outputs=n_outputs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xval_t, Yval_t = torch.from_numpy(Xval), torch.from_numpy(Yval)

    best_val, best_state, no_improve, epochs_run = float("inf"), None, 0, 0
    n = len(Xtr_t)

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), Yval_t).item()
        epochs_run += 1

        if val_loss < best_val - 1e-6:
            best_val, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val, epochs_run


def run_rolling_cv(
    df: pd.DataFrame,
    lookback: int = LOOKBACK, horizons=HORIZONS,
    min_train: int = MIN_TRAIN, n_folds: int = N_FOLDS,
    hidden_size: int = HIDDEN_SIZE, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
    min_val_size: int = 50,
):
    """
    Run expanding-window rolling CV: at each fold, fit on the training block
    (with a chronological tail slice held out for early stopping) and forecast
    the following test block. Returns forecasts in levels (comparable to the
    naive benchmark) and per-fold training diagnostics.

    The keyword args all default to the module constants used in production
    (`notebooks/03_models/lstm_baseline.ipynb` calls this with no overrides);
    they exist so tests can run the same logic on tiny synthetic data quickly.
    """
    levels = df[LEVEL_TARGET].to_numpy(dtype=np.float32)
    X, Y, origin_idx = make_windows(df, lookback=lookback, horizons=horizons)
    folds = make_rolling_folds(len(X), min_train=min_train, n_folds=n_folds)

    results, fold_diag = [], []

    for fold_id, (train_end, test_end) in enumerate(folds):
        val_size = max(int(train_end * 0.15), min_val_size)
        tr_end = train_end - val_size

        Xtr, Ytr = X[:tr_end], Y[:tr_end]
        Xval, Yval = X[tr_end:train_end], Y[tr_end:train_end]
        Xte = X[train_end:test_end]

        x_mean, x_std = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1)) + 1e-8
        y_mean, y_std = Ytr.mean(axis=0), Ytr.std(axis=0) + 1e-8

        model, best_val, epochs_run = _train_with_early_stopping(
            (Xtr - x_mean) / x_std, (Ytr - y_mean) / y_std,
            (Xval - x_mean) / x_std, (Yval - y_mean) / y_std,
            n_features=len(FEATURES), seed=SEED + fold_id,
            hidden_size=hidden_size, max_epochs=max_epochs, patience=patience,
        )

        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.from_numpy((Xte - x_mean) / x_std)).numpy()
        pred = pred_scaled * y_std + y_mean

        test_origins = origin_idx[train_end:test_end]
        for i, origin in enumerate(test_origins):
            last_level = levels[origin]
            origin_date = df.loc[origin, "date"]
            for h_idx, h in enumerate(horizons):
                results.append({
                    "origin_date": origin_date,
                    "horizon": h,
                    "actual": levels[origin + h],
                    "naive": last_level,
                    "lstm": last_level + pred[i, h_idx],
                    "fold": fold_id,
                })

        fold_diag.append({
            "fold": fold_id,
            "train_size": tr_end,
            "val_size": val_size,
            "test_size": test_end - train_end,
            "epochs_run": epochs_run,
            "best_val_loss": best_val,
        })

    return pd.DataFrame(results), pd.DataFrame(fold_diag)


def train_final_model(
    df: pd.DataFrame, val_frac: float = 0.15, seed: int = SEED,
    lookback: int = LOOKBACK, horizons=HORIZONS,
    hidden_size: int = HIDDEN_SIZE, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
):
    """
    Train one shallow LSTM on the full common sample (chronological train/val
    split, early stopping) for the downstream SHAP interpretability notebook.
    Not used for the rolling-CV forecast evaluation -- that's `run_rolling_cv`.
    """
    X, Y, origin_idx = make_windows(df, lookback=lookback, horizons=horizons)
    val_size = int(len(X) * val_frac)
    tr_end = len(X) - val_size

    Xtr, Ytr = X[:tr_end], Y[:tr_end]
    Xval, Yval = X[tr_end:], Y[tr_end:]

    x_mean, x_std = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1)) + 1e-8
    y_mean, y_std = Ytr.mean(axis=0), Ytr.std(axis=0) + 1e-8

    model, best_val, epochs_run = _train_with_early_stopping(
        (Xtr - x_mean) / x_std, (Ytr - y_mean) / y_std,
        (Xval - x_mean) / x_std, (Yval - y_mean) / y_std,
        n_features=len(FEATURES), seed=seed,
        hidden_size=hidden_size, max_epochs=max_epochs, patience=patience,
    )

    scaling = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, scaling, X, origin_idx, {"best_val_loss": best_val, "epochs_run": epochs_run}


def save_final_model(model: nn.Module, scaling: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scaling": scaling}, path)


def load_final_model(
    path: Path, n_features: int = len(FEATURES),
    hidden_size: int = HIDDEN_SIZE, n_outputs: int = len(HORIZONS),
):
    checkpoint = torch.load(path, weights_only=False)
    model = ShallowLSTM(n_features=n_features, hidden_size=hidden_size, n_outputs=n_outputs)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["scaling"]
