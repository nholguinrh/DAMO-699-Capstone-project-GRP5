"""
Unit tests for the multicollinearity diagnostic (Issue #67)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.diagnostics import compute_vif


def _independent_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })


def test_vif_near_one_for_independent_columns():
    df = _independent_df()
    out = compute_vif(df, ["a", "b", "c"])
    assert (out["vif"] < 2).all()


def test_vif_returns_one_row_per_column_sorted_descending():
    df = _independent_df()
    out = compute_vif(df, ["a", "b", "c"])
    assert list(out["feature"].sort_values()) == ["a", "b", "c"]
    assert (out["vif"].diff().dropna() <= 0).all()


def test_vif_explodes_for_exact_linear_dependence():
    """Mirrors issue #67's actual scenario: spread === yield_10y - yield_2y."""
    df = _independent_df()
    df["yield_2y"] = df["a"]
    df["yield_10y"] = df["a"] + df["b"]
    df["yield_spread_10y_2y"] = df["yield_10y"] - df["yield_2y"]  # === df["b"], exactly

    out = compute_vif(df, ["yield_2y", "yield_10y", "yield_spread_10y_2y"])
    assert (out["vif"] == np.inf).all()


def test_vif_drops_rows_with_missing_values():
    df = _independent_df()
    df.loc[0, "a"] = np.nan
    out = compute_vif(df, ["a", "b", "c"])
    assert out["vif"].notna().all()


def test_vif_raises_on_constant_feature_column():
    """Issue #96: a flat regressor (e.g. a pegged policy rate) must not
    silently make add_constant() skip the intercept."""
    df = _independent_df()
    df["flat"] = 5.0
    with pytest.raises(ValueError, match="exact constants"):
        compute_vif(df, ["a", "b", "flat"])


def test_vif_raises_on_zero_valued_constant_feature_column():
    """Issue #96 follow-up: statsmodels' has_constant='raise' only flags a
    *nonzero* constant, so an exact-zero column must be caught separately."""
    df = _independent_df()
    df["zero"] = 0.0
    with pytest.raises(ValueError, match="exact constants"):
        compute_vif(df, ["a", "b", "zero"])


def test_vif_raises_on_empty_overlap():
    """Issue #96: columns with no shared non-NaN rows must raise, not
    silently report VIF = inf for everything."""
    df = _independent_df()
    df["a"] = np.nan
    with pytest.raises(ValueError, match="overlapping non-NaN row"):
        compute_vif(df, ["a", "b", "c"])


def test_vif_raises_on_underdetermined_design_matrix():
    """Issue #96 follow-up: too few rows relative to columns is rank-deficient
    for the same reason as an empty overlap, and must raise rather than
    silently reporting every column as vif=inf."""
    df = _independent_df(n=3)
    with pytest.raises(ValueError, match="Only 3 overlapping"):
        compute_vif(df, ["a", "b", "c"])
