"""
Multicollinearity diagnostics (Issue #67)

Gold-layer feature store carries exact linear dependencies across yield/spread
columns (yield_spread_10y_2y === yield_10y - yield_2y === yield_spread_10y_5y +
yield_spread_5y_2y). No VAR/VECM/LSTM feature set currently combines columns
from that dependent set, but nothing guards against a future feature-set
change re-introducing it. compute_vif() is that guard, reusable at any point
a model's actual regressor set needs checking.
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def compute_vif(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Variance inflation factor for each of `columns` in `df`.

    Includes an intercept in the design matrix -- VAR/VECM here are always fit
    with a constant term, and VIF without one can understate collinearity
    among variables that share a common level/trend. Rows with any NaN across
    `columns` are dropped first.

    Returns a DataFrame with one row per column, sorted by VIF descending.
    A VIF of 1 means no correlation with the other regressors; conventional
    thresholds flag concern above 5 and serious concern above 10. Exact
    linear dependence (e.g. a === b - c) drives VIF to infinity.
    """
    X = add_constant(df[columns].dropna())
    return pd.DataFrame({
        "feature": columns,
        "vif": [variance_inflation_factor(X.values, X.columns.get_loc(c)) for c in columns],
    }).sort_values("vif", ascending=False).reset_index(drop=True)
