"""
Multicollinearity diagnostics (Issue #67)

Gold-layer feature store carries exact linear dependencies across yield/spread
columns (yield_spread_10y_2y === yield_10y - yield_2y === yield_spread_10y_5y +
yield_spread_5y_2y). No VAR/VECM/LSTM feature set currently combines columns
from that dependent set, but nothing guards against a future feature-set
change re-introducing it. compute_vif() is that guard, reusable at any point
a model's actual regressor set needs checking.
"""

import numpy as np
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
    thresholds flag concern above 5 and serious concern above 10.

    A column that's an exact linear combination of the others (e.g.
    a === b - c) makes the design matrix rank-deficient for that column, so
    its VIF is mathematically infinite -- reported as `np.inf` via an
    explicit `matrix_rank` check rather than relying on the large-but-finite
    float variance_inflation_factor's matrix inversion happens to return
    under floating-point noise.
    """
    X = add_constant(df[columns].dropna())
    full_rank = np.linalg.matrix_rank(X.values)

    vifs = []
    for c in columns:
        others = [col for col in X.columns if col != c]
        if np.linalg.matrix_rank(X[others].values) == full_rank:
            vifs.append(np.inf)
        else:
            vifs.append(variance_inflation_factor(X.values, X.columns.get_loc(c)))

    return pd.DataFrame({
        "feature": columns,
        "vif": vifs,
    }).sort_values("vif", ascending=False).reset_index(drop=True)
