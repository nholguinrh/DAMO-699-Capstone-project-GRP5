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

    Raises ValueError if `columns` has too few overlapping non-NaN rows to
    determine rank against the design matrix (including the intercept, that's
    len(columns) + 1 rows at minimum -- fewer than that and the matrix_rank
    comparison below can't distinguish "not enough data" from genuine
    collinearity, and would silently report every column as inf), or if one
    of `columns` is itself an exact constant -- silently omitting the
    intercept in that case would contradict the "always includes an
    intercept" guarantee above.
    """
    X = df[columns].dropna()
    if len(X) <= len(columns):
        raise ValueError(
            f"Only {len(X)} overlapping non-NaN row(s) across columns "
            f"{columns}; need more than {len(columns)} to compute VIF "
            "against a design matrix that includes an intercept."
        )
    constant_cols = [c for c in columns if X[c].nunique() <= 1]
    if constant_cols:
        raise ValueError(
            f"Columns {constant_cols} are exact constants over the "
            "overlapping non-NaN rows; VIF requires an intercept plus "
            "genuinely varying regressors."
        )
    # Backstop, not a substitute for the nunique check above -- has_constant="raise" only
    # catches a nonzero constant, so it alone would miss an exact-zero column.
    X = add_constant(X, has_constant="raise")
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
