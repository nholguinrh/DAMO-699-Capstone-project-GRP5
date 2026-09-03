# Code Review: Pull Request #106
**Title:** `feat(models): implement XGBoost tabular benchmark with forward CV and dashboard integration (close #101)`  
**Reviewers:** Lead Data Scientist & Principal Analytics Engineer  
**Repository:** `nholguinrh/DAMO-699-Capstone-project-GRP5`  
**Date:** September 3, 2026  
**Status:** **REJECTED / BLOCKED (CRITICAL METHODOLOGICAL RISKS IDENTIFIED)**

---

## Table of Contents
1. [Methodological Risk Assessment](#1-methodological-risk-assessment)
2. [Evaluation Dimensions Matrix](#2-evaluation-dimensions-matrix)
3. [Critical / Blocking Findings](#3-critical--blocking-findings)
   - [Finding 3.1: Multi-Step Forward Target Leakage Across Rolling CV Folds](#finding-31-multi-step-forward-target-leakage-across-rolling-cv-folds)
   - [Finding 3.2: In-Sample Residual Optimism and Ad-Hoc Quantile Mixing](#finding-32-in-sample-residual-optimism-and-ad-hoc-quantile-mixing)
   - [Finding 3.3: Primary Battery Hypothesis Dilution & FDR Multiplicity Corruption](#finding-33-primary-battery-hypothesis-dilution--fdr-multiplicity-corruption)
   - [Finding 3.4: Small-Sample Asymptotic Failure in Regime Clark-West Tests](#finding-34-small-sample-asymptotic-failure-in-regime-clark-west-tests)
   - [Finding 3.5: Common Evaluation Sample Truncation via Unconstrained Inner Merge](#finding-35-common-evaluation-sample-truncation-via-unconstrained-inner-merge)
4. [Performance & Memory Optimizations](#4-performance--memory-optimizations)
   - [Optimization 4.1: Elimination of Redundant Module Reloads in Streamlit](#optimization-41-elimination-of-redundant-module-reloads-in-streamlit)
   - [Optimization 4.2: In-Memory Caching for Gold-Layer Feature Ingestion](#optimization-42-in-memory-caching-for-gold-layer-feature-ingestion)
   - [Optimization 4.3: Safe Missing-Value Handling in Metric Aggregation](#optimization-43-safe-missing-value-handling-in-metric-aggregation)
5. [Missing Assertions & Test Cases](#5-missing-assertions--test-cases)
   - [Test 5.1: Prediction Interval Quantile Monotonicity on Committed Artifacts](#test-51-prediction-interval-quantile-monotonicity-on-committed-artifacts)
   - [Test 5.2: Temporal Embargo Non-Overlap in Multi-Step CV](#test-52-temporal-embargo-non-overlap-in-multi-step-cv)
   - [Test 5.3: Clark-West Primary Battery Multiplicity Invariance ($m=12$)](#test-53-clark-west-primary-battery-multiplicity-invariance-m12)
6. [Operational, Configuration & Determinism Fixes](#6-operational-configuration--determinism-fixes)
   - [6.1 Strict Dependency Pinning (`requirements.txt`)](#61-strict-dependency-pinning-requirementstxt)
   - [6.2 Valet API Series Identifier Corrections (`app.py`)](#62-valet-api-series-identifier-corrections-apppy)
   - [6.3 Line Ending Normalization (`.gitattributes`)](#63-line-ending-normalization-gitattributes)
7. [Remediation & Governance Roadmap](#7-remediation--governance-roadmap)

---

## 1. Methodological Risk Assessment

### Overall Risk Rating: **CRITICAL**

The analytical outputs produced by PR #106 are **neither mathematically sound nor operationally reliable** for production deployment or formal capstone thesis defense. 

While the architectural motivation—incorporating an institutional non-linear Machine Learning benchmark (XGBoost) into the forecasting battery—is sound, the underlying implementation contains structural flaws:
1. **Pervasive Look-Ahead Bias:** Cross-validation folds leak future test-set yield innovations into training targets across multi-step horizons ($h \in \{5, 20\}$).
2. **Severely Miscalibrated Prediction Intervals:** Uncertainty bands rely on in-sample training residuals from decision tree regressors, producing empirical undercoverage (e.g., $73.85\%$ empirical coverage against a nominal $90\%$ confidence target at $h=20$).
3. **Statistical Multiplicity Distortion:** Injecting exploratory machine learning models into the canonical 12-test primary battery inflates hypothesis cardinality from $m=12$ to $m=15$, corrupting Benjamini-Hochberg False Discovery Rate (FDR) critical thresholds across all econometric baselines (ARIMA, VAR, VECM).
4. **HAC Asymptotic Breakdown:** Running Clark-West tests on small sub-regimes ($N=22$ at $h=20$) with a 19-lag Newey-West/Bartlett kernel violates asymptotic regularity conditions, resulting in a spurious false-discovery claim ($p=0.0373$).
5. **Silent Sample Truncation:** Unconstrained inner merges in the dashboard data loader silently discard valid historical origin dates from core econometric models if the tabular model dates differ.

**Recommendation:** **BLOCK MERGE.** Revert or remediate via branch `feat/101-xgboost-benchmark-v2`.

---

## 2. Evaluation Dimensions Matrix

| Evaluation Dimension | Severity | Assessment Summary |
| :--- | :---: | :--- |
| **1. Methodological Integrity & Statistical Rigor** | **CRITICAL** | Target leakage in multi-step CV; in-sample residual optimism in conformal intervals; small-sample HAC breakdown; hypothesis family expansion under Benjamini-Hochberg FDR. |
| **2. Performance, Memory & Scalability** | **MEDIUM** | Top-level `importlib.reload()` executes on every Streamlit interaction; repeated disk parsing of raw CSVs in `load_common_sample()`; lack of vectorized metric aggregations. |
| **3. Reproducibility & Determinism** | **HIGH** | Unpinned dependency `xgboost>=2.0.0`; OS line-ending drift (CRLF vs LF rewriting 845 lines); silent exception swallowing during cross-pipeline alignment. |
| **4. Data Contracts & Validation** | **HIGH** | Breaking canonical model naming (`"VECM (6-var)"` $\to$ `"VECM"`, `"LSTM (Tuned)"` $\to$ `"LSTM"`); inner join silently alters common evaluation sample size; invalid Valet API series codes. |

---

## 3. Critical / Blocking Findings

### Finding 3.1: Multi-Step Forward Target Leakage Across Rolling CV Folds
- **File & Lines:** `src/xgboost_baseline.py:333-356`
- **Analytical Risk & Impact:**  
  The multi-horizon cumulative change target is constructed as:
  $$\Delta_h y_t = \sum_{k=1}^h \Delta y_{t+k} = y_{t+h} - y_t$$
  In `run_rolling_cv()`, folds are partitioned using `train_data = data.iloc[:train_end]` and `test_data = data.iloc[train_end:test_end]`. For $h > 1$, the training rows at index $train\_end - 1$ through $train\_end - h + 1$ incorporate realized return innovations occurring on or after $train\_end$. Because `test_data` begins at $train\_end$, the training labels literally contain future yield changes from the test fold. This generates artificial out-of-sample outperformance that will collapse in live forward testing.
- **Remediation Code Diff:**
```python
<<<<
    for fold_id, (train_end, test_end) in enumerate(folds):
        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end:test_end]

        X_train = train_data[feature_cols].values
        X_test = test_data[feature_cols].values

        for h in horizons:
            t_col = target_cols[h]
            y_train = train_data[t_col].values
====
    for fold_id, (train_end, test_end) in enumerate(folds):
        test_data = data.iloc[train_end:test_end]
        X_test = test_data[feature_cols].values

        for h in horizons:
            t_col = target_cols[h]

            # Enforce causal h-step embargo:
            # Slicing at train_end - h guarantees that target Delta_h y_t (spanning t+1..t+h)
            # contains zero realized innovations from test_data (which begins at train_end).
            embargo_end = train_end - h
            if embargo_end <= 0:
                raise ValueError(
                    f"Insufficient samples for train_end={train_end} and embargo h={h}"
                )
            train_data = data.iloc[:embargo_end]
            X_train = train_data[feature_cols].values
            y_train = train_data[t_col].values
>>>>
```

---

### Finding 3.2: In-Sample Residual Optimism and Ad-Hoc Quantile Mixing
- **File & Lines:** `src/xgboost_baseline.py:359-389`
- **Analytical Risk & Impact:**  
  1. Prediction intervals are calibrated using in-sample residuals: `val_residuals = y_train - model_point.predict(X_train)`. Decision trees minimize training loss aggressively, meaning in-sample residuals $|e_t|$ are substantially smaller than true generalization error. This violates conformal calibration assumptions and produces empirical coverage of only $73.85\%$ at $h=20$ for a $90\%$ nominal target.
  2. The code attempts to combine multi-quantile tree predictions with conformal residuals using `np.minimum(q05.predict(X_test), pred_diff - res_90)` and `np.maximum(...)`. This heuristic combines distinct uncertainty paradigms arbitrarily, fails to guarantee quantile monotonicity, and risks quantile crossing under non-linear tree partitioning.
- **Remediation Code Diff:**
```python
<<<<
            # --- Prediction intervals: Approach 1 (Quantile Regressors) or Approach 2 (Conformal Residuals) ---
            val_residuals = y_train - model_point.predict(X_train)
            res_90 = np.quantile(np.abs(val_residuals), 0.90)
            res_95 = np.quantile(np.abs(val_residuals), 0.95)

            if HAS_XGBOOST:
                # Approach 1: Native Quantile Loss objective in XGBoost
                q05 = create_model(loss="quantile", alpha=0.05, max_depth=max_depth,
                                   learning_rate=learning_rate, n_estimators=n_estimators)
                q95 = create_model(loss="quantile", alpha=0.95, max_depth=max_depth,
                                   learning_rate=learning_rate, n_estimators=n_estimators)
                q025 = create_model(loss="quantile", alpha=0.025, max_depth=max_depth,
                                    learning_rate=learning_rate, n_estimators=n_estimators)
                q975 = create_model(loss="quantile", alpha=0.975, max_depth=max_depth,
                                    learning_rate=learning_rate, n_estimators=n_estimators)

                q05.fit(X_train, y_train)
                q95.fit(X_train, y_train)
                q025.fit(X_train, y_train)
                q975.fit(X_train, y_train)

                lower_90_diff = np.minimum(q05.predict(X_test), pred_diff - res_90)
                upper_90_diff = np.maximum(q95.predict(X_test), pred_diff + res_90)
                lower_95_diff = np.minimum(q025.predict(X_test), pred_diff - res_95)
                upper_95_diff = np.maximum(q975.predict(X_test), pred_diff + res_95)
            else:
                # Approach 2: Conformal / empirical residual quantiles (calibrated, non-crossing)
                lower_90_diff = pred_diff - res_90
                upper_90_diff = pred_diff + res_90
                lower_95_diff = pred_diff - res_95
                upper_95_diff = pred_diff + res_95
====
            # --- Prediction intervals: Split-conformal calibration with temporal gap ---
            cal_ratio = 0.15
            n_tr = len(train_data)
            cal_size = max(int(n_tr * cal_ratio), 20)
            fit_end = n_tr - cal_size - h

            if fit_end >= 30:
                fit_data = train_data.iloc[:fit_end]
                cal_data = train_data.iloc[fit_end + h:]

                model_cal = create_model(
                    loss="squared_error",
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                )
                model_cal.fit(fit_data[feature_cols].values, fit_data[t_col].values)
                cal_preds = model_cal.predict(cal_data[feature_cols].values)
                cal_residuals = np.abs(cal_data[t_col].values - cal_preds)

                res_90 = float(np.quantile(cal_residuals, 0.90))
                res_95 = float(np.quantile(cal_residuals, 0.95))
            else:
                # Fallback for small fixtures in unit tests
                residuals = np.abs(y_train - model_point.predict(X_train))
                res_90 = float(np.quantile(residuals, 0.90))
                res_95 = float(np.quantile(residuals, 0.95))

            # Strictly enforce non-crossing quantile monotonicity (95% interval >= 90% interval)
            res_95 = max(res_95, res_90)

            lower_90_diff = pred_diff - res_90
            upper_90_diff = pred_diff + res_90
            lower_95_diff = pred_diff - res_95
            upper_95_diff = pred_diff + res_95
>>>>
```

---

### Finding 3.3: Primary Battery Hypothesis Dilution & FDR Multiplicity Corruption
- **File & Lines:** `src/model_comparison.py:592-611` & `tests/test_model_comparison.py:401-415`
- **Analytical Risk & Impact:**  
  The core econometric benchmark protocol pre-specifies a primary 12-test battery ($m=12$ hypotheses: 4 models $\times$ 3 horizons). PR #106 appends `("xgboost", "XGBoost", "xgboost")` directly into `primary_models`, inflating the test family to $m=15$. In Benjamini-Hochberg FDR control, the critical thresholds are:
  $$p_{(i)} \le \frac{i}{m} \cdot \alpha$$
  Changing $m$ from 12 to 15 mutates adjusted $q$-values across all existing econometric models, altering previously validated statistical conclusions in `outputs/clark_west_test_results.csv`. Furthermore, changing canonical identifiers (`"VECM"` vs `"VECM (6-var)"` and `"LSTM"` vs `"LSTM (Tuned)"`) violates data contracts with the reporting suite.
- **Remediation Code Diff:**
```python
<<<<
    primary_models = [
        ("arima_aic", "ARIMA-AIC", "core"),
        ("var_aic", "VAR-AIC", "core"),
        ("vecm", "VECM", "vecm"),
        ("lstm", "LSTM", "lstm"),
    ]

    # Issue #101: Include XGBoost if forecasts have been generated
    xgb_raw = load_xgboost(d_min, d_max)
    m_xgb = None
    if not xgb_raw.empty:
        try:
            m_xgb = merge_cross_pipeline(
                core, "arima_aic", core_calendar(),
                xgb_raw, "xgboost", xgboost_calendar(),
            )
            primary_models.append(("xgboost", "XGBoost", "xgboost"))
        except Exception:
            logger.warning("XGBoost cross-pipeline merge failed; skipping XGBoost in CW battery.")

    sensitivity_models = [
        ("arima_bic", "ARIMA-BIC", "core"),
        ("var_bic", "VAR-BIC", "core"),
    ]
====
    primary_models = [
        ("arima_aic", "ARIMA-AIC", "core"),
        ("var_aic", "VAR-AIC", "core"),
        ("vecm", "VECM (6-var)", "vecm"),
        ("lstm", "LSTM (Tuned)", "lstm"),
    ]

    sensitivity_models = [
        ("arima_bic", "ARIMA-BIC", "core"),
        ("var_bic", "VAR-BIC", "core"),
    ]

    # Preserve canonical primary battery (m=12) by placing XGBoost in sensitivity battery
    xgb_raw = load_xgboost(d_min, d_max)
    m_xgb = None
    if not xgb_raw.empty:
        m_xgb = merge_cross_pipeline(
            core, "arima_aic", core_calendar(),
            xgb_raw, "xgboost", xgboost_calendar(),
        )
        sensitivity_models.append(("xgboost", "XGBoost (Experimental)", "xgboost"))
>>>>
```

---

### Finding 3.4: Small-Sample Asymptotic Failure in Regime Clark-West Tests
- **File & Lines:** `src/model_comparison.py:742-746, 802-817`
- **Analytical Risk & Impact:**  
  `evaluate_regime_segmentation()` executes `clark_west_test` whenever `len(sub) >= 3`. In Regime 2 (Policy Plateau / Higher-for-Longer), there are only $N=22$ evaluation origins. At $h=20$, multi-step forecast errors exhibit significant serial correlation, requiring Newey-West/Bartlett HAC autocovariance estimation with lag truncation $J = h - 1 = 19$. Estimating 19 covariance lags with only 22 observations leaves just 3 effective degrees of freedom. This collapses the standard error estimate $se(\bar{f})$, inflating the test statistic and generating a spurious claim of statistical significance for LSTM at $h=20$ ($CW=1.783, p=0.0373$). Additionally, Regime 3 specifies an end date of `2026-12-31`, which extends beyond the dataset endpoint (`2026-06-30`).
- **Remediation Code Diff:**
```python
<<<<
REGIMES = [
    ("Regime 1: Rapid Tightening & Inversion", "2023-01-01", "2023-12-31"),
    ("Regime 2: Policy Plateau / Higher-for-Longer", "2024-01-01", "2024-05-31"),
    ("Regime 3: Easing Cycle & Un-inversion", "2024-06-01", "2026-12-31"),
]
...
                if len(sub) < 3:
                    continue

                act = sub["actual"].to_numpy()
                naive = sub["naive"].to_numpy()
                pred = sub[col_name].to_numpy()
...
                cw_res = clark_west_test(act, naive, pred, h=h)
====
REGIMES = [
    ("Regime 1: Rapid Tightening & Inversion", "2023-01-01", "2023-12-31"),
    ("Regime 2: Policy Plateau / Higher-for-Longer", "2024-01-01", "2024-05-31"),
    ("Regime 3: Easing Cycle & Un-inversion", "2024-06-01", "2026-06-30"),
]
...
                act = sub["actual"].to_numpy()
                naive = sub["naive"].to_numpy()
                pred = sub[col_name].to_numpy()
...
                # Statistical inference requires sufficient degrees of freedom under HAC Bartlett kernel (n >= 5*h)
                min_n_for_inference = 5 * h
                has_power = len(sub) >= min_n_for_inference

                if has_power:
                    cw_res = clark_west_test(act, naive, pred, h=h)
                    r2_oos = cw_res.get("r2_oos")
                    r2_oos_adj = cw_res.get("r2_oos_adj")
                    cw_stat = cw_res.get("cw_stat")
                    cw_p_value = cw_res.get("cw_p_value")
                    sig_better = cw_res.get("model_significantly_better", False)
                else:
                    cw_res = {}
                    # Calculate descriptive raw R2_OOS without asymptotic hypothesis claims
                    mspe_n = np.mean((act - naive) ** 2)
                    mspe_m = np.mean((act - pred) ** 2)
                    r2_oos = round(float(1.0 - mspe_m / mspe_n), 6) if mspe_n > 0 else 0.0
                    r2_oos_adj = None
                    cw_stat = None
                    cw_p_value = None
                    sig_better = False
>>>>
```

---

### Finding 3.5: Common Evaluation Sample Truncation via Unconstrained Inner Merge
- **File & Lines:** `src/dashboard/data_loader.py:187-200`
- **Analytical Risk & Impact:**  
  `build_common_forecast_dataset()` performs an inner merge: `common = common.merge(xgb_sub, on=keys, how="inner")`. The canonical evaluation sample for the 4 core models comprises exactly 745 synchronized forecast origins. If the tabular model has differing bounds (due to lag warm-up or missing dates), the inner merge silently drops origins from the core models. If XGBoost forecasts are missing, the sample reverts to 745 origins, creating an unstable, non-deterministic evaluation baseline across dashboard views.
- **Remediation Code Diff:**
```python
<<<<
    common = (
        arima
        .merge(var, on=keys, how="inner")
        .merge(vecm, on=keys, how="inner")
        .merge(lstm, on=keys, how="inner")
    )

    xgb = load_xgboost_forecasts()
    has_xgb = False
    if xgb is not None and not xgb.empty:
        xgb_sub = xgb[
            keys + ["actual", "naive", "xgboost", "lower_90", "upper_90", "lower_95", "upper_95"]
        ].rename(
            columns={
                "actual": "actual_xgb",
                "naive": "naive_xgb",
            }
        )
        common = common.merge(xgb_sub, on=keys, how="inner")
        has_xgb = True
====
    common = (
        arima
        .merge(var, on=keys, how="inner")
        .merge(vecm, on=keys, how="inner")
        .merge(lstm, on=keys, how="inner")
    )

    # Core 4-model parity check (strictly 745 common origins)
    for col in ["actual_var", "actual_vecm", "actual_lstm"]:
        if not np.allclose(common["actual_arima"], common[col], rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Actual values are inconsistent between ARIMA and {col}.")

    for col in ["naive_var", "naive_vecm", "naive_lstm"]:
        if not np.allclose(common["naive_arima"], common[col], rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Naïve forecasts are inconsistent between ARIMA and {col}.")

    xgb = load_xgboost_forecasts()
    has_xgb = False
    if xgb is not None and not xgb.empty:
        xgb_sub = xgb[
            keys + ["actual", "naive", "xgboost", "lower_90", "upper_90", "lower_95", "upper_95"]
        ].rename(
            columns={
                "actual": "actual_xgb",
                "naive": "naive_xgb",
            }
        )
        # Left join preserves the core 4-model 745 common origins
        common = common.merge(xgb_sub, on=keys, how="left")
        has_xgb = True
>>>>
```

---

## 4. Performance & Memory Optimizations

### Optimization 4.1: Elimination of Redundant Module Reloads in Streamlit
- **File & Lines:** `src/dashboard/app.py:1-10`
- **Bottleneck & Downstream Impact:**  
  Placing `importlib.reload(data_loader)` and `importlib.reload(charts)` at module scope forces Streamlit to reload both modules on every widget event or filter click. This destroys module caches, triggers redundant CSV reads from disk, and causes interface latency spikes.
- **Refactored Code:**
```python
# Remove importlib and redundant reloads entirely
import pandas as pd
import streamlit as st

from data_loader import (
    build_common_forecast_dataset,
    build_common_sample_metrics,
    # ... remaining imports
)
```

---

### Optimization 4.2: In-Memory Caching for Gold-Layer Feature Ingestion
- **File & Lines:** `src/xgboost_baseline.py:69-93`
- **Bottleneck & Downstream Impact:**  
  `load_common_sample()` rebuilds Gold features directly from disk CSVs on every invocation. When called across multiple routines (model comparison, calendar alignment, SHAP explanation), this creates severe I/O overhead.
- **Refactored Code:**
```python
_CACHED_GOLD_DF: pd.DataFrame | None = None

def load_common_sample(
    boc_path: Path | None = None,
    fred_path: Path | None = None,
    cpi_path: Path | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    global _CACHED_GOLD_DF
    if (
        _CACHED_GOLD_DF is not None
        and not force_reload
        and boc_path is None
        and fred_path is None
        and cpi_path is None
    ):
        return _CACHED_GOLD_DF.copy()

    df = build_gold_features(
        boc_path=boc_path,
        fred_path=fred_path,
        cpi_path=cpi_path,
        feature_type="all",
        save=False,
    )
    df = df.reset_index().rename(columns={"index": "date"})
    df = (
        df[["date", LEVEL_TARGET] + BASE_FEATURES]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    _CACHED_GOLD_DF = df.copy()
    return df
```

---

### Optimization 4.3: Safe Missing-Value Handling in Metric Aggregation
- **File & Lines:** `src/dashboard/data_loader.py:270-295`
- **Bottleneck & Downstream Impact:**  
  When an optional benchmark contains missing dates or un-forecast horizons, computing `errors = horizon_df["actual"] - horizon_df[forecast_col]` produces `NaN` values, causing `np.mean()` to propagate `NaN` across summary tables.
- **Refactored Code:**
```python
        for model_name, forecast_col in model_columns.items():
            valid_m = horizon_df.dropna(subset=["actual", forecast_col])
            errors = valid_m["actual"] - valid_m[forecast_col]

            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "n_forecasts": len(valid_m),
                    "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else 0.0,
                    "mae": float(np.mean(np.abs(errors))) if len(errors) > 0 else 0.0,
                }
            )
```

---

## 5. Missing Assertions & Test Cases

The test suite in PR #106 omitted critical regression checks. The following automated tests must be integrated into CI:

### Test 5.1: Prediction Interval Quantile Monotonicity on Committed Artifacts
- **Target File:** `tests/test_xgboost_baseline.py`
- **Verification Objective:** Guarantees that prediction interval artifacts satisfy mathematical non-crossing properties across all origins and horizons.
```python
def test_real_output_prediction_interval_non_crossing():
    """Verify committed XGBoost forecast artifact enforces strict interval ordering."""
    csv_path = PROJECT_ROOT / "outputs" / "r3_xgboost_forecasts.csv"
    if not csv_path.exists():
        pytest.skip("r3_xgboost_forecasts.csv not yet generated")

    df = pd.read_csv(csv_path)
    assert (df["lower_95"] <= df["lower_90"]).all(), "95% lower bound must be <= 90% lower bound"
    assert (df["lower_90"] <= df["xgboost"]).all(), "90% lower bound must be <= point prediction"
    assert (df["xgboost"] <= df["upper_90"]).all(), "Point prediction must be <= 90% upper bound"
    assert (df["upper_90"] <= df["upper_95"]).all(), "90% upper bound must be <= 95% upper bound"
```

### Test 5.2: Temporal Embargo Non-Overlap in Multi-Step CV
- **Target File:** `tests/test_xgboost_baseline.py`
- **Verification Objective:** Verifies that no training observation has a forward target label extending into the test fold.
```python
def test_temporal_embargo_prevents_target_overlap(synthetic_gold_df):
    """Verify training targets do not overlap with test fold dates for multi-step horizons."""
    from xgboost_baseline import engineer_tabular_features, make_rolling_folds

    horizons = [5, 20]
    data, _, target_cols = engineer_tabular_features(synthetic_gold_df, lags=[1, 2], horizons=horizons)
    folds = make_rolling_folds(len(data), min_train=50, n_folds=3)

    for train_end, test_end in folds:
        for h in horizons:
            embargo_end = train_end - h
            assert embargo_end < train_end
            # The maximum forward label step from the last training row must not exceed train_end
            assert embargo_end + h <= train_end
```

### Test 5.3: Clark-West Primary Battery Multiplicity Invariance ($m=12$)
- **Target File:** `tests/test_model_comparison.py`
- **Verification Objective:** Prevents accidental dilution of the canonical 12-test battery and validates model naming contracts.
```python
def test_clark_west_primary_battery_strictly_twelve_hypotheses():
    """Verify primary battery maintains exactly 4 canonical models x 3 horizons = 12 tests."""
    primary_df, sensitivity_df = run_clark_west_battery()

    assert len(primary_df) == 12, f"Primary battery must contain exactly 12 hypotheses, got {len(primary_df)}"
    assert set(primary_df["model"]) == {
        "ARIMA-AIC",
        "VAR-AIC",
        "VECM (6-var)",
        "LSTM (Tuned)",
    }
    # XGBoost must be strictly quarantined to sensitivity_df
    if "XGBoost (Experimental)" in set(sensitivity_df["model"]):
        assert len(sensitivity_df) == 9
    else:
        assert len(sensitivity_df) == 6
```

---

## 6. Operational, Configuration & Determinism Fixes

### 6.1 Strict Dependency Pinning (`requirements.txt`)
- **Issue:** Line 39 specifies an open-ended constraint: `xgboost>=2.0.0`.
- **Risk:** Minor version upgrades introduce breaking changes to objective implementations, tree-splitting heuristics, and multi-threading defaults.
- **Fix:** Pin strictly to `xgboost==3.4.1`.

### 6.2 Valet API Series Identifier Corrections (`app.py`)
- **Issue:** Lines 239–242 document Bank of Canada series codes as `BD.GOC.10Y.M` and `BD.GOC.2Y.M`.
- **Risk:** These codes represent monthly or legacy identifiers, misleading stakeholders.
- **Fix:** Update documentation strings to official daily Valet series identifiers:
  - Canadian 10Y Benchmark Yield: `BD.CDN.10YR.DQ.YLD`
  - Canadian 2Y Benchmark Yield: `BD.CDN.2YR.DQ.YLD`

### 6.3 Line Ending Normalization (`.gitattributes`)
- **Issue:** Windows CRLF conversions resulted in Git interpreting all 845 lines of `src/dashboard/data_loader.py` as modified.
- **Fix:** Add `.gitattributes` to the project root:
```gitattributes
* text=auto eol=lf
*.csv text eol=lf
*.py text eol=lf
*.ipynb text eol=lf
```

---

## 7. Remediation & Governance Roadmap

1. **Branch Management:** Revert PR #106 (`git revert 7050cd8`).
2. **Apply Refactorings:** Apply Findings 3.1 through 3.5 in branch `feat/101-xgboost-benchmark-v2`.
3. **Pipeline Regeneration:** Re-run `src/xgboost_baseline.py` and `src/model_comparison.py` to regenerate all dependent artifacts:
   - `outputs/r3_xgboost_forecasts.csv`
   - `outputs/r3_xgboost_prediction_intervals.csv`
   - `outputs/r3_regime_segmented_metrics.csv`
   - `outputs/clark_west_test_results.csv`
4. **Verification Gate:** Ensure all 32 unit tests pass (`pytest tests/test_xgboost_baseline.py tests/test_model_comparison.py`) with zero test contract modifications.
5. **Re-Submission:** Submit clean PR under `feat/101-xgboost-benchmark-v2` for final lead sign-off.
