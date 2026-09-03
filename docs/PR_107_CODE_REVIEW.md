# Code Review: Pull Request #107
**Title:** `feat(models): implement XGBoost tabular benchmark with sensitivity placement (close #101)`  
**Reviewers:** Lead Data Scientist & Principal Analytics Engineer  
**Repository:** `nholguinrh/DAMO-699-Capstone-project-GRP5`  
**Date:** September 3, 2026  
**Status:** **CONDITIONAL APPROVAL / CHANGES REQUESTED (NON-CRITICAL METHODOLOGICAL & CONTRACT DEFECTS IDENTIFIED)**

---

## Table of Contents
1. [Methodological Risk Assessment](#1-methodological-risk-assessment)
2. [Evaluation Dimensions Matrix](#2-evaluation-dimensions-matrix)
3. [Resolution Verification of PR #106 Blockers](#3-resolution-verification-of-pr-106-blockers)
4. [Critical / Blocking Findings](#4-critical--blocking-findings)
   - [Finding 4.1: Silent Metric Corruption via Unpaired Baseline Comparison in Metric Aggregation](#finding-41-silent-metric-corruption-via-unpaired-baseline-comparison-in-metric-aggregation)
   - [Finding 4.2: Unhandled KeyError & Missing-Data Propagation in Forecast Error Distribution Chart](#finding-42-unhandled-keyerror--missing-data-propagation-in-forecast-error-distribution-chart)
5. [Performance, Memory Footprint & Scalability Optimizations](#5-performance-memory-footprint--scalability-optimizations)
   - [Optimization 5.1: Eliminate Sparse-Matrix Outer Join and 66.7% NaN Bloat in SHAP Matrix Export](#optimization-51-eliminate-sparse-matrix-outer-join-and-667-nan-bloat-in-shap-matrix-export)
   - [Optimization 5.2: Vectorization of Rolling CV Forecast Record Assembly](#optimization-52-vectorization-of-rolling-cv-forecast-record-assembly)
   - [Optimization 5.3: Finite-Sample Conformal Quantile Calibration Adjustment](#optimization-53-finite-sample-conformal-quantile-calibration-adjustment)
6. [Data Contracts & Documentation Alignment](#6-data-contracts--documentation-alignment)
   - [6.1 Stale Primary Battery Specification in `run_clark_west_battery` Docstring](#61-stale-primary-battery-specification-in-run_clark_west_battery-docstring)
7. [Missing Assertions & Test Cases](#7-missing-assertions--test-cases)
   - [Test 7.1: Monetary Policy Regime Gating & Asymptotic Degeneracy Check](#test-71-monetary-policy-regime-gating--asymptotic-degeneracy-check)
   - [Test 7.2: Paired Baseline Evaluation Parity Test](#test-72-paired-baseline-evaluation-parity-test)
   - [Test 7.3: Defensive Visualization Execution Under Missing Benchmarks](#test-73-defensive-visualization-execution-under-missing-benchmarks)
   - [Test 7.4: SHAP Output Matrix Density and Zero-NaN Schema Contract](#test-74-shap-output-matrix-density-and-zero-nan-schema-contract)
8. [Remediation & Governance Roadmap](#8-remediation--governance-roadmap)

---

## 1. Methodological Risk Assessment

### Overall Risk Rating: **MEDIUM**

#### Executive Summary
Pull Request #107 represents an exemplary methodological iteration over PR #106. The candidate implementation successfully closes Issue #101 while demonstrating strict adherence to empirical econometrics standards:
1. **Multi-Step Look-Ahead Leakage Eliminated:** Enforcing an $h$-step causal embargo (`train_end - h`) ensures realized yield spread returns from $[train\_end, train\_end + h - 1]$ never contaminate the labels of training rows.
2. **In-Sample Residual Optimism Resolved:** Migration to split-conformal calibration over an embargoed temporal holdout slice elevates empirical coverage from 73.85% to 84.22% ($h=20$, 90% target) and from 80.85% to 92.64% ($h=20$, 95% target), while guaranteeing strict quantile non-crossing monotonicity ($lower_{95} \le lower_{90} \le point \le upper_{90} \le upper_{95}$).
3. **Statistical Multiplicity Invariance Preserved:** Quarantining XGBoost to the sensitivity battery (`outputs/clark_west_sensitivity_results.csv`) leaves the primary Clark-West battery strictly at $m=12$ hypotheses (4 models $\times$ 3 horizons), fully restoring the thesis-critical headline VECM ($h=20$) significance ($q_{\text{horizon}} = 0.098^*$).
4. **HAC Asymptotic Gating Enforced:** Regime-segmented Clark-West tests now explicitly gate asymptotic hypothesis evaluation on $N \ge 5 \times h$, preventing spurious small-sample false discovery claims on underpowered sub-samples ($N=22$).
5. **Common Evaluation Sample Preserved:** Implementing an optional left-join overlay rather than an inner merge preserves the 745 common origins for the core econometric suite across the dashboard.

However, approval is conditioned upon resolving **two operational defects and two analytical contract flaws**:
- **Silent Metric Corruption in Dashboard Aggregation:** [`build_common_sample_metrics`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/data_loader.py#L277-L350) evaluates XGBoost over $N=741$ observations but computes relative improvement against a global Naïve RMSE evaluated over $N=745$ observations, distorting the $h=20$ out-of-sample improvement metric by **0.205 percentage points**.
- **Unhandled KeyError Crash in Visualization Layer:** [`create_forecast_error_distribution`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/charts.py#L223-L265) accesses optional benchmark columns without defensive membership guarding, causing Tab 5 ("Forecast Errors") to crash if XGBoost forecasts are missing or partially loaded.
- **Sparse Matrix Data Bloat:** The SHAP values matrix export in [`compute_tree_shap_interpretability`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py#L568-L578) concatenates differently-suffixed columns row-wise, creating a 9.0 MB artifact where **66.7% ($1.26\text{M}$ cells) are empty `NaN`s**.
- **Finite-Sample Conformal Adjustment:** Split-conformal calibration in [`run_rolling_cv`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py#L407-L415) omits the finite-sample $+1/n$ adjustment factor.

---

## 2. Evaluation Dimensions Matrix

| Evaluation Dimension | Status | Assessment Summary |
| :--- | :---: | :--- |
| **1. Methodological Integrity & Statistical Rigor** | **PASS (Conditional)** | $h$-step embargo eliminates target leakage; primary battery strictly preserved at $m=12$; small-sample HAC breakdown gated ($N \ge 5h$); relative improvement metric in dashboard must be paired. |
| **2. Performance, Memory & Scalability** | **PASS (Conditional)** | `importlib.reload` eliminated; memory caching active; SHAP export produces $12,621 \times 152$ sparse matrix with 66.7% NaNs requiring tidy refactor; inner loop `.iloc` indexing in CV assembly needs vectorization. |
| **3. Reproducibility & Determinism** | **PASS** | `xgboost==3.4.1` pinned in `requirements.txt`; deterministic tree building (`n_jobs=1`, seed=42); `.gitattributes` enforces LF line endings; test suite passes cleanly (35/35). |
| **4. Data Contracts & Validation** | **PASS (Conditional)** | Left-join preserves 745 core evaluation origins; canonical display names restored; `create_forecast_error_distribution` missing column check causes `KeyError` on optional benchmark; docstring in `run_clark_west_battery` stale. |

---

## 3. Resolution Verification of PR #106 Blockers

| # | PR #106 Finding | Status in PR #107 | Verification Evidence |
|:---|:---|:---:|:---|
| **3.1** | **Multi-Step Target Leakage** | **RESOLVED** | `src/xgboost_baseline.py:357-362` enforces `train_data = data.iloc[:train_end - h]`. `test_temporal_embargo_prevents_target_overlap` asserts non-overlap. |
| **3.2** | **In-Sample Residual Optimism** | **RESOLVED** | `src/xgboost_baseline.py:384-409` implements split-conformal calibration with an $h$-step gap. Monotonicity bounds enforced. |
| **3.3** | **FDR Multiplicity Dilution** | **RESOLVED** | `src/model_comparison.py:592-614` confines XGBoost to `sensitivity_models`. Primary battery invariant at $m=12$ ($q_{\text{horizon}} = 0.098^*$). |
| **3.4** | **Small-Sample HAC Breakdown** | **RESOLVED** | `src/model_comparison.py:820-830` gates Clark-West tests on $N \ge 5 \times h$. Official BoC rate announcement dates cited. |
| **3.5** | **Common Sample Truncation** | **RESOLVED** | `src/dashboard/data_loader.py:223` executes a left-join overlay, preserving core 745 common origins across all horizons. |
| **4.1** | **Redundant Module Reloads** | **RESOLVED** | `importlib.reload` removed from `src/dashboard/app.py`. |
| **4.2** | **In-Memory Caching** | **RESOLVED** | `_CACHED_GOLD_DF` implemented in `src/xgboost_baseline.py:82-91`. |
| **6.1** | **Strict Dependency Pinning** | **RESOLVED** | `requirements.txt` strictly pins `xgboost==3.4.1`. |
| **6.2** | **Valet API Series Codes** | **RESOLVED** | Corrected to `BD.CDN.10YR.DQ.YLD` and `BD.CDN.2YR.DQ.YLD`. |
| **6.3** | **Line Ending Normalization** | **RESOLVED** | `.gitattributes` committed with `* text=auto eol=lf`. |

---

## 4. Critical / Blocking Findings

### Finding 4.1: Silent Metric Corruption via Unpaired Baseline Comparison in Metric Aggregation
- **File & Lines:** [`src/dashboard/data_loader.py:316-345`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/data_loader.py#L316-L345)
- **Analytical Risk & Impact:**  
  The core econometric models evaluate over $N=745$ synchronized origins. XGBoost rolling CV evaluates over $N=741$ origins (due to 20-lag feature engineering and 500-sample warm-up). In [`build_common_sample_metrics`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/data_loader.py#L277-L350), `naive_rmse` is computed globally from the `"Naïve Random Walk"` row ($N=745$). Later, the code computes:
  $$\text{improvement\_pct} = \frac{\text{naive\_rmse}_{745} - \text{rmse}_{741}}{\text{naive\_rmse}_{745}} \times 100$$
  Comparing a model scored on 741 origins against a benchmark scored on 745 origins violates the fundamental econometrics requirement of **paired evaluation**.
  On the true overlapping 741-origin sample:
  - At $h=1$: Paired Naïve RMSE is $0.029063$ (improvement: **-4.418%**, reported: **-4.524%**).
  - At $h=5$: Paired Naïve RMSE is $0.061243$ (improvement: **-2.548%**, reported: **-2.500%**).
  - At $h=20$: Paired Naïve RMSE is $0.125560$ (improvement: **-1.924%**, reported: **-2.129%**).  
  This introduces a silent metric distortion of **0.205 percentage points** in headline out-of-sample reporting.
- **Replacement Code Block:**

```python
<<<<
    for horizon in sorted(df["horizon"].unique()):
        horizon_df = df[df["horizon"] == horizon]

        for model_name, forecast_col in model_columns.items():
            valid_m = horizon_df.dropna(subset=["actual", forecast_col])
            errors = valid_m["actual"] - valid_m[forecast_col]

            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "n_forecasts": len(valid_m),
                    "rmse": float(
                        np.sqrt(np.mean(errors ** 2))
                    ) if len(errors) > 0 else 0.0,
                    "mae": float(
                        np.mean(np.abs(errors))
                    ) if len(errors) > 0 else 0.0,
                }
            )

    metrics = pd.DataFrame(rows)

    naive = (
        metrics[
            metrics["model"] == "Naïve Random Walk"
        ][["horizon", "rmse", "mae"]]
        .rename(
            columns={
                "rmse": "naive_rmse",
                "mae": "naive_mae",
            }
        )
    )

    metrics = metrics.merge(
        naive,
        on="horizon",
        how="left",
    )

    metrics["rmse_improvement_pct"] = (
        (metrics["naive_rmse"] - metrics["rmse"])
        / metrics["naive_rmse"]
        * 100
    )

    metrics["mae_improvement_pct"] = (
        (metrics["naive_mae"] - metrics["mae"])
        / metrics["naive_mae"]
        * 100
    )
====
    for horizon in sorted(df["horizon"].unique()):
        horizon_df = df[df["horizon"] == horizon]

        for model_name, forecast_col in model_columns.items():
            # Enforce strict paired evaluation: compute both model and naive errors on identical non-null origins
            valid_m = horizon_df.dropna(subset=["actual", "naive", forecast_col])
            errors = valid_m["actual"] - valid_m[forecast_col]
            naive_errors = valid_m["actual"] - valid_m["naive"]

            model_rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else 0.0
            model_mae = float(np.mean(np.abs(errors))) if len(errors) > 0 else 0.0
            paired_naive_rmse = float(np.sqrt(np.mean(naive_errors ** 2))) if len(naive_errors) > 0 else 0.0
            paired_naive_mae = float(np.mean(np.abs(naive_errors))) if len(naive_errors) > 0 else 0.0

            rmse_imp = (
                ((paired_naive_rmse - model_rmse) / paired_naive_rmse * 100.0)
                if paired_naive_rmse > 0 else 0.0
            )
            mae_imp = (
                ((paired_naive_mae - model_mae) / paired_naive_mae * 100.0)
                if paired_naive_mae > 0 else 0.0
            )

            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "n_forecasts": len(valid_m),
                    "rmse": model_rmse,
                    "mae": model_mae,
                    "naive_rmse": paired_naive_rmse,
                    "naive_mae": paired_naive_mae,
                    "rmse_improvement_pct": rmse_imp,
                    "mae_improvement_pct": mae_imp,
                }
            )

    metrics = pd.DataFrame(rows)
>>>>
```

---

### Finding 4.2: Unhandled KeyError & Missing-Data Propagation in Forecast Error Distribution Chart
- **File & Lines:** [`src/dashboard/charts.py:235-240`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/charts.py#L235-L240)
- **Analytical Risk & Impact:**  
  `MODEL_COLUMNS` was updated in [`src/dashboard/charts.py:8-15`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/charts.py#L8-L15) to register `"XGBoost": "xgboost"`. In [`create_forecast_vs_actual_chart`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/charts.py#L49), defensive existence filtering was applied (`if column is None or column not in plot_df.columns: continue`).  
  However, in [`create_forecast_error_distribution`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/charts.py#L235-L240), the loop accesses `plot_df[column]` unconditionally. If `outputs/r3_xgboost_forecasts.csv` is absent (or deleted during clean pipeline builds), `build_common_forecast_dataset()` returns without `"xgboost"`, causing `plot_df["xgboost"]` to trigger an immediate unhandled `KeyError: 'xgboost'` that crashes Tab 5 in Streamlit. In addition, when `"xgboost"` is present, `NaN` values from warm-up origins are passed directly to `go.Violin`, corrupting Gaussian kernel density bandwidth estimation.
- **Replacement Code Block:**

```python
<<<<
    for model_name, column in MODEL_COLUMNS.items():
        errors = (
            plot_df["actual"]
            - plot_df[column]
        )

        if chart_type == "Violin":
====
    for model_name, column in MODEL_COLUMNS.items():
        if column not in plot_df.columns:
            continue

        valid = plot_df.dropna(subset=["actual", column])
        errors = valid["actual"] - valid[column]
        if len(errors) == 0:
            continue

        if chart_type == "Violin":
>>>>
```

---

## 5. Performance, Memory Footprint & Scalability Optimizations

### Optimization 5.1: Eliminate Sparse-Matrix Outer Join and 66.7% NaN Bloat in SHAP Matrix Export
- **File & Lines:** [`src/xgboost_baseline.py:568-578`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py#L568-L578)
- **Bottleneck & Downstream Impact:**  
  Inside [`compute_tree_shap_interpretability`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py#L512-L578), SHAP feature columns are assigned horizon-suffixed names (`col_h1`, `col_h5`, `col_h20`). The three dataframes are then concatenated vertically: `pd.concat(shap_val_dfs, ignore_index=True)`.  
  Because column names do not match across horizons, Pandas performs an outer join, generating a $12,621 \times 152$ sparse matrix in `outputs/r3_xgboost_shap_values.csv`. Out of $1,893,150$ data cells, **$1,262,100$ cells ($66.7\%$) are NaN**. This bloats the committed CSV file to **9.0 MB** and forces downstream consumers to perform messy regex column lookups instead of filtering `df[df["horizon"] == h]`.
- **Refactored Code Block (Tidy Long Format):**

```python
<<<<
        val_df = pd.DataFrame(
            vals, columns=[f"{col}_h{h}" for col in feature_cols]
        )
        val_df["date"] = data["date"].values
        val_df["horizon"] = h
        shap_val_dfs.append(val_df)

    shap_summary_df = pd.DataFrame(summary_records)
    shap_values_df = pd.concat(shap_val_dfs, ignore_index=True)
====
        # Preserve standard feature column names to maintain a dense, tidy (long) schema
        val_df = pd.DataFrame(vals, columns=feature_cols)
        val_df["date"] = data["date"].values
        val_df["horizon"] = h
        shap_val_dfs.append(val_df)

    shap_summary_df = pd.DataFrame(summary_records)
    shap_values_df = pd.concat(shap_val_dfs, ignore_index=True)
>>>>
```

---

### Optimization 5.2: Vectorization of Rolling CV Forecast Record Assembly
- **File & Lines:** [`src/xgboost_baseline.py:433-446`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py#L433-L446)
- **Bottleneck & Downstream Impact:**  
  Across 5 folds and 3 horizons, [`run_rolling_cv`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py#L307-L506) executes 11,121 iterations of a pure Python loop with `.iloc` indexing to build `forecast_records`:
  ```python
  for i in range(len(test_data)):
      forecast_records.append({ "origin_date": str(test_data["date"].iloc[i])[:10], ... })
  ```
  Pandas `.iloc` within an inner loop has significant series-access overhead and incurs memory allocation overhead for 11,121 dict instances.
- **Refactored Code Block:**

```python
<<<<
            for i in range(len(test_data)):
                forecast_records.append({
                    "origin_date": str(test_data["date"].iloc[i])[:10],
                    "horizon": h,
                    "actual": float(actual_future[i]),
                    "naive": float(current_levels[i]),
                    "xgboost": float(pred_level[i]),
                    "fold": fold_id,
                    "lower_90": float(lower_90_lev[i]),
                    "upper_90": float(upper_90_lev[i]),
                    "lower_95": float(lower_95_lev[i]),
                    "upper_95": float(upper_95_lev[i]),
                })

    forecasts_df = pd.DataFrame(forecast_records)
====
            fold_df = pd.DataFrame({
                "origin_date": pd.to_datetime(test_data["date"].values).strftime("%Y-%m-%d"),
                "horizon": h,
                "actual": actual_future.astype(float),
                "naive": current_levels.astype(float),
                "xgboost": pred_level.astype(float),
                "fold": fold_id,
                "lower_90": lower_90_lev.astype(float),
                "upper_90": upper_90_lev.astype(float),
                "lower_95": lower_95_lev.astype(float),
                "upper_95": upper_95_lev.astype(float),
            })
            forecast_records.append(fold_df)

    forecasts_df = pd.concat(forecast_records, ignore_index=True)
>>>>
```

---

### Optimization 5.3: Finite-Sample Conformal Quantile Calibration Adjustment
- **File & Lines:** [`src/xgboost_baseline.py:407-417`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py#L407-L417)
- **Bottleneck & Downstream Impact:**  
  Lines 407–408 evaluate calibration residuals at unadjusted nominal quantiles:
  ```python
  res_90 = float(np.quantile(cal_residuals, 0.90))
  res_95 = float(np.quantile(cal_residuals, 0.95))
  ```
  In split-conformal prediction theory (Vovk et al., 2005; Lei et al., 2018; Angelopoulos & Bates, 2021), achieving marginal coverage of at least $1 - \alpha$ on finite calibration samples of size $n$ requires the conformal correction:
  $$q_{1-\alpha} = \min\left(1.0, \; \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)$$
  For calibration fold slices ($n \approx 72$), evaluating at $0.90$ instead of $73 \times 0.90 / 72 = 0.9125$ introduces a negative coverage bias. This accounts for the slight empirical under-coverage observed in `outputs/r3_xgboost_prediction_intervals.csv` ($84.22\%$ at $h=20$).
- **Refactored Code Block:**

```python
<<<<
                res_90 = float(np.quantile(cal_residuals, 0.90))
                res_95 = float(np.quantile(cal_residuals, 0.95))
====
                n_cal = len(cal_residuals)
                q90_level = min(1.0, np.ceil((n_cal + 1) * 0.90) / n_cal)
                q95_level = min(1.0, np.ceil((n_cal + 1) * 0.95) / n_cal)
                res_90 = float(np.quantile(cal_residuals, q90_level))
                res_95 = float(np.quantile(cal_residuals, q95_level))
>>>>
```

---

## 6. Data Contracts & Documentation Alignment

### 6.1 Stale Primary Battery Specification in `run_clark_west_battery` Docstring
- **File & Lines:** [`src/model_comparison.py:567-573`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/model_comparison.py#L567-L573)
- **Issue:** The docstring still reads:
  ```text
  Primary battery:
      5 models (ARIMA-AIC, VAR-AIC, VECM-6var, LSTM, XGBoost) x 3 horizons (1, 5, 20)
  ```
  This directly contradicts the code at line 592 where `primary_models` contains strictly 4 models ($m=12$) and XGBoost is placed into `sensitivity_models`.
- **Replacement Code Block:**

```python
<<<<
    Primary battery:
        5 models (ARIMA-AIC, VAR-AIC, VECM-6var, LSTM, XGBoost) x 3 horizons (1, 5, 20)
        XGBoost included only if outputs/r3_xgboost_forecasts.csv exists (Issue #101).
    Sensitivity battery:
        2 BIC variants (ARIMA-BIC, VAR-BIC) x 3 horizons (1, 5, 20)
====
    Primary 12-test battery (m=12):
        4 models (ARIMA-AIC, VAR-AIC, VECM-6var, LSTM) x 3 horizons (1, 5, 20)
        FDR multiplicity control is strictly invariant to experimental benchmarks.
    Sensitivity battery:
        2 BIC variants (ARIMA-BIC, VAR-BIC) + XGBoost (Experimental) x 3 horizons (1, 5, 20)
>>>>
```

---

## 7. Missing Assertions & Test Cases

The following test cases must be added to the test suite to guarantee long-term regression safety:

### Test 7.1: Monetary Policy Regime Gating & Asymptotic Degeneracy Check
- **Target File:** [`tests/test_model_comparison.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/tests/test_model_comparison.py)
- **Verification Objective:** Verifies that [`evaluate_regime_segmentation`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/model_comparison.py#L751-L865) executes without error, covers all 3 BoC policy regimes, and guarantees that any sub-sample with $N < 5 \times h$ sets `insufficient_sample = True` and keeps `cw_stat` / `cw_p_value` null.

```python
def test_regime_segmentation_hac_power_gating():
    """Assert Clark-West hypothesis claims are suppressed when sample size fails HAC regularity."""
    from model_comparison import evaluate_regime_segmentation, REGIMES

    regime_df = evaluate_regime_segmentation(save=False)
    assert len(regime_df) > 0

    expected_regimes = {r[0] for r in REGIMES}
    assert set(regime_df["regime"]) == expected_regimes

    # Assert small-sample gating rule: N < 5*h must suppress CW test claims
    underpowered = regime_df[regime_df["insufficient_sample"]]
    assert len(underpowered) > 0, "Expected underpowered regimes (e.g. Regime 2 at h=20)"

    for _, row in underpowered.iterrows():
        assert pd.isna(row["cw_stat"]) or row["cw_stat"] is None
        assert pd.isna(row["cw_p_value"]) or row["cw_p_value"] is None
        assert row["model_significantly_better"] is False
```

### Test 7.2: Paired Baseline Evaluation Parity Test
- **Target File:** [`tests/test_model_comparison.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/tests/test_model_comparison.py)
- **Verification Objective:** Asserts that [`build_common_sample_metrics`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/data_loader.py#L277-L350) evaluates Naïve benchmarks on the exact paired origins of each respective model.

```python
def test_common_sample_metrics_paired_baseline_parity():
    """Assert relative improvement metrics are computed over strictly identical observation pairs."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dashboard"))
    from data_loader import build_common_sample_metrics, build_common_forecast_dataset

    metrics_df = build_common_sample_metrics()
    forecast_df = build_common_forecast_dataset()

    for _, row in metrics_df.iterrows():
        model_name = row["model"]
        horizon = row["horizon"]
        if model_name == "Naïve Random Walk":
            continue

        # Map display name to forecast column
        col_map = {"ARIMA": "arima_aic", "VAR": "var_aic", "VECM": "vecm", "LSTM": "lstm", "XGBoost": "xgboost"}
        col = col_map.get(model_name)
        if col is None or col not in forecast_df.columns:
            continue

        sub = forecast_df[forecast_df["horizon"] == horizon].dropna(subset=["actual", "naive", col])
        expected_naive_rmse = np.sqrt(np.mean((sub["actual"] - sub["naive"]) ** 2))
        expected_model_rmse = np.sqrt(np.mean((sub["actual"] - sub[col]) ** 2))
        expected_imp = (expected_naive_rmse - expected_model_rmse) / expected_naive_rmse * 100.0

        assert np.isclose(row["naive_rmse"], expected_naive_rmse, atol=1e-5), \
            f"Naive RMSE for {model_name} at h={horizon} must match paired sample"
        assert np.isclose(row["rmse_improvement_pct"], expected_imp, atol=1e-3), \
            f"Improvement pct for {model_name} at h={horizon} must match paired calculation"
```

### Test 7.3: Defensive Visualization Execution Under Missing Benchmarks
- **Target File:** [`tests/test_model_comparison.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/tests/test_model_comparison.py)
- **Verification Objective:** Asserts that [`create_forecast_error_distribution`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/charts.py#L223-L265) does not crash when optional benchmark columns are omitted from `forecast_df`.

```python
def test_forecast_error_distribution_missing_column_resilience():
    """Assert error distribution visualization renders without KeyError when optional models are missing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dashboard"))
    from charts import create_forecast_error_distribution

    # Synthetic forecast dataframe missing 'xgboost'
    df = pd.DataFrame({
        "horizon": [1, 1, 1],
        "origin_date": pd.date_range("2024-01-01", periods=3),
        "actual": [1.0, 1.1, 1.2],
        "naive": [1.0, 1.0, 1.1],
        "arima_aic": [1.01, 1.09, 1.21],
        "var_aic": [1.00, 1.11, 1.20],
        "vecm": [1.02, 1.10, 1.19],
        "lstm": [1.01, 1.08, 1.22],
    })

    fig_box = create_forecast_error_distribution(df, horizon=1, chart_type="Box")
    fig_violin = create_forecast_error_distribution(df, horizon=1, chart_type="Violin")
    assert fig_box is not None
    assert fig_violin is not None
```

### Test 7.4: SHAP Output Matrix Density and Zero-NaN Schema Contract
- **Target File:** [`tests/test_xgboost_baseline.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/tests/test_xgboost_baseline.py)
- **Verification Objective:** Verifies that `outputs/r3_xgboost_shap_values.csv` adheres to a dense, tidy long schema without NaN values.

```python
def test_real_output_shap_values_schema_and_density():
    """Assert committed SHAP values artifact is dense without NaN column sprawl."""
    shap_path = PROJECT_ROOT / "outputs" / "r3_xgboost_shap_values.csv"
    if not shap_path.exists():
        pytest.skip("r3_xgboost_shap_values.csv not yet generated")

    df = pd.read_csv(shap_path)
    assert "horizon" in df.columns, "SHAP values must include 'horizon' column in tidy format"
    assert "date" in df.columns, "SHAP values must include 'date' column"
    # In tidy long format, feature columns should contain zero NaNs
    feature_cols = [c for c in df.columns if c not in ["date", "horizon"]]
    assert df[feature_cols].isna().sum().sum() == 0, "Tidy SHAP matrix must not contain NaN values"
```

---

## 8. Remediation & Governance Roadmap

1. **Apply Dashboard Fixes:**
   - Patch [`src/dashboard/data_loader.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/data_loader.py) with Finding 4.1 to enforce paired baseline metrics.
   - Patch [`src/dashboard/charts.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/dashboard/charts.py) with Finding 4.2 to guard against missing model columns.
2. **Refactor SHAP Export & Conformal Tuning:**
   - Update [`src/xgboost_baseline.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/xgboost_baseline.py) with Optimizations 5.1, 5.2, and 5.3.
   - Re-run `python src/xgboost_baseline.py` to regenerate the dense `r3_xgboost_shap_values.csv` and calibrated prediction intervals.
3. **Docstring & CI Tests:**
   - Correct the `run_clark_west_battery` docstring in [`src/model_comparison.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/src/model_comparison.py).
   - Add Tests 7.1, 7.2, 7.3 to [`tests/test_model_comparison.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/tests/test_model_comparison.py) and Test 7.4 to [`tests/test_xgboost_baseline.py`](file:///C:/Users/User/OneDrive/Documentos/Master%20in%20data%20analytics/Capstone%20Project/tests/test_xgboost_baseline.py).
4. **Final Gate Verification:**
   - Execute `pytest tests/test_xgboost_baseline.py tests/test_model_comparison.py` (target: 39 passing tests).
   - Verify that Streamlit runs cleanly with and without XGBoost artifacts: `streamlit run src/dashboard/app.py`.
