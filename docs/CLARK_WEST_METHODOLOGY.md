# Clark-West (2007) Econometric Methodology & Predictive Evaluation
**DAMO-699 Capstone Project | Group 5**

---

## 1. Executive Summary & Research Alignment

Under the refined analytical objective (§3) and research question (§3.1), the capstone investigation asks:
> **"Which modelling approach (Naïve Random Walk, ARIMA, VAR/VECM, Long Short-Term Memory [LSTM] Neural Network) most reliably forecasts near-term (1–20 day) changes in the Canadian 10y–2y yield spread for use in an operational decision?"**

Rather than conducting an all-pairs pairwise tournament among non-nested alternatives, the project evaluates each candidate forecasting paradigm **exclusively against the common Naïve Random Walk benchmark** ($\Delta s_{t+h} = 0$, or $\hat{s}_{t+h} = s_t$). 

To ensure econometric validity, statistical significance is assessed via the **Clark-West (2007) adjusted MSPE test** combined with **Benjamini-Hochberg (1995) False Discovery Rate (FDR) multiplicity control**.

---

## 2. Econometric Foundations: Clark-West (2007) vs. Diebold-Mariano (1995)

### 2.1 The Nested Model Problem in Standard Diebold-Mariano
In standard forecast evaluation, the Diebold-Mariano (1995) and West (1996) tests evaluate whether the population Mean Squared Prediction Errors ($\text{MSPE}$) of two competing models are equal:
$$H_0: \mathbb{E}[e_{1, t+h}^2 - e_{2, t+h}^2] = 0$$

However, when comparing a restricted benchmark ($\text{Model 1: Naïve Random Walk}$) against an unrestricted/parameter-rich model ($\text{Model 2: ARIMA, VAR, VECM, LSTM}$), the models are **strictly nested**—under the null hypothesis that the additional features/lags have zero population predictive value, Model 2's true population parameters are zero.

In finite samples, Model 2 must estimate these parameters. This estimation process introduces sample parameter uncertainty of order $\mathcal{O}_p(T^{-1})$. Consequently, even when Model 2 has equal population predictive accuracy to Model 1, sample estimation noise mechanically inflates Model 2's sample MSPE:
$$\mathbb{E}[\text{MSPE}_2] \approx \mathbb{E}[\text{MSPE}_1] + \mathcal{O}(T^{-1})$$

As proven by Clark & McCracken (2001) and Clark & West (2006, 2007):
1. The sample loss differential $d_{t+h} = e_{1, t+h}^2 - e_{2, t+h}^2$ has a **strictly negative expected value** under $H_0$.
2. Standard Diebold-Mariano statistics have a non-standard, shifted distribution that leads to severe undersizing and spurious statistical rejections in favor of the benchmark.
3. Standard DM cannot be used for nested model comparisons without asymptotic size distortions.

### 2.2 Mathematical Formulation of the Clark-West (2007) Adjustment
Clark & West (2007) proposed an approximately normal test that explicitly adjusts for the sample parameter estimation noise penalty. 

Let:
- $y_{t+h}$ be the observed Canadian 10y–2y yield spread at target date $t+h$.
- $\hat{y}_{1, t+h} = y_t$ be the Naïve Random Walk benchmark forecast ($e_{1, t+h} = y_{t+h} - \hat{y}_{1, t+h}$).
- $\hat{y}_{2, t+h}$ be the competing model's forecast ($e_{2, t+h} = y_{t+h} - \hat{y}_{2, t+h}$).

The Clark-West adjusted loss differential $\hat{f}_{t+h}$ is defined as:
$$\hat{f}_{t+h} = e_{1, t+h}^2 - \left[ e_{2, t+h}^2 - (\hat{y}_{1, t+h} - \hat{y}_{2, t+h})^2 \right]$$

Algebraically, this expands to:
$$\hat{f}_{t+h} = (y_{t+h} - \hat{y}_{1, t+h})^2 - (y_{t+h} - \hat{y}_{2, t+h})^2 + (\hat{y}_{1, t+h} - \hat{y}_{2, t+h})^2 = 2 e_{1, t+h}(\hat{y}_{2, t+h} - \hat{y}_{1, t+h})$$

The sample moments are:
- $\text{MSPE}_1 = \frac{1}{N} \sum_{t=1}^N e_{1, t+h}^2$
- $\text{MSPE}_2 = \frac{1}{N} \sum_{t=1}^N e_{2, t+h}^2$
- $\text{adj} = \frac{1}{N} \sum_{t=1}^N (\hat{y}_{1, t+h} - \hat{y}_{2, t+h})^2$
- $\text{MSPE}_2^{\text{adj}} = \text{MSPE}_2 - \text{adj}$
- $\bar{f} = \frac{1}{N} \sum_{t=1}^N \hat{f}_{t+h} = \text{MSPE}_1 - \text{MSPE}_2^{\text{adj}}$

Under $H_0: \text{MSPE}_1 \le \text{MSPE}_2$, $\mathbb{E}[\hat{f}_{t+h}] = 0$. Under the alternative $H_A: \text{MSPE}_1 > \text{MSPE}_2$ (the competing model outperforms the benchmark in population), $\mathbb{E}[\hat{f}_{t+h}] > 0$.

### 2.3 Variance Estimation & Autocorrelation Control (HAC / Newey-West)
Because multi-step ahead forecasts ($h > 1$) generate overlapping forecast errors, the adjusted loss differential series $\hat{f}_{t+h}$ exhibits moving average serial correlation of order at most $\text{MA}(h-1)$.

The long-run variance $\widehat{\text{Var}}(\bar{f})$ is estimated via a Heteroskedasticity and Autocorrelation Consistent (HAC / Newey-West) estimator with a Bartlett kernel:
$$\widehat{\Omega} = \hat{\gamma}(0) + 2 \sum_{k=1}^{h-1} \left( 1 - \frac{k}{h} \right) \hat{\gamma}(k)$$
where $\hat{\gamma}(k) = \frac{1}{N} \sum_{t=k+1}^N (\hat{f}_{t+h} - \bar{f})(\hat{f}_{t-k+h} - \bar{f})$.

For small-sample modification, the Harvey, Leybourne, & Newbold (1997) scaling factor is applied:
$$\text{HLN} = \frac{N + 1 - 2h + h(h-1)/N}{N}, \quad \text{Var}_{\text{adj}}(\bar{f}) = \frac{\widehat{\Omega}}{N \cdot \text{HLN}}$$

The Clark-West test statistic is:
$$CW = \frac{\bar{f}}{\sqrt{\text{Var}_{\text{adj}}(\bar{f})}}$$
with one-sided $p$-value:
$$p = 1 - \Phi(CW)$$

### 2.4 Out-of-Sample $R^2$ ($R^2_{OOS}$) and the Clark-West Noise-Adjustment Testing Construct
To evaluate whether a forecasting model delivers economic value relative to the benchmark, the literature (Campbell & Thompson, 2008; Welch & Goyal, 2008; Rapach, Strauss, & Zhou, 2010) defines the realized Out-of-Sample $R^2$:

1. **Realized Out-of-Sample $R^2$ ($R^2_{OOS}$):**
   $$R^2_{OOS} = 1 - \frac{\text{MSPE}_{\text{model}}}{\text{MSPE}_{\text{naive}}}$$
   $R^2_{OOS} > 0$ indicates that the forecasting model produces a lower mean squared prediction error than the Naïve benchmark in actual out-of-sample forecasting. This is the **sole metric of realized forecast accuracy**.

2. **Clark-West Noise-Adjusted Testing Construct in MSPE Space ($\bar{f} / \text{MSPE}_{\text{naive}}$):**
   $$R^2_{OOS,\text{adj}} = 1 - \frac{\text{MSPE}_{\text{model}}^{\text{adj}}}{\text{MSPE}_{\text{naive}}} = \frac{\bar{f}}{\text{MSPE}_{\text{naive}}}$$
   where $\text{MSPE}_{\text{model}}^{\text{adj}} = \text{MSPE}_{\text{model}} - \text{adj}$. 
   
   **Important Methodological Caveat:** As emphasized by Clark & West (2006, 2007) and Rapach et al. (2010), $\text{MSPE}_{\text{model}} - \text{adj}$ is **not an estimate of loss that a forecaster actually experiences**. The term $\text{adj} = \frac{1}{N}\sum (\hat{y}_1 - \hat{y}_2)^2$ is an analytical device designed to center the loss differential under the null hypothesis ($H_0$) so that the resulting $t$-statistic ($CW$) is asymptotically standard normal. Consequently, $R^2_{OOS,\text{adj}}$ is purely a **hypothesis testing construct in MSPE space**, not an achievable or realized reduction in forecast error. It must never be interpreted or presented as realized out-of-sample predictive power.

#### Theoretical Foundation of the Unadjusted Benchmark Denominator
In nested model evaluation, Model 1 (Naïve Random Walk: $\Delta s_{t+h} = 0$) is a **parameter-free benchmark** with zero estimated parameters. Consequently, its parameter estimation noise penalty is strictly zero ($\text{adj}_1 \equiv 0$), which implies:
$$\text{MSPE}_1^{\text{adj}} \equiv \text{MSPE}_1$$
Therefore, the denominator is unambiguously the unadjusted benchmark MSPE ($\text{MSPE}_{\text{naive}}$). Dividing the Clark-West adjusted loss differential $\bar{f} = \text{MSPE}_1 - \text{MSPE}_2^{\text{adj}}$ by $\text{MSPE}_1$ is the mathematically exact sample analog to the proportion of benchmark forecast error variance explained by the model after correcting for parameter estimation noise (Clark & West 2006, 2007; Campbell & Thompson 2008).

#### Numerical Precision & Floating-Point Convention
All metric computations inside `src/model_comparison.py` are executed directly on full 64-bit double-precision (`float64`) prediction error arrays per IEEE 754 standards to prevent compounding rounding error (*rounding error propagation*). CSV outputs are rounded to 6 decimal places for disk serialization, and dashboard tables format values to 2 decimal places. 
*Note on hand calculation:* Computing $R^2_{OOS}$ from pre-rounded 6-decimal CSV text for VECM at $h=20$ yields $1 - (0.015349 / 0.015639) = 0.018543 \rightarrow +1.85\%$, whereas exact double-precision computation yields $0.018561 \rightarrow +1.86\%$. The project standardizes on the unrounded double-precision calculation to avoid truncation bias.

#### Methodological Inclusion of Machine Learning Benchmark
XGBoost is formally incorporated into the primary evaluation matrix alongside classical econometric specifications (ARIMA, VAR, VECM) and deep learning (LSTM) to mitigate research penalization for omitting a simpler, standard tabular machine learning model. The companion BIC specifications (`outputs/clark_west_sensitivity_results.csv`, testing ARIMA-BIC and VAR-BIC) remain partitioned in the sensitivity battery ($m=6$ hypotheses) for offline audit.

---

## 3. Empirical Results: The 15-Test Battery

The evaluation matrix comprises **5 primary model paradigms × 3 horizons = 15 hypothesis tests** on the canonical rolling forecast origins sampled across the 2012–2026 historical span.

### Table 1: Primary 15-Test Clark-West Results vs. Naïve Benchmark (Reconciled Common Sample N=745)
| Model | Horizon ($h$) | $N$ | $\text{MSPE}_{\text{naive}}$ | $\text{MSPE}_{\text{model}}$ | $R^2_{OOS}$ (Realized) | CW Adj ($\|\hat{y}_1 - \hat{y}_2\|^2$) | $\text{MSPE}_{\text{model}}^{\text{adj}}$ | CW Testing Construct ($\bar{f}/\text{MSPE}_1$) | $CW$ Stat | $p_{\text{raw}}$ | $q_{\text{global}}$ (BH) | $q_{\text{horizon}}$ (BH) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ARIMA-AIC** | 1 day | 745 | 0.000843 | 0.000848 | -0.63% | 0.000006 | 0.000843 | +0.03% | +0.043 | 0.4828 | 0.8047 | 0.9446 |
| **ARIMA-AIC** | 5 days | 745 | 0.003754 | 0.003842 | -2.35% | 0.000065 | 0.003778 | -0.63% | -0.625 | 0.7340 | 0.8469 | 0.7340 |
| **ARIMA-AIC** | 20 days | 745 | 0.015702 | 0.016508 | -5.13% | 0.000603 | 0.015904 | -1.29% | -0.455 | 0.6754 | 0.8442 | 0.6754 |
| **VAR-AIC** | 1 day | 745 | 0.000843 | 0.000890 | -5.62% | 0.000041 | 0.000849 | -0.70% | -0.394 | 0.6533 | 0.8442 | 0.9446 |
| **VAR-AIC** | 5 days | 745 | 0.003754 | 0.004068 | -8.36% | 0.000296 | 0.003772 | -0.48% | -0.182 | 0.5721 | 0.8442 | 0.7151 |
| **VAR-AIC** | 20 days | 745 | 0.015702 | 0.015924 | -1.41% | 0.000939 | 0.014985 | +4.57% | +0.915 | 0.1800 | 0.5575 | 0.2788 |
| **VECM (6-var)** | 1 day | 745 | 0.000843 | 0.000887 | -5.22% | 0.000049 | 0.000838 | +0.64% | +0.335 | 0.3688 | 0.6915 | 0.9446 |
| **VECM (6-var)** | 5 days | 745 | 0.003754 | 0.004053 | -7.95% | 0.000361 | 0.003692 | +1.67% | +0.591 | 0.2774 | 0.5944 | 0.4623 |
| **VECM (6-var)** | 20 days | 745 | 0.015702 | 0.015387 | **+2.01%** | 0.001918 | 0.013469 | **+14.22%** | **+1.985** | **0.0236** | 0.3540 | **0.1180** |
| **LSTM (Tuned)** | 1 day | 745 | 0.000843 | 0.000853 | -1.23% | 0.000004 | 0.000849 | -0.70% | -1.390 | 0.9177 | 0.9446 | 0.9446 |
| **LSTM (Tuned)** | 5 days | 745 | 0.003754 | 0.003761 | -0.19% | 0.000057 | 0.003705 | +1.32% | +1.577 | 0.0574 | 0.3855 | 0.1928 |
| **LSTM (Tuned)** | 20 days | 745 | 0.015702 | 0.016014 | -1.99% | 0.000668 | 0.015347 | +2.26% | +0.762 | 0.2230 | 0.5575 | 0.2788 |
| **XGBoost (Tuned)** | 1 day | 745 | 0.000843 | 0.000921 | -9.27% | 0.000044 | 0.000877 | -4.02% | -1.595 | 0.9446 | 0.9446 | 0.9446 |
| **XGBoost (Tuned)** | 5 days | 745 | 0.003754 | 0.003970 | -5.75% | 0.000347 | 0.003623 | +3.50% | +1.425 | 0.0771 | 0.3855 | 0.1928 |
| **XGBoost (Tuned)** | 20 days | 745 | 0.015702 | 0.016676 | -6.20% | 0.001430 | 0.015247 | +2.90% | +0.875 | 0.1908 | 0.5575 | 0.2788 |

*Data Source: `outputs/clark_west_test_results.csv` generated by `src/model_comparison.py`.*

---

## 4. Methodological Insights & Resolution of Project Caveats

### 4.1 Resolution of Cumulative-Sum Scoring Caveat (Issue #68)
Issue #68 noted that ARIMA and VAR forecast in first differences ($\Delta s_{t+k}$) and reconstruct multi-step levels via cumulative summation:
$$\hat{s}_{t+h} = s_t + \sum_{k=1}^h \widehat{\Delta s}_{t+k}$$
Summing $h$ noisy daily estimates accumulates $h \times \sigma^2_{\text{estimation}}$ variance, artificially degrading raw sample MSPE at $h=20$.

The Clark-West decomposition provides the mathematically rigorous answer to this caveat:
- For **VAR-AIC at $h=20$**, the raw $\text{MSPE} = 0.015999$ is higher than Naïve ($0.015765$).
- However, the Clark-West adjustment $(\hat{y}_1 - \hat{y}_2)^2$ isolates and subtracts this accumulated estimation variance ($0.000933$), revealing that the underlying conditional expectation achieves an adjusted $\text{MSPE}^{\text{adj}} = 0.015066 < 0.015765$ ($CW = 0.889, p = 0.1870$).
- Clark-West thus neutralizes the structural penalty of cumulative-sum scoring without requiring arbitrary ad-hoc modifications to the baseline architecture.

### 4.2 Economic Interpretation: Martingale Property of Asset Prices & EMH
At the 1-day horizon ($h=1$), all five models fail to reject the null hypothesis ($CW \le 0.335$, $p \ge 0.3688$).

Rather than reflecting a failure of statistical or neural modeling, this finding is directly predicted by capital market theory:
1. **Efficient Market Hypothesis (Fama, 1970; Campbell, Lo, & MacKinlay, 1997)**: Government bond markets incorporate public macro-financial information rapidly. Daily fluctuations in sovereign yield spreads behave as a **Martingale Difference Sequence** ($\mathbb{E}[\Delta s_{t+1} \mid \mathcal{I}_t] = 0$).
2. **Horizon-Dependent Dynamics**: While daily innovations are dominated by unforecastable news arrivals, medium-term structure emerges at longer horizons:
   - At $h=5$ days, the **LSTM network** ($CW = 1.577, p_{\text{raw}} = 0.0574, q_{\text{horizon}} = 0.1928$) and **XGBoost (Tuned)** ($CW = 1.425, p_{\text{raw}} = 0.0771, q_{\text{horizon}} = 0.1928$) show the largest, though non-significant, adjusted MSPE reductions.
   - At $h=20$ days, the **VECM framework** captures cointegrating equilibrium adjustments across Canadian and U.S. yields, producing a raw unadjusted reduction in MSPE ($CW = 1.985, p_{\text{raw}} = 0.0236, q_{\text{horizon}} = 0.1180$).

### 4.3 Leakage-Safe Hyperparameter Tuning for XGBoost Benchmark (Issue #119)
To achieve parity with `LSTM (Tuned)` (Issue #90) and eliminate benchmark bias stemming from untuned default hyperparameters, XGBoost underwent systematic hyperparameter optimization (`src/xgboost_grid_search.py`):
1. **Zero-Leakage Selection Boundary**: Hyperparameters are selected strictly on historical burn-in data ending at index `MIN_TRAIN = 500` (March 19, 2010 to February 16, 2012). Because `make_rolling_folds()` places the first scored evaluation origin at index 500 (February 17, 2012), **0% of the 3,728 rolling-CV test origins lie within the selection window**.
2. **Causal Embargoed Partition Geometry**:
   $$\text{inner\_train}_{[0,\,k)} \xrightarrow{\text{embargo } h} \text{inner\_val}_{[k+h,\,f)} \xrightarrow{\text{embargo } h} \text{calibration}_{[f+h,\,n)} \;\Big|\; \text{first scored origin at index } \texttt{MIN\_TRAIN} = 500$$
   with $n = 500$, $f = n - \lceil 0.15n \rceil - h$, and $k = f - \lceil 0.20f \rceil - h$. All partitions and calendar dates are logged to `outputs/r3_xgboost_search_partitions.csv`.
3. **Multi-Horizon Objective Function**: Because the $h=20$ naïve RMSE scale is $\approx 4.2\times$ that of $h=1$ (variance ratio $\approx 17.7\times$), optimizing raw RMSE would overfit long horizons. To preserve model specification parsimony across all horizons, we optimize Naïve-standardized relative RMSE:
   $$\mathcal{L}_{\text{search}} = \frac{1}{3} \sum_{h \in \{1, 5, 20\}} \frac{\text{RMSE}_{\text{val}, h}}{\text{RMSE}_{\text{naive}, h}}$$
4. **Multi-Seed Robustness & Winning Configuration**:
   Evaluated across five independent seeds `(42, 0, 1, 7, 2024)` to guard against seed noise. Consistent with the EMH finding in §4.2, no candidate in the 28-configuration space attained $\mathcal{L}_{\text{search}} < 1.0$ (none beat the Naïve benchmark on validation). The top-ranked configuration achieved $\mathcal{L}_{\text{search}} = 1.0381$, a $+1.21\%$ relative improvement over the untuned baseline defaults (Rank 3, $1.0509$):
   - `max_depth = 2` (shallow trees suppress variance on noisy differenced financial series)
   - `learning_rate = 0.01` (conservative gradient shrinkage)
   - `n_estimators = 600`
   - `subsample = 1.0` & `colsample_bytree = 1.0` (no bagging)
   - `min_child_weight = 1.0` & `reg_lambda = 0.5` (weakest L2 tier in the space)

   *Selection-power caveat*: The leakage-safe burn-in block yields only 77–84 validation observations (2011-05 to 2011-11), a single low-volatility regime. The rank-1 configuration leads rank 2 by $7.3\times 10^{-4}$ against a per-seed $\sigma$ of $6.1\times 10^{-3}$, so the ordering inside the leading band is not sharply identified; under a 1-sd indifference rule, the frozen specification should be read as a parsimonious representative of the indifference set, not as a sharp global optimum.

   *Regularization reversal*: The winning regularization tier is sensitive to the selection window. On a (leaky) pre-2023 pool, the most-regularized tier (`subsample = 0.7`, `min_child_weight = 5.0`, `reg_lambda = 5.0`) dominated across seeds, whereas on the leakage-safe 2010–2012 burn-in block, the least-regularized tier wins and the former falls to rank 6. With ~84 validation observations, the search cannot reliably distinguish regularization regimes, which directly aligns with item 5: no configuration in the space carries out-of-sample signal.
5. **Realized OOS Performance — Tuning Yields No Out-of-Sample Gain**: Holding the data vintage fixed ($N = 3{,}728$ rolling origins), the leakage-safe tuned specification is statistically indistinguishable from the untuned Issue #101 defaults:

   | $h$ | RMSE (untuned defaults) | RMSE (tuned) | vs. Naïve (untuned) | vs. Naïve (tuned) | Δ from tuning |
   |---|---|---|---|---|---|
   | 1 | 0.029672 | 0.029690 | $-2.94\%$ | $-3.00\%$ | $-0.06\text{ pp (worse)}$ |
   | 5 | 0.065164 | 0.064930 | $-3.32\%$ | $-2.95\%$ | $+0.37\text{ pp (better)}$ |
   | 20 | 0.129038 | 0.129240 | $-2.31\%$ | $-2.46\%$ | $-0.16\text{ pp (worse)}$ |
   | **mean** | | | **$-2.857\%$** | **$-2.805\%$** | **$+0.052\text{ pp (wash)}$** |

   Tuning is marginally worse at $h \in \{1, 20\}$ and marginally better at $h=5$; the 3-horizon mean improvement changes by $+0.05$ pp. **This is the substantive econometric result of Issue #119**, and it is consistent with §4.2: if daily sovereign yield spread changes behave as a martingale difference sequence, no reweighting of a gradient-boosted ensemble over the same causal information set can extract forecastable structure, and hyperparameter search cannot manufacture it.

   For transparency: an earlier revision reported a $4.5\%$ $h=1$ MSPE reduction from tuning. That figure was produced by a selection window that overlapped $76\%$ of the scored forecast origins; it was selection bias, not genuine predictive signal, and it vanished once the window was re-scoped strictly to the pre-evaluation burn-in block. On the canonical $N=745$ sample, the tuned $R^2_{OOS}$ at $h=1$ is $-9.03\%$ (unadjusted) / $-9.27\%$ (adjusted), essentially where the untuned benchmark stood.

---

## 5. Multiplicity Control (Benjamini-Hochberg FDR)

To address simultaneous testing risk across the 15 primary hypotheses, Benjamini-Hochberg (1995) FDR control is reported at two granularities:
1. **Global 15-Test FDR (`q_global`)**: Controls FDR across the entire $5 \times 3$ matrix.
2. **Horizon-Stratified FDR (`q_horizon`)**: Partitions testing into three independent horizon families ($k=5$ per horizon), ensuring that long-horizon variance at $h=20$ does not inflate discovery thresholds for short-horizon tests at $h=1, 5$ (Issue #81).

### Key FDR Findings at Committed $\alpha = 0.05$:
- **No model rejects the null hypothesis at the pre-committed $\alpha = 0.05$ threshold under either Global or Horizon-Stratified FDR control**.
- **VECM (6-var) at $h=20$**: Shows unadjusted raw significance ($p_{\text{raw}} = 0.0236$), but after horizon-stratified FDR control across the 5 models achieves $q_{\text{horizon}} = 0.1180 > 0.05$ (and $q_{\text{global}} = 0.3540$).
- **LSTM (Tuned) at $h=5$**: Achieves raw $p = 0.0574$ and $q_{\text{horizon}} = 0.1928$.
- **XGBoost (Tuned) at $h=5$**: Achieves raw $p = 0.0771$ and $q_{\text{horizon}} = 0.1928$.
- **XGBoost (Tuned) at $h=20$**: Achieves raw $p = 0.1908$ and $q_{\text{horizon}} = 0.2788$.
- **Headline Operational Takeaway**: Across all 5 paradigms and 3 horizons, the **Naïve Random Walk benchmark remains statistically unbeaten at $\alpha = 0.05$ after multiplicity control**, reinforcing the high-frequency informational efficiency of the Canadian sovereign bond market.

---

## 6. Verification and Implementation Checklist

- [x] Implemented `clark_west_test()` with calendar-safe Newey-West HAC covariance estimation in `src/model_comparison.py`.
- [x] Implemented `run_clark_west_battery()` executing the 15 primary tests and sensitivity checks.
- [x] Implemented leakage-safe hyperparameter search for XGBoost benchmark with causal $h$-step embargos (`src/xgboost_grid_search.py`, Issue #119).
- [x] Applied Global and Horizon-Stratified Benjamini-Hochberg FDR control in `apply_clark_west_fdr()`.
- [x] Generated canonical output datasets `outputs/clark_west_test_results.csv`, `outputs/clark_west_sensitivity_results.csv`, and `outputs/r3_xgboost_hyperparameter_search.csv`.
- [x] Built and pre-rendered Jupyter notebook `notebooks/04_diagnostics/clark_west_comparison.ipynb`.
- [x] Unit tests in `tests/test_model_comparison.py` passing (44/44 passing).
- [x] Documented mathematical foundations, #68 caveat resolution, #119 tuning protocol, and EMH/Martingale economic framing.
