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

---

## 3. Empirical Results: The 12-Test Battery

The evaluation matrix comprises **4 primary model paradigms × 3 horizons = 12 hypothesis tests** on the canonical 745–750 business-day origin calendar ($2012\text{--}2026$).

### Table 1: Primary 12-Test Clark-West Results vs. Naïve Benchmark
| Model | Horizon ($h$) | $N$ | $\text{MSPE}_{\text{naive}}$ | $\text{MSPE}_{\text{model}}$ | CW Adj ($\|\hat{y}_1 - \hat{y}_2\|^2$) | $\text{MSPE}_{\text{model}}^{\text{adj}}$ | $CW$ Stat | $p_{\text{raw}}$ | $q_{\text{global}}$ (BH) | $q_{\text{horizon}}$ (BH) |
|---|---|---|---|---|---|---|---|---|---|---|
| **ARIMA-AIC** | 1 day | 750 | 0.000841 | 0.000847 | 0.000006 | 0.000841 | -0.080 | 0.5318 | 0.7872 | 0.8309 |
| **ARIMA-AIC** | 5 days | 750 | 0.003741 | 0.003828 | 0.000065 | 0.003764 | -0.615 | 0.7306 | 0.7970 | 0.7306 |
| **ARIMA-AIC** | 20 days | 750 | 0.015639 | 0.016423 | 0.000605 | 0.015818 | -0.402 | 0.6560 | 0.7872 | 0.6560 |
| **VAR-AIC** | 1 day | 750 | 0.000841 | 0.000887 | 0.000042 | 0.000846 | -0.314 | 0.6232 | 0.7872 | 0.8309 |
| **VAR-AIC** | 5 days | 750 | 0.003741 | 0.004050 | 0.000297 | 0.003753 | -0.123 | 0.5490 | 0.7872 | 0.7306 |
| **VAR-AIC** | 20 days | 750 | 0.015639 | 0.015832 | 0.000949 | 0.014883 | +0.970 | 0.1660 | 0.6230 | 0.2969 |
| **VECM (6-var)** | 1 day | 750 | 0.000841 | 0.000884 | 0.000050 | 0.000834 | +0.452 | 0.3257 | 0.6514 | 0.8309 |
| **VECM (6-var)** | 5 days | 750 | 0.003741 | 0.004033 | 0.000360 | 0.003673 | +0.645 | 0.2596 | 0.6230 | 0.5192 |
| **VECM (6-var)** | 20 days | 750 | 0.015639 | 0.015349 | 0.001910 | **0.013439** | **+1.973** | **0.0242** | 0.2904 | **0.0968** |
| **LSTM (Tuned)** | 1 day | 745 | 0.000843 | 0.000853 | 0.000004 | 0.000849 | -1.390 | 0.9177 | 0.9177 | 0.9177 |
| **LSTM (Tuned)** | 5 days | 745 | 0.003754 | 0.003761 | 0.000057 | **0.003705** | **+1.578** | **0.0573** | 0.3438 | 0.2292 |
| **LSTM (Tuned)** | 20 days | 745 | 0.015702 | 0.016014 | 0.000668 | 0.015347 | +0.763 | 0.2227 | 0.6230 | 0.2969 |

*Data Source: `outputs/clark_west_test_results.csv` generated by `src/model_comparison.py`.*

---

## 4. Methodological Insights & Resolution of Project Caveats

### 4.1 Resolution of Cumulative-Sum Scoring Caveat (Issue #68)
Issue #68 noted that ARIMA and VAR forecast in first differences ($\Delta s_{t+k}$) and reconstruct multi-step levels via cumulative summation:
$$\hat{s}_{t+h} = s_t + \sum_{k=1}^h \widehat{\Delta s}_{t+k}$$
Summing $h$ noisy daily estimates accumulates $h \times \sigma^2_{\text{estimation}}$ variance, artificially degrading raw sample MSPE at $h=20$.

The Clark-West decomposition provides the mathematically rigorous answer to this caveat:
- For **VAR-AIC at $h=20$**, the raw $\text{MSPE} = 0.015832$ is higher than Naïve ($0.015639$).
- However, the Clark-West adjustment $(\hat{y}_1 - \hat{y}_2)^2$ isolates and subtracts this accumulated estimation variance ($0.000949$), revealing that the underlying conditional expectation achieves an adjusted $\text{MSPE}^{\text{adj}} = 0.014883 < 0.015639$ ($CW = 0.970, p = 0.1660$).
- Clark-West thus neutralizes the structural penalty of cumulative-sum scoring without requiring arbitrary ad-hoc modifications to the baseline architecture.

### 4.2 Economic Interpretation: Martingale Property of Asset Prices & EMH
At the 1-day horizon ($h=1$), all four models fail to reject the null hypothesis ($CW \le 0.452$, $p \ge 0.3257$).

Rather than reflecting a failure of statistical or neural modeling, this finding is directly predicted by capital market theory:
1. **Efficient Market Hypothesis (Fama, 1970; Campbell, Lo, & MacKinlay, 1997)**: Government bond markets incorporate public macro-financial information rapidly. Daily fluctuations in sovereign yield spreads behave as a **Martingale Difference Sequence** ($\mathbb{E}[\Delta s_{t+1} \mid \mathcal{I}_t] = 0$).
2. **Horizon-Dependent Predictability**: While daily innovations are dominated by unforecastable news arrivals, predictable structure emerges at longer horizons:
   - At $h=5$ days, the **LSTM network** captures short-term nonlinear momentum ($CW = 1.578, p = 0.0573$).
   - At $h=20$ days, the **VECM framework** captures cointegrating equilibrium mean-reversion across Canadian and U.S. yields ($CW = 1.973, p = 0.0242, q_{\text{horiz}} = 0.0968$).

---

## 5. Multiplicity Control (Benjamini-Hochberg FDR)

To address the simultaneous testing risk across the 12 primary hypotheses, Benjamini-Hochberg (1995) FDR control is reported at two granularities:
1. **Global 12-Test FDR (`q_global`)**: Controls FDR across the entire $4 \times 3$ matrix.
2. **Horizon-Stratified FDR (`q_horizon`)**: Partitions testing into three independent horizon families ($k=4$ per horizon). This ensures that long-horizon variance at $h=20$ does not inflate discovery thresholds for short-horizon tests at $h=1, 5$ (Issue #81).

Under Horizon-Stratified FDR:
- **VECM (6-var) at $h=20$** achieves $q = 0.0968$, confirming significant predictive value at the 10% FDR threshold.
- **LSTM (Tuned) at $h=5$** achieves $q = 0.2292$ (near-significance at raw $p = 0.0573$).

---

## 6. Verification and Implementation Checklist

- [x] Implemented `clark_west_test()` with Newey-West HAC covariance estimation in `src/model_comparison.py`.
- [x] Implemented `run_clark_west_battery()` executing the 12 primary tests and sensitivity checks.
- [x] Applied Global and Horizon-Stratified Benjamini-Hochberg FDR control in `apply_clark_west_fdr()`.
- [x] Generated canonical output datasets `outputs/clark_west_test_results.csv` and `outputs/clark_west_sensitivity_results.csv`.
- [x] Built and pre-rendered Jupyter notebook `notebooks/04_diagnostics/clark_west_comparison.ipynb`.
- [x] Added 7 new unit tests in `tests/test_model_comparison.py` (25/25 passing).
- [x] Documented mathematical foundations, #68 caveat resolution, and EMH/Martingale economic framing.
