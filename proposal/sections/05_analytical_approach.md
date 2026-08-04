---
section: 05
title: Proposed Analytical Approach
owner: Lerneir
reviewer: Giti
status: draft
rubric_criteria: [Analytical Approach and Project Plan]
target_words: 375
last_rubric_score: null
---

<!-- Section content starts here. Formal academic register, APA 7, active voice,
     no first person unless the course docs explicitly allow it. -->

## 5. Proposed Analytical Approach

The analytical methodology follows a four-stage pipeline aligned with the analytics lifecycle: (1) data engineering and feature construction, (2) time-series diagnostics, (3) econometric and machine-learning modeling, and (4) model comparison and evaluation. This structure combines classical univariate and multivariate econometrics — incorporating both Vector Autoregression (VAR) and conditional Vector Error Correction Modeling (VECM) — with deep learning to comparatively assess which paradigm better captures yield curve dynamics under shifting monetary policy regimes, using the data sources described in Section 4.

### 5.1 Data Engineering and Feature Construction

A medallion architecture organizes the data pipeline into three layers. The **Bronze layer** extracts raw observations from each API endpoint using Python HTTP requests with randomized exponential back-off to comply with rate limits. The **Silver layer** transforms nested JSON responses into relational DataFrames, aligns observation dates across the three sources, and applies forward-fill imputation to handle missing values arising from weekends, Canadian statutory holidays, and cross-border calendar mismatches between Canadian and U.S. markets. Monthly CPI observations are merged to the daily time index using a merge-asof strategy that assigns each trading day the most recently published inflation reading.

The **Gold layer** engineers the econometric features that serve as model inputs:

- **Yield curve spreads:** The 10-year minus 2-year Government of Canada bond spread, a widely cited leading indicator of economic recessions (Estrella & Hardouvelis, 1991), and the Canada–U.S. 10-year sovereign spread, which captures cross-border interest rate differentials.
- **Momentum and volatility:** 20-day and 60-day rolling means and standard deviations of bond yields and the USD/CAD exchange rate.
- **Policy-shock indicators:** Binary variables marking Bank of Canada scheduled fixed announcement dates and the direction of rate changes.

### 5.2 Time-Series Diagnostics

Prior to model estimation, each variable undergoes stationarity assessment using the Augmented Dickey-Fuller (ADF) test (Dickey & Fuller, 1979). Non-stationary interest rate series are transformed using first-differencing (basis-point changes, $\Delta y_t = y_t - y_{t-1}$), the standard transformation for yields that may approach zero or turn negative; the USD/CAD exchange rate and CPI are transformed using log-differences where the logarithm is well defined. The VAR framework requires stationary inputs to avoid spurious regression. Additionally, a Johansen cointegration test screens for long-run equilibrium relationships among the level series (Johansen, 1991). If statistically significant cointegrating vectors are detected ($r \ge 1$), a Vector Error Correction Model (VECM) specification will be formally estimated alongside the stationary VAR, as cointegration implies that short-run deviations from long-run equilibrium contain predictive information that a purely differenced model discards.

### 5.3 Econometric and Machine-Learning Modeling

#### Autoregressive Integrated Moving Average (ARIMA)

To establish a univariate statistical baseline, the project implements an ARIMA(p, d, q) model (Box & Jenkins, 1970). Operating on the stationary yield-spread series identified in Section 5.2, ARIMA models future spread values as a linear combination of past spread observations (autoregressive terms, *p*) and past forecast errors (moving average terms, *q*), following *d* order differencing. Optimal order selection $(p, d, q)$ is determined programmatically using an automated Box-Jenkins procedure (Auto-ARIMA) guided by AIC and BIC minimization. The ARIMA model isolates purely autocorrelation-driven predictive signal, serving as the statistical benchmark against which multivariate macroeconomic features (VAR/VECM) and nonlinear neural network structures (LSTM) are evaluated.

#### Vector Autoregression (VAR) and Vector Error Correction Model (VECM)

The core econometric method models the dynamic interdependencies among the Bank of Canada's overnight rate, multi-maturity sovereign bond yields, the USD/CAD exchange rate, and CPI within a unified system. The optimal lag order *p* is selected using the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC) to balance model fit against overfitting risk (Lütkepohl, 2005). If Johansen tests establish cointegration, the system is specified as a VECM, augmenting the differenced VAR with an Error Correction Term ($\boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{Y}_{t-1}$) to model short-run adjustment toward long-run equilibrium.

Two standard multivariate interpretation tools complement the coefficient estimates:

- **Impulse Response Functions (IRF):** Trace how a one-standard-deviation shock to a single variable — for example, an unexpected change in the overnight rate — propagates across the system over a defined horizon, quantifying the speed and magnitude of monetary policy transmission to long-term yields.
- **Forecast Error Variance Decomposition (FEVD):** Identifies the proportion of forecast variance in each bond yield attributable to shocks from other variables, revealing which macroeconomic drivers explain the most variation in the term structure.

**Granger Causality tests** further validate directional lead-lag relationships, determining whether lagged values of the policy rate or exchange rate contain statistically significant predictive information for bond yields beyond what is captured by the yields' own history (Granger, 1969).

#### Long Short-Term Memory (LSTM) Neural Network

To evaluate whether nonlinear temporal patterns exist that linear VAR/VECM frameworks cannot capture, the project implements an LSTM recurrent neural network (Hochreiter & Schmidhuber, 1997). The LSTM architecture is well suited to financial time-series forecasting because its gating mechanism selectively retains or discards information across long sequences, mitigating the vanishing-gradient problem that limits standard recurrent networks. The LSTM receives the same Gold-layer feature set as the VAR/VECM model, ensuring a fair comparison. A rolling-window cross-validation strategy — rather than a single static train/test split — is employed to preserve the temporal ordering of observations and avoid look-ahead bias. Given the moderate sample size (~4,300 daily observations), the LSTM employs a shallow architecture — a single recurrent layer with dropout regularization and early stopping on validation loss — to mitigate overfitting risk inherent in training deep networks on limited financial time-series data. To support interpretability in the institutional investment context, SHapley Additive exPlanations (SHAP) values quantify each feature's marginal contribution to the LSTM's predictions (Lundberg & Lee, 2017), bridging the gap between black-box accuracy and decision-maker transparency.

### 5.4 Model Comparison and Evaluation

A Random Walk (naïve) model, which predicts tomorrow's yield as today's yield, serves as the non-parametric baseline floor; any proposed model must outperform this naïve forecast to demonstrate practical value. Forecasting accuracy is assessed across multiple horizons (1-day, 5-day, and 20-day ahead) using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). To determine whether differences in predictive performance among the ARIMA univariate baseline, VAR/VECM multivariate models, LSTM neural network, and naïve benchmark floor are statistically significant rather than attributable to sampling variation, the Diebold-Mariano test is applied (Diebold & Mariano, 1995). This comparative evaluation directly addresses the research question by identifying whether multivariate macroeconomic information or nonlinear deep learning structures provide statistically significant forecasting gains over univariate statistical baselines and naïve predictions under evolving monetary policy conditions.

This section does not commit to a final model selection; rather, it establishes a rigorous analytical direction that the project will refine during the exploratory data analysis and model development phases. Section 7 maps each analytical stage to the project timeline and team responsibilities, completing the analytics lifecycle from data acquisition through model deployment and stakeholder communication.
