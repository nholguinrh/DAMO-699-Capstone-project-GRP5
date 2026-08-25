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

The analytical methodology follows a four-stage pipeline aligned with the analytics lifecycle: (1) data engineering and feature construction, (2) time-series diagnostics, (3) statistical, econometric, and machine-learning forecasting, and (4) model comparison and evaluation. This structure enables a controlled comparison among modelling paradigms of different complexity — a Naïve Random Walk benchmark, ARIMA, VAR/VECM, and LSTM — to determine which approach most reliably forecasts near-term changes in the Canadian 10-year minus 2-year yield spread over 1-, 5-, and 20-day horizons.

### 5.1 Data Engineering and Feature Construction

A medallion architecture organizes the data pipeline into three layers. The **Bronze layer** extracts raw observations from each API endpoint using Python HTTP requests with randomized exponential back-off to comply with rate limits. The **Silver layer** transforms nested JSON responses into relational DataFrames, aligns observation dates across the three sources, and handles missing observations arising from weekends, Canadian statutory holidays, release schedules, and cross-border calendar differences between Canadian and U.S. markets. Monthly CPI observations are aligned to the daily time index using their publication dates so that each forecasting date receives only information that would have been available at that point in time, reducing the risk of look-ahead bias.

The **Gold layer** prepares the variables and engineered features used by the forecasting models. The common modelling dataset includes the Canadian 10-year minus 2-year yield spread together with macro-financial predictors such as the Bank of Canada overnight policy rate, U.S. Treasury yields, the Federal Funds Rate, CPI inflation, and the USD/CAD exchange rate. Additional derived features may include rolling measures of momentum and volatility where appropriate to the selected model specification.

The Canadian 10-year minus 2-year Government of Canada bond spread is the primary forecasting target. The spread is widely used as an indicator of the shape of the yield curve and has historically been associated with changes in economic conditions (Estrella & Hardouvelis, 1991).

### 5.2 Time-Series Diagnostics

Prior to model estimation, variables undergo stationarity assessment using the Augmented Dickey-Fuller (ADF) test (Dickey & Fuller, 1979). Non-stationary series are transformed as required before estimation to reduce the risk of spurious relationships. For the Canadian 10y–2y spread and other rate-based variables, first differences are used when supported by the diagnostic results.

The VAR framework requires stationary inputs for valid estimation. In addition, a Johansen cointegration test evaluates whether long-run equilibrium relationships exist among the relevant level series (Johansen, 1991). If statistically significant cointegrating vectors are detected, a Vector Error Correction Model (VECM) is estimated as an extension of the multivariate framework. This allows the project to evaluate whether modelling both short-run adjustments and long-run equilibrium relationships improves forecast reliability relative to the alternative forecasting approaches.

Diagnostics are used to determine appropriate specifications and assess model validity rather than to establish causal relationships among the macro-financial variables.

### 5.3 Statistical, Econometric, and Machine-Learning Modeling

#### Naïve Random Walk

The Naïve Random Walk serves as the common benchmark for the forecasting exercise. Under this approach, the most recently observed yield-spread level is carried forward as the forecast, which is equivalent to predicting no change from the current spread. Because it requires no estimated parameters and represents a difficult benchmark for many financial time series, it provides a practical performance floor against which the additional complexity of the other modelling approaches can be evaluated.

#### Autoregressive Integrated Moving Average (ARIMA)

The project implements an ARIMA(p, d, q) model as the univariate statistical forecasting approach (Box & Jenkins, 1970). ARIMA uses the historical dynamics of the Canadian 10y–2y spread to model future movements through autoregressive and moving-average components after the stationarity transformation required by the data.

Candidate orders are evaluated programmatically using the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC). The selected ARIMA specification provides a parsimonious statistical approach that can be compared with the Naïve benchmark and with the more complex multivariate and neural-network models.

#### Vector Autoregression (VAR) and Vector Error Correction Model (VECM)

The multivariate econometric approach models dynamic relationships among the yield spread and the shared macro-financial feature set, including policy rates, sovereign yields, inflation, and the USD/CAD exchange rate. VAR lag order is selected using the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to balance model fit and complexity (Lütkepohl, 2005).

If Johansen testing identifies statistically significant cointegration, a VECM specification is estimated to incorporate long-run equilibrium information alongside short-run adjustments.

Additional multivariate diagnostics support model interpretation:

- **Impulse Response Functions (IRF):** examine how shocks to variables within the system propagate through subsequent periods.
- **Forecast Error Variance Decomposition (FEVD):** estimates the proportion of forecast-error variance associated with innovations in each variable.
- **Granger Causality tests:** assess whether lagged information from one series contributes predictive information for another series beyond its own history (Granger, 1969).

These diagnostics support interpretation of the multivariate model and assessment of its forecasting behaviour; they are not used to make causal claims.

#### Long Short-Term Memory (LSTM) Neural Network

The project implements a Long Short-Term Memory (LSTM) recurrent neural network as the nonlinear machine-learning forecasting approach (Hochreiter & Schmidhuber, 1997). LSTM networks are designed to learn temporal dependencies from sequential data through gated memory mechanisms and therefore provide a modelling paradigm that differs substantially from the linear statistical and econometric alternatives.

The LSTM receives the same aligned Gold-layer feature set used in the multivariate modelling framework. Temporal ordering is preserved during model development to avoid look-ahead bias. A rolling or expanding time-series validation framework is used rather than a randomly shuffled train/test split.

Given the moderate sample size, the LSTM uses a shallow architecture with dropout regularization and early stopping on validation loss to limit overfitting. SHapley Additive exPlanations (SHAP) values are used to support interpretability by quantifying feature-level contributions to model predictions (Lundberg & Lee, 2017).

### 5.4 Model Comparison and Evaluation

Forecast performance is evaluated at **1-day, 5-day, and 20-day horizons**, consistent with the near-term forecasting scope defined in the research question.

Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) provide the primary measures of out-of-sample forecast accuracy. The same Naïve Random Walk serves as the common benchmark for ARIMA, VAR/VECM, and LSTM so that each modelling paradigm can be assessed relative to a consistent operational baseline.

Statistical evidence of forecasting improvement relative to the Naïve benchmark is evaluated using the **Clark-West test** (Clark & West, 2007). Rather than conducting unrestricted pairwise significance tests among every possible model combination, the evaluation focuses on whether each candidate forecasting approach provides evidence of improved predictive performance relative to the common Naïve Random Walk benchmark.

The final assessment does not rely on a single metric. Forecast accuracy, statistical evidence relative to the Naïve benchmark, model stability, interpretability, and implementation complexity are considered together when determining which modelling approach most reliably supports the operational forecasting decision.

This comparative framework directly addresses the research question by evaluating whether the additional complexity introduced by ARIMA, VAR/VECM, or LSTM produces sufficiently reliable near-term forecasts to justify their use instead of the simple Naïve benchmark.

This section establishes the analytical framework for the project. Section 7 maps the forecasting, evaluation, interpretation, and communication stages to the project timeline and team responsibilities.