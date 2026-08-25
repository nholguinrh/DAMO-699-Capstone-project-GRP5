---
section: 03
title: Analytical Objective
owner: Nelson
reviewer: Giti
status: draft
rubric_criteria: ["Research Question / Analytical Objective"]
target_words: 375
last_rubric_score: null
---

## 3. Analytical Objective

The primary objective of this project is to develop a reproducible forecasting and evaluation workflow for near-term changes in the Canadian 10-year minus 2-year government bond yield spread. The study compares four modelling approaches that represent increasing levels of statistical and computational complexity: a Naïve Random Walk benchmark, an Autoregressive Integrated Moving Average (ARIMA) univariate model, a multivariate Vector Autoregression (VAR) framework with a conditional Vector Error Correction Model (VECM) extension when cointegration is supported, and a Long Short-Term Memory (LSTM) neural network.

All models are evaluated within a common forecasting framework using the same prepared data, aligned evaluation periods, and forecast horizons of 1, 5, and 20 days. Predictive performance is assessed primarily through out-of-sample Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), with statistical comparison against the Naïve Random Walk benchmark using the Clark-West test. Model diagnostics, stability, interpretability, and implementation complexity are also considered when assessing the reliability of each modelling approach.

The purpose of this controlled comparison is not to isolate whether any individual macroeconomic variable provides incremental predictive value. Instead, the project evaluates which modelling paradigm most reliably forecasts near-term movements in the Canadian 10y–2y yield spread and whether the additional complexity of statistical, multivariate, or neural-network approaches produces meaningful forecasting improvements relative to a simple benchmark. The resulting evidence will support an operational decision regarding which forecasting approach is sufficiently accurate, stable, and interpretable for practical use.

### 3.1 Research Question

Which modelling approach (A Naïve-Random-Walk, ARIMA, VAR/VECM, Long Short-Term Memory (LSTM) Neural Network) most reliably forecasts near-term (1–20 day) changes in the Canadian 10y–2y yield spread for use in an operational decision?