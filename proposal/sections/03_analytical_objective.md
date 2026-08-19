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

The primary objective of this project is to develop a reproducible data analytics pipeline that integrates publicly available macro-financial data and evaluates the extent to which different time-series modeling paradigms can explain and forecast the slope of the Canadian yield curve. The study will conduct a controlled model comparison incorporating a naïve Random Walk benchmark floor, an Autoregressive Integrated Moving Average (ARIMA) univariate statistical baseline, a Vector Autoregression (VAR) framework — with a conditional Vector Error Correction Model (VECM) extension if Johansen tests detect cointegration — to capture multivariate linear interactions among monetary policy rates, sovereign bond yields, inflation, and exchange rates, and a Long Short-Term Memory (LSTM) recurrent neural network to evaluate whether nonlinear temporal dependencies enhance predictive power. Model performance will be evaluated using out-of-sample forecast accuracy, such as RMSE and MAE, alongside the interpretability and stability of estimated relationships, in order to determine which approach offers the best trade-off between predictive accuracy and explanatory transparency for yield-curve analysis.

### 3.1 Research Question

To what extent do multivariate econometric models (VAR/VECM) and deep learning architectures (LSTM) improve multi-horizon out-of-sample forecasting accuracy for the Canadian yield curve slope (10-year minus 2-year spread) compared to univariate statistical baselines (ARIMA) and a naïve random-walk floor, and does the inclusion of macroeconomic transmission variables (overnight policy rate, USD/CAD exchange rate, and CPI inflation) provide statistically significant predictive gains over yield-only dynamics?

