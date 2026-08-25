---
section: 06
title: Expected Outcomes and Contributions
owner: Mitchel
reviewer: Nelson
status: draft
rubric_criteria: ["Analytical Approach and Project Plan"]
target_words: 250
last_rubric_score: null
---

## 6. Expected Outcomes and Contributions

Building on the analytical objective defined in Section 3, the project is expected to produce a reproducible framework for forecasting near-term changes in the slope of the Canadian yield curve. The first deliverable will be a documented data pipeline that extracts, aligns, cleans, and transforms macro-financial time series from the Bank of Canada, the Federal Reserve Bank of St. Louis, and Statistics Canada. The pipeline will produce an analysis-ready dataset with documented provenance, transformations, and temporal-alignment procedures.

The project’s academic contribution will be empirical evidence on the relative forecasting performance of different modelling paradigms applied to the Canadian 10-year minus 2-year yield spread. The analysis will compare a Naïve Random Walk benchmark, ARIMA, VAR/VECM, and LSTM across common 1-day, 5-day, and 20-day forecast horizons.

Its methodological contribution will be a controlled comparison using aligned evaluation periods, chronological validation, consistent RMSE and MAE metrics, and Clark-West tests against the common Naïve Random Walk benchmark. The comparison will assess whether additional statistical, multivariate, or neural-network complexity produces sufficiently reliable forecasting improvements to justify differences in model stability, interpretability, and implementation complexity.

The practical contribution will be an interactive Streamlit dashboard presenting the yield-curve slope, Canada–U.S. yield differentials, rolling volatility, policy-rate changes, and model forecasts. The dashboard will translate the analytical findings into an accessible academic decision-support tool for fixed-income analysis while clearly communicating model uncertainty, comparative forecast performance, and data limitations.