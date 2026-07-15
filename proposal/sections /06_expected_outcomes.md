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
Building on the analytical objective defined in Section 3, the project is expected to produce an integrated and reproducible 
framework for examining changes in the Canadian yield curve using publicly available macro-financial time series. The first 
expected outcome is a documented pipeline that extracts, aligns, cleans, and transforms Bank of Canada data on the overnight 
policy rate, Government of Canada bond yields, and the USD/CAD exchange rate. This pipeline will create a consistent analytical 
dataset and support transparent replication of the study.

The analysis is also expected to generate empirical evidence on the temporal relationships among monetary policy, exchange-rate 
movements, and changes in the level and slope of the yield curve. Rather than assuming a specific relationship in advance, the 
project will quantify these dynamics and identify which variables provide useful explanatory or forecasting information.

A further outcome will be a consistent comparison of ARIMA, VAR, and LSTM models using the same forecast targets, chronological 
training and testing periods, and appropriate error metrics. This comparison will clarify the relative strengths, limitations, 
interpretability, and computational requirements of classical econometric and machine-learning approaches. The project will not 
presume that the most complex model will perform best.

The main practical contribution will be an executive Power BI dashboard that communicates yield-curve conditions, model outputs, 
and relevant risk indicators in an accessible format. Collectively, the dataset, reproducible code, model evaluation, and visual 
reporting will provide a transparent academic decision-support framework for analysts studying Canadian fixed-income markets, 
while also documenting limitations that must be considered before any real-world financial application.
