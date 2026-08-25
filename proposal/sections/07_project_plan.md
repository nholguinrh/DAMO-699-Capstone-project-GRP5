---
section: 07
title: Project Plan and Timeline
owner: Mitchel
reviewer: Nelson
status: draft
rubric_criteria: ["Analytical Approach and Project Plan"]
target_words: 250
last_rubric_score: null
---

## 7. Project Plan and Timeline

The project will follow a phased analytics lifecycle aligned with the course milestones and the dependencies among data preparation, modeling, evaluation, and communication. During Weeks 1 and 2, the team will finalize the problem definition, analytical objective, proposed data sources, methodology, and ethical considerations. The principal deliverable from this phase will be the completed proposal, submitted by July 17.

Weeks 3 to 5 will focus on data readiness. The team will extract and document the required series from the Bank of Canada, FRED, and Statistics Canada; construct the Bronze, Silver, and Gold data layers; align publication and observation dates; assess missingness; and engineer the proposed yield-spread, volatility, momentum, and policy-shock features. Exploratory analysis, stationarity testing, and implementation of the Naïve Random Walk benchmark will provide the preliminary analytical deliverable expected in GitHub by August 7.

During Weeks 6 to 8, the team will perform Johansen cointegration diagnostics and implement the ARIMA, VAR/VECM, and LSTM forecasting approaches using chronological validation and common 1-day, 5-day, and 20-day forecast horizons. Diagnostic testing, rolling- or expanding-window evaluation, RMSE and MAE comparison, and Clark-West tests against the common Naïve Random Walk benchmark will support an assessment of which modelling approach most reliably forecasts near-term changes in the Canadian 10y–2y yield spread. Forecast accuracy, statistical evidence relative to the benchmark, interpretability, stability, and implementation complexity will be considered together in the final assessment.

Weeks 9 and 10 will be dedicated to final synthesis. The comparative forecasting findings will inform the interactive Streamlit dashboard, operational recommendations, limitations, and final report. The presentation deck will be finalized by September 11, followed by submission of the report, dataset, and reproducible repository by September 13. Poster preparation and rehearsal will continue in Week 11.

Progress will be monitored through documented milestones, peer review, version control, and reproducibility checks to ensure that each phase produces an auditable deliverable before the subsequent stage begins.