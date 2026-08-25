\newpage

A Reproducible Analytics Framework for Forecasting the Canadian Fixed-Income Yield Curve to Support Decision-Makers and Policy Watchers

Course: DAMO 699 — Capstone Project

Program: Master of Data Analytics

Student Names: Giti [TBD — surname] · Mitchel [TBD — surname] · Lerneir [TBD — surname] · Nelson Holguin

Group Number: 5

Supervisor Name: [TBD — record per TIMELINE.md task "record supervisor name + meeting cadence"]

Institution: University of Niagara Falls

Submission Date: July 17, 2026

## 1. Introduction (Project Context)

This capstone project examines the Canadian yield curve using publicly available macro-financial data from the Bank of Canada and complementary public sources. The yield curve is a fundamental indicator in fixed-income analysis because its shape reflects market expectations regarding interest rates, inflation, and future economic activity.

Situated at the intersection of data engineering and time-series analytics, this study develops a reproducible data pipeline for collecting, preparing, and aligning observations of the Bank of Canada overnight policy rate, Government of Canada bond yields, U.S. interest rates, CPI inflation, and the USD/CAD exchange rate. These data support a controlled forecasting comparison focused on near-term changes in the Canadian 10-year minus 2-year yield spread.

The project evaluates four modelling approaches — a Naïve Random Walk benchmark, ARIMA, VAR/VECM, and a Long Short-Term Memory (LSTM) neural network — across 1-day, 5-day, and 20-day forecast horizons. The objective is to determine which modelling approach most reliably forecasts near-term yield-spread changes and whether additional model complexity provides sufficient forecasting improvement relative to the simple Naïve benchmark.

The project is particularly relevant within the Canadian financial context, where fixed-income instruments and sovereign bond yields are closely monitored by institutional investors, risk managers, and policy analysts. As a master's capstone in data analytics, the study demonstrates competencies in API-based data acquisition, temporal data preparation, time-series forecasting, model evaluation, and interactive analytical communication using reproducible public data.

## 2. Problem Definition

Reliable forecasting of near-term changes in the Canadian yield curve is relevant for fixed-income analysis, interest-rate risk management, and portfolio decision-making. However, the macro-financial variables associated with the yield curve are released at different frequencies and on different schedules, which creates challenges for temporal alignment, reproducibility, and consistent model evaluation.

The problem is particularly relevant because the Government of Canada yield curve is a benchmark for asset pricing, portfolio allocation, and monetary-policy assessment. Open public data is valuable in this context because it allows the analytical workflow to be verified, reproduced, and reviewed without relying on proprietary financial datasets.

The primary beneficiaries of a solution to this problem are fixed-income analysts, portfolio managers, risk managers, and monetary-policy watchers who use yield-curve information to assess interest-rate conditions, manage duration exposure, and support investment and risk-management decisions.

This project addresses the challenge by developing a reproducible forecasting framework that automates the collection, preparation, temporal alignment, and evaluation of publicly available macro-financial data. The analysis compares a Naïve Random Walk benchmark, ARIMA, VAR/VECM, and LSTM across common 1-day, 5-day, and 20-day horizons to determine which modelling approach most reliably forecasts near-term changes in the Canadian 10-year minus 2-year yield spread for use in an operational decision.

## 3. Analytical Objective

The primary objective of this project is to develop a reproducible forecasting and evaluation workflow for near-term changes in the Canadian 10-year minus 2-year government bond yield spread. The study compares four modelling approaches that represent increasing levels of statistical and computational complexity: a Naïve Random Walk benchmark, an Autoregressive Integrated Moving Average (ARIMA) univariate model, a multivariate Vector Autoregression (VAR) framework with a conditional Vector Error Correction Model (VECM) extension when cointegration is supported, and a Long Short-Term Memory (LSTM) neural network.

All models are evaluated within a common forecasting framework using the same prepared data, aligned evaluation periods, and forecast horizons of 1, 5, and 20 days. Predictive performance is assessed primarily through out-of-sample Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), with statistical comparison against the Naïve Random Walk benchmark using the Clark-West test. Model diagnostics, stability, interpretability, and implementation complexity are also considered when assessing the reliability of each modelling approach.

The purpose of this controlled comparison is not to isolate whether any individual macroeconomic variable provides incremental predictive value. Instead, the project evaluates which modelling paradigm most reliably forecasts near-term movements in the Canadian 10y–2y yield spread and whether the additional complexity of statistical, multivariate, or neural-network approaches produces meaningful forecasting improvements relative to a simple benchmark. The resulting evidence will support an operational decision regarding which forecasting approach is sufficiently accurate, stable, and interpretable for practical use.

### 3.1 Research Question

Which modelling approach (A Naïve-Random-Walk, ARIMA, VAR/VECM, Long Short-Term Memory (LSTM) Neural Network) most reliably forecasts near-term (1–20 day) changes in the Canadian 10y–2y yield spread for use in an operational decision?

## 4. Proposed Data Sources

This project adopts a fully programmatic, API-driven data acquisition strategy rather than relying on static downloaded files. Sourcing data through institutional web services supports reproducibility, automated pipeline refreshes, and transparent data provenance. Three complementary public data sources provide the Canadian and U.S. macro-financial time series required to construct the common forecasting dataset described in Section 3.

The data support the comparison of a Naïve Random Walk benchmark, ARIMA, VAR/VECM, and LSTM forecasting approaches. The Canadian 10-year minus 2-year yield spread is the primary forecasting target, while additional macro-financial variables provide the shared input set used by the multivariate and neural-network models and help define a common evaluation sample across modelling approaches.

### 4.1 Primary Source: Bank of Canada — Valet Web Services

The Bank of Canada's Valet API serves as the primary data source (Bank of Canada, 2024). This public REST interface requires no registration or authentication and provides structured JSON endpoints suitable for reproducible programmatic retrieval. The project retrieves three series groups spanning January 2009 through June 2026:

- **Monetary Policy Rate:** Target for the Overnight Rate (series `CBC20210`), representing the Bank of Canada's benchmark policy rate.

- **Benchmark Sovereign Bond Yields:** Government of Canada bond yields at 2-, 3-, 5-, 7-, 10-, and long-term maturities, including series `BD.CDN.2YR.DQ.YLD`, `BD.CDN.3YR.DQ.YLD`, `BD.CDN.5YR.DQ.YLD`, `BD.CDN.7YR.DQ.YLD`, `BD.CDN.10YR.DQ.YLD`, and `BD.CDN.LONG.DQ.YLD`. These series enable construction of the Canadian 10-year minus 2-year yield spread used as the primary forecasting target.

- **Foreign Exchange Rate:** Daily USD/CAD exchange-rate observations. Because of the Bank of Canada's 2017 methodology change, the daily series is reconstructed by combining the legacy noon-rate series `IEXE0101` with the current daily-average series `FXUSDCAD`.

The Bank of Canada data provide the core Canadian financial variables used throughout the forecasting workflow. Missing observations associated with weekends, statutory holidays, and differing market calendars are handled during the data-preparation stage using documented temporal-alignment procedures.

### 4.2 Complementary Source: U.S. Federal Reserve — FRED API

The Federal Reserve Economic Data (FRED) API, maintained by the Federal Reserve Bank of St. Louis, provides the 10-Year U.S. Treasury Constant Maturity Rate (`DGS10`) and the Federal Funds Effective Rate (`DFF`) (Federal Reserve Bank of St. Louis, 2024).

These series expand the common macro-financial feature set used by the VAR/VECM and LSTM models and allow the forecasting framework to incorporate information from U.S. interest-rate conditions. Observations are acquired programmatically through the FRED API using a registered API key stored outside the code repository. Because Canadian and U.S. market calendars differ, date alignment between the two sources forms an explicit preprocessing step.

### 4.3 Complementary Source: Statistics Canada — Consumer Price Index

Statistics Canada's Web Data Service provides the monthly Consumer Price Index (CPI), all items, not seasonally adjusted, from Table 18-10-0004-01 using vector ID `41690973` (Statistics Canada, 2024).

The pipeline retrieves the CPI index and derives year-over-year inflation (`cpi_yoy`) for use in the common macro-financial feature set. Because CPI is published monthly while the financial-market series are observed at daily frequency, CPI values are aligned to the forecasting timeline using their publication dates rather than their reference months.

This release-date alignment prevents look-ahead bias by ensuring that each forecasting observation contains only CPI information that would have been publicly available at that point in time. Missing or duplicated release dates are handled explicitly before the CPI series is merged with the daily modelling dataset.

### 4.4 Dataset Summary

| Source | Access | Frequency | Period | Approx. Size | Key Variables |
|:-------|:-------|:----------|:-------|:-------------|:--------------|
| Bank of Canada — Valet API | Public, no key | Daily | Jan 2009 – Jun 2026 | ~4,300 × 8 | `overnight_rate`, `yield_2y`, `yield_3y`, `yield_5y`, `yield_7y`, `yield_10y`, `yield_long`, `usdcad` |
| U.S. FRED API (`api.stlouisfed.org`) | Public, free key | Daily | 2009–2026 | ~4,300 × 2 | `us_treasury_10y`, `fed_funds_rate` |
| Statistics Canada Web Data Service | Public, no key | Monthly | 2009–2026 | ~210 × 1 | `cpi_all_items` |

Together, the three sources provide the target series and macro-financial inputs required for the project’s forecasting comparison. The Bank of Canada provides the primary Canadian yield and policy-rate data, FRED contributes complementary U.S. interest-rate information, and Statistics Canada provides inflation data. Their public availability, institutional provenance, and programmatic accessibility support the feasibility, transparency, and reproducibility of the analytical workflow.

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

## 6. Expected Outcomes and Contributions

Building on the analytical objective defined in Section 3, the project is expected to produce a reproducible framework for forecasting near-term changes in the slope of the Canadian yield curve. The first deliverable will be a documented data pipeline that extracts, aligns, cleans, and transforms macro-financial time series from the Bank of Canada, the Federal Reserve Bank of St. Louis, and Statistics Canada. The pipeline will produce an analysis-ready dataset with documented provenance, transformations, and temporal-alignment procedures.

The project’s academic contribution will be empirical evidence on the relative forecasting performance of different modelling paradigms applied to the Canadian 10-year minus 2-year yield spread. The analysis will compare a Naïve Random Walk benchmark, ARIMA, VAR/VECM, and LSTM across common 1-day, 5-day, and 20-day forecast horizons.

Its methodological contribution will be a controlled comparison using aligned evaluation periods, chronological validation, consistent RMSE and MAE metrics, and Clark-West tests against the common Naïve Random Walk benchmark. The comparison will assess whether additional statistical, multivariate, or neural-network complexity produces sufficiently reliable forecasting improvements to justify differences in model stability, interpretability, and implementation complexity.

The practical contribution will be an interactive Streamlit dashboard presenting the yield-curve slope, Canada–U.S. yield differentials, rolling volatility, policy-rate changes, and model forecasts. The dashboard will translate the analytical findings into an accessible academic decision-support tool for fixed-income analysis while clearly communicating model uncertainty, comparative forecast performance, and data limitations.

## 7. Project Plan and Timeline

The project will follow a phased analytics lifecycle aligned with the course milestones and the dependencies among data preparation, modeling, evaluation, and communication. During Weeks 1 and 2, the team will finalize the problem definition, analytical objective, proposed data sources, methodology, and ethical considerations. The principal deliverable from this phase will be the completed proposal, submitted by July 17.

Weeks 3 to 5 will focus on data readiness. The team will extract and document the required series from the Bank of Canada, FRED, and Statistics Canada; construct the Bronze, Silver, and Gold data layers; align publication and observation dates; assess missingness; and engineer the proposed yield-spread, volatility, momentum, and policy-shock features. Exploratory analysis, stationarity testing, and implementation of the Naïve Random Walk benchmark will provide the preliminary analytical deliverable expected in GitHub by August 7.

During Weeks 6 to 8, the team will perform Johansen cointegration diagnostics and implement the ARIMA, VAR/VECM, and LSTM forecasting approaches using chronological validation and common 1-day, 5-day, and 20-day forecast horizons. Diagnostic testing, rolling- or expanding-window evaluation, RMSE and MAE comparison, and Clark-West tests against the common Naïve Random Walk benchmark will support an assessment of which modelling approach most reliably forecasts near-term changes in the Canadian 10y–2y yield spread. Forecast accuracy, statistical evidence relative to the benchmark, interpretability, stability, and implementation complexity will be considered together in the final assessment.

Weeks 9 and 10 will be dedicated to final synthesis. The comparative forecasting findings will inform the interactive Streamlit dashboard, operational recommendations, limitations, and final report. The presentation deck will be finalized by September 11, followed by submission of the report, dataset, and reproducible repository by September 13. Poster preparation and rehearsal will continue in Week 11.

Progress will be monitored through documented milestones, peer review, version control, and reproducibility checks to ensure that each phase produces an auditable deliverable before the subsequent stage begins.

## 8. Ethical Considerations

### Data Governance and Compliance

The project will use publicly available, aggregate macro-financial time series and will not collect personal, confidential, or identifiable information. Although it does not involve human participants, all team members will complete TCPS 2 certification as a standard UNF academic research requirement. Data provenance, access conditions, transformations, imputation procedures, and publication-date alignment will be documented. Sources will be attributed using APA 7 conventions, including the Bank of Canada’s Valet API (Bank of Canada, 2024).

### Responsible Communication of Results

Forecasts will be presented with uncertainty, validation results, and model limitations. Granger-based findings and SHAP explanations will not be interpreted as evidence of economic causation. Potential bias arising from missing observations, different publication frequencies, look-ahead risk, and structural market-regime changes will be assessed. The dashboard will be presented as an academic decision-support tool, not financial advice, and its outputs should not replace independent professional judgment and validation.

\newpage

## 9. References

Bank of Canada. (2024). *Valet API: Web services for retrieving Bank of Canada data*. https://www.bankofcanada.ca/valet/docs

Box, G. E. P., & Jenkins, G. M. (1970). *Time series analysis: Forecasting and control*. Holden-Day.

Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics, 138*(1), 291–311. https://doi.org/10.1016/j.jeconom.2006.05.023

Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association, 74*(366), 427–431. https://doi.org/10.2307/2286348

Estrella, A., & Hardouvelis, G. A. (1991). The term structure as a predictor of real economic activity. *The Journal of Finance, 46*(2), 555–576. https://doi.org/10.1111/j.1540-6261.1991.tb02674.x

Federal Reserve Bank of St. Louis. (2024). *FRED: Federal Reserve Economic Data*. https://fred.stlouisfed.org/

Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica, 37*(3), 424–438. https://doi.org/10.2307/1912791

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica, 59*(6), 1551–1580. https://doi.org/10.2307/2938278

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Lütkepohl, H. (2005). *New introduction to multiple time series analysis*. Springer. https://doi.org/10.1007/978-3-540-27752-1

Statistics Canada. (2024). *Consumer Price Index, monthly, not seasonally adjusted* (Table 18-10-0004-01). https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401

