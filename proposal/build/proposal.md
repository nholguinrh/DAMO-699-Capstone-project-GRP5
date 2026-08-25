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

This capstone project examines the dynamics of the Canadian yield curve using publicly available macro-financial data provided by the Bank of Canada. The yield curve is a fundamental indicator in fixed-income analysis because its shape reflects market expectations regarding interest rates, inflation, and future economic activity.

Situated at the intersection of data engineering and time-series analytics, this study employs the Bank of Canada's Valet API to develop a reproducible data pipeline for collecting, preparing, and analyzing observations of the overnight policy rate, benchmark Government of Canada bond yields, and the USD/CAD exchange rate. These data series are used to investigate whether monetary policy and exchange-rate movements contribute to explaining and forecasting changes in the slope of the yield curve.

The project is particularly relevant within the Canadian financial context, where fixed-income instruments and sovereign bond yields are closely monitored by institutional investors, risk managers, and policy analysts. As a master's capstone in data analytics, this study provides a practical framework for demonstrating competencies in API-based data acquisition, temporal data preparation, econometric modeling, and analytical dashboard design using publicly available government data.

## 2. Problem Definition

Accurate interpretation of monetary policy, sovereign bond yields, and exchange-rate movements is essential for managing interest-rate risk and understanding changes in the Canadian fixed-income market. However, these indicators are released on different schedules and at different frequencies, which makes them difficult to align and analyze consistently. As a result, yield-curve analysis is often fragmented, time-consuming, and difficult to reproduce.

The problem is relevant because the Government of Canada yield curve is a benchmark for asset pricing, portfolio allocation, and monetary policy assessment. Open data is especially valuable in this context because it allows the full workflow to be verified, reproduced, and reviewed without the barriers associated with proprietary financial datasets.

The primary beneficiaries of a solution to this problem are fixed-income analysts, portfolio managers, risk managers, and monetary-policy watchers who depend on yield-curve information to price instruments, manage duration exposure, assess interest-rate risk, and interpret the Bank of Canada’s policy stance.

This project addresses that challenge by developing a reproducible data analytics framework that automates the collection, preparation, and analysis of publicly available macro-financial data from the Bank of Canada. The goal is to explain and forecast changes in the Canadian yield curve using a transparent and repeatable analytical process.

## 3. Analytical Objective 

The primary objective of this project is to develop a reproducible data analytics pipeline that integrates publicly available macro-financial data and evaluates the extent to which different time-series modeling paradigms can explain and forecast the slope of the Canadian yield curve. The study will conduct a controlled model comparison incorporating a naïve Random Walk benchmark floor, an Autoregressive Integrated Moving Average (ARIMA) univariate statistical model, a Vector Autoregression (VAR) framework — with a conditional Vector Error Correction Model (VECM) extension if Johansen tests detect cointegration — to capture multivariate linear interactions among monetary policy rates, sovereign bond yields, inflation, and exchange rates, and a Long Short-Term Memory (LSTM) recurrent neural network to evaluate whether nonlinear temporal dependencies enhance predictive power. Model performance will be evaluated using out-of-sample forecast accuracy, such as RMSE and MAE, alongside the interpretability and stability of estimated relationships, in order to determine which approach offers the best trade-off between predictive accuracy and explanatory transparency for yield-curve analysis. 

### 3.1 Research Question 

Which modelling approach (A Naïve-Random-Walk, ARIMA, VAR/VECM, Long Short-Term Memory (LSTM) Neural Network) most reliably forecasts near-term (1–20 day) changes in the Canadian 10y–2y yield spread for use in an operational decision?
 

## 4. Proposed Data Sources

This project adopts a fully programmatic, API-driven data acquisition strategy rather than relying on static downloaded files. Sourcing data through institutional web services ensures reproducibility, supports automated pipeline refreshes, and mirrors the data-engineering workflows used by quantitative research teams in institutional investment environments. Three complementary public data sources provide the macroeconomic and financial time-series data required to model yield curve dynamics under the analytical framework established in Section 3.

### 4.1 Primary Source: Bank of Canada — Valet Web Services

The Bank of Canada's Valet API serves as the primary data source (Bank of Canada, 2024). This public REST interface requires no registration or authentication and delivers version-controlled JSON endpoints suitable for long-term integration. The project retrieves three series groups spanning the period January 2, 2009 through June 30, 2026:

- **Monetary Policy Rate:** Target for the Overnight Rate (series `CBC20210`), representing the benchmark cost of short-term borrowing in the Canadian economy.
- **Benchmark Sovereign Bond Yields:** Government of Canada bond yields at 2-, 3-, 5-, 7-, 10-, and 30-year maturities (specifically series `BD.CDN.2YR.DQ.YLD`, `BD.CDN.3YR.DQ.YLD`, `BD.CDN.5YR.DQ.YLD`, `BD.CDN.7YR.DQ.YLD`, `BD.CDN.10YR.DQ.YLD`, and `BD.CDN.LONG.DQ.YLD`), enabling computation of the yield curve slope (10-year minus 2-year spread) and related cross-maturity dynamics central to the analytical objective.
- **Foreign Exchange Rate:** Daily USD/CAD exchange rate observations, capturing cross-border capital-flow dynamics that influence long-term yields. Because of the Bank of Canada's 2017 methodology change, the daily series is reconstructed by stitching together the legacy noon rate series `IEXE0101` (covering 2009-01-01 to 2017-04-28) and the active daily average rate series `FXUSDCAD` (covering 2017-05-01 to present).

The resulting dataset comprises approximately 4,300 daily observations across eight analytical variables (plus the observation date). A known limitation involves missing values on weekends and Canadian statutory holidays, which the data-engineering pipeline addresses through forward-fill imputation (as detailed in Section 5). To mitigate the operational risk of API outages or schema changes — inherent to any public web service without a formal service-level agreement — the pipeline persists each extraction as a timestamped local JSON cache, enabling reproducible offline analysis.

### 4.2 Complementary Source: U.S. Federal Reserve — FRED API

The Federal Reserve Economic Data (FRED) API, maintained by the Federal Reserve Bank of St. Louis, provides the 10-Year U.S. Treasury Constant Maturity Rate (`DGS10`) and the Federal Funds Effective Rate (`DFF`) (Federal Reserve Bank of St. Louis, 2024). These series enable computation of the Canada–U.S. sovereign yield spread, a critical variable for modeling open-economy interest rate transmission and USD/CAD exchange rate dynamics within the VAR framework. The program acquires observations through the dedicated API endpoint (`api.stlouisfed.org`) using a free registered API key stored as an environment variable outside the code repository, ensuring both reliable connection routing and secure credential management. Because U.S. and Canadian market holiday calendars differ, date alignment between the two sources constitutes a necessary preprocessing step.

### 4.3 Complementary Source: Statistics Canada — Consumer Price Index

Statistics Canada's Web Data Service provides the monthly Consumer Price Index (CPI), all items, not seasonally adjusted (Table 18-10-0004-01, queried programmatically via the `getDataFromVectorByReferencePeriodRange` service method using vector ID `41690973`) (Statistics Canada, 2024). The pipeline retrieves the raw CPI index level (base year 2002 = 100); the appropriate stationarity transformation — log-differencing to approximate monthly inflation — is applied during the diagnostics stage described in Section 5. CPI captures domestic inflation expectations, a fundamental macroeconomic driver of the yield curve slope and a theoretically motivated input to the econometric model. This source is publicly accessible without authentication. The primary limitation is the monthly reporting frequency versus the daily frequency of the other two sources, which requires temporal aggregation or merge-asof alignment during feature engineering. To prevent look-ahead bias, the merge key uses the Statistics Canada publication date — not the CPI reference month — ensuring that each trading day is paired only with inflation data that was publicly available at that point in time.

### 4.4 Dataset Summary

| Source | Access | Frequency | Period | Approx. Size | Key Variables |
|:-------|:-------|:----------|:-------|:-------------|:--------------|
| Bank of Canada — Valet API | Public, no key | Daily | Jan 2009 – Jun 2026 | ~4,300 × 8 | `overnight_rate`, `yield_2y`, `yield_3y`, `yield_5y`, `yield_7y`, `yield_10y`, `yield_long`, `usdcad` |
| U.S. FRED API (`api.stlouisfed.org`) | Public, free key | Daily | 2009–2026 | ~4,300 × 2 | `us_treasury_10y`, `fed_funds_rate` |
| Statistics Canada Web Data Service | Public, no key | Monthly | 2009–2026 | ~210 × 1 | `cpi_all_items` |

These three sources are appropriate for the analytical objective because they collectively capture the monetary policy channel (overnight and fed funds rates), the term structure of interest rates (multi-maturity sovereign yields), the cross-border transmission mechanism (yield spreads and exchange rates), and the inflation expectations channel (CPI) — the four macroeconomic dimensions central to the research question. All sources are publicly available, well-documented, and actively maintained by central statistical agencies, ensuring both feasibility and reproducibility of the analytical pipeline.

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

A Random Walk (naïve) model, which predicts tomorrow's yield as today's yield, serves as the non-parametric baseline floor; any proposed model must outperform this naïve forecast to demonstrate practical value. Forecasting accuracy is assessed across multiple horizons (1-day, 5-day, and 20-day ahead) using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). To determine whether differences in predictive performance among the ARIMA univariate baseline, VAR/VECM multivariate models, LSTM neural network, and naïve benchmark floor are statistically significant rather than attributable to sampling variation, the Clark West test is applied (reference pending). This comparative evaluation directly addresses the research question by identifying whether multivariate macroeconomic information or nonlinear deep learning structures provide statistically significant forecasting gains over univariate statistical baselines and naïve predictions under evolving monetary policy conditions.

This section does not commit to a final model selection; rather, it establishes a rigorous analytical direction that the project will refine during the exploratory data analysis and model development phases. Section 7 maps each analytical stage to the project timeline and team responsibilities, completing the analytics lifecycle from data acquisition through model deployment and stakeholder communication.

## 6. Expected Outcomes and Contributions

Building on the analytical objective defined in Section 3, the project is expected to produce a reproducible framework for examining and forecasting changes in the slope of the Canadian yield curve. The first deliverable will be a documented data pipeline that extracts, aligns, cleans, and transforms macro-financial time series from the Bank of Canada, the Federal Reserve Bank of St. Louis, and Statistics Canada. The pipeline will produce an analysis-ready dataset with documented provenance, transformations, and temporal alignment procedures.

The project’s academic contribution will be empirical evidence on the relationships among monetary-policy rates, sovereign bond yields, inflation, and the USD/CAD exchange rate. Specifically, the analysis will assess whether these variables provide explanatory or forecasting information for movements in the Canadian 10-year minus 2-year yield spread.

Its methodological contribution will be a controlled comparison of ARIMA, VAR/VECM, LSTM, and naïve random-walk forecasts using common prediction horizons, chronological validation, and consistent performance metrics. This comparison will help determine whether gains in forecasting accuracy justify differences in interpretability, stability, and model complexity.

The practical contribution will be an executive Power BI dashboard presenting the yield-curve slope, Canada–U.S. yield differentials, rolling volatility, policy-rate changes, and model forecasts. The dashboard will translate the analytical findings into an accessible academic decision-support tool for fixed-income analysis while clearly communicating model uncertainty and data limitations.

## 7. Project Plan and Timeline

The project will follow a phased analytics lifecycle aligned with the course milestones and the dependencies among data preparation, modeling, evaluation, and communication. During Weeks 1 and 2, the team will finalize the problem definition, analytical objective, proposed data sources, methodology, and ethical considerations. The principal deliverable from this phase will be the completed proposal, submitted by July 17.

Weeks 3 to 5 will focus on data readiness. The team will extract and document the required series from the Bank of Canada, FRED, and Statistics Canada; construct the Bronze, Silver, and Gold data layers; align publication and observation dates; assess missingness; and engineer the proposed yield-spread, volatility, momentum, and policy-shock features. Exploratory analysis, stationarity testing, and a naïve random-walk benchmark will provide the preliminary analytical deliverable expected in GitHub by August 7.

During Weeks 6 to 8, the team will perform Johansen cointegration diagnostics and implement ARIMA, VAR/VECM, and LSTM models using chronological validation and common forecast horizons. Diagnostic testing, rolling-window evaluation, and comparison through RMSE, MAE, and the Diebold–Mariano test will support a model-selection decision based on predictive accuracy, interpretability, and robustness.

Weeks 9 and 10 will be dedicated to final synthesis. The selected findings will inform the Power BI dashboard, practical recommendations, limitations, and final report. The presentation deck will be finalized by September 11, followed by submission of the report, dataset, and reproducible repository by September 13. Poster preparation and rehearsal will continue in Week 11.

Progress will be monitored through documented milestones, peer review, version control, and reproducibility checks to ensure that each phase produces an auditable deliverable before the subsequent stage begins.

## 8. Ethical Considerations

### Data Governance and Compliance

The project will use publicly available, aggregate macro-financial time series and will not collect personal, confidential, or identifiable information. Although it does not involve human participants, all team members will complete TCPS 2 certification as a standard UNF academic research requirement. Data provenance, access conditions, transformations, imputation procedures, and publication-date alignment will be documented. Sources will be attributed using APA 7 conventions, including the Bank of Canada’s Valet API (Bank of Canada, 2024).

### Responsible Communication of Results

Forecasts will be presented with uncertainty, validation results, and model limitations. Granger-based findings and SHAP explanations will not be interpreted as evidence of economic causation. Potential bias arising from missing observations, different publication frequencies, look-ahead risk, and structural market-regime changes will be assessed. The dashboard will be presented as an academic decision-support tool, not financial advice, and its outputs should not replace independent professional judgment and validation.

\newpage

## 9. References

Bank of Canada. (2024). *Valet API: Web services for retrieving Bank of Canada data*. https://www.bankofcanada.ca/valet/docs

Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, *74*(366), 427–431. https://doi.org/10.2307/2286348

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, *13*(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599

Estrella, A., & Hardouvelis, G. A. (1991). The term structure as a predictor of real economic activity. *The Journal of Finance*, *46*(2), 555–576. https://doi.org/10.1111/j.1540-6261.1991.tb02674.x

Federal Reserve Bank of St. Louis. (2024). *FRED: Federal Reserve Economic Data*. https://fred.stlouisfed.org/

Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, *37*(3), 424–438. https://doi.org/10.2307/1912791

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, *59*(6), 1551–1580. https://doi.org/10.2307/2938278

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Lütkepohl, H. (2005). *New introduction to multiple time series analysis*. Springer. https://doi.org/10.1007/978-3-540-27752-1

Statistics Canada. (2024). *Consumer Price Index, monthly, not seasonally adjusted* (Table 18-10-0004-01). https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401

