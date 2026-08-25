---
section: 04
title: Proposed Data Sources
owner: Lerneir
reviewer: Mitchel
status: draft
rubric_criteria: [Data Sources and Feasibility]
target_words: 375
last_rubric_score: null
---

<!-- Section content starts here. Formal academic register, APA 7, active voice,
     no first person unless the course docs explicitly allow it. -->

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