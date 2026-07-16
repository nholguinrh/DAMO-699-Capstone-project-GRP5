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

This project adopts a fully programmatic, API-driven data acquisition strategy rather than relying on static downloaded files. Sourcing data through institutional web services ensures reproducibility, supports automated pipeline refreshes, and mirrors the data-engineering workflows used by quantitative research teams in institutional investment environments. Three complementary public data sources provide the macroeconomic and financial time-series data required to model yield curve dynamics under the analytical framework established in Section 3.

### 4.1 Primary Source: Bank of Canada — Valet Web Services

The Bank of Canada's Valet API serves as the primary data source (Bank of Canada, 2024). This public REST interface requires no registration or authentication and delivers version-controlled JSON endpoints suitable for long-term integration. The project retrieves three series groups spanning the period 2009 to present (mid-2026):

- **Monetary Policy Rate:** Target for the Overnight Rate (series `CBC20210`), representing the benchmark cost of short-term borrowing in the Canadian economy.
- **Benchmark Sovereign Bond Yields:** Government of Canada bond yields at 2-, 3-, 5-, 7-, 10-, and 30-year maturities (specifically series `BD.CDN.2YR.DQ.YLD`, `BD.CDN.3YR.DQ.YLD`, `BD.CDN.5YR.DQ.YLD`, `BD.CDN.7YR.DQ.YLD`, `BD.CDN.10YR.DQ.YLD`, and `BD.CDN.LONG.DQ.YLD`), enabling construction of the full term structure of interest rates.
- **Foreign Exchange Rate:** Daily USD/CAD exchange rate observations, capturing cross-border capital-flow dynamics that influence long-term yields. Because of the Bank of Canada's 2017 methodology change, the daily series is reconstructed by stitching together the legacy noon rate series `IEXE0101` (covering 2009-01-01 to 2017-04-28) and the active daily average rate series `FXUSDCAD` (covering 2017-05-01 to present).

The resulting dataset comprises approximately 4,300 daily observations across nine variables. A known limitation involves missing values on weekends and Canadian statutory holidays, which the data-engineering pipeline addresses through forward-fill imputation (as detailed in Section 5). To mitigate the operational risk of API outages or schema changes — inherent to any public web service without a formal service-level agreement — the pipeline persists each extraction as a timestamped local JSON cache, enabling reproducible offline analysis.

### 4.2 Complementary Source: U.S. Federal Reserve — FRED API

The Federal Reserve Economic Data (FRED) API, maintained by the Federal Reserve Bank of St. Louis, provides the 10-Year U.S. Treasury Constant Maturity Rate (`DGS10`) and the Federal Funds Effective Rate (`DFF`) (Federal Reserve Bank of St. Louis, 2024). These series enable computation of the Canada–U.S. sovereign yield spread, a critical variable for modeling open-economy interest rate transmission and USD/CAD exchange rate dynamics within the VAR framework. The program acquires observations through the dedicated API endpoint (`api.stlouisfed.org`) using a free registered API key rather than standard web downloads, ensuring reliable connection routing. Because U.S. and Canadian market holiday calendars differ, date alignment between the two sources constitutes a necessary preprocessing step.

### 4.3 Complementary Source: Statistics Canada — Consumer Price Index

Statistics Canada's Web Data Service provides the monthly Consumer Price Index (CPI), all items, not seasonally adjusted (Table 18-10-0004-01, queried programmatically via the `getDataFromVectorByReferencePeriodRange` service method using vector ID `41690973`) (Statistics Canada, 2024). CPI captures domestic inflation expectations, a fundamental macroeconomic driver of the yield curve slope and a theoretically motivated input to the econometric model. This source is publicly accessible without authentication. The primary limitation is the monthly reporting frequency versus the daily frequency of the other two sources, which requires temporal aggregation or merge-asof alignment during feature engineering. To prevent look-ahead bias, the merge key uses the Statistics Canada publication date — not the CPI reference month — ensuring that each trading day is paired only with inflation data that was publicly available at that point in time.

### 4.4 Dataset Summary

| Source | Access | Frequency | Period | Approx. Size | Key Variables |
|:-------|:-------|:----------|:-------|:-------------|:--------------|
| Bank of Canada — Valet API | Public, no key | Daily | 2009–2026 | ~4,300 × 9 | `overnight_rate`, `yield_2y`…`yield_10y`, `yield_long`, `usdcad` |
| U.S. FRED API (`api.stlouisfed.org`) | Public, free key | Daily | 2009–2026 | ~4,300 × 2 | `us_treasury_10y`, `fed_funds_rate` |
| Statistics Canada Web Data Service | Public, no key | Monthly | 2009–2026 | ~210 × 1 | `cpi_all_items` |

These three sources are appropriate for the analytical objective because they collectively capture the monetary policy channel (overnight and fed funds rates), the term structure of interest rates (multi-maturity sovereign yields), the cross-border transmission mechanism (yield spreads and exchange rates), and the inflation expectations channel (CPI) — the four macroeconomic dimensions central to the research question. All sources are publicly available, well-documented, and actively maintained by central statistical agencies, ensuring both feasibility and reproducibility of the analytical pipeline.
