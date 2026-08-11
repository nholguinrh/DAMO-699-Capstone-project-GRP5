# Power BI Exploratory Data Analysis

## Purpose

This Power BI dashboard provides an interactive exploratory analysis of the Canadian yield curve, with particular focus on the 10-year minus 2-year Government of Canada yield spread.

The objective is to complement the Python-based EDA by providing a stakeholder-oriented visual view of:

- the historical behaviour of the 10Y–2Y spread,
- yield-curve inversions,
- relationships across bond maturities,
- and changes in rolling volatility over time.

## Data Source

The dashboard uses the processed Bank of Canada dataset generated locally through the selected Round 1 Path A data pipeline:

`data/processed/bank_of_canada_data.csv`

The processed dataset is regenerated locally and is not committed to the repository.

## Dashboard Pages

### 1. Yield Spread Overview

This page focuses on the Canadian 10Y–2Y yield spread over time.

It includes:

- the historical spread,
- a zero reference line used as the inversion threshold,
- the latest available spread,
- the latest observation date,
- the minimum observed spread,
- and an interactive date-range slicer.

Preliminary visual findings:

- The spread shows clear changes in behaviour across the sample period.
- Periods of positive and negative slope are visible.
- The most pronounced inversion occurs approximately during the 2022–2024 period.
- The minimum observed spread is approximately -1.32 percentage points.
- The latest available observation is June 30, 2026.
- The latest spread is approximately 0.64 percentage points.

### 2. Yield Curve Comparison

This page examines the behaviour of yields across maturities.

It includes:

- a direct comparison between the 2-year and 10-year Government of Canada yields,
- and a cross-maturity comparison including 2Y, 3Y, 5Y, 7Y, 10Y, and long-term yields.

Preliminary visual findings:

- The 2-year and 10-year yields generally move together but their relative positions change over time.
- Periods where the 2-year yield exceeds the 10-year yield correspond to negative values of the 10Y–2Y spread.
- Yields across maturities show strong visual co-movement.
- Relative differences across maturities change over time, producing changes in the slope and shape of the yield curve.

These observations motivate further statistical testing of multivariate relationships but do not by themselves establish stationarity, cointegration, or causality.

### 3. Rolling Volatility

This page examines how the variability of the 10Y–2Y spread changes over time.

Two rolling standard-deviation measures are shown:

- 20-observation rolling volatility,
- 60-observation rolling volatility.

The shorter window provides a more localized view of recent variability, while the longer window provides a smoother view of more persistent volatility conditions.

Preliminary visual findings:

- Spread volatility is not constant over time.
- Periods of relatively low volatility are interrupted by clusters of substantially higher volatility.
- Elevated volatility is visible around several major changes in yield-curve behaviour.

This suggests that forecast difficulty may vary across time and motivates evaluating forecasting models across different market or volatility regimes.

## Rolling Volatility Implementation Note

The initial rolling-volatility measures calculated moving windows dynamically inside the visual and exceeded the available Power BI query resources.

To improve efficiency, a referenced Power Query table named `Spread Volatility Data` was created containing:

- `date`
- `yield_spread_10y_2y`
- a sequential observation index

Rows with missing spread values were excluded before indexing.

The 20- and 60-observation rolling standard deviations are calculated using this sequential observation index. This avoids treating weekends, holidays, or missing market observations as artificial trading-day observations and significantly reduces computational cost.

## Screenshots

Dashboard screenshots are stored in:

`report/powerbi/screenshots/`

- `01_yield_spread_overview.png`
- `02_yield_curve_comparison.png`
- `03_rolling_volatility.png`

## Interpretation

The findings reported above are preliminary visual EDA observations. They are intended to guide subsequent statistical analysis and forecasting work.

They should not be interpreted as evidence of causality, stationarity, or cointegration without additional statistical testing.