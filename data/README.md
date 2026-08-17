# Data

- `raw/` — original, unmodified source datasets as obtained.
- `processed/` — cleaned and preprocessed datasets ready for analysis.

## Setup & API Credentials

For instructions on setting up API credentials (`.env`) and running the data collection pipeline, see [`docs/SETUP.md`](../docs/SETUP.md).


## Data dictionary

<!-- Placeholder. One row per variable, filled in once the dataset is chosen. -->

| Variable | Type | Description | Units / Values | Source |
| -------- | ---- | ----------- | -------------- | ------ |
| `date` | Date | Business day timestamp (YYYY-MM-DD) | Date | Bank of Canada / FRED trading calendar |
| `overnight_rate` | Float | Bank of Canada key policy overnight interest rate | Percent (%) | Bank of Canada (Valet API) |
| `yield_2y` | Float | Canadian Benchmark Bond Yield (2-Year) | Percent (%) | Bank of Canada (Valet API) |
| `yield_3y` | Float | Canadian Benchmark Bond Yield (3-Year) | Percent (%) | Bank of Canada (Valet API) |
| `yield_5y` | Float | Canadian Benchmark Bond Yield (5-Year) | Percent (%) | Bank of Canada (Valet API) |
| `yield_7y` | Float | Canadian Benchmark Bond Yield (7-Year) | Percent (%) | Bank of Canada (Valet API) |
| `yield_10y` | Float | Canadian Benchmark Bond Yield (10-Year) | Percent (%) | Bank of Canada (Valet API) |
| `yield_long` | Float | Canadian Long-term Benchmark Bond Yield (>10 Years) | Percent (%) | Bank of Canada (Valet API) |
| `usdcad` | Float | Stitched CAD/USD spot exchange rate (CAD per 1 USD) | Currency Ratio | Bank of Canada (Valet API) |
| `us_treasury_10y` | Float | US 10-Year Treasury Constant Maturity Rate (DGS10) | Percent (%) | St. Louis FRED |
| `fed_funds_rate` | Float | US Effective Federal Funds Rate (DFF) | Percent (%) | St. Louis FRED |
| `cpi_yoy` | Float | Canadian CPI YoY percentage change (release-date aligned) | Percent (%) | Statistics Canada (Table 18-10-0004-01) |
| `cpi_all_items` | Float | Canadian CPI All-items index level (release-date aligned) | Index (2002=100) | Statistics Canada (Table 18-10-0004-01) |
| `yield_spread_10y_2y` | Float | Primary target: 10Y minus 2Y Canadian yield spread | Percentage Points | Derived (`yield_10y - yield_2y`) |
| `yield_spread_10y_5y` | Float | Curve sub-spread: 10Y minus 5Y Canadian yield spread | Percentage Points | Derived (`yield_10y - yield_5y`) |
| `yield_spread_5y_2y` | Float | Curve sub-spread: 5Y minus 2Y Canadian yield spread | Percentage Points | Derived (`yield_5y - yield_2y`) |
| `d_yield_spread_10y_2y` | Float | Daily 1st difference of 10Y-2Y yield spread (stationary) | Percentage Points | Derived (`diff(1)`) |
| `d_yield_spread_10y_5y` | Float | Daily 1st difference of 10Y-5Y yield spread (stationary) | Percentage Points | Derived (`diff(1)`) |
| `d_yield_spread_5y_2y` | Float | Daily 1st difference of 5Y-2Y yield spread (stationary) | Percentage Points | Derived (`diff(1)`) |
| `d_overnight_rate` | Float | Daily 1st difference of BoC overnight rate | Percentage Points | Derived (`diff(1)`) |
| `d_yield_2y` | Float | Daily 1st difference of Canadian 2Y yield | Percentage Points | Derived (`diff(1)`) |
| `d_yield_5y` | Float | Daily 1st difference of Canadian 5Y yield | Percentage Points | Derived (`diff(1)`) |
| `d_yield_10y` | Float | Daily 1st difference of Canadian 10Y yield | Percentage Points | Derived (`diff(1)`) |
| `d_us_treasury_10y` | Float | Daily 1st difference of US 10Y Treasury yield | Percentage Points | Derived (`diff(1)`) |
| `d_fed_funds_rate` | Float | Daily 1st difference of US Federal Funds rate | Percentage Points | Derived (`diff(1)`) |
| `d_usdcad` | Float | Daily 1st difference of USDCAD exchange rate | Absolute Change | Derived (`diff(1)`) |
| `d_cpi_yoy` | Float | Daily 1st difference of Canadian CPI YoY percentage change | Percentage Points | Derived (`diff(1)`) |

## Data citation (APA 7)

Cite every dataset in the references, even public or self-collected ones.
Format:

> Author/Organization. (Year). *Title of dataset* [Data set]. Publisher. URL or DOI

Example:

> Statistics Canada. (2023). *Consumer Trends Dataset, 2022* [Data set]. Government of Canada. https://doi.org/xxxxx
