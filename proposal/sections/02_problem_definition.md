## 2. Problem Definition

Reliable forecasting of near-term changes in the Canadian yield curve is relevant for fixed-income analysis, interest-rate risk management, and portfolio decision-making. However, the macro-financial variables associated with the yield curve are released at different frequencies and on different schedules, which creates challenges for temporal alignment, reproducibility, and consistent model evaluation.

The problem is particularly relevant because the Government of Canada yield curve is a benchmark for asset pricing, portfolio allocation, and monetary-policy assessment. Open public data is valuable in this context because it allows the analytical workflow to be verified, reproduced, and reviewed without relying on proprietary financial datasets.

The primary beneficiaries of a solution to this problem are fixed-income analysts, portfolio managers, risk managers, and monetary-policy watchers who use yield-curve information to assess interest-rate conditions, manage duration exposure, and support investment and risk-management decisions.

This project addresses the challenge by developing a reproducible forecasting framework that automates the collection, preparation, temporal alignment, and evaluation of publicly available macro-financial data. The analysis compares a Naïve Random Walk benchmark, ARIMA, VAR/VECM, and LSTM across common 1-day, 5-day, and 20-day horizons to determine which modelling approach most reliably forecasts near-term changes in the Canadian 10-year minus 2-year yield spread for use in an operational decision.