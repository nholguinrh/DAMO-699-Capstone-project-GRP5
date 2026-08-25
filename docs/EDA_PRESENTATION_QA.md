# Exploratory Data Analysis (EDA) — Oral Presentation Practice Q&A

This document consolidates the **Presentation Readiness Checks** and oral rehearsal questions originally created during the exploratory data analysis phase in [`notebooks/01_eda/eda.ipynb`](../notebooks/01_eda/eda.ipynb).

These practice questions and model answers are designed to prepare team members to defend data preprocessing choices, statistical properties, and econometric interpretations out loud during milestone presentations, committee reviews, and stakeholder Q&A.

---

## Table of Contents

1. [Section 2: Frequency Alignment & CPI Lag](#1-frequency-alignment--cpi-lag)
2. [Section 3: Missingness Map & Data Construction](#2-missingness-map--data-construction)
3. [Section 4: Yield Curve Overview & Slope Dynamics](#3-yield-curve-overview--slope-dynamics)
4. [Section 5: Marginal Distributions & Macro Regimes](#4-marginal-distributions--macro-regimes)
5. [Section 6: Correlation in Levels vs. First Differences](#5-correlation-in-levels-vs-first-differences)
6. [Section 7: ADF Stationarity Pre-Checks & Integration Orders](#6-adf-stationarity-pre-checks--integration-orders)
7. [Section 9: Whole-EDA Cross-Cutting Questions](#7-whole-eda-cross-cutting-questions)

---

## 1. Frequency Alignment & CPI Lag

*Source: `notebooks/01_eda/eda.ipynb` §2 (Aligning three different frequencies onto one daily frame)*

### Questions to Rehearse

1. **Why is CPI aligned onto the daily frame using `release_date` instead of `reference_month`? What would go wrong for backtesting if we used `reference_month` instead?**
2. **StatCan publishes CPI roughly 3 weeks after the reference month ends. Why does that lag matter?**
3. **Why is YoY inflation computed with `pct_change(12)` on the monthly series instead of `pct_change(252)` on the daily forward-filled series? What specifically breaks with the daily version?**

<details>
<summary><strong>Show Model Answers</strong></summary>

- **Q1:** `reference_month` is the period the number *describes*; `release_date` is when it became public. Joining on `reference_month` would let the model "see" a CPI print before Statistics Canada actually published it — information leakage (look-ahead bias) that would make backtests look better than any real-time strategy could actually perform.
- **Q2:** Any date before the release date has to use the *previous* month's CPI print, not the current one — the 3-week lag is why a naive join creates a block of dates where the "latest" CPI a market participant could observe is older than you'd expect.
- **Q3:** `pct_change(252)` assumes exactly 252 business days between "now" and "one year ago," but the real count varies (250–253) due to holidays and leap years. On a step-function series that is flat for ~21 business days between releases, that drift can pull the comparison across a release-date boundary, producing an artificial jump. `pct_change(12)` on the monthly series compares the correct calendar month every time before forward-filling.

</details>

---

## 2. Missingness Map & Data Construction

*Source: `notebooks/01_eda/eda.ipynb` §3 (Missingness map)*

### Questions to Rehearse

1. **Name the three distinct causes of missing data in `daily`, in your own words.**
2. **Which of the three is missing *by design*, not because of a data collection problem?**
3. **Roughly what share of rows are affected by unexplained holiday-calendar gaps?**
4. **Why is it wrong to just call `daily.dropna()` and move on, instead of understanding each cause separately first?**

<details>
<summary><strong>Show Model Answers</strong></summary>

- **Q1:** 
  1. `usdcad_legacy` vs. `usdcad_current` split — Bank of Canada switched series methodology on 2017-05-01.
  2. `cpi_all_items` / `cpi_yoy` — null at the start of the sample before the first release and during the 12-month YoY warm-up window.
  3. Yield and rate columns — small scattered gaps from Canadian vs. US bank holiday calendar mismatches.
- **Q2:** The USD/CAD legacy/current split — each column is structurally null on one side of the 2017-05-01 methodology change by construction, not because of missing collection.
- **Q3:** ~180 rows out of ~4,552, or roughly 4% of total sample observations.
- **Q4:** A blind `dropna()` would silently discard ~4,550+ rows if applied before understanding that most of the "missingness" is either structural (USD/CAD split) or expected (CPI's monthly frequency) — you would lose almost the entire dataset to solve an issue that is largely expected structure.

</details>

---

## 3. Yield Curve Overview & Slope Dynamics

*Source: `notebooks/01_eda/eda.ipynb` §4 (Yield curve overview)*

### Questions to Rehearse

1. **Is the Canadian yield curve currently (last observation, 2026-06-30) inverted or normal? What number backs that up?**
2. **What is the full historical range of the 10y-2y spread (min/max) across the sample, and what does the negative extreme represent economically?**
3. **Trap Question: True or false — the curve was continuously inverted from mid-2022 through 2023. Look closely before answering.**
4. **Looking at the latest snapshot across all six tenors, is the curve strictly increasing from 2y to long (a "normal" upward-sloping curve), or is any segment flat/inverted?**
5. **Why is an inverted 10y-2y spread treated as a leading recession indicator? (One sentence, Expectations Hypothesis framing.)**

<details>
<summary><strong>Show Model Answers</strong></summary>

- **Q1:** Normal / upward-sloping — the 10y-2y spread is **+0.64%** on 2026-06-30, which is positive (not inverted).
- **Q2:** The historical range is roughly **-1.32% to +2.32%**. The negative extreme (-1.32%) represents the deepest inversion episode in the sample, where short-term yields (2y) priced meaningfully higher than long-term yields (10y).
- **Q3:** **False.** The spread crosses zero repeatedly — roughly three dozen separate in/out flips between mid-2022 and late 2024 — rather than staying inverted in one unbroken block.
- **Q4:** Strictly increasing on the latest observation date: 2y = 2.74%, 3y = 2.84%, 5y = 3.01%, 7y = 3.14%, 10y = 3.38%, Long = 3.77% — a textbook upward-sloping normal curve.
- **Q5:** Under the Expectations Hypothesis, an inverted curve signals that market participants expect future short-term rates to fall significantly (e.g., via central bank easing in response to economic deceleration), causing short-term rates to rise above long-term yields.

</details>

---

## 4. Marginal Distributions & Macro Regimes

*Source: `notebooks/01_eda/eda.ipynb` §5 (Distributions)*

### Questions to Rehearse

1. **What is the overnight rate's full range across the whole sample? What range did it stay within during the 2009–2021 ZIRP era specifically, versus after 2022?**
2. **The Bank of Canada's inflation target band is 1–3% (midpoint 2%). Where does the latest CPI YoY value in this dataset sit relative to that band?**
3. **CPI YoY ranges from about -0.4% to +8.1% across the full sample. Which real-world period does the ~8% reading most plausibly correspond to, and could you defend that from this notebook alone or would you need to pull in another source?**
4. **Why does `yield_spread_10y_2y` show a left tail instead of a symmetric bell shape?**

<details>
<summary><strong>Show Model Answers</strong></summary>

- **Q1:** Full range: 0.25% to 5.00%. During the ZIRP era (2009–2021), it fluctuated between 0.25% and 1.75%. Post-2022, it climbed up to 5.00% — the bimodal histogram reflects these two distinct policy regimes side by side.
- **Q2:** Latest CPI YoY ≈ 3.2%, slightly above the upper ceiling of the Bank of Canada's 1–3% target band.
- **Q3:** The ~8.1% peak corresponds to the 2022 global post-COVID inflationary surge (supply chain bottlenecks, energy shocks, and fiscal/monetary stimulus). The EDA notebook shows the timing and magnitude; defending the underlying economic mechanism requires citing external authoritative sources (e.g., Bank of Canada Monetary Policy Reports).
- **Q4:** The left tail reflects inversion regimes — periods where the yield curve inverts ($spread < 0$) are infrequent, asymmetric macroeconomic events rather than symmetrical Gaussian variations around the positive mean.

</details>

---

## 5. Correlation in Levels vs. First Differences

*Source: `notebooks/01_eda/eda.ipynb` §6 (Correlation — levels vs. first differences)*

### Questions to Rehearse

1. **`cpi_yoy` vs `yield_10y` correlates at ~0.41 in levels but drops to ~0.01 in first differences. Explain that gap in one sentence — what trap does it expose?**
2. **`yield_2y` vs `yield_10y` co-move at ~0.80 in first differences. Why would you *expect* two points on the same curve to move together that tightly?**
3. **`yield_10y` vs `us_treasury_10y` co-move at ~0.85 in first differences. What does that say about how connected the Canadian and US long ends are day-to-day?**
4. **Why is it wrong to correlate CPI's daily change (flat on ~96% of days) against daily yield changes and conclude "CPI has no relationship to yields"?**

<details>
<summary><strong>Show Model Answers</strong></summary>

- **Q1:** The 0.41 level correlation is a spurious trend artifact from two drifting non-stationary series; the ~0.01 differenced correlation confirms that daily fluctuations have near-zero contemporaneous correlation.
- **Q2:** Adjacent tenors on the term structure share a common underlying factor (the level factor driven by broad monetary policy and macroeconomic sentiment), so shocks systematically shift the entire curve in the same direction.
- **Q3:** A 0.85 differenced correlation demonstrates strong cross-border capital market integration and tightly synchronized sovereign bond pricing between Canada and the United States.
- **Q4:** Monthly CPI forward-filled to daily frequency creates a step function where ~96% of daily changes are exactly zero. Differenced daily correlation with volatile daily yields is mechanically attenuated to zero by construction (frequency mismatch), not because macroeconomic inflation is irrelevant to bond yields.

</details>

---

## 6. ADF Stationarity Pre-Checks & Integration Orders

*Source: `notebooks/01_eda/eda.ipynb` §7 (ADF stationarity pre-checks)*

### Questions to Rehearse

1. **In plain language, what does $p < 0.05$ mean in an Augmented Dickey-Fuller (ADF) test? What is the null hypothesis?**
2. **Every level series has $p > 0.05$ and every first-differenced series has $p \approx 0$. What does that combination tell you about the order of integration of these series?**
3. **`yield_spread_10y_2y` has $p = 0.48$ in levels. Why can't that number alone be used to claim the spread is stationary, or that the components are cointegrated? What test would actually be needed?**
4. **Based on this table, why does VAR baseline modeling need to run on differenced data rather than raw levels?**

<details>
<summary><strong>Show Model Answers</strong></summary>

- **Q1:** The null hypothesis ($H_0$) is that the time series contains a unit root (is non-stationary). A p-value $p < 0.05$ allows us to reject $H_0$ at the 5% significance level and conclude the series is covariance stationary.
- **Q2:** This confirms the textbook $I(1)$ (integrated of order 1) property: each series is non-stationary in levels but becomes stationary after taking first differences ($\Delta y_t$).
- **Q3:** A p-value of 0.48 fails to reject the unit root null for the spread in isolation. Cointegration formally requires that a linear combination of non-stationary $I(1)$ series is stationary $I(0)$, which must be tested via system methods (Johansen cointegration trace and maximum eigenvalue tests) in a VAR framework rather than an individual univariate ADF test.
- **Q4:** Fitting VAR models directly on non-stationary $I(1)$ levels without cointegration error-correction produces spurious regressions, non-standard asymptotic distributions, and unreliable inference (Granger & Newbold, 1974). Differencing to $I(0)$ guarantees stationarity.

</details>

---

## 7. Whole-EDA Cross-Cutting Questions

*Source: `notebooks/01_eda/eda.ipynb` §9 (Summary for the team)*

### Questions to Rehearse

1. **Give the 90-second version: what does this notebook establish about the data before downstream modeling starts, and why does the sequence of sections matter (missingness → yield curve → distributions → correlation → stationarity)?**
2. **"Is the Canadian yield curve inverted right now?" — answer with a specific number, not a vague impression.**
3. **"Why should I trust the differenced correlation heatmap over just eyeballing the levels chart?" — answer without saying "it's more accurate," say *why*.**
4. **What has this EDA explicitly *not* proven yet (cointegration rank, ARCH effects, ARIMA lag orders, PCA level/slope/curvature decomposition), and why is that a deliberate scoping decision rather than an omission?**
5. **If asked "why does the CPI calculation matter if it's just one column," what's the answer?**

<details>
<summary><strong>Show Model Answers</strong></summary>

- **Q1:** The notebook establishes data fidelity (identifying structural vs. calendar missingness), characterizes yield curve slope regimes (highlighting historical inversions), proves that all rate and yield series are $I(1)$ non-stationary, and distinguishes real day-to-day co-movement from spurious level trends. The order matters because data cleanliness must precede distributional inspection, and stationarity must precede correlation interpretations.
- **Q2:** No, the yield curve is not inverted — the 10y-2y spread is **+0.64%** as of June 30, 2026, and yields are strictly monotonically increasing across all six tenors (from 2.74% at 2y to 3.77% at Long).
- **Q3:** Pearson correlation on non-stationary level series is mechanically inflated by shared secular trends over multi-year periods. Differencing isolates true high-frequency co-movements and avoids confounding drift with economic co-dependence.
- **Q4:** Formal Johansen cointegration rank testing, ARCH-LM volatility clustering checks, ACF/PACF model order identification, and Nelson-Siegel/PCA yield curve decompositions are downstream econometric modeling steps that require specialized model specifications (VAR/VECM/ARIMA), scoped deliberately into subsequent analytical stages rather than preliminary EDA.
- **Q5:** `cpi_yoy` serves as a core macroeconomic feature across multivariate baselines (VAR, VECM, LSTM). Calculation errors (such as look-ahead bias or frequency misalignment jumps) directly distort econometric estimation, impulse response functions, and out-of-sample forecasts.

</details>
